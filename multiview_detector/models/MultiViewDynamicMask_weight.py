import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewDynamicMask(nn.Module):
    def __init__(self,
                 num_views: int = 7,
                 init_thresholds: list = [0.03] * 7,
                 threshold_min: float = 0.0,
                 threshold_max: float = 0.1,
                 learnable_scale: bool = False):
        super().__init__()
        assert len(init_thresholds) == num_views, "初始阈值数量必须等于视角数"
        assert threshold_min < threshold_max, "阈值下限必须小于上限"

        # 配置参数
        self.num_views = num_views
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        # 将基准阈值直接定义为可训练参数
        self.base_thresholds = nn.ParameterList([
            self._init_sigmoid_param(t) for t in init_thresholds
        ])

        # 动态缩放系数
        self.scale = nn.Parameter(torch.tensor(0.5)) if learnable_scale else 0.5

        # 视角权重系数
        self.view_weights = None

    # 新增方法：获取当前各视角基准阈值（不含动态偏移）
    def get_current_thresholds(self) -> torch.Tensor:
        """ 直接返回基准阈值（不包含动态调整部分） """
        return self.last_thresholds

    def _init_sigmoid_param(self, init_value: float) -> nn.Parameter:
        """初始化方法保持不变"""
        clamped = torch.clamp(torch.tensor(init_value), self.threshold_min, self.threshold_max)
        sigmoid_space = (clamped - self.threshold_min) / (self.threshold_max - self.threshold_min)
        return nn.Parameter(torch.logit(sigmoid_space))

    @property
    def last_thresholds(self) -> torch.Tensor:
        """动态属性：获取各视角基准阈值（无动态偏移）"""
        with torch.no_grad():
            return torch.stack([
                self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(t)
                for t in self.base_thresholds
            ])

    # 新增方法：获取最后一次前向传播的动态阈值（需先执行前向计算）
    def get_last_dynamic_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """ 获取包含动态调整的阈值（需传入输入张量重新计算） """
        with torch.no_grad():
            return self._compute_dynamic_thresholds(x).squeeze()

    def _compute_dynamic_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """动态阈值计算（无需存储中间结果）"""
        B, V, C, H, W = x.shape
        channel_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [B,V,1,1,1]
        dynamic_offset = self.scale * channel_mean

        thresholds = []
        for v in range(self.num_views):
            # 动态调整后的阈值
            adjusted = self.base_thresholds[v] + dynamic_offset[:, v]
            threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(adjusted)
            thresholds.append(threshold)

        return torch.stack(thresholds, dim=1)  # [B,V,1,1,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """改进后的前向传播，包含视角贡献度加权"""
        B, V, C, H, W = x.shape
        thresholds = self._compute_dynamic_thresholds(x)  # [B,V,1,1,1]

        masks = []
        mask_ratios = []  # 存储各视角有效像素比例
        for v in range(V):
            x_v = x[:, v, :, :, :]
            energy = torch.mean(x_v, dim=1, keepdim=True)  # [B,1,H,W]

            # 生成二值化掩码（前向传播用）
            mask_forward = (energy > thresholds[:, v]).float()  # [B,1,H,W]
            # 生成梯度可传的掩码（反向传播用）
            mask_backward = torch.sigmoid((energy - thresholds[:, v]) * 10)
            mask = mask_forward + (mask_backward - mask_backward.detach())  # [B,1,H,W]

            # 记录有效像素比例，因为掩码是二值化的（0或1），所以均值即为有效像素的比例（仅用前向部分计算）
            mask_ratios.append(mask_forward.mean(dim=[1, 2, 3]))

            masks.append(mask.expand(-1, C, -1, -1))  # [B,C,H,W]

        # 计算各视角权重
        mask_ratios = torch.stack(mask_ratios, dim=1)  # [B,V]
        view_weights = F.softmax(mask_ratios, dim=1)  # [B,V]
        self.view_weights = view_weights
        # 加权融合
        masks = torch.stack(masks, dim=1)  # [B,V,C,H,W]
        weighted_features = x * masks
        # 残差连接
        residual_features = weighted_features + x  # [B,V,C,H,W]
        # 使用视角权重进行融合（需要扩展维度对齐）
        view_weights = view_weights.view(B, V, 1, 1, 1)  # [B,V,1,1,1]
        return torch.sum(residual_features * view_weights, dim=1)  # [B,C,H,W]


if __name__ == '__main__':
    print('################### 测试MultiViewDynamicMask')
    DynamicMask = MultiViewDynamicMask()
    print(DynamicMask)

    # 生成测试输入
    test_input = torch.randn([1, 7, 128, 120, 360])
    output = DynamicMask(test_input)  # 必须执行前向传播
    print('输出形状:', output.shape)  # 应为 [1, 128, 120, 360]
    print('视角权重: ',DynamicMask.view_weights)

    print('\n################### 获取阈值并打印')
    print("基准阈值（不含动态调整）:")
    base_thresholds = DynamicMask.get_current_thresholds()
    for v in range(7):
        print(f"视角{v + 1}: {base_thresholds[v].item():.4f}")

    print("\n实际应用阈值（含动态调整）:")
    dynamic_thresholds = DynamicMask.get_last_dynamic_thresholds(test_input)
    for v in range(7):
        print(f"视角{v + 1}: {dynamic_thresholds[v].item():.4f}")