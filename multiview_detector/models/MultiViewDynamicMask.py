import torch
import torch.nn as nn

class MultiViewDynamicMask(nn.Module):
    def __init__(self,
                 num_views: int = 7,
                 init_thresholds: list = [0.1] * 7,
                 threshold_min: float = 0.0,
                 threshold_max: float = 0.2,
                 learnable_scale: bool = False):
        super().__init__()
        assert len(init_thresholds) == num_views, "初始阈值数量必须等于视角数"
        assert threshold_min < threshold_max, "阈值下限必须小于上限"

        self.num_views = num_views
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        # 将基准阈值直接定义为可训练参数
        self.base_thresholds = nn.ParameterList([
            self._init_sigmoid_param(t) for t in init_thresholds
        ])
        # 动态缩放系数
        self.scale = nn.Parameter(torch.tensor(1.0)) if learnable_scale else 0.5

    # 初始化函数
    def _init_sigmoid_param(self, init_value: float) -> nn.Parameter:
        clamped = torch.clamp(torch.tensor(init_value), self.threshold_min, self.threshold_max)     # 输入截断（Clamping），输入值在范围内，因此未被截断。
        sigmoid_space = (clamped - self.threshold_min) / (self.threshold_max - self.threshold_min)  # 归一化（Sigmoid Space），将输入线性映射到 [0, 1]
        return nn.Parameter(torch.logit(sigmoid_space))

    ############################################## 获取当前各视角基准阈值（不含动态偏移）
    def get_current_thresholds(self) -> torch.Tensor:
        return self.last_thresholds
    @property
    def last_thresholds(self) -> torch.Tensor:
        """动态属性：获取各视角基准阈值（无动态偏移）"""
        with torch.no_grad():
            return torch.stack([
                self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(t)
                for t in self.base_thresholds
            ])

    ############################################## 获取最后一次前向传播的动态阈值（需先执行前向计算）
    def get_last_dynamic_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._compute_dynamic_thresholds(x).squeeze()

    # 动态阈值计算（无需存储中间结果）
    def _compute_dynamic_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = x.shape                             # 1 7 128 120 360
        channel_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1,7,1,1,1]
        dynamic_offset = self.scale * channel_mean          # [1,7,1,1,1]
        thresholds = []
        for v in range(self.num_views):
            adjusted = self.base_thresholds[v] + dynamic_offset[:, v] # {tensor(1,1,1,1)}
            threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(adjusted)
            thresholds.append(threshold)
        return torch.stack(thresholds, dim=1) # [B,V,1,1,1]

    ############################################## 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = x.shape

        ######## thresholds = self._compute_dynamic_thresholds(x)
        channel_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1,7,1,1,1]
        dynamic_offset = self.scale * channel_mean          # [1,7,1,1,1]
        thresholds = []
        for v in range(V):
            # 动态调整后的阈值
            adjusted = self.base_thresholds[v] + dynamic_offset[:, v]                                               # {tensor(1,1,1,1)}
            threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(adjusted)    # {tensor(1,1,1,1)}
            thresholds.append(threshold)
        thresholds = torch.stack(thresholds, dim=1)  # [B,V,1,1,1]

        masks = []
        for v in range(V):
            x_v = x[:, v, :, :, :]                                          # {tensor(1,128,120,360)}
            energy = torch.mean(x_v, dim=1, keepdim=True)                   # {tensor(1,1,120,360)}
            mask_forward = (energy > thresholds[:, v]).float()              # {tensor(1,1,120,360)}
            mask_backward = torch.sigmoid((energy - thresholds[:, v]) * 10) # {tensor(1,1,120,360)}
            mask = mask_forward + (mask_backward - mask_backward.detach())  # {tensor(1,1,120,360)}
            masks.append(mask.expand(-1, C, -1, -1))
        masks = torch.stack(masks, dim=1)               # {tensor(1,7,128,120,360)} = {list: 7}
        weighted_features = x * masks                   # {tensor(1,7,128,120,360)} * {tensor(1,7,128,120,360)}

        residual_features = weighted_features + x       # {tensor(1,7,128,120,360)}

        return residual_features, masks
        # return torch.mean(residual_features, dim=1)   # {tensor(1,128,120,360)}




if __name__ == '__main__':
    print('################### 测试MultiViewDynamicMask')
    DynamicMask = MultiViewDynamicMask()
    print(DynamicMask)

    # 生成测试输入
    test_input = torch.randn([1, 6, 128, 120, 360])
    output = DynamicMask(test_input)  # 必须执行前向传播

    print('输出形状:', output.shape)  # 应为 [1, 128, 120, 360]

    print('\n################### 获取阈值并打印')
    base_thresholds = DynamicMask.get_current_thresholds()
    dynamic_thresholds = DynamicMask.get_last_dynamic_thresholds(test_input)
    for v in range(6):
        print(f"视角{v + 1} 基准阈值（不含动态调整）: {base_thresholds[v].item():.4f}, 实际应用阈值（含动态调整）: {dynamic_thresholds[v].item():.4f}")
