import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = 'cpu'
##############################################
#########          fusion3          ##########
##############################################
def create_coord_map(img_size, with_r=False):
    H, W = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        grid_r = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, grid_r], dim=1)
    return ret

class fusion3(nn.Module):
    def __init__(self, num_cam, Rworld_shape, hidden_dim=128):
        super(fusion3, self).__init__()
        # 瓶颈器 nn.LeakyReLU(negative_slope=0.01) & nn.ReLU()
        self.coord_map = create_coord_map(np.array(Rworld_shape))
        self.bottleneck = nn.Sequential(nn.Conv2d(898, hidden_dim, kernel_size=3, padding=1), nn.ReLU())

        # 三组编码器
        # self.world_feat0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), )
        # self.world_feat1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.LeakyReLU(negative_slope=0.01))
        # self.world_feat2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=4, dilation=4), nn.LeakyReLU(negative_slope=0.01))

        # 三组采样器
        self.sampler_groups = [1, 2, 4, 8, 16, 32, 64, 128]     # 采样组数         [低分组1,2,4  中分组8,16  高分组32,64,128]
        self.sampler_kernel_size = [1, 3, 5]                    # 产生偏移的卷积核   [1x1仅依赖通道信息, 3x3捕获局部邻域时空关系, 5x5可能引入无关区域噪声]
        self.sampler_local_windows = [3, 5, 7, 9]               # 余弦相似度的范围   [高频细节    中层语义   全局结构]
        self.sampler_direction_feat = 'sim'                     # ['sim','sim_concat','sim_sim']
        #############################################################
        self.sampler0 = LocalSimGuidedSampler(in_channels=128, groups=64, kernel_size=3, local_window=7, direction_feat='sim_concat', dilation=2)
        # self.sampler1 = LocalSimGuidedSampler(in_channels=128, groups=4, kernel_size=3, local_window=7, direction_feat='sim_concat', dilation=2)
        # self.sampler2 = LocalSimGuidedSampler(in_channels=128, groups=4, kernel_size=3, local_window=9, direction_feat='sim_concat', dilation=2)

        # 掩码
        ##############################################
        from multiview_detector.models.MultiViewDynamicMask import MultiViewDynamicMask
        self.mask = MultiViewDynamicMask()
        ##############################################

        # 卷积增强
        self.feat_enhanced = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                                        # nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.ReLU(),
                                        # nn.Conv2d(128, 128, 3, padding=4, dilation=4), nn.ReLU(),
                                        )

        # 统一设备
        # self.coord_map.to(device)
        # self.world_feat0.to(device)
        # self.world_feat1.to(device)
        # self.world_feat2.to(device)
        # self.sampler0.to(device)
        # self.sampler1.to(device)
        # self.sampler2.to(device)

    def forward(self, x, visualize):      # x = {tensor(1,7,128,120,360)}
        # for_offset = torch.mean(x, dim=1)
        for_offset = self.mask(x)   # for_offset = {tensor(1,128,120,360)}

        B, N, C, H, W = x.shape
        x = x.view(B, N * C, H, W)
        x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1)
        x = self.bottleneck(x)
        x = self.sampler0(for_offset, x)

        # x0 = self.sampler0(for_offset, self.world_feat0(x))     # ( {tensor(1,128,120,360)}用于产生偏移 , {tensor(1,128,120,360)}用于被采样 )
        # x0 = self.world_feat0(x)
        # x1 = self.sampler1(for_offset, self.world_feat1(x0))    # ( {tensor(1,128,120,360)}用于产生偏移 , {tensor(1,128,120,360)}用于被采样 )
        # x1 = self.world_feat1(x0)
        # x2 = self.sampler2(for_offset, self.world_feat2(x1))    # ( {tensor(1,128,120,360)}用于产生偏移 , {tensor(1,128,120,360)}用于被采样 )
        # x2 = self.world_feat2(x1)
        x = self.feat_enhanced(x)
        return x




try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:
    def xavier_init(module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = 'normal') -> None:
        if hasattr(module, 'weight') and module.weight is not None:
            assert distribution in ['uniform', 'normal']
            nn.init.xavier_uniform_(module.weight, gain=gain) if distribution == 'uniform' else nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    def constant_init(module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
##############################################
#########  相似度模块（包含偏移和采样）   ##########
##############################################
def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # 展平输入张量中每个点及其周围KxK范围内的点(B, CxKxK, HW)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1) # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = similarity.view(B, k * k - 1, H, W)        # 将结果重塑回[B, KxK-1, H, W]的形状
    return similarity


class LocalSimGuidedSampler(nn.Module):
    def __init__(self,
                 in_channels,
                 groups        =4,
                 kernel_size   =1,
                 local_window  =3,
                 norm          =True,
                 direction_feat='sim_concat',
                 dilation      =1):
        super().__init__()
        self.groups                          = groups
        self.kernel_size                     = kernel_size
        self.local_window                    = local_window
        self.direction_feat                  = direction_feat
        self.dilation_for_compute_similarity = dilation

        if self.direction_feat == 'sim':                # out_channels = 2 * groups
            self.offset = nn.Conv2d(              local_window ** 2 - 1, 2 * groups, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(in_channels + local_window ** 2 - 1, 2 * groups, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == 'sim_sim':
            self.offset = nn.Conv2d(2*           (local_window ** 2 - 1), 2 * groups, kernel_size=kernel_size, padding=kernel_size // 2)
            self.dilation_for_compute_similarity = [1,2]
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        self.norm = nn.GroupNorm(in_channels // 8, in_channels) if norm else nn.Identity()       # (分成16组8, 通道数128)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        result = F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, H, W)
        return result

    def forward(self, lr_x, feat2sample):
        if self.direction_feat == 'sim':
            lr_x = self.norm(lr_x)
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=self.dilation_for_compute_similarity, sim='cos')
        elif self.direction_feat == 'sim_concat':
            lr_x = self.norm(lr_x)
            lr_sim = torch.cat([
                lr_x,
                compute_similarity(lr_x, self.local_window, dilation=self.dilation_for_compute_similarity, sim='cos')
            ], dim=1)
        elif self.direction_feat == 'sim_sim':
            lr_x = self.norm(lr_x)
            lr_sim = torch.cat([
                compute_similarity(lr_x, self.local_window, dilation=2, sim='cos'),
                compute_similarity(lr_x, self.local_window, dilation=1, sim='cos')
            ], dim=1)

        offset = self.offset(lr_sim)
        result = self.sample(feat2sample, offset)
        return result


##############################################
#########           main()          ##########
##############################################
if __name__ == '__main__':
    print('################### 测试fusion3')
    print('                        └──子模块LocalSimGuidedSampler')
    in_feat = torch.randn([1, 7, 128, 120, 360])
    model = fusion3(7, [120, 360], 128).to(device)
    print(model)
    print(model(in_feat, False).shape)

    print('################### 测试fusion3 的 sampler')
    # sampler = model.sampler0
    # sampler.to(device)
    # print(f'[kernel_size={sampler.kernel_size}, groups={sampler.groups}] {sampler.direction_feat} [local_window={sampler.local_window}, dilation={sampler.dilation_for_compute_similarity}]')
    # print(sampler(torch.randn([1, 128, 120, 360]), torch.randn([1, 128, 120, 360])).shape)
    #
    # sampler = model.sampler1
    # sampler.to(device)
    # print(f'[kernel_size={sampler.kernel_size}, groups={sampler.groups}] {sampler.direction_feat} [local_window={sampler.local_window}, dilation={sampler.dilation_for_compute_similarity}]')
    # print(sampler(torch.randn([1, 128, 120, 360]), torch.randn([1, 128, 120, 360])).shape)

    # sampler = model.sampler2
    # sampler.to(device)
    # print(f'[kernel_size={sampler.kernel_size}, groups={sampler.groups}] {sampler.direction_feat} [local_window={sampler.local_window}, dilation={sampler.dilation_for_compute_similarity}]')
    # print(sampler(torch.randn([1, 128, 120, 360]), torch.randn([1, 128, 120, 360])).shape)