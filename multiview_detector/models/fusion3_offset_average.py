import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = 'cuda:0'
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
        self.coord_map = create_coord_map(np.array(Rworld_shape))
        self.world_feat = nn.Sequential(nn.Conv2d(898, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),)

        self.sampler_in_channels    = 128           # 128
        self.sampler_groups         = 16             # [1, 2, 4, 8, 16, 32, 64, 128]
        self.sampler_local_windows  = 7             # [3, 5, 7]
        self.sampler_direction_feat = 'sim_concat'
        self.sampler_dilation       = 2
        self.sampler = LocalSimGuidedSampler(
            in_channels   =self.sampler_in_channels,
            groups        =self.sampler_groups,
            kernel_size=3,
            local_window  =self.sampler_local_windows,
            direction_feat=self.sampler_direction_feat,
            dilation      =self.sampler_dilation,     # 2比1效果好
        )
        # 统一设备
        # self.coord_map.to(device)
        # self.world_feat.to(device)
        # self.sampler.to(device)

    def forward(self, x, visualize=False):      # x = {tensor(1,7,128,120,360)}
        if self.sampler_in_channels == 128:
            # x = torch.nn.Softmax(dim=1)(x)
            for_offset = torch.mean(x, dim=1)        # mean_x = {tensor(1,128,120,360)}
        else:
            pass
            ######### 直接拼接 concat_x = {tensor(1,896,120,360)}
            # for_offset = x.view(1, 7 * 128, 120, 360)
            ######### 分组拼接
            # group0 = torch.mean(x[:, [0, 1], :, :, :], dim=1)   # 合并0-1帧
            # group1 = torch.mean(x[:, [2, 3], :, :, :], dim=1)   # 合并2-3帧
            # group2 = torch.mean(x[:, [4, 5], :, :, :], dim=1)   # 合并4-5帧
            # group3 = x[:, 6:7, :, :, :].squeeze(1)              # 保留第6帧
            # for_offset = torch.cat([group0, group1, group2, group3], dim=1)  # (1,4*128,120,360)拼接合并结果 [注意维度顺序]
            ######### 先softmax再平均


        B, N, C, H, W = x.shape
        x = x.view(B, N * C, H, W)
        x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1)
        x = self.world_feat(x)
        x = self.sampler(for_offset, x)     # (mean_x用于产生offset  x用于被采样的)
        return x

##############################################
#########  相似度模块（包含偏移和采样）   ##########
##############################################
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

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # 展平输入张量中每个点及其周围KxK范围内的点(B, CxKxK, HW)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)
    if sim == 'cos':    # 计算余弦相似度
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':  # 计算点积相似度
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
                 scale            =1,
                 style            ='lp',
                 groups           =4,
                 use_direct_scale =True,
                 kernel_size      =1,
                 local_window     =3,
                 sim_type         ='dot',
                 norm             =True,
                 direction_feat   ='sim_concat',
                 dilation=1):
        super().__init__()
        # self.scale          = scale
        # self.style          = style
        # self.sim_type       = sim_type
        self.groups         = groups
        self.local_window   = local_window      # 3，5，7，9
        self.direction_feat = direction_feat
        self.dilation_for_compute_similarity = dilation  # 1, 2

        out_channels = 2 * groups
        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == 'sim_concat':
            self.offset   = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.offset_1 = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2)
            self.offset_2 = nn.Conv2d(in_channels + local_window ** 2 - 1, out_channels // 4, kernel_size=kernel_size, padding=kernel_size // 2)

        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        # self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels) if norm else nn.Identity()
        self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels) if norm else nn.Identity()       # (分组数16, 通道数128)

    def sample(self, x, offset, groups):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        result = F.grid_sample(x.reshape(B * groups, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, H, W)
        return result

    # def sample(self, x, offset):
    #     B, _, H, W = offset.shape
    #     offset = offset.view(B, 2, -1, H, W)
    #     coords_h = torch.arange(H) + 0.5
    #     coords_w = torch.arange(W) + 0.5
    #     coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
    #     normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
    #     coords = 2 * (coords + offset) / normalizer - 1
    #     coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
    #     result = F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear', align_corners=False, padding_mode="border").view(B, -1, H, W)
    #     return result


    def forward(self, lr_x, feat2sample):
        if self.direction_feat == 'sim':
            lr_x = self.norm_lr(lr_x)
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=self.dilation_for_compute_similarity, sim='cos')
        elif self.direction_feat == 'sim_concat':
            lr_x = self.norm_lr(lr_x)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=self.dilation_for_compute_similarity, sim='cos')], dim=1)

        offset = self.offset(lr_sim)
        result = self.sample(feat2sample, offset, groups=self.groups)

        offset_1 = self.offset_1(lr_sim)
        result_1 = self.sample(feat2sample, offset_1, groups=self.groups // 2)

        offset_2 = self.offset_2(lr_sim)
        result_2 = self.sample(feat2sample, offset_2, groups=self.groups // 4)

        averaged = torch.stack([result, result_1, result_2], dim=0).mean(dim=0)

        return averaged


##############################################
#########           main()          ##########
##############################################
if __name__ == '__main__':
    print('################### 测试fusion3')
    print('                        └──子模块LocalSimGuidedSampler')
    in_feat = torch.randn([1, 7, 128, 120, 360])
    model = fusion3(7, [120, 360], 128)
    print(model)
    print(model(in_feat).shape)
    torch.save(model.state_dict(), '第二个模块fusion3_model_896.pt')

    lr_x = torch.randn([1, 128, 120, 360])      # 128 256 512 896
    feat2sample = torch.randn([1, 128, 120, 360])
    sampler = model.sampler
    print('余弦窗口local_window     =', sampler.local_window)
    print('   ├──dilation          =', sampler.dilation_for_compute_similarity)
    print('   ├──sim & sim_concate =', sampler.direction_feat)
    print('   └──采样组数groups      =', sampler.groups)
    print(sampler(lr_x, feat2sample).shape)


