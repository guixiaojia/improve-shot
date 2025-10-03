import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
# from multiview_detector.models.trans_world_feat import create_pos_embedding


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


class ConvWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, stride=2, reduction=None):
        super(ConvWorldFeat, self).__init__()
        # self.downsample = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, stride, 1), nn.ReLU(), )
        # self.coord_map = create_coord_map(np.array(Rworld_shape) // stride)
        self.coord_map = create_coord_map(np.array(Rworld_shape))
        self.reduction = reduction
        if self.reduction is None:
            combined_input_dim = base_dim * num_cam + 2
        elif self.reduction == 'sum':
            combined_input_dim = base_dim + 2
        else:
            raise Exception
        self.world_feat = nn.Sequential(nn.Conv2d(combined_input_dim, hidden_dim, 3, padding=1),
                                        nn.ReLU(),
                                        # nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                                        # nn.ReLU()
                                        # nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
                                        # nn.ReLU(),
                                        # nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
                                        # nn.ReLU(),
                                        )
        # self.upsample = nn.Sequential(nn.Upsample(Rworld_shape, mode='bilinear', align_corners=False),
        #                               nn.Conv2d(hidden_dim, base_dim, 3, 1, 1),
        #                               nn.ReLU(), )

        # 掩码
        ##############################################
        # from multiview_detector.models.MultiViewDynamicMask import MultiViewDynamicMask
        from multiview_detector.models.MultiViewDynamicMask_theta import MultiViewDynamicMask
        self.mask = MultiViewDynamicMask()
        ##############################################
        # 采样器
        ##############################################
        from multiview_detector.models.fusion3_ablation import fusion3
        self.sampler = fusion3()
        ##############################################

    # def forward(self, x, visualize=False):
    #     ##############################################
    #     # 掩码
    #     x = self.mask(x)
    #     ##############################################
    #     B, N, C, H, W = x.shape     # 1 7 128 120 360
    #     x = x.view(B * N, C, H, W)  # {tensor(7,128,60,180)}
    #     x = self.downsample(x)      # {tensor(7,128,60,180)}
    #     ##############################################
    #     # for_offset
    #     for_offset = torch.mean(x, dim=0, keepdim=True)
    #     ##############################################
    #     _, _, H, W = x.shape
    #     if self.reduction is None:
    #         x = x.view(B, N * C, H, W)  # 1 7 128 60 180
    #     elif self.reduction == 'sum':
    #         x = x.sum(dim=1)
    #     else:
    #         raise Exception
    #     x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1) # {tensor(1,898,60,180)}
    #     x = self.world_feat(x)  # {tensor(1,128,60,180)}
    #     ##############################################
    #     # sampler
    #     x = self.sampler(for_offset, x)
    #     ##############################################
    #     x = self.upsample(x)
    #     return x

    # 移除downsample upsample
    def forward(self, x, visualize=False):
        ##############################################
        # 版本二 for_offset
        # for_offset = torch.mean(x, dim=1)
        ##############################################
        ##############################################
        # 掩码
        x, masks = self.mask(x)            # 1 7 128 120 360
        ##############################################
        B, N, C, H, W = x.shape     # 1 7 128 120 360
        x = x.view(B * N, C, H, W)  # {tensor(7,128,60,180)}
        ##############################################
        # 版本一 for_offset
        for_offset = torch.mean(x, dim=0, keepdim=True)
        ##############################################
        _, _, H, W = x.shape
        x = x.view(B, N * C, H, W)
        x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1)
        x = self.world_feat(x)      # {tensor(1,128,60,180)}
        ##############################################
        # sampler
        x = self.sampler(for_offset, masks, x) # sampler(   )
        ##############################################
        return x

# class DeformConvWorldFeat(nn.Module):
#     def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, ):
#         super(DeformConvWorldFeat, self).__init__()
#         self.pos_embedding = create_pos_embedding(Rworld_shape, base_dim // 2)
#         self.deform_pos = nn.ModuleList([nn.Conv2d(base_dim, 9 * 2, 1) for _ in range(num_cam)])
#         self.deform_conv = nn.ModuleList([DeformConv2d(base_dim, base_dim, 3, padding=1) for _ in range(num_cam)])
#         self.merge_linear = nn.Sequential(nn.Conv2d(base_dim * num_cam, hidden_dim, 1),
#                                           nn.ReLU())
#         self.world_feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#                                         nn.ReLU(),
#                                         nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
#                                         nn.ReLU(),
#                                         nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
#                                         nn.ReLU(), )
#
#     def forward(self, x, visualize=False):
#         B, N, C, H, W = x.shape
#         feats = []
#         for n in range(N):
#             feat = x[:, n] + self.pos_embedding.to(x.device)
#             pos = self.deform_pos[n](feat)
#             feat = F.relu(self.deform_conv[n](feat, pos))
#             feats.append(feat)
#         feats = torch.cat(feats, dim=1)
#         feats = self.merge_linear(feats)
#         return self.world_feat(feats)


def test():
    in_feat = torch.zeros([1, 6, 128, 120, 360])
    model = ConvWorldFeat(6, [120, 360], 128)
    out_feat = model(in_feat)
    pass


if __name__ == '__main__':
    test()
