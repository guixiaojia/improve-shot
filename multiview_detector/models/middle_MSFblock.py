import torch.nn as nn
import torch
from kornia.geometry.transform import warp_perspective
from multiview_detector.models.original_MSFblock import MSFblock

class middle_MSFblock(nn.Module):
    def __init__(self):
        super(middle_MSFblock, self).__init__()
        self.depth_v5 = nn.ModuleDict({
            f'{i}': nn.Sequential(nn.Conv2d(128, 32, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 4, 1, bias=False))
            for i in range(7)
        })
        self.feat_before_merge_v5 = nn.ModuleDict({
            f'{i}': MSFblock(128)
            for i in range(7)
        })
        self.Rworld_shape = [120,360]

    def forward(self, img_feature_all, proj_mats):
        views = []
        for i in range(7):
            depth = self.depth_v5[str(i)](img_feature_all[i, :, :, :].unsqueeze(0)).softmax(dim=1)
            depths = []
            for j in range(4):
                in_feat = img_feature_all[i, :, :, :].unsqueeze(0) * depth[:, j, :, :]
                out_feat = warp_perspective(in_feat, proj_mats[j][i, :, :].unsqueeze(0), self.Rworld_shape)
                depths.append(out_feat)
            view = self.feat_before_merge_v5[str(i)](depths[0], depths[1], depths[2], depths[3])
            views.append(view)
        warped_feat = torch.cat(views, dim=0)
        return warped_feat








if __name__ == '__main__':
    img_feature_all = torch.rand(7, 128, 90, 160)
    proj_mats = [torch.rand(7,3,3),
                 torch.rand(7,3,3),
                 torch.rand(7,3,3),
                 torch.rand(7,3,3)]
    my = middle_MSFblock()
    result = my(img_feature_all, proj_mats )
    print(result.shape)
