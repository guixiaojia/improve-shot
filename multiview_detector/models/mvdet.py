import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
device = 'cuda:0'

class mvdet(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to(device)
            self.base_pt2 = base[split:].to(device)
            out_channel = 512
        elif arch == 'resnet18':
            # base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            # split = 7
            # self.base_pt1 = base[:split].to(device)
            # self.base_pt2 = base[split:].to(device)
            self.base = nn.Sequential(*list(resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True]).children())[:-2])
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to(device)
        # self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
        #                                     # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
        #                                     nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to(device)

        # world_feat & map_classifier
        ##############################################
        self.world_feat = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                                        nn.ReLU(),
                                        # nn.Conv2d(128, 128, 3, padding=4, dilation=4),
                                        # nn.ReLU()
                                        ).to(device)
        self.map_classifier = nn.Conv2d(512, 1, 1, 1, bias=False).to(device)
        ##############################################

        # 掩码
        ##############################################
        from multiview_detector.models.MultiViewDynamicMask import MultiViewDynamicMask
        self.mask = MultiViewDynamicMask()
        self.mask.to(device)
        ##############################################

        # 采样器
        ##############################################
        from multiview_detector.models.fusion3_ablation import fusion3
        self.sampler = fusion3()
        self.sampler.to(device)
        ##############################################
        pass



    def _base(self, x):
        return self.base(x)
    def _map_classifier(self, world_features):
        return self.map_classifier(world_features)
    def _world_feat(self, x):
        return

    # def forward(self, imgs, visualize=False):
    #     B, N, C, H, W = imgs.shape
    #     assert N == self.num_cam
    #     world_features = []
    #     imgs_result = []
    #     for cam in range(self.num_cam):
    #         img_feature = self.base_pt1(imgs[:, cam].to(device))
    #         img_feature = self.base_pt2(img_feature.to(device))
    #         img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
    #         img_res = self.img_classifier(img_feature.to(device))
    #         imgs_result.append(img_res)
    #         proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(device)
    #         world_feature = warp_perspective(img_feature.to(device), proj_mat, self.reducedgrid_shape)
    #         if visualize:
    #             plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
    #             plt.show()
    #             plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
    #             plt.show()
    #         world_features.append(world_feature.to(device))
    #
    #     ##############################################
    #     world_features = torch.stack(world_features, dim=1)  # 沿着 dim=1 合并
    #     world_features = self.mask(world_features)
    #     world_features = list(torch.unbind(world_features, dim=1))  # 沿着 dim=1 拆分并转化为list
    #     ##############################################
    #
    #     world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(device)], dim=1)
    #     if visualize:
    #         plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
    #         plt.show()
    #     # map_result = self.map_classifier(world_features.to(device))
    #     map_result = checkpoint(self._map_classifier, world_features)   # 检查点优化
    #     map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
    #
    #     if visualize:
    #         plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
    #         plt.show()
    #     return map_result, imgs_result



    def forward(self, imgs, visualize=True):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            # img_feature = self.base_pt1(imgs[:, cam].to(device))
            # img_feature = self.base_pt2(img_feature.to(device))
            # img_feature = self.base(imgs[:, cam])
            img_feature = checkpoint(self._base, imgs[:, cam].to(device))  # 检查点优化
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to(device))
            imgs_result.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(device)
            world_feature = warp_perspective(img_feature.to(device), proj_mat, self.reducedgrid_shape)
            world_features.append(world_feature.to(device))

        ##############################################
        # 掩码
        world_features = torch.stack(world_features, dim=1) # 沿着 dim=1 合并
        world_features = self.mask(world_features)      # {tensor(1,7,512,120,360)}
        #####################
        # for_offset
        for_offset = torch.mean(world_features, dim=1, keepdim=False)
        #####################
        world_features= list(torch.unbind(world_features, dim=1))   # 沿着 dim=1 拆分并转化为list
        ##############################################

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(device)], dim=1)   # {tensor(1,3586,120,360)}
        map_result = self.world_feat(world_features)  # {tensor(1,128,120,360)}  # map_result = self.map_classifier()
        # map_result = checkpoint(self._world_feat, world_features)
        #####################
        # sampler
        map_result = self.sampler(for_offset, map_result)
        #####################
        # map_result = self.map_classifier(map_result)
        map_result = checkpoint(self._map_classifier, map_result)  # 检查点优化
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        # return map_result, imgs_result
        return (map_result, None), (imgs_result, None, None)




    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
