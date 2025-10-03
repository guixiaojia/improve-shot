import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg11
from kornia.geometry.transform import warp_perspective
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap
from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
from torch.utils.checkpoint import checkpoint
device = 'cuda:0'
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc

# def create_reference_map(dataset, n_points=4, downsample=2, visualize=False):
#     H, W = dataset.Rworld_shape  # H,W; N_row,N_col
#     H, W = H // downsample, W // downsample
#     ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32), torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
#     ref = torch.stack((ref_x, ref_y), -1).reshape([-1, 2])
#     if n_points == 4:
#         zs = [0, 0, 0, 0]
#     elif n_points == 8:
#         zs = [-0.4, -0.2, 0, 0, 0.2, 0.4, 1, 1.8]
#     else:
#         raise Exception
#     ref_maps = torch.zeros([H * W, dataset.num_cam, n_points, 2])
#     world_zoom_mat = np.diag([dataset.world_reduce * downsample, dataset.world_reduce * downsample, 1])
#     Rworldgrid_from_worldcoord_mat = np.linalg.inv(dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)
#     for cam in range(dataset.num_cam):
#         mat_0 = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam], dataset.base.extrinsic_matrices[cam])
#         for i, z in enumerate(zs):
#             mat_z = Rworldgrid_from_worldcoord_mat @ get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam], dataset.base.extrinsic_matrices[cam], z / dataset.base.worldcoord_unit)
#             img_pts = project_2d_points(np.linalg.inv(mat_z), ref)
#             ref_maps[:, cam, i, :] = torch.from_numpy(project_2d_points(mat_0, img_pts))
#         pass
#         if visualize:
#             fig, ax = plt.subplots()
#             field_x = (ref_maps[:, cam, 3, 0] - ref_maps[:, cam, 1, 0]).reshape([H, W])
#             field_y = (ref_maps[:, cam, 3, 1] - ref_maps[:, cam, 1, 1]).reshape([H, W])
#             ax.streamplot(ref_x.numpy(), ref_y.numpy(), field_x.numpy(), field_y.numpy())
#             ax.set_aspect('equal', 'box')
#             ax.invert_yaxis()
#             # plt.show()
#     ref_maps[:, :, :, 0] /= W
#     ref_maps[:, :, :, 1] /= H
#     return ref_maps


class SHOT(nn.Module):
    def __init__(self, dataset, arch='resnet18', world_feat_arch='conv', bottleneck_dim=128, outfeat_dim=64, dropout=0.5, depth_scales=4):
        super().__init__()
        self.Rimg_shape = dataset.Rimg_shape
        self.Rworld_shape = dataset.Rworld_shape
        self.img_reduce = dataset.img_reduce
        self.depth_scales = depth_scales
        self.dataset_num = dataset.num_cam

        # world grid change to xy indexing, z in meters by default, projection matrices: img feat -> world feat
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)
        worldcoord_from_imgcoord_mats = [np.array([get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam], dataset.base.extrinsic_matrices[cam], dataset.base.depth_margin * i)
                                                   for cam in range(dataset.num_cam)])
                                         for i in range(self.depth_scales)]
        proj_mats, B = {}, 1
        for i in range(self.depth_scales):
            proj_mats[i] = [torch.from_numpy(Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[i][j]) for j in range(dataset.num_cam)]
        for i in range(self.depth_scales):
            proj = torch.stack(proj_mats[i]).float()[None].repeat([B, 1, 1, 1])
            proj_mats[i] = nn.Parameter(proj.view([-1, 3, 3]), requires_grad=False).to(device)
        self.proj_mats = proj_mats

        if arch == 'vgg11':
            self.base = vgg11(pretrained=True).features
            self.base[-1] = nn.Identity()
            self.base[-4] = nn.Identity()
            base_dim = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        else:
            raise Exception('只能二选一[vgg11, resnet18]')

        if bottleneck_dim:
            # self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 3, padding=2, dilation=2), nn.ReLU(), nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=2, dilation=2))
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(dropout))
            base_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)

        # world feat
        if world_feat_arch == 'conv':
            # from multiview_detector.models.conv_world_feat import ConvWorldFeat
            from multiview_detector.models.conv_world_feat_mask import ConvWorldFeat
            self.world_feat = ConvWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == 'trans':
            pass
        elif world_feat_arch == 'deform_conv':
            pass
        elif world_feat_arch == 'deform_trans':
            pass
        elif world_feat_arch == 'freqfusion':
            ##############################################
            ######## 消融模块2
            ##############################################
            # from multiview_detector.models.fusion3 import fusion3
            # from multiview_detector.models.fusion3_ablation import fusion3
            # self.world_feat = fusion3(7, [120, 360], 128)
            # from multiview_detector.models.fusion3_offset_average import fusion3
            # from multiview_detector.models.fusion3_cascade_sampling import fusion3
            # from multiview_detector.models.fusion3_cascade_sampling_version2 import fusion3
            # from multiview_detector.models.fusion3_cascade_sampling_version3 import fusion3
            # from multiview_detector.models.fusion3_bottleneck_cascade_sampling import fusion3
            # from multiview_detector.models.fusion3_bottleneck_sampling import fusion3
            # self.world_feat = fusion3(7, [120, 360], 128)
            pass
        else:
            raise Exception

        self.depth_classifier = nn.Sequential(nn.Conv2d(base_dim, 64, 1),
                                              nn.ReLU(),
                                              nn.Conv2d(64, self.depth_scales, 1, bias=False))
        ##############################################
        ######## 原始SHOT
        ##############################################
        self.feat_before_merge = nn.ModuleDict({
            f'{i}': nn.Conv2d(base_dim, base_dim, 3, padding=1)
            for i in range(self.depth_scales)       # 1 2 3 self.depth_scales
        })
        ##############################################
        ######## v2
        ##############################################
        # self.feat_before_merge_v2 = nn.ModuleDict({
        #     f'{i}': nn.Conv2d(base_dim, base_dim, 3, padding=1)
        #     for i in range(self.depth_scales * 7)
        # })
        ##############################################
        ######## v3
        ##############################################
        # self.feat_before_merge_v3 = nn.ModuleDict({
        #     f'{i}': nn.Conv2d(base_dim, base_dim, 1)
        #     for i in range(self.depth_scales * self.dataset_num)
        # })
        ##############################################
        ######## v4
        ##############################################
        # from multiview_detector.models.original_MSFblock import MSFblock
        # self.feat_before_merge_v4 = nn.ModuleDict({
        #     f'{i}': MSFblock(128)
        #     for i in range(7)
        # })
        ##############################################
        ######## v5
        ##############################################
        # from multiview_detector.models.middle_MSFblock import middle_MSFblock
        # self.depth_v5 = middle_MSFblock()

        ##############################################
        ######## 消融模块1
        ##############################################
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_1 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_2 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_3 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_4 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_5 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_6 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_8 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_9 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_10 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_12 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_15 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_16 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_AdaptiveAvgPool2d_unique                         import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_case1_DynamicGAP                                 import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_case4_DynamicAdaptivePooling                     import middle_MSFblock
        from multiview_detector.models.Middle_MSFblock_case7_ShareWeightMultiAdaptivePool2d             import middle_MSFblock
        # self.depth_v5 = middle_MSFblock()
        # from multiview_detector.models.Middle_MSFblock_case8_ShareWeightMultiAdaptivePool2d_alpha_beta  import middle_MSFblock
        # from multiview_detector.models.Middle_MSFblock_case9_ShareWeightMultiAdaptivePool2d_alpha       import middle_MSFblock

        # world heads
        self.world_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.world_offset = output_head(base_dim, outfeat_dim, 2)
        pass

    def warp_perspective(self, img_feature_all, proj_mats):
        warped_feat = 0
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1)  # [b*n,d,h,w]
        for i in range(self.depth_scales):
            in_feat = img_feature_all * depth_select[:, i][:, None]
            out_feat = warp_perspective(in_feat, proj_mats[i], self.Rworld_shape)
            warped_feat += self.feat_before_merge[f'{i}'](out_feat)     # [b*n,c,h,w]
        return warped_feat
    def warp_perspective_v2(self, img_feature_all, proj_mats):
        warped_feat = 0
        ##############################################
        #########(depth_classifier): Sequential(
        #########(0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        #########(1): ReLU()
        #########(2): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), bias=False) )
        ##############################################
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1)  # [b*n,d,h,w]

        ##############################################
        ######## (feat_before_merge_v2): ModuleDict(
        ########(0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ########  ……
        ########(27): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        ##############################################
        in_feat_0 = img_feature_all * depth_select[:, 0][:, None]  # {Tensor:(7,128,45,80)} = {Tensor:(7,128,45,80)} 广播机制逐元素相乘 {Tensor:(7,1,45,80)}
        in_feat_1 = img_feature_all * depth_select[:, 1][:, None]  # {Tensor:(7,128,45,80)} = {Tensor:(7,128,45,80)} 广播机制逐元素相乘 {Tensor:(7,1,45,80)}
        in_feat_2 = img_feature_all * depth_select[:, 2][:, None]  # {Tensor:(7,128,45,80)} = {Tensor:(7,128,45,80)} 广播机制逐元素相乘 {Tensor:(7,1,45,80)}
        in_feat_3 = img_feature_all * depth_select[:, 3][:, None]  # {Tensor:(7,128,45,80)} = {Tensor:(7,128,45,80)} 广播机制逐元素相乘 {Tensor:(7,1,45,80)}
        out_feat_0 = warp_perspective(in_feat_0, proj_mats[0], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,45,80)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_1 = warp_perspective(in_feat_1, proj_mats[1], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,45,80)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_2 = warp_perspective(in_feat_2, proj_mats[2], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,45,80)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_3 = warp_perspective(in_feat_3, proj_mats[3], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,45,80)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_00 = out_feat_0[0, :, :, :].unsqueeze(0)
        out_feat_01 = out_feat_0[1, :, :, :].unsqueeze(0)
        out_feat_02 = out_feat_0[2, :, :, :].unsqueeze(0)
        out_feat_03 = out_feat_0[3, :, :, :].unsqueeze(0)
        out_feat_04 = out_feat_0[4, :, :, :].unsqueeze(0)
        out_feat_05 = out_feat_0[5, :, :, :].unsqueeze(0)
        out_feat_06 = out_feat_0[6, :, :, :].unsqueeze(0)
        out_feat_10 = out_feat_1[0, :, :, :].unsqueeze(0)
        out_feat_11 = out_feat_1[1, :, :, :].unsqueeze(0)
        out_feat_12 = out_feat_1[2, :, :, :].unsqueeze(0)
        out_feat_13 = out_feat_1[3, :, :, :].unsqueeze(0)
        out_feat_14 = out_feat_1[4, :, :, :].unsqueeze(0)
        out_feat_15 = out_feat_1[5, :, :, :].unsqueeze(0)
        out_feat_16 = out_feat_1[6, :, :, :].unsqueeze(0)
        out_feat_20 = out_feat_2[0, :, :, :].unsqueeze(0)
        out_feat_21 = out_feat_2[1, :, :, :].unsqueeze(0)
        out_feat_22 = out_feat_2[2, :, :, :].unsqueeze(0)
        out_feat_23 = out_feat_2[3, :, :, :].unsqueeze(0)
        out_feat_24 = out_feat_2[4, :, :, :].unsqueeze(0)
        out_feat_25 = out_feat_2[5, :, :, :].unsqueeze(0)
        out_feat_26 = out_feat_2[6, :, :, :].unsqueeze(0)
        out_feat_30 = out_feat_3[0, :, :, :].unsqueeze(0)
        out_feat_31 = out_feat_3[1, :, :, :].unsqueeze(0)
        out_feat_32 = out_feat_3[2, :, :, :].unsqueeze(0)
        out_feat_33 = out_feat_3[3, :, :, :].unsqueeze(0)
        out_feat_34 = out_feat_3[4, :, :, :].unsqueeze(0)
        out_feat_35 = out_feat_3[5, :, :, :].unsqueeze(0)
        out_feat_36 = out_feat_3[6, :, :, :].unsqueeze(0)
        warped_feat += torch.cat([ self.feat_before_merge_v2[f'{0}'](out_feat_00),
                                   self.feat_before_merge_v2[f'{1}'](out_feat_01),
                                   self.feat_before_merge_v2[f'{2}'](out_feat_02),
                                   self.feat_before_merge_v2[f'{3}'](out_feat_03),
                                   self.feat_before_merge_v2[f'{4}'](out_feat_04),
                                   self.feat_before_merge_v2[f'{5}'](out_feat_05),
                                   self.feat_before_merge_v2[f'{6}'](out_feat_06)], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v2[f'{7}'](out_feat_10),
                                  self.feat_before_merge_v2[f'{8}'](out_feat_11),
                                  self.feat_before_merge_v2[f'{9}'](out_feat_12),
                                  self.feat_before_merge_v2[f'{10}'](out_feat_13),
                                  self.feat_before_merge_v2[f'{11}'](out_feat_14),
                                  self.feat_before_merge_v2[f'{12}'](out_feat_15),
                                  self.feat_before_merge_v2[f'{13}'](out_feat_16)], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v2[f'{14}'](out_feat_20),
                                  self.feat_before_merge_v2[f'{15}'](out_feat_21),
                                  self.feat_before_merge_v2[f'{16}'](out_feat_22),
                                  self.feat_before_merge_v2[f'{17}'](out_feat_23),
                                  self.feat_before_merge_v2[f'{18}'](out_feat_24),
                                  self.feat_before_merge_v2[f'{19}'](out_feat_25),
                                  self.feat_before_merge_v2[f'{20}'](out_feat_26)], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v2[f'{21}'](out_feat_30),
                                  self.feat_before_merge_v2[f'{22}'](out_feat_31),
                                  self.feat_before_merge_v2[f'{23}'](out_feat_32),
                                  self.feat_before_merge_v2[f'{24}'](out_feat_33),
                                  self.feat_before_merge_v2[f'{25}'](out_feat_34),
                                  self.feat_before_merge_v2[f'{26}'](out_feat_35),
                                  self.feat_before_merge_v2[f'{27}'](out_feat_36)], dim=0)
        return warped_feat      # 要求{Tensor:(7,128,120,360)}
    def warp_perspective_v3(self, img_feature_all, proj_mats):
        warped_feat = 0
        ##############################################
        #########(depth_classifier): Sequential(
        #########(0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        #########(1): ReLU()
        #########(2): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), bias=False) )
        ##############################################
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1)  # [b*n,d,h,w]

        ##############################################
        ######## (feat_before_merge_v2): ModuleDict(
        ########(0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ########  ……
        ########(27): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        ##############################################
        in_feat_0 = img_feature_all * depth_select[:, 0][:, None]  # {Tensor:(7,128,90,160)} = 广播机制逐元素相乘 {Tensor:(7,128,90,160)} * {Tensor:(7,1,90,160)}
        in_feat_1 = img_feature_all * depth_select[:, 1][:, None]  # {Tensor:(7,128,90,160)} = 广播机制逐元素相乘 {Tensor:(7,128,90,160)} * {Tensor:(7,1,90,160)}
        in_feat_2 = img_feature_all * depth_select[:, 2][:, None]  # {Tensor:(7,128,90,160)} = 广播机制逐元素相乘 {Tensor:(7,128,90,160)} * {Tensor:(7,1,90,160)}
        in_feat_3 = img_feature_all * depth_select[:, 3][:, None]  # {Tensor:(7,128,90,160)} = 广播机制逐元素相乘 {Tensor:(7,128,90,160)} * {Tensor:(7,1,90,160)}

        out_feat_0 = warp_perspective(in_feat_0, proj_mats[0], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_1 = warp_perspective(in_feat_1, proj_mats[1], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_2 = warp_perspective(in_feat_2, proj_mats[2], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_3 = warp_perspective(in_feat_3, proj_mats[3], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])

        warped_feat += torch.cat([self.feat_before_merge_v3[f'{0}'](out_feat_0[0, :, :, :].unsqueeze(0)),  # {Tensor:(1,128,120,360)} = 卷积( {Tensor:(1,128,120,360)} )
                                        self.feat_before_merge_v3[f'{1}'](out_feat_0[1, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{2}'](out_feat_0[2, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{3}'](out_feat_0[3, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{4}'](out_feat_0[4, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{5}'](out_feat_0[5, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{6}'](out_feat_0[6, :, :, :].unsqueeze(0))], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v3[f'{7}'](out_feat_1[0, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{8}'](out_feat_1[1, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{9}'](out_feat_1[2, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{10}'](out_feat_1[3, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{11}'](out_feat_1[4, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{12}'](out_feat_1[5, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{13}'](out_feat_1[6, :, :, :].unsqueeze(0))], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v3[f'{14}'](out_feat_2[0, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{15}'](out_feat_2[1, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{16}'](out_feat_2[2, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{17}'](out_feat_2[3, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{18}'](out_feat_2[4, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{19}'](out_feat_2[5, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{20}'](out_feat_2[6, :, :, :].unsqueeze(0))], dim=0)
        warped_feat += torch.cat([self.feat_before_merge_v3[f'{21}'](out_feat_3[0, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{22}'](out_feat_3[1, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{23}'](out_feat_3[2, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{24}'](out_feat_3[3, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{25}'](out_feat_3[4, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{26}'](out_feat_3[5, :, :, :].unsqueeze(0)),
                                        self.feat_before_merge_v3[f'{27}'](out_feat_3[6, :, :, :].unsqueeze(0))], dim=0)
        return warped_feat      # 要求{Tensor:(7,128,120,360)}

    def warp_perspective_v3_refactoring(self, img_feature_all, proj_mats):
        warped_feat = 0
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1)  # [b*n, d, h, w]

        # 生成输入特征
        in_feat = [img_feature_all * depth_select[:, i][:, None] for i in range(4)]  # List[4xTensor:(7,128,90,160)]

        # 生成透视变换后的特征
        out_feat = [warp_perspective(feat, proj_mats[i], self.Rworld_shape) for i, feat in enumerate(in_feat)]  # List[4xTensor:(7,128,120,360)]

        # 特征融合处理
        for i in range(4):
            # 计算当前组的起始索引
            start_idx = i * self.dataset_num
            # 生成当前组的处理结果
            warped_feat += torch.cat([
                self.feat_before_merge_v3[f'{start_idx + k}'](out_feat[i][k, :, :, :].unsqueeze(0))
                for k in range(self.dataset_num)
            ], dim=0)

        return warped_feat  # [dataset_num, c, H, W]

    def warp_perspective_v4(self, img_feature_all, proj_mats):
        warped_feat = 0
        ##############################################
        #########(depth_classifier): Sequential(
        #########(0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        #########(1): ReLU()
        #########(2): Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), bias=False) )
        ##############################################
        depth_select = self.depth_classifier(img_feature_all).softmax(dim=1)  # {Tensor:(7,4,90,160)} = 卷积( {Tensor:(7,128,90,160)} )

        in_feat_0 = img_feature_all * depth_select[:, 0][:, None]  # {Tensor:(7,128,90,160)} = {Tensor:(7,128,90,160)} 广播机制逐元素相乘 {Tensor:(7,1,90,160)}
        in_feat_1 = img_feature_all * depth_select[:, 1][:, None]  # {Tensor:(7,128,90,160)} = {Tensor:(7,128,90,160)} 广播机制逐元素相乘 {Tensor:(7,1,90,160)}
        in_feat_2 = img_feature_all * depth_select[:, 2][:, None]  # {Tensor:(7,128,90,160)} = {Tensor:(7,128,90,160)} 广播机制逐元素相乘 {Tensor:(7,1,90,160)}
        in_feat_3 = img_feature_all * depth_select[:, 3][:, None]  # {Tensor:(7,128,90,160)} = {Tensor:(7,128,90,160)} 广播机制逐元素相乘 {Tensor:(7,1,90,160)}

        out_feat_0 = warp_perspective(in_feat_0, proj_mats[0], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_1 = warp_perspective(in_feat_1, proj_mats[1], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_2 = warp_perspective(in_feat_2, proj_mats[2], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        out_feat_3 = warp_perspective(in_feat_3, proj_mats[3], self.Rworld_shape)  # {Tensor:(7,128,120,360)} = 透视变换函数({Tensor:(7,128,90,160)}, {Tensor:(7,3,3)}, [120,360])
        ##############################################
        ######## self.feat_before_merge_v4: ModuleDict(
        ########(0): MSFblock(128)
        ########  ……
        ########(6): MSFblock(128))
        ##############################################
        view0 = self.feat_before_merge_v4[f'{0}'](out_feat_0[0, :, :, :].unsqueeze(0),
                                                  out_feat_1[0, :, :, :].unsqueeze(0),
                                                  out_feat_2[0, :, :, :].unsqueeze(0),
                                                  out_feat_3[0, :, :, :].unsqueeze(0))
        view1 = self.feat_before_merge_v4[f'{1}'](out_feat_0[1, :, :, :].unsqueeze(0),
                                                  out_feat_1[1, :, :, :].unsqueeze(0),
                                                  out_feat_2[1, :, :, :].unsqueeze(0),
                                                  out_feat_3[1, :, :, :].unsqueeze(0))
        view2 = self.feat_before_merge_v4[f'{2}'](out_feat_0[2, :, :, :].unsqueeze(0),
                                                  out_feat_1[2, :, :, :].unsqueeze(0),
                                                  out_feat_2[2, :, :, :].unsqueeze(0),
                                                  out_feat_3[2, :, :, :].unsqueeze(0))
        view3 = self.feat_before_merge_v4[f'{3}'](out_feat_0[3, :, :, :].unsqueeze(0),
                                                  out_feat_1[3, :, :, :].unsqueeze(0),
                                                  out_feat_2[3, :, :, :].unsqueeze(0),
                                                  out_feat_3[3, :, :, :].unsqueeze(0))
        view4 = self.feat_before_merge_v4[f'{4}'](out_feat_0[4, :, :, :].unsqueeze(0),
                                                  out_feat_1[4, :, :, :].unsqueeze(0),
                                                  out_feat_2[4, :, :, :].unsqueeze(0),
                                                  out_feat_3[4, :, :, :].unsqueeze(0))
        view5 = self.feat_before_merge_v4[f'{5}'](out_feat_0[5, :, :, :].unsqueeze(0),
                                                  out_feat_1[5, :, :, :].unsqueeze(0),
                                                  out_feat_2[5, :, :, :].unsqueeze(0),
                                                  out_feat_3[5, :, :, :].unsqueeze(0))
        view6 = self.feat_before_merge_v4[f'{6}'](out_feat_0[6, :, :, :].unsqueeze(0),
                                                  out_feat_1[6, :, :, :].unsqueeze(0),
                                                  out_feat_2[6, :, :, :].unsqueeze(0),
                                                  out_feat_3[6, :, :, :].unsqueeze(0))
        warped_feat = torch.cat([view0, view1, view2, view3, view4, view5, view6], dim=0)
        return warped_feat
    def warp_perspective_v5(self, img_feature_all, proj_mats):
        return self.depth_v5(img_feature_all, proj_mats)

    def _base(self, x):
        return self.base(x)

    def _bottleneck(self, x):
        return self.bottleneck(x)

    def _base_bottleneck(self, x):
        x = self.base(x)
        return self.bottleneck(x)



    def forward(self, imgs, M, logdir=None, visualize=False, epoch=None):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        inverse_affine_mats = torch.inverse(M.view([B * N, 3, 3]))
        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ torch.from_numpy(np.diag([self.img_reduce, self.img_reduce, 1])).view(1, 3, 3).repeat(B * N, 1, 1).float()   # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        proj_mats = [self.proj_mats[i].repeat(B, 1, 1, 1).view(B * N, 3, 3).float() @ imgcoord_from_Rimggrid_mat.to(device) for i in range(self.depth_scales)   ]            # Rworldgrid(xy)_from_Rimggrid(xy)
        if visualize and epoch==1:
            assert logdir is not None
            denorm = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ##############################################
            # epoch
            #   └──imgs
            #       ├──augimg1.png
            #       ├──augimg2.png
            #       ├──augimg3.png
            #       ├──augimg4.png
            #       ├──augimg5.png
            #       ├──augimg6.png
            #       ├──augimg7.png
            #       ├──depth1_augimgproj1.png   depth2_augimgproj1.png  depth3_augimgproj1.png  depth4_augimgproj1.png
            #       ├──depth1_augimgproj2.png   depth2_augimgproj2.png  depth3_augimgproj2.png  depth4_augimgproj2.png
            #       ├──depth1_augimgproj3.png   depth2_augimgproj3.png  depth3_augimgproj3.png  depth4_augimgproj3.png
            #       ├──depth1_augimgproj4.png   depth2_augimgproj4.png  depth3_augimgproj4.png  depth4_augimgproj4.png
            #       ├──depth1_augimgproj5.png   depth2_augimgproj5.png  depth3_augimgproj5.png  depth4_augimgproj5.png
            #       ├──depth1_augimgproj6.png   depth2_augimgproj6.png  depth3_augimgproj6.png  depth4_augimgproj6.png
            #       ├──depth1_augimgproj7.png   depth2_augimgproj7.png  depth3_augimgproj7.png  depth4_augimgproj7.png
            ##############################################
            for cam in range(N):
                visualize_img = T.ToPILImage()(denorm(imgs.detach())[cam * B])
                visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/augimg{cam + 1}.png'))
            for i in range(self.depth_scales):
                proj_imgs = warp_perspective(T.Resize(self.Rimg_shape)(imgs), proj_mats[i], self.Rworld_shape).view(B, N, 3, self.Rworld_shape[0], self.Rworld_shape[1])
                for cam in range(N):
                    visualize_img = T.ToPILImage()(denorm(proj_imgs.detach())[0, cam])
                    visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/depth{i + 1}_augimgproj{cam + 1}.png'))

        # 统一设备: 上游 中游 下游
        imgs.to(device)
        self.base.to(device)
        self.bottleneck.to(device)
        self.img_heatmap.to(device)
        self.img_offset.to(device)
        self.img_wh.to(device)

        # self.depth_classifier.to(device)
        # self.feat_before_merge.to(device)
        # self.depth_v5.to(device)

        self.world_feat.to(device)
        self.world_heatmap.to(device)
        self.world_offset.to(device)


        # 上游
        imgs_feat = self.base(imgs)
        # imgs_feat = checkpoint(self._base, imgs)               # 检查点优化
        imgs_feat = self.bottleneck(imgs_feat)
        # imgs_feat = checkpoint(self._bottleneck, imgs_feat)    # 检查点优化
        # imgs_feat = checkpoint(self._base_bottleneck, imgs)    # 检查点优化

        _, C, H, W   = imgs_feat.shape
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset  = self.img_offset(imgs_feat)
        imgs_wh      = self.img_wh(imgs_feat)
        if visualize:
            ##############################################
            # epoch
            #   └──imgs
            #       ├──augimgfeat1.png
            #       ├──augimgfeat2.png
            #       ├──augimgfeat3.png
            #       ├──augimgfeat4.png
            #       ├──augimgfeat5.png
            #       ├──augimgfeat6.png
            #       ├──augimgfeat7.png
            ##############################################
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
                visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/augimgfeat{cam + 1}.png'))


        # 中游
        H, W = self.Rworld_shape
        world_feat = self.warp_perspective(imgs_feat, proj_mats).view(B, N, C, H, W)
        # world_feat = self.warp_perspective_v3_refactoring(imgs_feat, proj_mats).view(B, N, C, H, W)
        # world_feat = self.warp_perspective_v5(imgs_feat, proj_mats).view(B, N, C, H, W)
        # world_feat = checkpoint(lambda x, mats: self.warp_perspective(x, mats).view(B, N, C, H, W), imgs_feat, proj_mats)    # 检查点优化
        # world_feat = checkpoint(lambda x, mats: self.warp_perspective_v3_refactoring(x, mats).view(B, N, C, H, W), imgs_feat, proj_mats)  # 检查点优化
        # world_feat = checkpoint(lambda x, mats: self.warp_perspective_v5(x, mats).view(B, N, C, H, W), imgs_feat, proj_mats)    # 检查点优化
        if visualize:
            ##############################################
            # epoch
            #   └──imgs
            #       ├──projfeat1.png
            #       ├──projfeat2.png
            #       ├──projfeat3.png
            #       ├──projfeat4.png
            #       ├──projfeat5.png
            #       ├──projfeat6.png
            #       ├──projfeat7.png
            ##############################################
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(world_feat[0, cam].detach(), dim=0).cpu())
                visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/projfeat{cam + 1}.png'))

        # 统一设备
        world_feat.to(device)
        self.world_feat.to(device)
        self.world_heatmap.to(device)
        self.world_offset.to(device)

        # 下游
        world_feat = self.world_feat(world_feat, visualize=visualize)
        # world_feat = checkpoint(lambda x: self.world_feat(x, visualize=False), world_feat, use_reentrant=False) if not visualize else self.world_feat(world_feat, visualize=visualize)    # 检查点优化
        world_heatmap = self.world_heatmap(world_feat)
        world_offset = self.world_offset(world_feat)
        if visualize:
            ##############################################
            # epoch
            #   └──imgs
            #       ├──worldfeatall.png
            #       ├──worldres.png
            ##############################################
            visualize_img = array2heatmap(torch.norm(world_feat[0].detach(), dim=0).cpu())
            visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/worldfeatall.png'))
            visualize_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            visualize_img.save(os.path.join(logdir, f'imgs/epoch{epoch}/worldres.png'))

        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)

