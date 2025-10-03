import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.loss import *
from multiview_detector.utils.nms import nms
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.image_utils import add_heatmap_to_image, img_color_denormalize
device = 'cuda:0'
class PerspectiveTrainer(object):
    def __init__(self, model, logdir, cls_thres=0.4, alpha=1.0, use_mse=False, id_ratio=0, visualize=False):
        super(PerspectiveTrainer, self).__init__()
        self.model          = model
        self.mse_loss       = nn.MSELoss()
        self.focal_loss     = FocalLoss()
        self.regress_loss   = RegL1Loss()
        self.ce_loss        = RegCELoss()
        self.cls_thres      = cls_thres
        self.logdir         = logdir
        self.denormalize    = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.alpha          = alpha
        self.use_mse        = use_mse
        self.id_ratio       = id_ratio



    def train(self, epoch, dataloader, optimizer, scaler, scheduler=None, log_interval=100):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.to(device)
            self.model = self.model.to(device)
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats)

            if self.use_mse:
                loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + self.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))
            else:
                loss_w_hm    = self.focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off   = self.regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                w_loss       = loss_w_hm + loss_w_off
                loss_img_hm  = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh  = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                img_loss     = loss_img_hm + loss_img_off + loss_img_wh * 0.1
                loss         = w_loss + img_loss / N * self.alpha

            t_f = time.time()
            t_forward += t_f - t_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, LR: {optimizer.param_groups[0]["lr"]}, Time: {t_epoch:.1f}, maxima: {world_heatmap.max():.3f}')
                pass
        return losses / len(dataloader)






    def test(self, epoch, dataloader, res_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.to(device)
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            # with autocast():
            with torch.no_grad():
                (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats, self.logdir, visualize, epoch)
                if self.use_mse:
                    loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + self.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))
                else:
                    loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
                    loss = loss_w_hm
            losses += loss.item()
            if res_fpath is not None:
                xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(), reduce=dataloader.dataset.world_reduce)
                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]
                for b in range(B):
                    ids = scores[b].squeeze() > self.cls_thres
                    pos, s = positions[b, ids], scores[b, ids, 0]
                    res = torch.cat([torch.ones([len(s), 1]) * frame[b], pos], dim=1)
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    res_list.append(res)
        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            ##############################################
            # epoch
            #   ├──world.jpg
            #   ├──cam1_foot.jpg
            #   ├──cam2_foot.jpg
            #   ├──cam3_foot.jpg
            #   ├──cam4_foot.jpg
            #   ├──cam5_foot.jpg
            #   ├──cam6_foot.jpg
            #   ├──cam7_foot.jpg
            ##############################################
            fig = plt.figure()  # 创建一个新的图形对象
            subplt0 = fig.add_subplot(211, title="output")  # 添加第一个子图
            subplt1 = fig.add_subplot(212, title="target")  # 添加第二个子图
            subplt0.imshow(world_heatmap.cpu().detach().numpy().squeeze())  # 将world_heatmap移动到CPU，移除梯度信息，转换为NumPy数组，去除单维度，使用imshow在第一个子图中显示
            subplt1.imshow(world_gt['heatmap'].squeeze())  # 将world_gt字典中的heatmap键对应的值，去除单维度，在第二个子图中显示
            plt.savefig(os.path.join(self.logdir, f'imgs/epoch{epoch}/world.jpg'))  # 将整个图形保存为一个JPEG文件
            plt.close(fig)  # 关闭当前图形对象，释放资源。
            # N = 7
            # for i in range(N):
            #     heatmap_foot = imgs_heatmap[i].detach().cpu().numpy().squeeze()
            #     img = self.denormalize(data[0, i]).cpu().numpy().squeeze().transpose([1, 2, 0])
            #     img = Image.fromarray((img * 255).astype('uint8'))
            #     foot_cam_result = add_heatmap_to_image(heatmap_foot, img)
            #     foot_cam_result.save(os.path.join(self.logdir, f'imgs/epoch{epoch}/cam{i + 1}_foot.jpg'))


        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(dataloader.dataset.gt_fpath), dataloader.dataset.base.__name__)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0
        print(f'Test, loss: {losses / len(dataloader):.6f}, Time: {t_epoch:.3f}')
        # print(f"α: {self.model.middle_MSFblock.feat_before_merge_v5:.4f} | 池化尺寸: {}")
        return losses / len(dataloader), moda



    """打印Middle_MSFblock_case7_ShareWeightMultiAdaptivePool2d中所有的alpha值"""
    # def print_alphas(self):
    #     with torch.no_grad():
    #         alphas = []
    #         for i in range(7):
    #             msf_block = self.model.depth_v5.feat_before_merge_v5[str(i)]
    #             alpha_raw = msf_block.alpha
    #             alphas.append(alpha_raw)
    #         print(alphas)

    """ 打印Middle_MSFblock_case8_ShareWeightMultiAdaptivePool2d_alpha_beta中所有的alpha值 和 beta值 """
    # def print_alphas_beta(self):
    #     with torch.no_grad():
    #         alphas, beta = [], []
    #         for i in range(7):
    #             msf_block = self.model.depth_v5.feat_before_merge_v5[str(i)]
    #             # block_alphas = [param.data.cpu().numpy() for param in msf_block.alphas]
    #             # block_beta = msf_block.beta.data.cpu().numpy()
    #             # softmax归一化（dim=0）
    #             block_alphas = [F.softmax(param, dim=0).cpu().numpy() for param in msf_block.alphas]
    #             block_beta = F.softmax(msf_block.beta, dim=0).cpu().numpy()
    #
    #             alphas.append(block_alphas)
    #             beta.append(block_beta)
    #         print("===== Alpha Beta =====")
    #         for block_idx, (block_alpha, block_beta) in enumerate(zip(alphas, beta)):
    #             print(f"View {block_idx}:")
    #             for branch_idx, alpha in enumerate(block_alpha):
    #                 print(f"  alpha Branch_{branch_idx}:", alpha)
    #             print(f"  beta:", block_beta)

    """ 打印Middle_MSFblock_case9_ShareWeightMultiAdaptivePool2d_alpha中所有的alpha值 """
    # def print_alphas(self):
    #     with torch.no_grad():
    #         alphas = []
    #         for i in range(7):
    #             msf_block = self.model.depth_v5.feat_before_merge_v5[str(i)]
    #             alphas.append(F.softmax(msf_block.alpha, dim=0).cpu().numpy())
    #         print("===== Alpha =====")
    #         for view_idx, alpha in enumerate(alphas):
    #             print(f"View {view_idx}:", alpha)

    """打印 LocalSimGuidedSampler 中 四大参数 """
    # def print_sampler(self):
    #     with torch.no_grad():
    #         pool_sizes = self.model.depth_v5.feat_before_merge_v5['0'].pool_sizes
    #         print(f'pool_sizes={pool_sizes}')
    #         sampler = self.model.world_feat.sampler
    #         print('余弦窗口大小local_window      =', sampler.local_window)
    #         print('   ├──膨胀系数dilation        =', sampler.dilation_for_compute_similarity)
    #         print('   ├──二选一sim & sim_concate =', sampler.direction_feat)
    #         print('   └──采样组数groups           =', sampler.groups)


    """打印 LocalSimGuidedSampler 中 三个采样器的参数 """
    def print_sampler(self):
        with torch.no_grad():
            # pool_sizes = self.model.depth_v5.feat_before_merge_v5['0'].pool_sizes
            # print(f'pool_sizes={pool_sizes}')
            sampler0 = self.model.world_feat.sampler0
            # sampler1 = self.model.world_feat.sampler1
            # sampler2 = self.model.world_feat.sampler2
            print(f'[kernel_size={sampler0.kernel_size}, groups={sampler0.groups}] {sampler0.direction_feat} [local_window={sampler0.local_window}, dilation={sampler0.dilation_for_compute_similarity}]')
            # print(f'[kernel_size={sampler1.kernel_size}, groups={sampler1.groups}] {sampler1.direction_feat} [local_window={sampler1.local_window}, dilation={sampler1.dilation_for_compute_similarity}]')
            # print(f'[kernel_size={sampler2.kernel_size}, groups={sampler2.groups}] {sampler2.direction_feat} [local_window={sampler2.local_window}, dilation={sampler2.dilation_for_compute_similarity}]')

            # sampler0_1 = self.model.world_feat.sampler0_1
            # sampler1_1 = self.model.world_feat.sampler1_1
            # sampler2_1 = self.model.world_feat.sampler2_1
            # print(f'[kernel_size={sampler0_1.kernel_size}, groups={sampler0_1.groups}] {sampler0_1.direction_feat} [local_window={sampler0_1.local_window}, dilation={sampler0_1.dilation_for_compute_similarity}]')
            # print(f'[kernel_size={sampler1_1.kernel_size}, groups={sampler1_1.groups}] {sampler1_1.direction_feat} [local_window={sampler1_1.local_window}, dilation={sampler1_1.dilation_for_compute_similarity}]')
            # print(f'[kernel_size={sampler2_1.kernel_size}, groups={sampler2_1.groups}] {sampler2_1.direction_feat} [local_window={sampler2_1.local_window}, dilation={sampler2_1.dilation_for_compute_similarity}]')

            feat_enhanced = self.model.world_feat.feat_enhanced
            print(feat_enhanced)

    """打印 mask 阈值"""
    def print_thresholds(self):
        with torch.no_grad():
            print('################### 获取阈值并打印')
            thresholds = self.model.world_feat.mask.get_current_thresholds()
            for v in range(len(thresholds)):
                print(f"视角{v + 1}: {thresholds[v].item():.8f}")
            # print(f'视角概率分布: {self.model.world_feat.mask.view_weights}')
            if hasattr(self.model.world_feat.mask, 'theta'):        # 检查是否存在 theta 属性
                print(f"当前theta: {self.model.world_feat.mask.theta:.4f}")
            else:
                pass