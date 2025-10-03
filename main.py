import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'     # 128 256 512
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keep_learning = False
path = f'/home/wangzc/exps/GJJ_Proj/improve_SHOT/logs/wildtrack/SHOT_for_keeplearning/MultiviewDetector.pth'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from multiview_detector.datasets import *
from multiview_detector.models.shot import SHOT
# from multiview_detector.models.mvdetr import MVDeTr
# from multiview_detector.models.boostershot import BoosterSHOT
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID',              action='store_true')
    parser.add_argument('--semi_supervised',   type=float, default=0)
    parser.add_argument('--id_ratio',          type=float, default=0)
    parser.add_argument('--cls_thres',         type=float, default=0.6)
    parser.add_argument('--alpha',             type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse',           type=str2bool, default=False)
    parser.add_argument('--arch',              type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('-d', '--dataset',     type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=0)
    parser.add_argument('-b', '--batch_size',  type=int, default=1)
    parser.add_argument('--dropout',           type=float, default=0.0)
    parser.add_argument('--dropcam',           type=float, default=0.0)
    parser.add_argument('--model',             type=str, default='SHOT', choices=['mvdet', 'SHOT', 'BoosterSHOT', 'MVDeTr'])  # 'MVDeTr', 'Test', 'SHOT', 'BoosterSHOT'
    parser.add_argument('--optimizer',         type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--depth_scales',      type=int, default=4)
    parser.add_argument('--epochs',            type=int, default=200)
    parser.add_argument('--lr',                type=float, default=5e-4)
    # added to adjust OneCycleLR final learning rate (if needed)
    parser.add_argument('--base_lr_ratio',     type=float, default=0.1)
    parser.add_argument('--final_div_factor',  type=float, default=1e4)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--resume',            type=str, default=None)
    parser.add_argument('--visualize',         action='store_true')
    parser.add_argument('--seed',              type=int, default=2021)
    parser.add_argument('--deterministic',     type=str2bool, default=False)
    parser.add_argument('--augmentation',      type=str2bool, default=True)
    parser.add_argument('--world_feat',        type=str, default='conv', choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio', 'freqfusion'])
    parser.add_argument('--bottleneck_dim',    type=int, default=128)
    parser.add_argument('--outfeat_dim',       type=int, default=0)
    ##############################################
    # 世界地图的尺寸 = [480,1440] * 1 / world_reduce
    # 输入模型的尺寸 = [1080,1920] * 8 / img_reduce
    ##############################################
    parser.add_argument('--world_reduce',      type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce',        type=int, default=12)
    parser.add_argument('--img_kernel_size',   type=int, default=10)
    args = parser.parse_args()

    ##############################################
    # debug_mode
    # seed
    # deterministic
    ##############################################
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    if args.seed is not None:
        random.seed(args.seed)                  # 随机数生成器
        np.random.seed(args.seed)               # numpy随机数生成器
        torch.manual_seed(args.seed)            # Torch随机数生成器
        torch.cuda.manual_seed(args.seed)       # CUDA随机数生成器
        torch.cuda.manual_seed_all(args.seed)   # 所有CUDA随机数生成器

    if args.deterministic:
        torch.backends.cudnn.deterministic = True   # 设置 PyTorch 的 cuDNN 库为确定性模式
        torch.backends.cudnn.benchmark = False      # 禁用 cuDNN 的自动寻找最适合当前配置的算法的功能
        torch.autograd.set_detect_anomaly(True)     # 设置 PyTorch 自动梯度检测异常功能
    else:
        torch.backends.cudnn.benchmark = True       # 启用 cuDNN 的自动寻找最适合当前配置的算法的功能





    ##############################################
    # base
    # frameDataset
    # DataLoader
    ##############################################
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('/home/wangzc/exps/GJJ_Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('/home/wangzc/exps/GJJ_Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, world_reduce=args.world_reduce, img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size, img_kernel_size=args.img_kernel_size, semi_supervised=args.semi_supervised, dropout=args.dropcam, augmentation=args.augmentation)
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce, img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size, img_kernel_size=args.img_kernel_size)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)


    ##############################################
    # logging = logs/wildtrack/aug_freqfusion_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK24_10_2024-12-26_15-12-08
    ##############################################
    if args.resume is None:
        logdir = f'logs/' \
                 f'{args.dataset}/'\
                 f'{"debug_" if is_debug else ""}'\
                 f'{"SS_" if args.semi_supervised else ""}' \
                 f'{"aug_" if args.augmentation else ""}' \
                 f'{args.world_feat}'\
                 f'_lr{args.lr}' \
                 f'_baseR{args.base_lr_ratio}' \
                 f'_neck{args.bottleneck_dim}' \
                 f'_out{args.outfeat_dim}' \
                 f'_alpha{args.alpha}' \
                 f'_id{args.id_ratio}' \
                 f'_drop{args.dropout}' \
                 f'_dropcam{args.dropcam}' \
                 f'_worldRK{args.world_reduce}_{args.world_kernel_size}' \
                 f'_imgRK{args.img_reduce}_{args.img_kernel_size}' \
                 f'_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'imgs'), exist_ok=True)
        if args.model == 'ABCDet':
            for i in range(args.depth_scales):
                os.makedirs(os.path.join(logdir, f'imgs/{i + 1}'), exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        logdir = f'logs/{args.dataset}/{args.resume}'
    print(logdir)
    print(vars(args))



    ##############################################
    # model
    # param_dicts
    # optim
    # GradScaler
    # scheduler
    # PerspectiveTrainer
    ##############################################
    if args.model == 'SHOT':
        model = SHOT(train_set, args.arch, world_feat_arch=args.world_feat, bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, dropout=args.dropout, depth_scales=args.depth_scales)
        print(model)
        ##############################################
        # 中断模型地址
        ##############################################
        if keep_learning == True:
            model.load_state_dict(torch.load(path))
            print('成功加载中断的权重,继续训练')
        ##############################################
        # 打印模型参数量
        ##############################################
        # import torch.nn as nn
        # print("-" * 50)
        # for name, module in model.named_children():
        #     print(f"{name}: {type(module).__name__}")
        #
        # # 打印总参数量
        # total_params = sum(p.numel() for p in model.parameters())
        # print("\n" + "=" * 50)
        # print(f"Total model parameters: {total_params:,}")
        # print("=" * 50)
        #
        # # 单独分析world_feat模块中的卷积层参数
        # print("\nAnalyzing ConvWorldFeat module:")
        # print("=" * 50)
        # total_conv_params = 0
        # for name, module in model.world_feat.named_modules():
        #     # 跳过顶级模块本身
        #     if name == '':
        #         continue
        #
        #     # 检查是否为卷积层
        #     if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        #         print(f"\n[Layer]: {name}")
        #         print(f"  Type: {type(module).__name__}")
        #         print(f"  Input channels: {module.in_channels}")
        #         print(f"  Output channels: {module.out_channels}")
        #         print(f"  Kernel size: {module.kernel_size}")
        #         print(f"  Stride: {module.stride}")
        #         print(f"  Padding: {module.padding}")
        #
        #         # 统计参数量
        #         params = sum(p.numel() for p in module.parameters())
        #         total_conv_params += params
        #         print(f"  Parameters: {params:,}")
        #
        #         # 打印权重形状
        #         print(f"  Weight shape: {list(module.weight.shape)}")
        #         if module.bias is not None:
        #             print(f"  Bias shape: {list(module.bias.shape)}")
        #
        # print("\n" + "-" * 50)
        # print(f"Total convolution parameters in world_feat: {total_conv_params:,}")
        # print("=" * 50)
        # exit()
    elif args.model == 'BoosterSHOT':
        # model = BoosterSHOT(train_set, args.arch, world_feat_arch=args.world_feat, bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, dropout=args.dropout, depth_scales=args.depth_scales).cuda()
        # print(model)
        # exit()
        pass
    elif args.model == 'MVDeTr':
        # model = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat, bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, dropout=args.dropout).cuda()
        # print(model)
        # exit()
        pass
    elif args.model == 'mvdet':
        from multiview_detector.models.mvdet import mvdet
        model = mvdet(train_set, args.arch)
        args.use_mse = True
    else:
        raise Exception('The selected model is not supported.')
    param_dicts = [{"params": [p for n, p in model.named_parameters() if 'base' not in n and p.requires_grad], },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, }, ]

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise Exception('The selected optimizer is not supported.')

    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, final_div_factor=args.final_div_factor, steps_per_epoch=len(train_loader), epochs=args.epochs)
    trainer = PerspectiveTrainer(model, logdir, args.cls_thres, args.alpha, args.use_mse, args.id_ratio, args.visualize)
    # trainer.print_sampler()


    ##############################################
    # list for learning_curve.jpg
    # training for epochs
    ##############################################
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []
    res_fpath = os.path.join(logdir, 'test.txt')
    max_moda = 0.0
    max_moda_epoch = 1
    if args.resume is None:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            epochdir = logdir + f'/imgs/epoch{epoch}'
            os.makedirs(epochdir, exist_ok=True)

            trainer.print_thresholds()
            train_loss      = trainer.train(epoch, train_loader, optimizer, scaler, scheduler)
            test_loss, moda = trainer.test(epoch, test_loader, res_fpath, visualize=True)


            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            test_loss_s.append(test_loss)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
            if moda >= max_moda:
                max_moda = moda
                max_moda_epoch = epoch
                torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
            print('最大moda:', max_moda, '最大moda的epoch是:', max_moda_epoch)

        # 这两行代码与else的功能一样，只有路径不一样
        model.load_state_dict(torch.load(os.path.join(logdir, 'MultiviewDetector.pth')))
        model.eval()
    else:
        model.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
        model.eval()
