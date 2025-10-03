"""SHISRCNet: Super-resolution And Classification Network For Low-resolution Breast Cancer Histopathology Image"""
import torch.nn as nn
import torch

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels
        # self.project = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x0,x1,x2,x3):
        y0 = x0
        y1 = x1
        y2 = x2
        y3 = x3
        # 通过池化聚合全局信息,然后通过1×1conv建模通道相关性: (B,C,H,W)-->GAP-->(B,C,1,1)-->SE1-->(B,C,1,1)
        y0_weight = self.SE1(self.gap(x0))
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))
        y3_weight = self.SE4(self.gap(x3))
        # 将多个尺度的全局信息进行拼接: (B,C,4,1),首先通过sigmoid函数获得通道描述符表示, 然后通过softmax函数,求每个尺度的权重: (B,C,4,1)--> (B,C,4,1)
        weight = self.softmax(self.Sigmoid(torch.cat([y0_weight,
                                                            y1_weight,
                                                            y2_weight,
                                                            y3_weight],2)))
        # weight[:,:,0]:(B,C,1); (B,C,1)-->unsqueeze-->(B,C,1,1)
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        # 将权重与对应的输入进行逐元素乘法: (B,C,1,1) * (B,C,H,W)= (B,C,H,W), 然后将多个尺度的输出进行相加
        x_att = (y0_weight*y0+
                 y1_weight*y1+
                 y2_weight*y2+
                 y3_weight*y3)
        return x_att
        # return self.project(x_att)


if __name__ == '__main__':
    x0 = torch.rand(1, 128, 120, 360)
    x1 = torch.rand(1, 128, 120, 360)
    x2 = torch.rand(1, 128, 120, 360)
    x3 = torch.rand(1, 128, 120, 360)
    Model = MSFblock(in_channels=128)
    result =Model(x0,x1,x2,x3)
    print(result.shape)                             # (1,128,120,360)
    concate = torch.cat([x0,x1,x2,x3],dim=0)
    print(concate.shape)                            # (4, 128, 120, 360)
