import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2
import numpy as np
from .swintransformer import *

class ONEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.one_conv(x)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    
class ONEConvCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            CBAM(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.one_conv(x)

class TWOConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.two_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.two_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.two_conv1(x) + x
        x = self.relu1(x)
        x = self.two_conv2(x) + x
        x = self.relu2(x)
        return x

class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d):
        super(NeighborFeatureAggregation, self).__init__()
        self.in_d = in_d
        self.CatChannels = in_d[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        # self.downsample = nn.AvgPool2d(stride=2, kernel_size=2)
        # self.downsample = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2d1 = TWOConv(self.in_d[0], self.in_d[0])
        self.conv2d2 = TWOConv(self.in_d[1], self.in_d[1])
        self.conv2d3 = TWOConv(self.in_d[2], self.in_d[2])
        self.conv2d4 = TWOConv(self.in_d[3], self.in_d[3])
        self.conv2d5 = TWOConv(self.in_d[4], self.in_d[4])


        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = ONEConv(self.in_d[0], self.CatChannels)
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = ONEConv(self.in_d[1], self.CatChannels)
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = ONEConv(self.in_d[2], self.CatChannels)
        self.h4_Cat_hd4_conv = ONEConv(self.in_d[3], self.CatChannels)
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = ONEConv(self.in_d[4], self.CatChannels)
        self.conv4d_1 = ONEConvCBAM(self.UpChannels, self.UpChannels)

        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = ONEConv(self.in_d[0], self.CatChannels)
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = ONEConv(self.in_d[1], self.CatChannels)
        self.h3_Cat_hd3_conv = ONEConv(self.in_d[2], self.CatChannels)
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = ONEConv(self.in_d[4], self.CatChannels)
        self.conv3d_1 = ONEConvCBAM(self.UpChannels, self.UpChannels)

        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = ONEConv(self.in_d[0], self.CatChannels)
        self.h2_Cat_hd2_conv = ONEConv(self.in_d[1], self.CatChannels)
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = ONEConv(self.in_d[4], self.CatChannels)
        self.conv2d_1 = ONEConvCBAM(self.UpChannels, self.UpChannels)

        self.h1_Cat_hd1_conv = ONEConv(self.in_d[0], self.CatChannels)
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = ONEConv(self.UpChannels, self.CatChannels)
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = ONEConv(self.in_d[4], self.CatChannels)
        self.conv1d_1 = ONEConvCBAM(self.UpChannels, self.UpChannels)

        self.outconv1 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(self.in_d[4], 1, 3, padding=1)
        self.upscore5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore1 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5):
        c1 = torch.abs(x1_1 - x2_1)
        c2 = torch.abs(x1_2 - x2_2)
        c3 = torch.abs(x1_3 - x2_3)
        c4 = torch.abs(x1_4 - x2_4)
        c5 = torch.abs(x1_5 - x2_5)

        # d1 = self.downsample(c1)
        # d1 = F.interpolate(d1, scale_factor=(2, 2), mode='bilinear')
        d1 = self.conv2d1(c1)
        d2 = self.conv2d2(c2)
        d3 = self.conv2d3(c3)
        d4 = self.conv2d4(c4)
        d5 = self.conv2d5(c5)

        
        h1_PT_hd4 = self.h1_PT_hd4_conv(self.h1_PT_hd4(d1))
        h2_PT_hd4 = self.h2_PT_hd4_conv(self.h2_PT_hd4(d2))
        h3_PT_hd4 = self.h3_PT_hd4_conv(self.h3_PT_hd4(d3))
        h4_Cat_hd4 = self.h4_Cat_hd4_conv(d4)
        hd5_UT_hd4 = self.hd5_UT_hd4_conv(self.hd5_UT_hd4(d5))
        hd4 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        h1_PT_hd3 = self.h1_PT_hd3_conv(self.h1_PT_hd3(d1))
        h2_PT_hd3 = self.h2_PT_hd3_conv(self.h2_PT_hd3(d2))
        h3_Cat_hd3 = self.h3_Cat_hd3_conv(d3)
        hd4_UT_hd3 = self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))
        hd5_UT_hd3 = self.hd5_UT_hd3_conv(self.hd5_UT_hd3(d5))
        hd3 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))

        h1_PT_hd2 = self.h1_PT_hd2_conv(self.h1_PT_hd2(d1))
        h2_Cat_hd2 = self.h2_Cat_hd2_conv(d2)
        hd3_UT_hd2 = self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))
        hd4_UT_hd2 = self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))
        hd5_UT_hd2 = self.hd5_UT_hd2_conv(self.hd5_UT_hd2(d5))
        hd2 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))

        h1_Cat_hd1 = self.h1_Cat_hd1_conv(d1)
        hd2_UT_hd1 = self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))
        hd3_UT_hd1 = self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))
        hd4_UT_hd1 = self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))
        hd5_UT_hd1 = self.hd5_UT_hd1_conv(self.hd5_UT_hd1(d5))
        hd1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
        d5 = self.upscore5(d5)
        d5 = self.outconv5(d5)
        d4 = self.upscore4(hd4)
        d4 = self.outconv4(d4)
        d3 = self.upscore3(hd3)
        d3 = self.outconv3(d3)
        d2 = self.upscore2(hd2)
        d2 = self.outconv2(d2)
        d1 = self.upscore1(hd1)
        d1 = self.outconv1(d1)  #

        return torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)

class ONEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.one_conv(x)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.residualBlock1 = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0])
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.residualBlock2 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[1], self.channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[1])
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.residualBlock3 = nn.Sequential(
            nn.Conv2d(self.channels[2], self.channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[2], self.channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[2])
        )
        self.relu3 = nn.ReLU(inplace=True)
        self.residualBlock4 = nn.Sequential(
            nn.Conv2d(self.channels[3], self.channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[3], self.channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[3])
        )
        self.relu4 = nn.ReLU(inplace=True)
        self.residualBlock5 = nn.Sequential(
            nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[4])
        )
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x1,x2,x3,x4,x5,y1,y2,y3,y4,y5):
        out1 = self.residualBlock1(x1) + y1
        out1 = self.relu1(out1)
        out2 = self.residualBlock2(x2) + y2
        out2 = self.relu2(out2)
        out3 = self.residualBlock3(x3) + y3
        out3 = self.relu3(out3)
        out4 = self.residualBlock4(x4) + y4
        out4 = self.relu4(out4)
        out5 = self.residualBlock5(x5) + y5
        out5 = self.relu5(out5)
        

        return out1,out2,out3,out4,out5
class TCD_Net(nn.Module):
    def __init__(self, n_channels=6, n_classes=1):
        super(TCD_Net, self).__init__()

        self.swin1 = Swin_Model(chan=16, dim=16, patch_size=1, num_heads=4,
                                input_res=[128, 128], window_size=8, qkv_bias=True)
        self.swin2 = Swin_Model(chan=24, dim=24, patch_size=1, num_heads=4,
                                input_res=[64, 64], window_size=8, qkv_bias=True)
        self.swin3 = Swin_Model(chan=32, dim=32, patch_size=1, num_heads=4,
                                input_res=[32, 32], window_size=8, qkv_bias=True)
        # self.se1 = SEBlock(64)
        self.swin4 = Swin_Model(chan=96, dim=96, patch_size=1, num_heads=4,
                                input_res=[16, 16], window_size=8, qkv_bias=True)
        self.swin5 = Swin_Model(chan=320, dim=320, patch_size=1, num_heads=4,
                                input_res=[8, 8], window_size=8, qkv_bias=True)
    def forward(self, x1,x2,x3,x4,x5):
        x1 = self.swin1(x1)
        x2 = self.swin2(x2)
        x3 = self.swin3(x3)
        x4 = self.swin4(x4)
        x5 = self.swin5(x5)

        return x1,x2,x3,x4,x5

    
class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        channles = [16, 24, 32, 96, 320]
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        self.ResidualBlock = ResidualBlock(channles)
        self.swb = TCD_Net(6, 1)
        self.swa = NeighborFeatureAggregation(channles)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        y1_1, y1_2, y1_3, y1_4, y1_5 = self.swb(x1_1, x1_2, x1_3, x1_4, x1_5)
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.ResidualBlock(x1_1, x1_2, x1_3, x1_4, x1_5,y1_1, y1_2, y1_3, y1_4, y1_5)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        y2_1, y2_2, y2_3, y2_4, y2_5 = self.swb(x2_1, x2_2, x2_3, x2_4, x2_5)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.ResidualBlock(x2_1, x2_2, x2_3, x2_4, x2_5, y2_1, y2_2, y2_3, y2_4, y2_5)
        a, b, c, d, e = self.swa(x1_1, x1_2, x1_3, x1_4, x1_5, x2_1, x2_2, x2_3, x2_4, x2_5)

        return a, b, c, d, e
