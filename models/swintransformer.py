# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np


##single_Swin_block

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H//window_size, W//window_size, window_size, window_size, C]
    x = x.reshape(-1, window_size, window_size, C)  # [B*num_window, window_size, window_size, C]

    return x


def window_reverse(windows, window_size, H, W): # [B*num_windows, window_size, window_size, C]
    # windows:[B*num_window, window_size, window_size, C]
    B = int(windows.shape[0] // (H / window_size * W / window_size)) #计算批量大小 B：通过窗口的总数除以一个图片中的窗口数量。
    # x: [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H//window_size, window_size, W//window_size, window_size, C]

    x = x.reshape(B, H, W, -1)
    return x


def generate_mask(input_res, window_size, shift_size):
    H, W, = input_res
    Hp = int(np.ceil(H / window_size)) * window_size #Hp, Wp：将输入分辨率向上取整到窗口大小的倍数，确保整齐划分
    Wp = int(np.ceil(W / window_size)) * window_size

    image_mask = torch.zeros((1, Hp, Wp, 1)) #image_mask：创建全零掩码张量，形状 [1, Hp, Wp, 1] 表示单个批量（batch），调整后的分辨率 (Hp, Wp)，最后一维为 1（存储区域编号）。
    h_slice = (slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)
               )

    w_slice = (slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)
               )

    cnt = 0 #为每个区域分配唯一编号
    for h in h_slice:
        for w in w_slice:
            image_mask[:, h, w, :] = cnt
            cnt += 1
    mask_window = window_partition(image_mask, window_size) # [B*num_window, window_size, window_size, C] 这里的B和C都为1其实就是[num_window, window_size, window_size, 1]
    mask_window = mask_window.reshape(-1, window_size * window_size) #[num_window, window_size*window_size]

    attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2) #将每个窗口的区域编号展平，便于后续注意力计算[num_window, window_size*window_size, window_size*window_size]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class Patch_Embeding(nn.Module):
    def __init__(self, chan=3, dim=96, patch_size=4):
        super().__init__()
        self.patch = nn.Conv2d(chan, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x)  # [B, C, H, W] , C = dim 不重复的卷积
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, C]
        x = self.norm(x)
        return x
#输入进swin blocks的就是B HW C

class Patch_Merging(nn.Module):
    def __init__(self, input_res, dim):  #256  64
        super().__init__()
        self.resolution = input_res         #256
        self.dim = dim                 #64

        self.reduction = nn.Linear(dim, dim)          #256  256
        self.norm = nn.LayerNorm(dim)              #256

    def forward(self, x):
        # x: [B, num_patches, C]
        H, W = self.resolution    #256  256
        B, _, C = x.shape
        x = x.reshape(B, H, W, C)   #1,256,256,64
        x = x.reshape(B, -1, C)
        x = self.reduction(x)
        x = self.norm(x)

        return x

# Swin_block


class window_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads
        prehead_dim = dim // self.num_heads
        self.scale = prehead_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
         # [B*num_windows, window_size*window_size, dim]
        B, num_patches, total_dim = x.shape

        qkv = self.qkv(x)  # [B*num_window, num_patches, 3*embed_dim] num_patches=window_size*window_size

        qkv = qkv.reshape(B, num_patches, 3, self.num_heads,
                          total_dim // self.num_heads)  # [B*num_window, num_patches, 3, num_heads, prehead_dim]

        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_window, num_heads, num_patches, prehead_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*num_window, num_heads, num_patches, prehead_dim]

        atten = (q @ k.transpose(-2, -1)) * self.scale  # [B*num_window, num_heads, num_patches, num_patches]
        if mask is None:
            atten = atten.softmax(dim=-1)
        else:
            # mask: [num_window, num_patches, num_patches] \\\\\[num_window, window_size*window_size, window_size*window_size]
            # atten: [B*num_window, num_head, num_patches, num_patches]
            atten = atten.reshape(B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1])
            # reshape_atten [B, num_window, num_head, num_patches, num_patches]
            # mask [1, num_window, 1, num_patches, num_patches]
            atten = atten + mask.unsqueeze(1).unsqueeze(0).to(atten.device)

            # atten = atten + mask.unsqueeze(1).unsqueeze(0).cuda()  # atten = atten + mask.unsqueeze(1).unsqueeze(0)
            atten = atten.reshape(-1, self.num_heads, mask.shape[1],
                                  mask.shape[1])  # [B*num_window, num_head, num_patches, num_patches] 权重矩阵
            atten = atten.softmax(dim=-1)

        atten = atten @ v  ## [B*num_window, num_heads, num_patches, prehead_dim]
        atten = atten.transpose(1, 2)  # [B*num_window, num_patches+1, num_heads, prehead_dim]
        atten = atten.reshape(B, num_patches, total_dim)  # [B*num_windows, window_size*window_size, dim]
        out = self.proj(atten)

        return out


class MLP(nn.Module):
    def __init__(self, in_dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * mlp_ratio)
        self.actlayer = nn.GELU()
        self.fc2 = nn.Linear(mlp_ratio * in_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)  # [B, num_patches+1, hidden_dim]
        x = self.actlayer(x)
        x = self.fc2(x)  # [B, num_patches+1, out_dim]
        x = self.actlayer(x)

        return x


# swin_encode & Patch_Merging
class Swin_Block(nn.Module):
    def __init__(self, dim, num_heads, Out_res, window_size, qkv_bias=False, shift_size=0):
        super().__init__()

        self.dim = dim
        self.resolution = Out_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.atten_norm = nn.LayerNorm(dim)
        self.atten = window_attention(dim, num_heads, qkv_bias)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=4)

        # self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x):
        # x:[B, num_patches, embed_dim]
        H, W = self.resolution #self.resolution = Out_res
        B, N, C = x.shape #传入的输入x
        assert N == H * W

        h = x
        x = self.atten_norm(x) #self.atten_norm = nn.LayerNorm(dim)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) #表示在第 1 维（高度，H）和第 2 维（宽度，W）上分别向上和向左移位 self.shift_size 个位置
            atten_mask = generate_mask(input_res=self.resolution, window_size=self.window_size,
                                       shift_size=self.shift_size)
        else:
            shift_x = x
            atten_mask = None

        x_window = window_partition(shift_x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_window = x_window.reshape(-1, self.window_size * self.window_size, C)  # [B*num_windows, window_size*window_size, C]
        atten_window = self.atten(x_window, mask=atten_mask) # [B*num_windows, window_size*window_size, dim] dim = c
        atten_window = atten_window.reshape(-1, self.window_size, self.window_size, C) # [B*num_windows, window_size, window_size, C]
        x = window_reverse(atten_window, self.window_size, H, W)  # [B, H, W, C]
        x = x.reshape(B, -1, C)  # [B, HW, C]
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class Swin_Model(nn.Module):
    def __init__(self,
                 chan,
                 dim,
                 patch_size,
                 num_heads,
                 input_res,
                 window_size,
                 qkv_bias=None
                 ):
        super().__init__()


        self.patch_embed = Patch_Embeding(chan, dim, patch_size)
        self.W_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=0)
        self.SW_MSA_block = Swin_Block(dim, num_heads, input_res, window_size, qkv_bias, shift_size=window_size // 2)
        self.patch_merging = Patch_Merging(input_res, dim)

    def forward(self, x):
        #输入是B C H W
        x = self.patch_embed(x)
        # 输入进swin blocks的就是B HW C
        x = self.W_MSA_block(x)
#出来是B HW C
        x = self.SW_MSA_block(x)
#出来是B HW C
        out = self.patch_merging(x) #B HW C 经过这一代码依然还是B HW C

        h, w = int(np.sqrt(out.size(1))), int(np.sqrt(out.size(1)))
        x = out.permute(0, 2, 1)
        x = x.contiguous().view(out.size(0), out.size(2), h, w)
        return x # B C H W




# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # Global average pooling
#         x_se = self.global_pooling(x).view(x.size(0), -1)
#
#         # Squeeze phase
#         x_se = self.fc1(x_se)
#         x_se = self.relu(x_se)
#         x_se = self.fc2(x_se)
#
#         # Excitation phase
#         x_se = self.sigmoid(x_se).view(x.size(0), x.size(1), 1, 1)
#
#         # Scale the input
#         x = x * x_se
#
#         return x

