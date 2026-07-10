# -*- coding: utf-8 -*-
# @Time    : 2024/12/8
# @Author  : Xiaoqi Liang
# @GitHub  : https://github.com/liangjiaxiaoqi

import timm
import torch
import torch.nn as nn
from einops import rearrange

from utils.ops.tensor_ops import cus_sample

import torch.nn.functional as F
import math
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath, trunc_normal_
# from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
    ):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=False))


class StackedCBRBlock(nn.Sequential):
    def __init__(self, in_c, out_c, num_blocks=1, kernel_size=3):
        assert num_blocks >= 1
        super().__init__()

        if kernel_size == 3:
            kernel_setting = dict(kernel_size=3, stride=1, padding=1)
        elif kernel_size == 1:
            kernel_setting = dict(kernel_size=1)
        else:
            raise NotImplementedError

        cs = [in_c] + [out_c] * num_blocks
        self.channel_pairs = self.slide_win_select(cs, win_size=2, win_stride=1, drop_last=True)
        self.kernel_setting = kernel_setting

        for i, (i_c, o_c) in enumerate(self.channel_pairs):
            self.add_module(name=f"cbr_{i}", module=ConvBNReLU(i_c, o_c, **self.kernel_setting))

    @staticmethod
    def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
        num_items = len(items)
        i = 0
        while i + win_size <= num_items:
            yield items[i : i + win_size]
            i += win_stride

        if not drop_last:
            # 对于最后不满一个win_size的切片，保留
            yield items[i : i + win_size]


class ConvFFN(nn.Module):
    def __init__(self, dim, out_dim=None, ffn_expand=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            StackedCBRBlock(dim, dim * ffn_expand, num_blocks=2, kernel_size=3),
            nn.Conv2d(dim * ffn_expand, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        x = self.norm(x)
        return x


class PosCNN_PEG(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN_PEG, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),)
        self.s = s

    def forward(self, x, H=None, W=None):
        # B, C, H, W = x.shape
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchwiseTokenReEmbedding:
    @staticmethod
    def encode(x, nh, ph, pw):
        return rearrange(x, "b (nh hd) (nhp ph) (nwp pw) -> b nh (hd ph pw) (nhp nwp)", nh=nh, ph=ph, pw=pw)

    @staticmethod
    def decode(x, nhp, ph, pw):
        return rearrange(x, "b nh (hd ph pw) (nhp nwp) -> b (nh hd) (nhp ph) (nwp pw)", nhp=nhp, ph=ph, pw=pw)


class GARSA(nn.Module):
    def __init__(self, dim, p, nh=2):
        super().__init__()
        self.p = p
        self.nh = nh
        self.scale = (dim // nh * self.p ** 2) ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # add code
        seg_dim = dim // 4
        self.seg_dim = seg_dim
        self.seg = 4

        self.aggq1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.aggq2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.aggq3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)

        self.aggk1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.aggk2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.aggk3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)

        self.aggv1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.aggv2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.aggv3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        # add code

    @staticmethod
    def channel_shufflle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(b, c, groups, -1, w)
        # x = x.permute(0, 1, 3, 2, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        # x = x.reshape(b, c, -1, w)
        return x

    def forward(self, q, kv=None, need_weights: bool = False):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        # z = q

        q = torch.chunk(q, 4, dim=1)
        k = torch.chunk(k, 4, dim=1)
        v = torch.chunk(v, 4, dim=1)

        q0 = q[0]
        q1 = self.aggq1(q[1])
        q2 = self.aggq2(q[2])
        q3 = self.aggq3(q[3])

        k0 = k[0]
        k1 = self.aggk1(k[1])
        k2 = self.aggk2(k[2])
        k3 = self.aggk3(k[3])

        v0 = v[0]
        v1 = self.aggv1(v[1])
        v2 = self.aggv2(v[2])
        v3 = self.aggv3(v[3])

        q0 = PatchwiseTokenReEmbedding.encode(q0, nh=self.nh, ph=self.p, pw=self.p)
        q1 = PatchwiseTokenReEmbedding.encode(q1, nh=self.nh, ph=self.p, pw=self.p)
        q2 = PatchwiseTokenReEmbedding.encode(q2, nh=self.nh, ph=self.p, pw=self.p)
        q3 = PatchwiseTokenReEmbedding.encode(q3, nh=self.nh, ph=self.p, pw=self.p)
        k0 = PatchwiseTokenReEmbedding.encode(k0, nh=self.nh, ph=self.p, pw=self.p)
        k1 = PatchwiseTokenReEmbedding.encode(k1, nh=self.nh, ph=self.p, pw=self.p)
        k2 = PatchwiseTokenReEmbedding.encode(k2, nh=self.nh, ph=self.p, pw=self.p)
        k3 = PatchwiseTokenReEmbedding.encode(k3, nh=self.nh, ph=self.p, pw=self.p)
        v0 = PatchwiseTokenReEmbedding.encode(v0, nh=self.nh, ph=self.p, pw=self.p)
        v1 = PatchwiseTokenReEmbedding.encode(v1, nh=self.nh, ph=self.p, pw=self.p)
        v2 = PatchwiseTokenReEmbedding.encode(v2, nh=self.nh, ph=self.p, pw=self.p)
        v3 = PatchwiseTokenReEmbedding.encode(v3, nh=self.nh, ph=self.p, pw=self.p)

        q = torch.cat([q0, q1, q2, q3], dim=1)
        k = torch.cat([k0, k1, k2, k3], dim=1)
        v = torch.cat([v0, v1, v2, v3], dim=1)

        q = self.channel_shufflle(q, 4)
        k = self.channel_shufflle(k, 4)
        v = self.channel_shufflle(v, 4)

        qk = torch.einsum("bndx, bndy -> bnxy", q, k) * self.scale
        qk = qk.softmax(-1)
        qkv = torch.einsum("bnxy, bndy -> bndx", qk, v)

        qkv = PatchwiseTokenReEmbedding.decode(qkv, nhp=H // self.p, ph=self.p, pw=self.p)

        x = self.proj(qkv)
        if not need_weights:
            return x
        else:
            # average attention weights over heads
            return x, qk.mean(dim=1)


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, kv=None):
        if kv is None:
            kv = x
        input = x
        x = self.dwconv(x)
        kv = self.dwconv(kv)
        # x1, x2 = self.f1(x), self.f2(x)
        x1, x2 = self.f1(x), self.f2(kv)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class CRA(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.xq_PEG = PosCNN_PEG(in_chans=dim1, embed_dim=dim1)
        self.xkv_PEG = PosCNN_PEG(in_chans=dim1, embed_dim=dim1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xq, xkv=None):
        if xkv is None:
            xkv = xq
        B, C, h, w = xq.shape
        xq = xq.reshape(B, C, h*w).permute(0, 2, 1)
        xq = self.xq_PEG(xq,H=h,W=w)
        xkv = xkv.reshape(B, C, h * w).permute(0, 2, 1)
        xkv = self.xkv_PEG(xkv, H=h, W=w)
        B, N, C = xq.shape

        q = self.q(xq).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = xkv.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).reshape(B, C, h, w)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, input_resolution, p, nh, ffn_expand):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.sa = GARSA(dim, p=p, nh=nh)
        self.ca = CRA(dim1=dim, num_heads=nh, pool_ratio=input_resolution[0])  # 16,8,input_resolution[0]
        self.alpha = nn.Parameter(data=torch.zeros(1))
        self.beta = nn.Parameter(data=torch.zeros(1))

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)

    def forward(self, x):
        normed_x = self.norm1(x)
        x = x + self.alpha.sigmoid() * self.sa(normed_x) + self.beta.sigmoid() * self.ca(normed_x)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, p, input_resolution, nh=4, ffn_expand=1):
        super().__init__()
        self.rgb_norm2 = nn.BatchNorm2d(dim)
        self.depth_norm2 = nn.BatchNorm2d(dim)

        self.depth_to_rgb_sa = GARSA(dim, p=p, nh=nh)
        self.depth_to_rgb_ca = CRA(dim1=dim, num_heads=nh, pool_ratio=input_resolution[0])#8
        self.rgb_alpha = nn.Parameter(data=torch.zeros(1))
        self.rgb_beta = nn.Parameter(data=torch.zeros(1))

        self.rgb_to_depth_sa = GARSA(dim, p=p, nh=nh)
        self.rgb_to_depth_ca = CRA(dim1=dim, num_heads=nh, pool_ratio=input_resolution[0])#8
        self.depth_alpha = nn.Parameter(data=torch.zeros(1))
        self.depth_beta = nn.Parameter(data=torch.zeros(1))

        self.norm3 = nn.BatchNorm2d(2 * dim)
        self.ffn = ConvFFN(dim=2 * dim, ffn_expand=ffn_expand, out_dim=2 * dim)

        self.norm3_rgb = nn.BatchNorm2d(dim)
        self.ffn_rgb = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)
        self.norm3_depth = nn.BatchNorm2d(dim)
        self.ffn_depth = ConvFFN(dim=dim, ffn_expand=ffn_expand, out_dim=dim)

    def forward(self, rgb, depth):
        normed_rgb = self.rgb_norm2(rgb)
        normed_depth = self.depth_norm2(depth)
        transd_rgb = self.rgb_alpha.sigmoid() * self.depth_to_rgb_sa(normed_rgb, normed_depth) + self.rgb_beta.sigmoid() * self.depth_to_rgb_ca(normed_rgb, normed_depth)
        rgb_rgbd = rgb + transd_rgb
        transd_depth = self.depth_alpha.sigmoid() * self.rgb_to_depth_sa(normed_depth, normed_rgb) + self.depth_beta.sigmoid() * self.rgb_to_depth_ca(normed_depth, normed_rgb)
        depth_rgbd = depth + transd_depth

        rgb_rgbd = rgb_rgbd + self.ffn_rgb(self.norm3_rgb(rgb_rgbd))
        depth_rgbd = depth_rgbd + self.ffn_depth(self.norm3_depth(depth_rgbd))
        return rgb_rgbd, depth_rgbd


class CMFU(nn.Module):
    def __init__(self, in_dim, embed_dim, input_resolution, p, nh, ffn_expand):
        super().__init__()
        self.p = p
        self.rgb_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.depth_cnn_proj = nn.Sequential(
            StackedCBRBlock(in_c=in_dim, out_c=embed_dim), nn.Conv2d(embed_dim, embed_dim, 1)
        )

        self.rgb_imct = nn.ModuleList(SelfAttention(embed_dim, input_resolution=input_resolution,nh=nh, p=p, ffn_expand=ffn_expand) for i in range(3))  # H=W=[8,16,32,64]3
        self.depth_imct = nn.ModuleList(SelfAttention(embed_dim, input_resolution=input_resolution,nh=nh, p=p, ffn_expand=ffn_expand) for i in range(3))#3

        self.cmct = nn.ModuleList(CrossAttention(embed_dim, input_resolution=input_resolution, nh=nh, p=p, ffn_expand=ffn_expand) for i in range(2))#2
        self.norm3 = nn.BatchNorm2d(2 * embed_dim)
        self.cst = nn.ModuleList(SelfAttention(2 * embed_dim, input_resolution=input_resolution, nh=nh, p=p, ffn_expand=ffn_expand) for i in range(5))#5

        self.top_rgbd_imct = nn.ModuleList(SelfAttention(2 * embed_dim, input_resolution=input_resolution, nh=nh, p=p, ffn_expand=ffn_expand) for i in range(3))#3

        self.Block = Block(2 * embed_dim)

        self.alpha = nn.Parameter(data=torch.zeros(1))
        self.beta = nn.Parameter(data=torch.zeros(1))

    def forward(self, rgb, depth, top_rgbd=None):
        """输入均为NCHW"""
        rgb = self.rgb_cnn_proj(rgb)
        depth = self.depth_cnn_proj(depth)

        # IMCT
        for t in self.rgb_imct:
            rgb = t(rgb)
        for t in self.depth_imct:
            depth = t(depth)

        # CMCT
        for t in self.cmct:
            rgb, depth = t(rgb, depth)

        rgbd = self.norm3(torch.cat([rgb, depth], dim=1))

        # DBIP
        if top_rgbd is not None:
            top_rgbd_ = top_rgbd
            for t in self.top_rgbd_imct:
                top_rgbd = t(top_rgbd)
            top_rgbd = top_rgbd_ + top_rgbd
            rgbd = rgbd + self.alpha.sigmoid() * top_rgbd + self.beta.sigmoid() * self.Block(rgbd, top_rgbd_)

        # CST
        for t in self.cst:
            rgbd = t(rgbd)
        return rgbd


class MC2FNet_ResNet(nn.Module):
    def __init__(self, ps=(8, 8, 8, 8), embed_dim=64, pretrained=None): # ps=(8, 8, 8, 8)
        super().__init__()
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        self.depth_encoder: nn.Module = timm.create_model(
            model_name="resnet101d", features_only=True, out_indices=range(1, 5)
        )
        if pretrained:
            self.rgb_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
            self.depth_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)

        self.cmfus = nn.ModuleList(
            [
                CMFU(in_dim=c, embed_dim=embed_dim, input_resolution=input_resolution, p=p, nh=2, ffn_expand=1)  # nh=2
                for i, (p, c, input_resolution) in
                enumerate(zip(ps, (2048, 1024, 512, 256), ([8//8, 8//8], [16//16, 16//16], [32//32, 32//32], [64//64, 64//64])))
            ]
        )

        # predictor
        self.predictor = nn.ModuleList()
        self.predictor.append(StackedCBRBlock(embed_dim * 2, embed_dim))
        self.predictor.append(StackedCBRBlock(embed_dim, 32))
        self.predictor.append(nn.Conv2d(32, 1, 1))

    def forward(self, data):
        rgb_feats = self.rgb_encoder(data["image"])
        depth_feats = self.depth_encoder(data["depth"].repeat(1, 3, 1, 1))

        # to cnn decoder for fusion
        x = self.cmfus[0](rgb=rgb_feats[3], depth=depth_feats[3])
        x = self.cmfus[1](rgb=rgb_feats[2], depth=depth_feats[2], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmfus[2](rgb=rgb_feats[1], depth=depth_feats[1], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.cmfus[3](rgb=rgb_feats[0], depth=depth_feats[0], top_rgbd=cus_sample(x, scale_factor=2))

        # predictor
        x = self.predictor[0](cus_sample(x, scale_factor=2))
        x = self.predictor[1](cus_sample(x, scale_factor=2))
        x = self.predictor[2](x)
        return x
