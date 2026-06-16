# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from abc import abstractmethod
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import adapt_input_conv, resize_pos_embed
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # elif isinstance(layer, SpatialTransformer):
            #     x = layer(x, context)
            else:
                x = layer(x)
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()

        self.down_4 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim * 2)
        self.down_8 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16_chan = max(out_dims[2], in_dim * 2)
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_16_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_16_chan),
            nn.GELU(),

            nn.Conv2d(self.down_16_chan, self.down_16_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_16_chan),
            nn.Conv2d(self.down_16_chan, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.GELU(),
            nn.Conv2d(self.down_32_chan, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.GELU(),
            nn.Conv2d(self.down_32_chan, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),

            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)  # 112
        x_down_8 = self.down_8(x)  # 56
        x_down_16 = self.down_16(x)  # 28
        x_down_32 = self.down_32(x)  # 14

        return [x_down_4, x_down_8, x_down_16, x_down_32]
class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        # if l_mask is not None:
        #     l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        #     key = key * l_mask  # (B, key_channels, N_l)
        #     value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        # l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        # sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = torch.nn.functional.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out
def make_zero_conv( channels):
    return TimestepEmbedSequential(zero_module(nn.Conv2d(channels, channels, 1, padding=0)))
class Swin(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8,use_zero_conv=False,**kwargs):
        super(Swin, self).__init__(global_pool='',**kwargs)
        self.patch_embed_coords = PatchEmbed(
            img_size=(kwargs['img_size'][0]*stride,kwargs['img_size'][1]*stride), patch_size=stride,
            in_chans=3, embed_dim=kwargs['embed_dim'])

        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']

        self.residual_language = SpatialImageLanguageAttention(v_in_channels=kwargs['embed_dim'], l_in_channels=768,
                                                               key_channels=kwargs['embed_dim'], value_channels=kwargs['embed_dim'])
        self.neck = SimpleFPN(kwargs['embed_dim'])

        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = nn.Sequential(*[make_zero_conv(128),make_zero_conv(256),
                              make_zero_conv(512),
                              make_zero_conv(1024)])






    def forward_features(self, x,coord_features,prompt):
        x = self.patch_embed(x)
        x = x + self.patch_embed_coords(coord_features)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        prompt = self.residual_language(x,prompt)
        x = x + prompt

        B, N, C = x.shape

        grid_size = self.patch_embed.grid_size

        x = x.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        outputs = self.neck(x)

        if self.use_zero_conv:
            for i in range(len(outputs)):
                outputs[i] = self.zero_conv[i](outputs[i])

        
        return outputs



