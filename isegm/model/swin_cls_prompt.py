import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers.patch_embed import PatchEmbed

from isegm.model.asppl import ASPP
from isegm.model.models_swin import SimpleFPN, make_zero_conv


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels)
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
            nn.Linear(self.l_in_channels, self.key_channels),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Linear(self.l_in_channels, self.value_channels),
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
        n_l =1 # value.size(-1)
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

class Swin_ClsPrompt(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8,use_zero_conv=False,use_gate=False,task='click',**kwargs):
        super(Swin_ClsPrompt, self).__init__(global_pool='',**kwargs)
        self.task = task
        if self.task == 'click':
            self.patch_embed_coords = PatchEmbed(
                img_size=(kwargs['img_size'][0]*stride,kwargs['img_size'][1]*stride), patch_size=stride,
                in_chans=3, embed_dim=kwargs['embed_dim'])

        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']

        self.residual_language = nn.ModuleList()
        for i in range(kwargs['depths'][0]+1):
            self.residual_language.append(SpatialImageLanguageAttention(v_in_channels=kwargs['embed_dim'], l_in_channels=512,
                                                               key_channels=kwargs['embed_dim'], value_channels=kwargs['embed_dim']))
        self.neck = SimpleFPN(kwargs['embed_dim'])

        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = nn.Sequential(*[make_zero_conv(128),make_zero_conv(256),
                              make_zero_conv(512),
                              make_zero_conv(1024)])

        self.use_gate = use_gate
        if self.use_gate:
            assert self.use_zero_conv ,"Gate is used with zero conv!"
            self.gate= nn.Sequential(*[nn.Sequential(*[nn.Conv2d(128,128,kernel_size=1),  nn.Tanh()]),
                                       nn.Sequential(*[nn.Conv2d(256,256,kernel_size=1), nn.Tanh()]),
                                       nn.Sequential(*[nn.Conv2d(512,512,kernel_size=1), nn.Tanh()]),
                                       nn.Sequential(*[nn.Conv2d(1024,1024,kernel_size=1), nn.Tanh()]),])

    def forward_features(self, x,coord_features,prompt):
        x = self.patch_embed(x)
        x = x + self.residual_language[0](x,prompt)
        if self.task == 'click':
            x = x + self.patch_embed_coords(coord_features)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        assert len(self.layers)==1
        for i,l in enumerate(self.layers[0].blocks):

            x = l(x)
            x=x + self.residual_language[i+1](x,prompt)
        x = self.norm(x)  # B L C

        B, N, C = x.shape

        grid_size = self.patch_embed.grid_size

        x = x.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        outputs = self.neck(x)

        if self.use_zero_conv:
            for i in range(len(outputs)):
                if self.use_gate:
                    outputs[i] = self.gate[i](outputs[i]) * self.zero_conv[i](outputs[i])
                else:
                    outputs[i] = self.zero_conv[i](outputs[i])

        
        return outputs


class Swin_ASPP_ClsPrompt(timm.models.swin_transformer.SwinTransformer):

    def __init__(self, stride=8, use_zero_conv=False, task='click', **kwargs):
        super(Swin_ASPP_ClsPrompt, self).__init__(global_pool='', **kwargs)
        self.task = task
        if self.task == 'click':
            self.patch_embed_coords = PatchEmbed(
                img_size=(kwargs['img_size'][0] * stride, kwargs['img_size'][1] * stride), patch_size=stride,
                in_chans=3, embed_dim=kwargs['embed_dim'])

        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']

        self.residual_language = nn.ModuleList()
        for i in range(kwargs['depths'][0] + 1):
            self.residual_language.append(
                SpatialImageLanguageAttention(v_in_channels=kwargs['embed_dim'], l_in_channels=512,
                                              key_channels=kwargs['embed_dim'], value_channels=kwargs['embed_dim']))
        self.neck = ASPP(kwargs['embed_dim'],256,[6, 12, 18])

        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = make_zero_conv(256)

    def forward_features(self, x, coord_features, prompt):
        x = self.patch_embed(x)
        x = x + self.residual_language[0](x, prompt)
        if self.task == 'click':
            x = x + self.patch_embed_coords(coord_features)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        assert len(self.layers) == 1
        for i, l in enumerate(self.layers[0].blocks):
            x = l(x)
            x = x + self.residual_language[i + 1](x, prompt)
        x = self.norm(x)  # B L C

        B, N, C = x.shape

        grid_size = self.patch_embed.grid_size

        x = x.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        outputs = self.neck(x)

        if self.use_zero_conv:
            outputs = self.zero_conv(outputs)

        return outputs
