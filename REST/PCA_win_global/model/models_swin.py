# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import adapt_input_conv, resize_pos_embed

from model.util import random_masking


class Swin(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8,global_pool=False, **kwargs):
        super(Swin, self).__init__(**kwargs)
        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm



    def forward_features(self, x,mask_ratio=None):
        mask_pos = None
        if mask_ratio is not None:
            mask_pos= random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1,1,*x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos
            # mask_pos = mask_pos.flatten(1,3)
            # print(mask_pos.shape)
            # print(x.shape,mask_pos.shape)
            # print(x[mask_pos.squeeze()==1])
        x = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        
        return x,mask_pos

    def forward(self, x, mask_ratio=None):

        x = x ** 2
        x,mask_pos = self.forward_features(x,mask_ratio)
        logits = self.forward_head(x)
        if self.num_classes == 1:
            logits = torch.sigmoid(logits)
        if mask_ratio is not None:
            return logits,mask_pos
        return logits

