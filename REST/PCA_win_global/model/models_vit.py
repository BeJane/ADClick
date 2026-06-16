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


class ViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, stride=8,global_pool=False, **kwargs):
        super(ViT, self).__init__(**kwargs)
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

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x

        return outcome,mask_pos


    def forward(self, x, mask_ratio=None):
        x = x ** 2
        x,mask_pos = self.forward_features(x,mask_ratio)
        logits = self.forward_head(x)[:,1:]
        if self.num_classes == 1:
            logits = torch.sigmoid(logits)
        if mask_ratio is not None:
            return logits,mask_pos
        return logits
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='',load_patch_embed=False):
        _load_weights(self, checkpoint_path, prefix,load_patch_embed)

@torch.no_grad()
def _load_weights(model: ViT, checkpoint_path: str, prefix: str = '',load_patch_embed=False):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if load_patch_embed:

        if hasattr(model.patch_embed, 'backbone'):
            # hybrid
            backbone = model.patch_embed.backbone
            stem_only = not hasattr(backbone, 'stem')
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
            stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
            stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                        for r in range(3):
                            getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                            getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                            getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                            block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                            block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
            embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        else:
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))

        model.patch_embed.proj.weight.copy_(embed_conv_w)
        model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_prefix_tokens', 1),
            model.patch_embed.grid_size
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

