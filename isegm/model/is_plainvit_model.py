import math

import torch
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .models_swin import Swin, make_zero_conv
from .swin_cls_prompt import Swin_ClsPrompt


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024],use_zero_conv=False):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

        self.use_zero_conv = use_zero_conv
        if self.use_zero_conv:
            self.zero_conv = nn.ModuleList()
            for  out_channels in out_dims:
                self.zero_conv.append(make_zero_conv(out_channels))

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)#112
        x_down_8 = self.down_8(x)#56
        x_down_16 = self.down_16(x)#28
        x_down_32 = self.down_32(x)#14

        if self.use_zero_conv:
            x_down_4 = self.zero_conv[0](x_down_4)
            x_down_8 = self.zero_conv[1](x_down_8)
            x_down_16 = self.zero_conv[2](x_down_16)
            x_down_32 = self.zero_conv[3](x_down_32)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
            residual_backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
            prompt_mode='prompt',
            task='click',
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split
        self.task = task
        if task == 'click':

            self.patch_embed_coords = PatchEmbed(
                img_size= backbone_params['img_size'],
                patch_size=backbone_params['patch_size'],
                in_chans=3 if self.with_prev_mask else 2,
                embed_dim=backbone_params['embed_dim'],
            )

        self.backbone = VisionTransformer(**backbone_params)
        if prompt_mode == 'cls_prompt':
            self.residual_backbone = Swin_ClsPrompt(**residual_backbone_params)
        else:
            self.residual_backbone = Swin(**residual_backbone_params)

        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image,residual,prompt, coord_features=None):

        residual_features = self.residual_backbone.forward_features(residual,coord_features,prompt)
        if self.task == 'click':
            coord_features = self.patch_embed_coords(coord_features)
        else:
            coord_features = None
        backbone_features = self.backbone.forward_backbone(image, coord_features, self.random_split)
        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        backbone_features = backbone_features

        multi_scale_features = self.neck(backbone_features)
        for i in range(len(multi_scale_features)):

            multi_scale_features[i] = multi_scale_features[i] + residual_features[i]

        return {'instances': self.head(multi_scale_features), 'instances_aux': None}
