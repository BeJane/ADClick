import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from model.transformer_helper import BaseDecodeHead, resize


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
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

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class SimpleFPN_4_32(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        # self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        # self.down_4 = nn.Sequential(
        #     nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
        #     nn.GroupNorm(1, self.down_4_chan),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
        #     nn.GroupNorm(1, self.down_4_chan // 2),
        #     nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
        #     nn.GroupNorm(1, out_dims[0]),
        #     nn.GELU()
        # )
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.Conv2d(self.down_4_chan, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16_chan = max(out_dims[2], in_dim * 2)
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_16_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_16_chan),
            nn.Conv2d(self.down_16_chan, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan//2, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan//2),
            nn.GELU(),
            nn.Conv2d(self.down_32_chan//2, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]

class Fusion_coord(SimpleFPN_4_32):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__(in_dim, out_dims)

        self.coords_4 = nn.Conv2d(3, out_dims[0], kernel_size=4, stride=4, bias=True)

        self.coords_8 = nn.Conv2d(3, out_dims[1], kernel_size=8, stride=8, bias=True)
        self.coords_16 = nn.Conv2d(3, out_dims[2], kernel_size=16, stride=16, bias=True)
        self.coords_32 = nn.Conv2d(3, out_dims[3], kernel_size=32, stride=32, bias=True)

        self.gate_4 = nn.Sequential(
            nn.Conv2d(out_dims[0], out_dims[0],kernel_size=1,stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_dims[0], out_dims[0],kernel_size=1,stride=1, bias=False),
            nn.Tanh()
        )
        self.gate_8 = nn.Sequential(
            nn.Conv2d(out_dims[1], out_dims[1],kernel_size=1,stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_dims[1], out_dims[1],kernel_size=1,stride=1, bias=False),
            nn.Tanh()
        )
        self.gate_16 = nn.Sequential(
            nn.Conv2d(out_dims[2], out_dims[2],kernel_size=1,stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_dims[2], out_dims[2],kernel_size=1,stride=1, bias=False),
            nn.Tanh()
        )
        self.gate_32 = nn.Sequential(
            nn.Conv2d(out_dims[3], out_dims[3],kernel_size=1,stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_dims[3], out_dims[3],kernel_size=1,stride=1, bias=False),
            nn.Tanh()
        )
    def forward(self, x,coord_features):
        coords_4 = self.coords_4(coord_features)
        # print(coords_4.shape)
        x_down_4 = self.down_4(x) + coords_4 * self.gate_4(coords_4)
        coords_8 = self.coords_8(coord_features)
        x_down_8 = self.down_8(x) + coords_8 * self.gate_8(coords_8)
        coords_16 = self.coords_16(coord_features)
        x_down_16 = self.down_16(x) + coords_16 * self.gate_16(coords_16)
        coords_32 = self.coords_32(coord_features)
        x_down_32 = self.down_32(x) + coords_32 * self.gate_32(coords_32)

        return [x_down_4, x_down_8, x_down_16, x_down_32]

class SwinTransfomerSegHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, upsample='x1', interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.unsample = upsample
        self.out_channels = {'x1': self.channels, 'x2': self.channels * 2,
            'x4': self.channels * 4}[upsample]

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.out_channels * num_inputs,
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, self.out_channels // 2, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.Conv2d(self.out_channels // 2, self.out_channels // 2, 1),
            nn.GroupNorm(1, self.out_channels // 2),
            nn.GELU()
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels // 2, self.out_channels // 4, 2, stride=2),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.Conv2d(self.out_channels // 4, self.out_channels // 4, 1),
            nn.GroupNorm(1, self.out_channels // 4),
            nn.GELU()
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        if self.unsample == 'x2':
            out = self.up_conv1(out)

        if self.unsample == 'x4':
            out = self.up_conv2(self.up_conv1(out))

        out = self.cls_seg(out)

        return out