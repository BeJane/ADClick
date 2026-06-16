from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_norm_layer(norm: str):
    norm = {
        "BN": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
    }[norm.upper()]
    return norm


def get_act_layer(act: str):
    act = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }[act.lower()]
    return act


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        conv_kwargs=None,
        norm_layer=None,
        norm_kwargs=None,
        act_layer=None,
        act_kwargs=None,
    ):
        super(ConvNormAct2d, self).__init__()

        conv_kwargs = {}
        if norm_layer:
            conv_kwargs["bias"] = False
        if padding == "same" and stride > 1:
            # if kernel_size is even, -1 is must
            padding = (kernel_size - 1) // 2

        self.conv = self._build_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_kwargs,
        )
        self.norm = None
        if norm_layer:
            norm_kwargs = {}
            self.norm = get_norm_layer(norm_layer)(
                num_features=out_channels, **norm_kwargs
            )
        self.act = None
        if act_layer:
            act_kwargs = {}
            self.act = get_act_layer(act_layer)(**act_kwargs)

    def _build_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        conv_kwargs,
    ):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **conv_kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvNormAct2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    norm_layer="BN",
                    act_layer="RELU",
                ),
            )
        )
        for atrous_rate in atrous_rates:
            conv_norm_act = ConvNormAct2d
            modules.append(
                conv_norm_act(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1 if atrous_rate == 1 else 3,
                    padding=0 if atrous_rate == 1 else atrous_rate,
                    dilation=atrous_rate,
                    norm_layer="BN",
                    act_layer="RELU",
                )
            )

        self.aspp_feature_extractors = nn.ModuleList(modules)
        self.aspp_fusion_layer = ConvNormAct2d(
            (1 + len(atrous_rates)) * output_channels,
            output_channels,
            kernel_size=3,
            norm_layer="BN",
            act_layer="RELU",
        )

    def forward(self, x):
        res = []
        for aspp_feature_extractor in self.aspp_feature_extractors:
            res.append(aspp_feature_extractor(x))
        res[0] = F.interpolate(
            input=res[0], size=x.shape[2:], mode="bilinear", align_corners=False
        )  # resize back after global-avg-pooling layer
        res = torch.cat(res, dim=1)
        res = self.aspp_fusion_layer(res)
        return res
