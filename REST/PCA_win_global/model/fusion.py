from typing import Tuple

from torch import nn, Tensor


class FusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int] = (512,1024),
        out_channels: Tuple[int] = (512,1024),
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_func = act_func
        self.with_cp = with_cp
        self.n_outputs = len(out_channels)
        self._build_fuse_layers()

    def _build_fuse_layers(self):
        self.blocks = nn.ModuleList([])
        n_inputs = len(self.in_channels)
        for i, outc in enumerate(self.out_channels):
            blocks = nn.ModuleList([])

            start = 0
            end = n_inputs
            for j in range(start, end):
                inc = self.in_channels[j]
                if j == i:
                    blocks.append(nn.Identity())
                elif j < i:
                    block = [
                        nn.Conv2d(
                            inc,
                            inc,
                            kernel_size=2 ** (i - j) + 1,
                            stride=2 ** (i - j),
                            dilation=1,
                            padding=2 ** (i - j) // 2,
                            groups=inc,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inc),
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),
                    ]

                    blocks.append(nn.Sequential(*block))

                else:
                    block = [
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),
                    ]

                    block.append(
                        nn.Upsample(
                            scale_factor=2 ** (j - i),
                            mode="bilinear",
                        ),
                    )
                    blocks.append(nn.Sequential(*block))
            self.blocks.append(blocks)

        self.act = nn.ModuleList([self.act_func() for _ in self.out_channels])

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:

        out = [None] * len(self.blocks)
        n_inputs = len(x)

        for i, (blocks, act) in enumerate(zip(self.blocks, self.act)):
            start = 0
            end = n_inputs
            for j, block in zip(range(start, end), blocks):
                out[i] = block(x[j]) if out[i] is None else out[i] + block(x[j])
            out[i] = act(out[i])

        return out
