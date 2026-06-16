from matplotlib import pyplot as plt
from torch import nn

from model.destseg_util import make_layer, BasicBlock, ASPP
from model.util import prepocess_residual, random_masking


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=1024,residual_method='square',out_channels=2):
        super().__init__()
        self.residual_method = residual_method
        print(f'model residual:{self.residual_method}')
        self.res = make_layer(BasicBlock, inplanes, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x, mask_ratio=None):
        x = prepocess_residual(x, self.residual_method)
        mask_pos = None
        if mask_ratio is not None:
            mask_pos = random_masking(x, mask_ratio)
            mask_pos = mask_pos.view(-1, 1, *x.shape[2:])
            # print(x[mask_pos.squeeze()==1])
            x = x * mask_pos

        x = self.res(x)
        x = self.head(x)
        # plt.imshow(x[-1,-1].cpu().detach().numpy())
        # plt.show()
        x = x.permute(0, 2, 3, 1).flatten(1,2).contiguous()
        # x = torch.sigmoid(x)
        if mask_pos is not None: return x, mask_pos
        return x
