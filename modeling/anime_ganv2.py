
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Layer_Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.layer_norm(x, x.size()[1:])

def truncated_normal_(tensor, mean=0., std=0.1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class Conv2DNormLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode='reflect',
                              bias=bias)
        self.LayerNorm = Layer_Norm()
        self.LRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.Conv(x)
        x = self.LayerNorm(x)
        x = self.LRelu(x)
        return x

class InvertedRes_Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride):
        super().__init__()
        self.add_op = (in_channels == out_channels and stride == 1)
        bottleneck_dim = round(expansion_ratio * in_channels)
        # pw
        self.pw = Conv2DNormLReLU(in_channels, bottleneck_dim, kernel_size=1)
        # dw
        self.dw = nn.Sequential(
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=bottleneck_dim,
                padding_mode='reflect'
            ),
            Layer_Norm(),
            nn.LeakyReLU(0.2)
        )
        # pw & linear
        self.pw_linear = nn.Sequential(
            nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1,  bias=False, padding_mode='reflect'),
            Layer_Norm()
        )

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        out = self.pw_linear(out)
        if self.add_op:
            out += x
        return out

class Generator(nn.Module):
    def __init__(self, dataset='', in_channels=3):
        super().__init__()
        self.name = f'generator_{dataset}'
        self.A = nn.Sequential(
            Conv2DNormLReLU(in_channels, 32, kernel_size=7, padding=3),
            Conv2DNormLReLU(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2DNormLReLU(64, 64, kernel_size=3, padding=1)
        )
        self.B = nn.Sequential(
            Conv2DNormLReLU(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
        )
        self.C = nn.Sequential(
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
            InvertedRes_Block(128, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            InvertedRes_Block(256, 256, 2, 1),
            Conv2DNormLReLU(256, 128, kernel_size=3, padding=1)
        )
        self.D = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1),
            Conv2DNormLReLU(128, 128, kernel_size=3, padding=1)
        )
        self.E = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Conv2DNormLReLU(128, 64, kernel_size=3, padding=1),
            Conv2DNormLReLU(64, 64, kernel_size=3, padding=1),
            Conv2DNormLReLU(64, 32, kernel_size=7, padding=3)
        )
        self.F = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, bias=False, padding_mode='reflect'),
            nn.Tanh()
        )
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # variance_scaling_initializer
                # https://docs.w3cub.com/tensorflow~python/tf/contrib/layers/variance_scaling_initializer
                truncated_normal_(m.weight, mean=0., std=math.sqrt(1.3 * 2.0 / m.in_channels))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        x = self.C(x)
        x = self.D(x)
        x = self.E(x)
        x = self.F(x)
        return x


def conv_sn(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    return spectral_norm(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    )


class Discriminator(nn.Module):
    def __init__(self,args, in_channels=3, channels=64, n_dis=2):
        super().__init__()
        self.name = f'discriminator_{args.dataset}'
        channels = channels // 2
        self.first = nn.Sequential(
            conv_sn(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        second_list = []
        channels_in = channels
        for _ in range(n_dis):
            second_list += [
                conv_sn(channels_in, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2)
            ]
            second_list += [
                conv_sn(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                Layer_Norm(),
                nn.LeakyReLU(0.2)
            ]
            channels_in = channels * 4
            channels *= 2
        self.second = nn.Sequential(*second_list)

        self.third = nn.Sequential(
            conv_sn(channels_in, channels * 2, kernel_size=3, stride=1, padding=1),
            Layer_Norm(),
            nn.LeakyReLU(0.2),
            conv_sn(channels * 2, 1, kernel_size=3, stride=1, padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        return x