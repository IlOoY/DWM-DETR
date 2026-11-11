"""Final-stage deformable residual module for the R18 backbone.

Core-module excerpt from the supplied DWM-DETR implementation.
Adapted from RT-DETR / RT-DETRv2; Copyright(c) 2023 lyuwenyu.
Original computation is retained; framework registration and unrelated code are omitted.
See NOTICE.md for upstream attribution and license terms.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

def get_activation(act: str, inplace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inplace
    
    return m


class DCNv2(nn.Module):
    """
    DCNv2 wrapper.

    The main convolution weight is named `weight`, same as nn.Conv2d,
    so pretrained Conv2d weights can be loaded into this module when
    strict=False is used.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # DCNv2 needs offset_x, offset_y and modulation mask.
        # Therefore output channels = 3 * kernel_size * kernel_size.
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        # Initialize offset and mask branch to zero.
        # offset starts from 0, and mask starts from sigmoid(0)=0.5.
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        out = self.offset_mask_conv(x)

        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        padding=None,
        bias=False,
        act=None,
        use_dcn=False
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 if padding is None else padding

        # Only replace spatial convolutions with DCNv2.
        # 1x1 conv is kept as normal Conv2d.
        if use_dcn and kernel_size > 1:
            self.conv = DCNv2(ch_in, ch_out, kernel_size, stride, padding, bias=bias)
        else:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=bias)

        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        ch_in,
        ch_out,
        stride,
        shortcut,
        act='relu',
        variant='b',
        use_dcn=False
    ):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        # For ResNet18/34, both 3x3 convolutions can be replaced by DCNv2
        # when use_dcn=True.
        self.branch2a = ConvNormLayer(
            ch_in,
            ch_out,
            3,
            stride,
            act=act,
            use_dcn=use_dcn
        )

        self.branch2b = ConvNormLayer(
            ch_out,
            ch_out,
            3,
            1,
            act=None,
            use_dcn=use_dcn
        )

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


def build_deformable_stage5(in_channels=256, out_channels=512, act='relu'):
    """Two vd residual blocks, each with two 3x3 DCNv2 layers.

    The first block downsamples by 2 and uses a projection shortcut.
    The second block keeps the resolution and uses the identity shortcut.
    """
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, stride=2, shortcut=False,
                   act=act, variant='d', use_dcn=True),
        BasicBlock(out_channels, out_channels, stride=1, shortcut=True,
                   act=act, variant='d', use_dcn=True),
    )
