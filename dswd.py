"""Dual-stream spatial/Haar wavelet downsampling (DSWD).

Core-module excerpt from the supplied DWM-DETR implementation.
Adapted from RT-DETR / RT-DETRv2; Copyright(c) 2023 lyuwenyu.
Original computation is retained; framework registration and unrelated code are omitted.
See NOTICE.md for upstream attribution and license terms.
"""

import torch
import torch.nn as nn

def get_activation(act: str, inpace: bool=True):
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
        m.inplace = inpace
    
    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class HaarWaveletDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, act='silu'):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Identity() if act is None else get_activation(act)
        )

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 0::2, 1::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 1::2, 1::2]
        ll = (x0 + x1 + x2 + x3) * 0.5
        lh = (x0 - x1 + x2 - x3) * 0.5
        hl = (x0 + x1 - x2 - x3) * 0.5
        hh = (x0 - x1 - x2 + x3) * 0.5
        return self.reduction(torch.cat([ll, lh, hl, hh], dim=1))


class DualStreamDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, act='silu'):
        super().__init__()
        self.spatial_conv = ConvNormLayer(in_channels, out_channels, 3, 2, act=act)
        self.wavelet_branch = HaarWaveletDownsample(in_channels, out_channels, act=act)
        self.alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return self.spatial_conv(x) + torch.sigmoid(self.alpha) * self.wavelet_branch(x)
