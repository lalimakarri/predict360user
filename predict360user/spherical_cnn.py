# predict360user/models/spherical_cnn.py
import torch
import torch.nn as nn

# from your SPVP360 repo
from sconv.module import SphericalConv
from spool import SphericalPooling
from spad import SphericalPad


class SphericalConv2d(nn.Module):
    """Adapter around SPVP360’s SphericalConv to look like Conv2d."""
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, bias=True,
                 padding="sphere", radius=None):
        super().__init__()
        self.conv = SphericalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            radius=radius,
        )

    def forward(self, x):
        return self.conv(x)


class SphericalMaxPool2d(nn.Module):
    """Adapter around SPVP360’s SphericalPooling to look like MaxPool2d."""
    def __init__(self, kernel_size=2, stride=None, mode="max"):
        super().__init__()
        self.pool = SphericalPooling(
            kernel_size=kernel_size,
            stride=stride or kernel_size,
            mode=mode,
        )

    def forward(self, x):
        return self.pool(x)
