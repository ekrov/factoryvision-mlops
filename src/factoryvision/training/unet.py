"""A small U-Net for binary industrial-defect segmentation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions that refine features at one resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample once, then learn features at the lower resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample, join the matching encoder features, and refine them."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        # The chosen 256 x 640 input is divisible by all four pooling steps.
        # This fallback also keeps the block safe for other image dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Four-level U-Net that returns one defect logit per image pixel.

    The output is deliberately a logit rather than a sigmoid probability.
    BCEWithLogitsLoss applies the sigmoid internally and is numerically safer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.encoder1 = DoubleConv(in_channels, channels[0])
        self.encoder2 = DownBlock(channels[0], channels[1])
        self.encoder3 = DownBlock(channels[1], channels[2])
        self.encoder4 = DownBlock(channels[2], channels[3])
        self.bottleneck = DownBlock(channels[3], channels[3] * 2)

        self.decoder4 = UpBlock(channels[3] * 2, channels[3])
        self.decoder3 = UpBlock(channels[3], channels[2])
        self.decoder2 = UpBlock(channels[2], channels[1])
        self.decoder1 = UpBlock(channels[1], channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        bottleneck = self.bottleneck(encoder4)

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)

        return self.output(decoder1)
