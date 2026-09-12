from __future__ import annotations

import torch
import torch.nn as nn


class FourierSpectralBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            ffted = torch.fft.rfft2(xf, norm="ortho")
            real, imag = ffted.real, ffted.imag
            out_real = self.conv_real(real) - self.conv_imag(imag)
            out_imag = self.conv_real(imag) + self.conv_imag(real)
            out = torch.fft.irfft2(torch.complex(out_real, out_imag), s=(h, w), norm="ortho")
        return out.to(x.dtype)


class FFCBlock(nn.Module):
    def __init__(self, channels: int, use_spectral: bool = True):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spectral = FourierSpectralBlock(channels) if use_spectral else nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        f = self.spectral(x)
        return self.fuse(torch.cat([s, f], dim=1)) + x
