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
            spec = torch.fft.rfft2(xf, norm="ortho")
            real, imag = spec.real, spec.imag
            out_real = self.conv_real(real) - self.conv_imag(imag)
            out_imag = self.conv_real(imag) + self.conv_imag(real)
            m = out_real.shape[-1]  # w//2 + 1
            # Full-length 2D Hermitian spectrum in real domain; identical to the
            # training-time onesided irfft2 (conjugation mirrors BOTH axes, with
            # the DC row as its own partner), but valid for ONNX inverse DFT.
            # Mirrors use index_select (positive dynamic indices) so DirectML
            # can execute them (Slice with negative steps is CPU-only).
            if w % 2 == 0:
                src_re, src_im = out_real[..., 1:m - 1], out_imag[..., 1:m - 1]
            else:
                src_re, src_im = out_real[..., 1:], out_imag[..., 1:]
            l = src_re.shape[-1]
            iw = (l - 1) - torch.arange(l, device=src_re.device)
            ih = h - torch.arange(h, device=src_re.device)
            ih = torch.where(ih == h, torch.zeros_like(ih), ih)
            mir_re = src_re.index_select(-1, iw).index_select(-2, ih)
            mir_im = (-src_im).index_select(-1, iw).index_select(-2, ih)
            full_re = torch.cat([out_real, mir_re], dim=-1)
            full_im = torch.cat([out_imag, mir_im], dim=-1)
            full = torch.complex(full_re, full_im)
            out = torch.fft.ifft2(full, norm="ortho").real
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
