"""Learned spectral prefilter: маленькая UNet на log-спектрограмме -> мульт.

Вместо handcrafted Butterworth-notch: сеть сама предсказывает, где в спектре
резать (диффузные боковые полосы растра, межточечные гармоники), а где нет
(контент). Стартует как тождество (bias выхода +4 -> sigmoid ~0.98), обучается
совместно с основной сетью end-to-end. Стоимость: rFFT 512² (~1 мс) + UNet
на 257×129 с base16 — доли миллисекунды против 177 мс основной сети.

ONNX-валидность: rfft2 -> |.| -> log10 -> conv -> sigmoid -> умножение ->
irfft2 (onesided inverse валиден как DFT(inverse=1, onesided=1)? НЕТ — та же
лава, что в Этапе 3: обратный onesided DFT невалиден по ONNX-спеке.
Для экспорта маску нужно будет применять через полный эрмитов спектр
(тот же приём, что в ffblock._hermitian_ifft) — TODO на этапе экспорта.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMaskUNet(nn.Module):
    """logmag (N,C,H,W/2+1) -> mask (N,C,H,W/2+1) в (0,1)."""

    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GELU())
        self.e3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1), nn.GELU())
        self.d1 = nn.Sequential(nn.Conv2d(base * 2 + base, base, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(base, in_channels, 1)
        nn.init.constant_(self.out.bias, 4.0)  # sigmoid(4) ~ 0.982 -> старт ≈ identity

    def forward(self, logmag: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(logmag)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        m = self.mid(e3)
        m = F.interpolate(m, size=e2.shape[-2:], mode="nearest")
        m = self.d2(torch.cat([m, e2], dim=1))
        m = F.interpolate(m, size=e1.shape[-2:], mode="nearest")
        m = self.d1(torch.cat([m, e1], dim=1))
        return torch.sigmoid(self.out(m))


class LearnedSpectralPrefilter(nn.Module):
    """Применяет маску к спектру входного изображения (per-channel).

    forward: x (N,3,H,W) [0,1] -> x_masked (N,3,H,W). Дифференцируемо.
    """

    def __init__(self, base: int = 16):
        super().__init__()
        self.mask_net = SpectralMaskUNet(in_channels=3, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            n, c, h, w = xf.shape
            mean = xf.mean(dim=(-2, -1), keepdim=True)
            spec = torch.fft.rfft2(xf - mean, norm="ortho")
            logmag = torch.log10(spec.abs() + 1e-8)
            lm = (logmag - logmag.mean(dim=(-2, -1), keepdim=True)) / (
                logmag.std(dim=(-2, -1), keepdim=True) + 1e-5
            )
            mask = self.mask_net(lm)
            out = torch.fft.irfft2(spec * mask, s=(h, w), norm="ortho") + mean
        return out.to(x.dtype)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
