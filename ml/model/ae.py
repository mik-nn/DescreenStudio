"""Автоэнкодер без skip-соединений — ответ на вопрос «а что если не UNet».

Та же бюджетная рамка, что у FourierUNet (base96/L3), но информация от
входа к выходу идёт ТОЛЬКО через bottleneck (никаких skip-add). Глобальный
residual-выход сохранён: out = clamp(x - f(x)).

Если AE проиграет UNet с тем же числом параметров — гипотеза «skip не нужны»
закрыта измерением, а не интуицией.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.model.ffblock import FFCBlock, FFCBlockV2, NAFBlock


class _Down(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _Up(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SpectralAE(nn.Module):
    """Conv-AE: inc(3->96) -> [96] -> down -> [192] -> down -> [384]
    -> bottleneck blocks -> up -> [192] -> up -> [96] -> outc(96->3)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 96,
        enc_blocks: int = 1,
        bottleneck_blocks: int = 3,
        dec_blocks: int = 1,
        block: str = "ffc",  # "ffc" | "naf"
        spectral_mode: str = "plain",  # для ffc-блоков
        spectral_l0: bool = False,  # полноразмерный спектр (96ch@512²) — дорого
        skip: bool = False,  # UNet-style skip-add (иначе — чистый автоэнкодер)
    ):
        super().__init__()
        def make(ch: int, n: int, spectral: bool) -> nn.Sequential:
            if block == "naf":
                return nn.Sequential(*[NAFBlock(ch) for _ in range(n)])
            return nn.Sequential(
                *[FFCBlockV2(ch, use_spectral=spectral, spectral_mode=spectral_mode) for _ in range(n)]
            )

        self.inc = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        self.enc1 = make(base, enc_blocks, spectral=spectral_l0)
        self.down1 = _Down(base, base * 2)
        self.enc2 = make(base * 2, enc_blocks, spectral=True)
        self.down2 = _Down(base * 2, base * 4)
        self.bottleneck = make(base * 4, bottleneck_blocks, spectral=True)
        self.up2 = _Up(base * 4, base * 2)
        self.dec2 = make(base * 2, dec_blocks, spectral=True)
        self.up1 = _Up(base * 2, base)
        self.dec1 = make(base, dec_blocks, spectral=spectral_l0)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.nn.functional.gelu(self.inc(x))
        e1 = self.enc1(e)
        e = self.down1(e1)
        e2 = self.enc2(e)
        e = self.down2(e2)
        b = self.bottleneck(e)
        d = self.up2(b)
        if self.skip:
            d = d + e2
        d = self.dec2(d)
        d = self.up1(d)
        if self.skip:
            d = d + e1
        d = self.dec1(d)
        residual = self.outc(d)
        base = x[:, :3] if x.shape[1] > 3 else x
        return torch.clamp(base - residual, 0.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
