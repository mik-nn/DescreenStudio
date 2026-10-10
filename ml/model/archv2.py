"""Обёртка архитектуры v2: learned spectral prefilter + coarse-edge канал
+ FourierUNet(snorm). Все опции независимы, флаги через конструктор.

- use_lpf: LearnedSpectralPrefilter до сети (совместно обучается, стартует
  с тождества).
- use_edge_ch: 4-й входной канал — градиент люмы на 4× уменьшенном масштабе
  (контуры без растра: на /4 период растра 6–7 px схлопывается в алиас,
  поэтому берём градиент после avg_pool — структурные контент-контуры,
  upsample nearest обратно). Идея Qwen «уменьшение 4х -> контуры -> маска».
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.model.spmask import LearnedSpectralPrefilter
from ml.model.unet import FourierUNet


class CoarseEdge4(nn.Module):
    """(N,3,H,W) -> (N,1,H,W): градиент люмы на /4-м масштабе, обратный апскейл."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        luma = (x[:, :3] * torch.tensor([0.299, 0.587, 0.114], device=x.device)
                .view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        coarse = F.avg_pool2d(luma, kernel_size=4, stride=4)
        gx = coarse[:, :, :, 2:] - coarse[:, :, :, :-2]
        gy = coarse[:, :, 2:, :] - coarse[:, :, :-2, :]
        g = torch.sqrt(gx[:, :, 1:-1, :] ** 2 + gy[:, :, :, 1:-1] ** 2 + 1e-12)
        g = (g - g.mean(dim=(-2, -1), keepdim=True)) / (g.std(dim=(-2, -1), keepdim=True) + 1e-5)
        return F.interpolate(g, size=x.shape[-2:], mode="nearest")


class ArchV2(nn.Module):
    def __init__(self, use_lpf: bool = False, use_edge_ch: bool = False,
                 spectral_mode: str = "norm", base: int = 96):
        super().__init__()
        self.use_lpf = use_lpf
        self.use_edge_ch = use_edge_ch
        self.lpf = LearnedSpectralPrefilter() if use_lpf else None
        self.edge = CoarseEdge4() if use_edge_ch else None
        in_ch = 4 if use_edge_ch else 3
        self.net = FourierUNet(
            in_channels=in_ch, base=base, levels=3,
            enc_blocks=(1, 1), bottleneck_blocks=3, dec_blocks=(1, 1),
            spectral=(False, True, True), spectral_mode=spectral_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, :3]
        cur = self.lpf(base) if self.lpf is not None else base
        if self.edge is not None:
            cur = torch.cat([cur, self.edge(base)], dim=1)
        return self.net(cur)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
