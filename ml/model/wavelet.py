from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.model.ffblock import FFCBlock


def haar_kernels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    base = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [-0.5, -0.5]],
            [[0.5, -0.5], [0.5, -0.5]],
            [[0.5, -0.5], [-0.5, 0.5]],
        ],
        dtype=dtype,
    ).unsqueeze(1)
    return base


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", haar_kernels())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = self.kernels.to(x.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(x, w, stride=2, groups=c)


class IWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", haar_kernels())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 4
        w = self.kernels.to(x.dtype).repeat(c, 1, 1, 1)
        return F.conv_transpose2d(x, w, stride=2, groups=c)


def _blocks(ch: int, n: int, spectral: bool) -> nn.Sequential:
    return nn.Sequential(*[FFCBlock(ch, use_spectral=spectral) for _ in range(n)])


class MWNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 96,
        levels: int = 3,
        enc_blocks: tuple[int, ...] = (1, 1),
        bottleneck_blocks: int = 3,
        dec_blocks: tuple[int, ...] = (1, 1),
        spectral: tuple[bool, ...] = (False, True, True),
    ):
        super().__init__()
        assert levels in (2, 3)
        assert len(enc_blocks) == levels - 1 and len(dec_blocks) == levels - 1
        assert len(spectral) == levels
        self.inc = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        chs = [base * 2**i for i in range(levels)]
        self.encs = nn.ModuleList(_blocks(chs[i], enc_blocks[i], spectral[i]) for i in range(levels - 1))
        self.downs = nn.ModuleList([DWT() for _ in range(levels - 1)])
        self.shrink = nn.ModuleList(
            [nn.Conv2d(chs[i] * 4, chs[i + 1], kernel_size=1) for i in range(levels - 1)]
        )
        self.bottleneck = _blocks(chs[-1], bottleneck_blocks, spectral[-1])
        self.expand = nn.ModuleList(
            [nn.Conv2d(chs[i + 1], chs[i] * 4, kernel_size=1) for i in range(levels - 2, -1, -1)]
        )
        self.ups = nn.ModuleList([IWT() for _ in range(levels - 1)])
        self.decs = nn.ModuleList(
            _blocks(chs[i], dec_blocks[i], spectral[i]) for i in range(levels - 2, -1, -1)
        )
        self.outc = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)
        self.base = base
        self.levels = levels

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = []
        e = F.gelu(self.inc(x))
        for enc, down, shrink in zip(self.encs, self.downs, self.shrink):
            e = enc(e)
            feats.append(e)
            e = shrink(down(e))
        b = self.bottleneck(e)
        d = b
        for up, expand, dec, skip in zip(self.ups, self.expand, self.decs, reversed(feats)):
            d = dec(up(expand(d)) + skip)
        residual = self.outc(d)
        out = torch.clamp(x - residual, 0.0, 1.0)
        if return_features:
            return out, b
        return out

    def bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        e = F.gelu(self.inc(x))
        for enc, down, shrink in zip(self.encs, self.downs, self.shrink):
            e = enc(e)
            e = shrink(down(e))
        return self.bottleneck(e)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
