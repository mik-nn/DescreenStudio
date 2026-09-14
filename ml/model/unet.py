from __future__ import annotations

import torch
import torch.nn as nn

from ml.model.ffblock import FFCBlock


class Down(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def _blocks(ch: int, n: int, spectral: bool) -> nn.Sequential:
    return nn.Sequential(*[FFCBlock(ch, use_spectral=spectral) for _ in range(n)])


class FourierUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 64,
        levels: int = 3,
        enc_blocks: tuple[int, ...] = (1, 1),
        bottleneck_blocks: int = 2,
        dec_blocks: tuple[int, ...] = (1, 1),
        spectral: tuple[bool, ...] = (True, True, True),
    ):
        super().__init__()
        assert levels in (2, 3)
        assert len(enc_blocks) == levels - 1 and len(dec_blocks) == levels - 1
        assert len(spectral) == levels
        self.inc = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        chs = [base * 2**i for i in range(levels)]
        self.encs = nn.ModuleList(_blocks(chs[i], enc_blocks[i], spectral[i]) for i in range(levels - 1))
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(levels - 1))
        self.bottleneck = _blocks(chs[-1], bottleneck_blocks, spectral[-1])
        self.ups = nn.ModuleList(Up(chs[i + 1], chs[i]) for i in range(levels - 2, -1, -1))
        self.decs = nn.ModuleList(
            _blocks(chs[i], dec_blocks[i], spectral[i]) for i in range(levels - 2, -1, -1)
        )
        self.outc = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feats = []
        e = torch.nn.functional.gelu(self.inc(x))
        for enc, down in zip(self.encs, self.downs):
            e = enc(e)
            feats.append(e)
            e = down(e)
        b = self.bottleneck(e)
        d = b
        for up, dec, skip in zip(self.ups, self.decs, reversed(feats)):
            d = dec(up(d) + skip)
        residual = self.outc(d)
        out = torch.clamp(x - residual, 0.0, 1.0)
        if return_features:
            return out, b
        return out

    def bottleneck_features(self, x: torch.Tensor) -> torch.Tensor:
        e = torch.nn.functional.gelu(self.inc(x))
        for enc, down in zip(self.encs, self.downs):
            e = enc(e)
            e = down(e)
        return self.bottleneck(e)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def arch_config(self) -> dict:
        return {
            "base": self.inc.out_channels,
            "levels": len(self.downs) + 1,
            "enc_blocks": [len(b) for b in self.encs],
            "bottleneck_blocks": len(self.bottleneck),
            "dec_blocks": [len(b) for b in self.decs],
        }


LEGACY_PREFIX_MAP = {
    "enc1.": "encs.0.0.",
    "down1.": "downs.0.",
    "enc2.": "encs.1.0.",
    "down2.": "downs.1.",
    "up2.": "ups.0.",
    "dec2.": "decs.0.0.",
    "up1.": "ups.1.",
    "dec1.": "decs.1.0.",
}


def load_legacy_state_dict(model: FourierUNet, legacy: dict) -> None:
    remapped = {}
    for k, v in legacy.items():
        for old, new in LEGACY_PREFIX_MAP.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
