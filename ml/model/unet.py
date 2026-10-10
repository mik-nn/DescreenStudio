from __future__ import annotations

import torch
import torch.nn as nn

from ml.model.ffblock import FFCBlock, FFCBlockV2, TransformerBottle
from ml.model.film import COND_DIM, FiLMCondition


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


def _blocks(ch: int, n: int, spectral: bool, spectral_mode: str = "plain",
            ff_norm: bool = False, ff_zero_init: bool = False) -> nn.Sequential:
    return nn.Sequential(*[FFCBlockV2(ch, use_spectral=spectral, spectral_mode=spectral_mode,
                                      ff_norm=ff_norm, ff_zero_init=ff_zero_init) for _ in range(n)])


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
        spectral_mode: str = "plain",
        bottleneck_attn: bool = False,  # TransformerBottle (MDTA) после FFC bottleneck
        cond_dim: int = 0,  # >0: FiLM-conditioning bottleneck на вектор A0 (0 = run7-поведение)
        aux: bool = False,  # aux-головы декодера для deep supervision (только трейн, в инференсе отбрасываются)
        ff_norm: bool = False,  # LayerNorm на выходе spectral-ветвей (фикс pile-up run12)
        ff_zero_init: bool = False,  # zero-init последних conv spectral-ветви (отдельно от нормы)
    ):
        super().__init__()
        assert levels in (2, 3)
        assert len(enc_blocks) == levels - 1 and len(dec_blocks) == levels - 1
        assert len(spectral) == levels
        self.cond_dim = cond_dim
        self.has_aux = aux
        self.inc = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        chs = [base * 2**i for i in range(levels)]
        self.encs = nn.ModuleList(_blocks(chs[i], enc_blocks[i], spectral[i], spectral_mode, ff_norm, ff_zero_init) for i in range(levels - 1))
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(levels - 1))
        self.bottleneck = _blocks(chs[-1], bottleneck_blocks, spectral[-1], spectral_mode, ff_norm, ff_zero_init)
        self.film = FiLMCondition(cond_dim, chs[-1]) if cond_dim > 0 else None
        self.attn = TransformerBottle(chs[-1]) if bottleneck_attn else None
        self.ups = nn.ModuleList(Up(chs[i + 1], chs[i]) for i in range(levels - 2, -1, -1))
        self.decs = nn.ModuleList(
            _blocks(chs[i], dec_blocks[i], spectral[i], spectral_mode, ff_norm, ff_zero_init) for i in range(levels - 2, -1, -1)
        )
        self.outc = nn.Conv2d(base, out_channels, kernel_size=3, padding=1)
        # aux-головы на стадиях декодера кроме финальной (прямое предсказание RGB):
        # levels=3 -> [голова 1/2 (chs[1])]; создаются ТОЛЬКО при aux=True,
        # иначе state_dict несовместим со старыми чекпоинтами
        self.aux_heads = nn.ModuleList(
            nn.Conv2d(chs[i], out_channels, kernel_size=1) for i in range(levels - 2, 0, -1)
        ) if aux else nn.ModuleList()
        if aux and len(self.aux_heads) == 0:
            raise ValueError("aux=True needs levels>=3 (no intermediate decoder stages)")

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                return_features: bool = False,
                return_aux: bool = False) -> torch.Tensor | tuple:
        feats = []
        e = torch.nn.functional.gelu(self.inc(x))
        for enc, down in zip(self.encs, self.downs):
            e = enc(e)
            feats.append(e)
            e = down(e)
        b = self.bottleneck(e)
        if self.film is not None and cond is not None:
            b = self.film(b, cond)
        if self.attn is not None:
            b = self.attn(b)
        d = b
        auxs: list[torch.Tensor] = []
        stages = list(zip(self.ups, self.decs, reversed(feats)))
        for si, (up, dec, skip) in enumerate(stages):
            d = dec(up(d) + skip)
            if si < len(self.aux_heads):
                auxs.append(self.aux_heads[si](d))
        residual = self.outc(d)
        base = x[:, :3] if x.shape[1] > 3 else x
        out = torch.clamp(base - residual, 0.0, 1.0)
        if return_features:
            return out, b
        if return_aux:
            if not self.has_aux:
                raise ValueError("return_aux=True needs aux=True model")
            return out, auxs
        return out

    def bottleneck_features(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        e = torch.nn.functional.gelu(self.inc(x))
        for enc, down in zip(self.encs, self.downs):
            e = enc(e)
            e = down(e)
        b = self.bottleneck(e)
        if self.film is not None and cond is not None:
            b = self.film(b, cond)
        return b

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
