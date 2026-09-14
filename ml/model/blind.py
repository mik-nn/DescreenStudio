from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

PRESETS = ["magazine", "newspaper", "fine"]
DPIS = [300, 600]


def label_of(preset: str, dpi: int) -> int:
    return PRESETS.index(preset) * len(DPIS) + DPIS.index(dpi)


def load_label_map(synth_root: str | Path, split: str) -> dict[str, int]:
    man = [json.loads(l) for l in open(Path(synth_root) / split / "manifest.jsonl", encoding="utf-8")]
    return {r["id"]: label_of(r["preset"], r["scan_dpi"]) for r in man}


class BlindHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 6):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )
        self.n_classes = n_classes

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(feats).flatten(1))
