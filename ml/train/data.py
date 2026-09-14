from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DescreenDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train"):
        d = Path(root) / split
        self.scans = sorted((d / "scan").glob("*.png"))
        self.gts = sorted((d / "gt").glob("*.png"))
        assert len(self.scans) == len(self.gts) and len(self.scans) > 0
        assert all(s.stem == g.stem for s, g in zip(self.scans, self.gts))

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        scan = np.asarray(Image.open(self.scans[i]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(self.gts[i]).convert("RGB"), dtype=np.float32) / 255.0
        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)
        return to_t(scan), to_t(gt), self.scans[i].stem
