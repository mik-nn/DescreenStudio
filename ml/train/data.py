from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DescreenDataset(Dataset):
    def __init__(self, root: str | Path | list, split: str = "train", extra_dct: bool = False,
                 cond: bool = False, cond_drop: float = 0.0):
        roots = [Path(r) for r in (root if isinstance(root, (list, tuple)) else [root])]
        self.scans: list[Path] = []
        self.gts: list[Path] = []
        self.roots: list[Path] = []
        for r in roots:
            d = r / split
            scans = sorted((d / "scan").glob("*.png"))
            gts = sorted((d / "gt").glob("*.png"))
            assert len(scans) == len(gts) and len(scans) > 0, f"bad split {split} in {r}"
            assert all(s.stem == g.stem for s, g in zip(scans, gts))
            self.scans += scans
            self.gts += gts
            self.roots += [r] * len(scans)
        self.extra_dct = extra_dct
        # FiLM-conditioning: A0-sidecar {stem: [r, angle_deg, conf]} per root.
        # Ключ — (имя корня, stem): p-id пересекаются между корнями
        # (synth/run4/gap все p00000-...), голый stem давал бы чужой замер.
        # Отсутствие файла или id -> нулевой вектор.
        self.cond = cond
        self.cond_drop = cond_drop
        self._a0: dict[tuple[str, str], list[float]] = {}
        if cond:
            from ml.model.film import build_cond_vec
            for r in roots:
                p = r / split / "a0.json"
                if p.exists():
                    import json
                    for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                        self._a0[(r.name, k)] = build_cond_vec(*v)

    def __len__(self) -> int:
        return len(self.scans)

    def cond_vec(self, root_name: str, stem: str) -> torch.Tensor:
        import random as _rng
        v = self._a0.get((root_name, stem), [0.0, 0.0, 1.0, 0.0])
        # cond_drop>0 задаётся только train-датасету (val всегда честный замер)
        if self.cond_drop > 0 and _rng.random() < self.cond_drop:
            v = [0.0, 0.0, 1.0, 0.0]
        return torch.tensor(v, dtype=torch.float32)

    def __getitem__(self, i: int) -> tuple:
        scan = np.asarray(Image.open(self.scans[i]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(self.gts[i]).convert("RGB"), dtype=np.float32) / 255.0
        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)
        if self.extra_dct:
            from ml.train.dctfeat import dct_channels

            lum = (0.299 * scan[:, :, 0] + 0.587 * scan[:, :, 1] + 0.114 * scan[:, :, 2]).astype(np.float32)
            feat = dct_channels(lum)
            scan = np.concatenate([scan, feat], axis=-1)
        stem = self.scans[i].stem
        if self.cond:
            return to_t(scan), to_t(gt), stem, self.cond_vec(self.roots[i].name, stem)
        return to_t(scan), to_t(gt), stem
