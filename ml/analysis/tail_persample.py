#!/usr/bin/env python
"""Шаг 2b: per-sample PSNR run7 на 3 val-сетах (GPU, reuse eval_models.load_model).

Выход: ml/analysis/_tail_persample.json — {set: [{id, psnr}]}, n=300 на сет.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.eval_models import load_model  # noqa: E402

SETS = ["Data/synth", "Data/synth_run4", "Data/synth_hd"]
N = 300


@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_model("ml/train/_run7")
    model = model.to(device)
    use_dct = bool(cfg.get("extra_dct"))
    if use_dct:
        from ml.train.dctfeat import dct_channels
    out: dict = {}
    for s in SETS:
        root = ROOT / s
        scans = sorted((root / "val" / "scan").glob("*.png"))[:N]
        rows = []
        for i in range(0, len(scans), 4):
            chunk = scans[i:i + 4]
            xs, ys = [], []
            for p in chunk:
                x = np.asarray(Image.open(p), dtype=np.float32) / 255.0
                y = np.asarray(Image.open(root / "val" / "gt" / p.name), dtype=np.float32) / 255.0
                xs.append(torch.from_numpy(x).permute(2, 0, 1))
                ys.append(torch.from_numpy(y).permute(2, 0, 1))
            xb = torch.stack(xs).to(device)
            yb = torch.stack(ys).to(device)
            if use_dct:
                feats = [dct_channels(np.asarray(Image.open(p), dtype=np.float32) / 255.0) for p in chunk]
                fb = torch.from_numpy(np.stack(feats)).to(device)
                xb = torch.cat([xb, fb], dim=1)
            pred = model(xb).clamp(0, 1)
            for j, p in enumerate(chunk):
                mse = torch.mean((pred[j] - yb[j]) ** 2).item()
                rows.append({"id": p.stem, "psnr": round(10.0 * math.log10(1.0 / max(mse, 1e-12)), 3)})
        ps = [r["psnr"] for r in rows]
        print(f"{s}: n={len(rows)} mean={np.mean(ps):.2f} p10={np.percentile(ps,10):.2f} min={min(ps):.2f}", flush=True)
        out[s] = rows
    outp = ROOT / "ml" / "analysis" / "_tail_persample.json"
    outp.write_text(json.dumps(out), encoding="utf-8")
    print("saved:", outp)


if __name__ == "__main__":
    main()
