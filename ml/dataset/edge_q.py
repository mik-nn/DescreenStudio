"""Edge-density sidecar for hard boosting: mean gradient magnitude per train
GT patch -> <split>/edge_q.json {id: q}. Deterministic, no randomness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def edge_density(p: Path) -> float:
    im = Image.open(p).convert("RGB")
    im.load()
    a = np.asarray(im, dtype=np.float32) / 255.0
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    gy, gx = np.gradient(lum)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data/synth_run4")
    args = ap.parse_args()
    gt_dir = Path(args.root) / "train" / "gt"
    out = {}
    files = sorted(gt_dir.glob("*.png"))
    for i, p in enumerate(files):
        out[p.stem] = round(edge_density(p), 6)
        if (i + 1) % 1000 == 0:
            print(f"edge_q {i + 1}/{len(files)}", flush=True)
    (Path(args.root) / "train" / "edge_q.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"edge_q done: {len(out)}", flush=True)


if __name__ == "__main__":
    main()
