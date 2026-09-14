from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ml.simulator.compose import print_masks, simulate_print
from ml.simulator.config import ScanConfig, preset
from ml.simulator.preview import center_crop
from ml.simulator.scan import simulate_scan

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=20)
    ap.add_argument("--gt-size", type=int, default=512)
    ap.add_argument("--out", default="ml/analysis/_plates")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "scan").mkdir(parents=True, exist_ok=True)
    (out / "plates").mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "Data" / "gt" / "dev").glob("*.jpg"))[: args.photos]
    for i, f in enumerate(files):
        im = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        gt = center_crop(im, args.gt_size)
        pcfg = preset("magazine", oversample=8, seed=100 + i)
        scfg = ScanConfig(seed=200 + i)
        printed = simulate_print(gt, pcfg)
        scan, _ = simulate_scan(printed, pcfg, scfg)
        masks = print_masks(gt, pcfg)
        Image.fromarray((scan * 255 + 0.5).astype(np.uint8)).save(out / "scan" / f"{f.stem}.png")
        for ch, m in masks.items():
            down = np.asarray(
                Image.fromarray((m * 255 + 0.5).astype(np.uint8)).resize(
                    (args.gt_size, args.gt_size), Image.BOX
                ),
                dtype=np.float32,
            ) / 255.0
            np.save(out / "plates" / f"{f.stem}_{ch}.npy", down)
        print(f"{i + 1}/{len(files)} {f.stem}", flush=True)


if __name__ == "__main__":
    main()
