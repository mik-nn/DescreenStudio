#!/usr/bin/env python
"""A0-sidecar для FiLM-conditioning: замер растра на каждом train/val скане.

Выход: <root>/<split>/a0.json — {stem: [r, angle_deg, conf]}.
Считается один раз офлайн (CPU); лоадер подхватывает без цены в обучении.
Схема расширяема: hd2 считается тем же скриптом после сборки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402


def run_root(root: str, splits: tuple[str, ...]) -> None:
    cfg = P0Config().spectrum
    for split in splits:
        scan_dir = ROOT / root / split / "scan"
        scans = sorted(scan_dir.glob("*.png"))
        outp = ROOT / root / split / "a0.json"
        done: dict = {}
        if outp.exists():
            done = json.loads(outp.read_text(encoding="utf-8"))
        todo = [p for p in scans if p.stem not in done]
        print(f"{root}/{split}: total={len(scans)} todo={len(todo)}", flush=True)
        for i, p in enumerate(todo):
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            est = estimate_raster(luma(img), cfg)
            done[p.stem] = [round(est.r, 5), round(est.angle_deg, 2), round(est.confidence, 4)]
            if (i + 1) % 500 == 0:
                print(f"{root}/{split}: {i + 1}/{len(todo)}", flush=True)
        outp.write_text(json.dumps(done), encoding="utf-8")
        print(f"{root}/{split}: saved {len(done)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="Data/synth,Data/synth_run4,Data/synth_gap")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    for r in args.roots.split(","):
        run_root(r.strip(), tuple(s.strip() for s in args.splits.split(",")))


if __name__ == "__main__":
    main()
