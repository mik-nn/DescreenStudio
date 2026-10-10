#!/usr/bin/env python
"""Sweep силы нотча (Phase 2b): префильтр-only при radius 4/6/9px на 3 сканах.
Вопрос: сколько запаса осталось в классике (resid vs sharp tradeoff).
Без сети, CPU. Выход: таблица + metrics.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb, save_rgb  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.benchmark.eval_realscan import fmt  # noqa: E402

CROP = 2048
SCANS = ["Data/RealScan/1821145u_1.tif", "Data/RealScan/Magazine.tif",
         "Data/RealScan/Pack.tif"]
RADII = [4.0, 6.0, 9.0]


def main() -> None:
    base = P0Config()
    outdir = ROOT / "Data" / "RealScan" / "_notchsweep"
    outdir.mkdir(parents=True, exist_ok=True)
    table: dict = {}
    for rel in SCANS:
        src = load_rgb(str(ROOT / rel))
        s = min(CROP, src.shape[0], src.shape[1])
        y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
        src = src[y:y + s, x:x + s]
        stem = Path(rel).stem.replace(" ", "_")
        est = estimate_raster(luma(src), base.spectrum)
        print(f"=== {stem} (A0 r={est.r:.4f} conf={est.confidence:.2f})", flush=True)
        for rad in RADII:
            cfg = dataclasses.replace(
                base,
                notch=dataclasses.replace(base.notch, radius_roi_px=rad),
                edge_notch=dataclasses.replace(
                    base.edge_notch, radius_ref_roi=base.edge_notch.radius_ref_roi),
            )
            ch = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
            cur = auto_notch_rgb(src, ch, cfg.notch, cfg.spectrum.measure_roi)
            cur, _ = adaptive_notch_rgb(cur, ch, cfg.edge_notch,
                                       cfg.edge_notch.radius_ref_roi)
            bands = detect_lf_bands(luma(cur), cfg.lf)
            cur = subtract_lf_rgb(cur, bands, cfg.lf)
            save_rgb(str(outdir / f"{stem}_r{rad:g}.png"), cur)
            rep = evaluate(src, cur, cfg.spectrum)
            table.setdefault(stem, {})[f"r{rad:g}"] = fmt(rep)
            print(f"  r={rad:g}: {fmt(rep)}", flush=True)
    (outdir / "metrics.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("saved:", outdir / "metrics.json")


if __name__ == "__main__":
    main()
