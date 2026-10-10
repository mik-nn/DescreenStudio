#!/usr/bin/env python
"""Рычаг 3 (2026-09-21): постобработка после сети, не трогая backbone.

Идея: NN сглаживает растр ценой потери резкости (sharp 0.40 на Magazine,
0.83–0.90 на 1821145u). Проверяем: восстанавливает ли лёгкий unsharp-mask
резкость БЕЗ роста остаточного растра (resid) и DC-сдвига.

Протокол: полная P0/E-цепочка до net_tiled — один раз, с кэшем net_raw
(.npy в outdir/_cache). Дальше сетка постобработки — чистый CPU:
  blended = adaptive_blend(pre, net_raw, src)
  bgswap  = return_background(src, blended, sigma)
  tone    = linear_tone(bgswap, src)
  final   = clip(tone + amount * (tone - gauss(tone, sigma_unsharp)))

Использование:
    python ml/benchmark/postprocess_grid.py --scans Data/RealScan/... --models run3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.eval_models import load_model  # noqa: E402
from ml.benchmark.eval_realscan import net_tiled  # noqa: E402
from ml.preproc.blend import adaptive_blend  # noqa: E402
from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402


def unsharp(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return img
    blur = np.stack([gaussian_filter(img[..., c], sigma) for c in range(3)], axis=-1)
    return np.clip(img + amount * (img - blur), 0.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["run3"])
    ap.add_argument("--outdir", default=r"Data\RealScan\_post0921")
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--amounts", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.8])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 1.5])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = P0Config()
    outdir = ROOT / args.outdir
    cache_dir = outdir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}

    models = {n: load_model(f"ml/train/_{n}")[0].to(device).eval() for n in args.models}

    for scan_path in args.scans:
        src = load_rgb(str(ROOT / scan_path))
        if args.crop:
            s = min(args.crop, src.shape[0], src.shape[1])
            y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
            src = src[y:y + s, x:x + s]
        stem = Path(scan_path).stem.replace(" ", "_")

        # prefilter-цепочка до сети
        est = estimate_raster(luma(src), cfg.spectrum)
        ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
        cur = auto_notch_rgb(src, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
        cur, _ = adaptive_notch_rgb(cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
        bands = detect_lf_bands(luma(cur), cfg.lf)
        pre = subtract_lf_rgb(cur, bands, cfg.lf)

        base = evaluate(src, pre, cfg.spectrum)
        print(f"\n=== {stem}  prefilter: resid={base.residual_peak_xmean:6.1f} sharp={base.sharp_ratio:.2f}", flush=True)

        for name, model in models.items():
            raw_path = cache_dir / f"{stem}_{name}_netraw.npy"
            if raw_path.exists():
                net_raw = np.load(raw_path)
            else:
                net_raw = net_tiled(pre, model, False, cfg.tile.tile, cfg.tile.overlap, device)
                np.save(raw_path, net_raw)
            blended = adaptive_blend(pre, net_raw, src, cfg.blend)
            toned = linear_tone(return_background(src, blended, default_sigma(cfg)), src)
            for amt in args.amounts:
                for sg in args.sigmas:
                    out = unsharp(toned, amt, sg)
                    rep = evaluate(src, out, cfg.spectrum)
                    key = f"{stem}/{name}/a{amt:.1f}/s{sg:.1f}"
                    table[key] = dict(resid=round(rep.residual_peak_xmean, 1), sharp=round(rep.sharp_ratio, 3),
                                      dc=[round(v, 4) for v in rep.dc_shift], lf=round(rep.lf_ratio, 3))
                    print(f"  {name} amount={amt:.1f} sigma={sg:.1f}: "
                          f"resid={rep.residual_peak_xmean:6.1f} sharp={rep.sharp_ratio:.2f} "
                          f"DC={rep.dc_shift[0]:+.3f} LF={rep.lf_ratio:.2f}", flush=True)

    (outdir / "grid.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("\nsaved:", outdir / "grid.json")


if __name__ == "__main__":
    main()
