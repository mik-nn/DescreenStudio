#!/usr/bin/env python
"""Фикс blend-регрессии: весовая карта не должна нести растр.

Варианты:
  base   — blend_map(src) (текущее поведение, эталон регрессии)
  pre    — blend_map(pre) — карта из очищенного изображения
  smooth4/smooth8 — blend_map(src), карта сглажена Гауссом sigma=4/8
                    (период растра ~6.7 px -> сглаживание убивает модуляцию)

Метрики: energy пика в кольце A0 (r±20%, угол±10°), evaluate() resid/sharp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.analyze_resid_1821145u import peak_in_ring  # noqa: E402
from ml.preproc.blend import adaptive_blend, blend_map  # noqa: E402
from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402


def main() -> None:
    cfg = P0Config()
    src = load_rgb(str(ROOT / "Data/RealScan/1821145u_1.tif"))
    s = min(2048, src.shape[0], src.shape[1])
    y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
    src = src[y:y + s, x:x + s]
    est = estimate_raster(luma(src), cfg.spectrum)
    r0, ang0 = est.r, est.angle_deg

    ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
    cur = auto_notch_rgb(src, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
    cur, _ = adaptive_notch_rgb(cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
    bands = detect_lf_bands(luma(cur), cfg.lf)
    pre = subtract_lf_rgb(cur, bands, cfg.lf)

    net_raw = np.load(ROOT / "Data/RealScan/_post0921/_cache/1821145u_1_run3_netraw.npy")

    w_src = blend_map(src, cfg.blend)[..., None]
    variants = {
        "base": w_src,
        "pre": blend_map(pre, cfg.blend)[..., None],
        "smooth4": gaussian_filter(w_src[..., 0], 4.0)[..., None],
        "smooth8": gaussian_filter(w_src[..., 0], 8.0)[..., None],
    }
    for name, w in variants.items():
        blended = np.clip(pre * (1.0 - w) + net_raw * w, 0.0, 1.0).astype(np.float32)
        toned = linear_tone(return_background(src, blended, default_sigma(cfg)), src)
        rep = evaluate(src, toned, cfg.spectrum)
        r, a, e = peak_in_ring(luma(toned), r0, ang0)
        # насколько сама карта модулирована растром: энергия ВЧ карты
        w_hf = float(np.abs(np.fft.fft2(w[..., 0] * np.hanning(s)[:, None] * np.hanning(s)[None, :])).mean())
        print(f"{name:8s}: peak r={r:.4f} ang={a:5.1f} E={e:7.1f} | "
              f"resid={rep.residual_peak_xmean:6.1f} sharp={rep.sharp_ratio:.2f} "
              f"DC={rep.dc_shift[0]:+.3f} LF={rep.lf_ratio:.2f} | w_hf={w_hf:.2f}", flush=True)


if __name__ == "__main__":
    main()
