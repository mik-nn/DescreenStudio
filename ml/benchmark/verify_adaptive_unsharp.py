#!/usr/bin/env python
"""Проверка adaptive_unsharp на кэше net_raw (рычаг 3, 2026-09-21).

Берёт кэш сырого выхода сети из postprocess_grid (_cache/*_netraw.npy),
прогоняет остаток цепочки (blend -> bgswap -> lintone -> adaptive_unsharp)
и сравнивает с базовым (amount=0).

Использование:
    python ml/benchmark/verify_adaptive_unsharp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.blend import adaptive_blend  # noqa: E402
from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402
from ml.preproc.unsharp import adaptive_unsharp  # noqa: E402

CACHE = ROOT / "Data/RealScan/_post0921/_cache"
SCANS = {
    "1821145u_1": "Data/RealScan/1821145u_1.tif",
    "Offset1": "Data/RealScan/Offset1.tif",
    "Magazine": "Data/RealScan/Magazine.tif",
}


def main() -> None:
    cfg = P0Config()
    for stem, scan_path in SCANS.items():
        src = load_rgb(str(ROOT / scan_path))
        s = min(2048, src.shape[0], src.shape[1])
        y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
        src = src[y:y + s, x:x + s]
        est = estimate_raster(luma(src), cfg.spectrum)
        ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
        cur = auto_notch_rgb(src, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
        cur, _ = adaptive_notch_rgb(cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
        bands = detect_lf_bands(luma(cur), cfg.lf)
        pre = subtract_lf_rgb(cur, bands, cfg.lf)

        raw_files = sorted(CACHE.glob(f"{stem}_*_netraw.npy"))
        for rf in raw_files:
            net_raw = np.load(rf)
            toned = linear_tone(return_background(src, adaptive_blend(pre, net_raw, src, cfg.blend),
                                                  default_sigma(cfg)), src)
            base = evaluate(src, toned, cfg.spectrum)
            out, amt, sharp = adaptive_unsharp(toned, src, cfg.spectrum)
            rep = evaluate(src, out, cfg.spectrum)
            print(f"{stem:12s} {rf.stem.replace(stem + '_', '').replace('_netraw', ''):6s} "
                  f"base: resid={base.residual_peak_xmean:6.1f} sharp={base.sharp_ratio:.2f} | "
                  f"adaptive(amt={amt:.2f}): resid={rep.residual_peak_xmean:6.1f} "
                  f"sharp={rep.sharp_ratio:.2f} DC={rep.dc_shift[0]:+.3f} LF={rep.lf_ratio:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
