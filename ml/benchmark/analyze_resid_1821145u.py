#!/usr/bin/env python
"""Анализ: откуда на 1821145u_1 растёт resid (53.5 prefilter -> 90 run3)?

Гипотезы:
A. net_raw сети уже содержит пик растра (модель его не дожала/породила).
B. Пик появляется на стадиях blend/bgswap/lintone.
C. Пик смещён по радиусу/углу относительно ожидаемого (метрика меряет на r ожидания).

Протокол: спектр люмы (лог) центрального ROI на src, pre, net_raw, blend, tone.
Пик ищем в кольце r∈[0.8,1.2]×r0, угол ±10° от A0 (estimate_raster).
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
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.paperbg import default_sigma, linear_tone, return_background  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402


def peak_in_ring(img_luma: np.ndarray, r0: float, ang0: float, roi: int = 1024):
    """Энергия максимального пика в кольце вокруг (r0, ang0); возвращает (r, ang, energy)."""
    h, w = img_luma.shape
    y0, x0 = (h - roi) // 2, (w - roi) // 2
    g = img_luma[y0:y0 + roi, x0:x0 + roi]
    g = g * np.hanning(roi)[:, None] * np.hanning(roi)[None, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    fy = np.fft.fftshift(np.fft.fftfreq(roi))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(roi))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.degrees(np.arctan2(fy, fx)) % 180.0
    d_ang = np.abs(((ang - ang0) + 90) % 180 - 90)
    mask = (r > r0 * 0.8) & (r < r0 * 1.2) & (d_ang < 10) & (r > 0.02)
    if not mask.any():
        return None
    idx = np.unravel_index(np.argmax(np.where(mask, F, 0)), F.shape)
    return float(r[idx]), float(ang[idx]), float(F[idx])


def main() -> None:
    cfg = P0Config()
    stem = "1821145u_1"
    src = load_rgb(str(ROOT / "Data/RealScan/1821145u_1.tif"))
    s = min(2048, src.shape[0], src.shape[1])
    y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
    src = src[y:y + s, x:x + s]

    est = estimate_raster(luma(src), cfg.spectrum)
    r0, ang0 = est.r, est.angle_deg
    print(f"A0: r={r0:.4f} ang={ang0:.1f} conf={est.confidence:.2f}")

    ch_peaks = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
    cur = auto_notch_rgb(src, ch_peaks, cfg.notch, cfg.spectrum.measure_roi)
    cur, _ = adaptive_notch_rgb(cur, ch_peaks, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
    bands = detect_lf_bands(luma(cur), cfg.lf)
    pre = subtract_lf_rgb(cur, bands, cfg.lf)

    net_raw = np.load(ROOT / "Data/RealScan/_post0921/_cache" / f"{stem}_run3_netraw.npy")
    blended = adaptive_blend(pre, net_raw, src, cfg.blend)
    toned = linear_tone(return_background(src, blended, default_sigma(cfg)), src)

    # med-baseline: медиана энергии кольца той же площади без пикового сектора
    stages = {"src": src, "pre": pre, "net_raw": net_raw, "blend": blended, "tone": toned}
    for name, img in stages.items():
        lu = luma(img)
        res = peak_in_ring(lu, r0, ang0)
        if res:
            r, a, e = res
            print(f"{name:8s}: peak r={r:.4f} ang={a:6.1f} energy={e:8.1f}")
        else:
            print(f"{name:8s}: no peak in ring")
    # broadband-энергия ВН (нормировка): средняя энергия спектра вне пиков
    for name, img in stages.items():
        lu = luma(img)
        g = lu * np.hanning(s)[:, None] * np.hanning(s)[None, :]
        F = np.abs(np.fft.fft2(g))
        print(f"{name:8s}: broadband HF energy (r>0.1 mean) = {F[np.sqrt(np.fft.fftfreq(s)[:,None]**2 + np.fft.fftfreq(s)[None,:]**2) > 0.1].mean():.2f}")


if __name__ == "__main__":
    main()
