#!/usr/bin/env python
"""Метрика остаточного полиграфического растра (детектор в стиле DSP v3.7).

2D FFT люмы -> логарифм магнитуды -> sigma-thresholding в полосе частот
растра -> избыточная энергия пиков над порогом. Чем меньше, тем чище вывод.

Использование:
    python residual_raster.py <img1> [img2 ...] [--screen-r 0.443]
"""
import argparse
import numpy as np
from PIL import Image


def luma(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float64) / 255.0


def residual_metrics(gray: np.ndarray, r_lo: float = 0.15, r_hi: float = 0.95,
                     k_sigma: float = 3.0):
    h, w = gray.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = wy * wx
    spec = np.fft.fftshift(np.fft.fft2(gray * win))
    mag = np.abs(spec)
    logmag = np.log10(mag + 1e-12)

    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    band = (r >= r_lo) & (r <= r_hi)

    vals = logmag[band]
    thr = vals.mean() + k_sigma * vals.std()
    above = logmag > thr
    above &= band

    excess = float((logmag[above] - thr).sum())
    peak = float(logmag[above].max() - thr) if above.any() else 0.0
    frac = float(above.sum()) / float(band.sum())
    # положение сильнейшего пика
    if above.any():
        idx = np.argmax(np.where(above, logmag, -np.inf))
        py, px = np.unravel_index(idx, logmag.shape)
        fyk, fxk = fy[py, 0], fx[0, px]
        peak_r = float(np.hypot(fxk, fyk))
        peak_deg = float(np.degrees(np.arctan2(fyk, fxk)))
    else:
        peak_r, peak_deg = 0.0, 0.0
    return dict(thr=thr, excess=excess, peak=peak, frac=frac,
                peak_r=peak_r, peak_deg=peak_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--screen-r", type=float, default=None,
                    help="ожидаемая частота растра, цикл/px (для справки)")
    args = ap.parse_args()

    print(f"{'image':<44} {'thr':>6} {'excess':>9} {'peak':>6} {'frac%':>6} "
          f"{'r_peak':>7} {'deg':>6}")
    for p in args.images:
        m = residual_metrics(luma(p))
        print(f"{p.split(chr(92))[-1] or p.split('/')[-1]:<44} "
              f"{m['thr']:6.2f} {m['excess']:9.1f} {m['peak']:6.2f} "
              f"{m['frac'] * 100:6.3f} {m['peak_r']:7.4f} {m['peak_deg']:6.1f}")
    if args.screen_r:
        print(f"(ожидаемая частота растра: {args.screen_r} цикл/px)")


if __name__ == "__main__":
    main()
