"""Contrast-enhanced 2D spectrum + symmetric peak detection (v3.7 parity).

Pipeline per ROI: luma -> log(1+|FFT|) [DC centered] -> box top-hat
(log - boxblur(log)) -> threshold mean+std*sensitivity outside DC radius ->
greedy 3x3 local maxima with Hermitian symmetry check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SpectrumConfig


@dataclass(frozen=True)
class Peak:
    r: float  # cycles/px
    angle_deg: float  # atan2(fy, fx), degrees
    x: int  # spectrum pixel, fftshifted frame
    y: int
    strength: float  # top-hat value at peak


@dataclass(frozen=True)
class RasterEstimate:
    r: float
    angle_deg: float
    confidence: float  # 0..1, margin of fundamental above threshold
    harmonic_r: float | None  # sqrt(2) diagonal harmonic if found
    peaks: tuple[Peak, ...]


def luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.float64)


def log_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    return np.log1p(np.abs(f))


def box_blur(a: np.ndarray, win: int) -> np.ndarray:
    """Box mean via integral image (no scipy dependency)."""
    r = win // 2
    padded = np.pad(a, r, mode="reflect")
    ii = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = a.shape
    yy0 = np.broadcast_to(np.arange(h)[:, None], (h, w))
    xx0 = np.broadcast_to(np.arange(w)[None, :], (h, w))
    s = ii[yy0 + win, xx0 + win] - ii[yy0, xx0 + win] - ii[yy0 + win, xx0] + ii[yy0, xx0]
    return s / float(win * win)


def tophat_contrast(logmag: np.ndarray, win: int) -> np.ndarray:
    return np.clip(logmag - box_blur(logmag, win), 0.0, None)


def _freq_grids(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - n / 2) / n
    fy = (yy - n / 2) / n
    return fx, fy, np.sqrt(fx * fx + fy * fy)


def detect_peaks(
    diff: np.ndarray, cfg: SpectrumConfig
) -> tuple[list[Peak], float]:
    """Greedy symmetric peak detection. Returns (peaks, threshold)."""
    n = diff.shape[0]
    assert diff.shape == (n, n)
    fx, fy, r = _freq_grids(n)
    valid = r > cfg.dc_radius_px / n
    vals = diff[valid]
    mean, std = float(vals.mean()), float(vals.std())
    thr = mean + std * cfg.sensitivity
    order = np.argsort(diff[valid])[::-1]
    ys, xs = np.nonzero(valid)
    visited = np.zeros_like(diff, dtype=bool)
    peaks: list[Peak] = []
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        if visited[y, x]:
            continue
        v = float(diff[y, x])
        if v <= thr:
            break
        y0, y1 = max(0, y - 1), min(n, y + 2)
        x0, x1 = max(0, x - 1), min(n, x + 2)
        if diff[y0:y1, x0:x1].max() > v + 1e-12:
            continue
        sx, sy = n - x, n - y
        if sx >= n:
            sx = 0
        if sy >= n:
            sy = 0
        if float(diff[sy, sx]) <= thr * cfg.sym_ratio:
            continue
        peaks.append(
            Peak(
                r=float(r[y, x]),
                angle_deg=float(np.degrees(np.arctan2(fy[y, x], fx[y, x]))),
                x=x,
                y=y,
                strength=v,
            )
        )
        visited[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
        visited[max(0, sy - 1) : sy + 2, max(0, sx - 1) : sx + 2] = True
        if len(peaks) >= cfg.max_peaks:
            break
    return peaks, thr


def estimate_raster(gray: np.ndarray, cfg: SpectrumConfig) -> RasterEstimate:
    """Single-ROI raster estimate with sqrt(2) harmonic check."""
    n = cfg.measure_roi
    h, w = gray.shape
    y = max(0, h // 2 - n // 2)
    x = max(0, w // 2 - n // 2)
    roi = gray[y : y + n, x : x + n]
    diff = tophat_contrast(log_magnitude(roi), cfg.tophat_win)
    peaks, thr = detect_peaks(diff, cfg)
    if not peaks:
        return RasterEstimate(0.0, 0.0, 0.0, None, ())
    fund = peaks[0]
    harmonic_r: float | None = None
    for p in peaks[1:]:
        ratio = p.r / fund.r if fund.r > 0 else 0.0
        d_ang = abs((p.angle_deg - fund.angle_deg + 180.0) % 180.0 - 90.0)
        # diagonal harmonic of a square screen: ~sqrt(2) farther, ~45 deg off
        ang45 = abs(abs((p.angle_deg - fund.angle_deg + 90.0) % 180.0 - 90.0) - 45.0)
        if abs(ratio - np.sqrt(2.0)) < 0.08 and (d_ang < 12.0 or ang45 < 12.0):
            harmonic_r = p.r
            break
    confidence = float(np.clip((fund.strength - thr) / (thr * 5.0 + 1e-12), 0.0, 1.0))
    return RasterEstimate(fund.r, fund.angle_deg, confidence, harmonic_r, tuple(peaks))


def harmonic_family(r: float, angle_deg: float) -> list[tuple[float, float]]:
    """Predicted notch targets of a square screen: fundamental, 2nd harmonic
    (same angle) and the two sqrt(2) diagonal harmonics (+-45 deg)."""
    s2 = float(np.sqrt(2.0))
    return [
        (r, angle_deg),
        (2.0 * r, angle_deg),
        (s2 * r, angle_deg + 45.0),
        (s2 * r, angle_deg - 45.0),
    ]


def raster_map(
    gray: np.ndarray, cfg: SpectrumConfig, patch: int = 256
) -> np.ndarray:
    """Per-patch raster presence (confidence>=0.3) grid for masking/FM zones."""
    h, w = gray.shape
    ys = range(0, max(1, h - patch + 1), patch)
    xs = range(0, max(1, w - patch + 1), patch)
    grid = np.zeros((len(list(ys)), len(list(xs))), dtype=np.float32)
    for j, y in enumerate(range(0, max(1, h - patch + 1), patch)):
        for i, x in enumerate(range(0, max(1, w - patch + 1), patch)):
            roi = gray[y : y + patch, x : x + patch]
            diff = tophat_contrast(log_magnitude(roi), cfg.tophat_win)
            peaks, thr = detect_peaks(diff, cfg)
            grid[j, i] = 1.0 if peaks and (peaks[0].strength - thr) / (thr * 5.0 + 1e-12) >= 0.3 else 0.0
    return grid
