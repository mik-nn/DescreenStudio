"""No-GT quality metrics for real scans.

- residual_peak: strongest symmetric peak above the raster band (raster leftovers)
- lf_ratio: axial LF-band energy out/src (banding amplified or kept?)
- sharp_ratio: Laplacian-variance out/src (detail eaten or kept?)
- dc_shift: per-channel mean(out-src) (tone darkening/brightening)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SpectrumConfig
from .spectrum import _freq_grids, log_magnitude, luma, tophat_contrast


@dataclass(frozen=True)
class NoGTReport:
    residual_peak_xmean: float
    residual_peak_r: float
    lf_ratio: float
    sharp_ratio: float
    dc_shift: tuple[float, float, float]


def _lapvar(g: np.ndarray) -> float:
    lap = g[:-2, :-2] + g[:-2, 2:] + g[2:, :-2] + g[2:, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def center_roi(a: np.ndarray, s: int = 1024) -> np.ndarray:
    h, w = a.shape[:2]
    s = min(s, h, w)
    return a[h // 2 - s // 2 : h // 2 + s // 2, w // 2 - s // 2 : w // 2 + s // 2]


def residual_peak(gray: np.ndarray, r_min: float = 0.05) -> tuple[float, float]:
    g = center_roi(gray)
    n = g.shape[0]
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(f)
    mean = mag.mean()
    _, _, r = _freq_grids(n)
    m = mag.copy()
    m[r < r_min] = 0.0
    j, i = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(m[j, i] / mean), float(r[j, i])


def lf_energy_ratio(
    src: np.ndarray,
    out: np.ndarray,
    lo: float = 0.006,
    hi: float = 0.05,
    axis_tol: float = 8.0,
) -> float:
    """Absolute LF-band energy out/src (NOT normalized by total energy, so
    that HF removal elsewhere does not inflate the ratio)."""

    def energy(g: np.ndarray) -> float:
        g = center_roi(g)
        n = g.shape[0]
        f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
        fx, fy, r = _freq_grids(n)
        ang = np.degrees(np.arctan2(fy, fx))
        d_axis = np.minimum(
            np.abs((ang + 90.0) % 180.0 - 90.0), np.abs((ang + 180.0) % 180.0 - 90.0)
        )
        sel = (r >= lo) & (r < hi) & (d_axis <= axis_tol)
        p = np.abs(f) ** 2
        return float(p[sel].sum())

    return energy(out) / (energy(src) + 1e-12)


def evaluate(src_rgb: np.ndarray, out_rgb: np.ndarray, cfg: SpectrumConfig) -> NoGTReport:
    _ = cfg
    gs, go = luma(src_rgb), luma(out_rgb)
    xm, rr = residual_peak(go)
    return NoGTReport(
        residual_peak_xmean=xm,
        residual_peak_r=rr,
        lf_ratio=lf_energy_ratio(gs, go),
        sharp_ratio=_lapvar(center_roi(go)) / (_lapvar(center_roi(gs)) + 1e-12),
        dc_shift=tuple(float((out_rgb[:, :, c] - src_rgb[:, :, c]).mean()) for c in range(3)),
    )
