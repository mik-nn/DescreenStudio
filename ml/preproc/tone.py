"""Tone handling: measurement (tone curve) + histogram alignment.

The net shifts tone (DC and per-brightness curve). Fix cheaply without
retraining: match the output histogram to the input one, per channel.
Deterministic (no randomness).
"""

from __future__ import annotations

import numpy as np

from .config import ToneConfig


def tone_curve(
    src: np.ndarray, out: np.ndarray, cfg: ToneConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Mean output brightness per input-brightness bin (luma)."""
    lum_s = 0.299 * src[:, :, 0] + 0.587 * src[:, :, 1] + 0.114 * src[:, :, 2]
    lum_o = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    edges = np.linspace(0.0, 1.0, cfg.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(lum_s.ravel(), edges) - 1, 0, cfg.bins - 1)
    means = np.array(
        [lum_o.ravel()[idx == b].mean() if (idx == b).any() else np.nan for b in range(cfg.bins)]
    )
    return centers, means


def match_histogram(out: np.ndarray, ref: np.ndarray, cfg: ToneConfig) -> np.ndarray:
    """Per-channel CDF matching of `out` onto `ref` (both [0,1] RGB)."""
    res = np.empty_like(out)
    edges = np.linspace(0.0, 1.0, cfg.bins + 1)
    for c in range(3):
        o = out[:, :, c].ravel()
        r = ref[:, :, c].ravel()
        ho, _ = np.histogram(o, bins=cfg.bins, range=(0.0, 1.0))
        hr, _ = np.histogram(r, bins=cfg.bins, range=(0.0, 1.0))
        cdf_o = np.cumsum(ho).astype(np.float64)
        cdf_o /= cdf_o[-1] + 1e-12
        cdf_r = np.cumsum(hr).astype(np.float64)
        cdf_r /= cdf_r[-1] + 1e-12
        lut = np.interp(cdf_o, cdf_r, edges[:-1] + 0.5 / cfg.bins)
        res[:, :, c] = lut[np.clip(np.digitize(o, edges) - 1, 0, cfg.bins - 1)].reshape(
            out.shape[:2]
        )
    return np.clip(res, 0.0, 1.0)
