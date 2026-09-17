"""Global axial LF banding: detect + subtract.

Source paper/scanner banding shows as strong peaks strictly on the frequency
axes (0 / +-90 deg) at very low r (period ~100px @800DPI). Content varies per
ROI, banding does not -- so a band counts as global only if the same (r,
axis) repeats across >= min_rois ROIs (corners + center).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LFConfig
from .notch import apply_notch_channel, notch_mask
from .spectrum import _freq_grids, log_magnitude


@dataclass(frozen=True)
class LFBand:
    r: float
    angle_deg: float  # snapped to nearest axis (0 / 90)
    axis: str  # 'h' (varies along x) or 'v'


def _rois(h: int, w: int, s: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, max(0, w - s)),
        (max(0, h - s), 0),
        (max(0, h - s), max(0, w - s)),
        (max(0, h // 2 - s // 2), max(0, w // 2 - s // 2)),
    ]


def detect_lf_bands(gray: np.ndarray, cfg: LFConfig) -> list[LFBand]:
    h, w = gray.shape
    s = min(cfg.roi_size, h, w)
    votes: list[tuple[float, str]] = []
    for y, x in _rois(h, w, s):
        roi = gray[y : y + s, x : x + s]
        # NOTE: raw log-magnitude here, NOT top-hat: the LF hump is broader
        # than any top-hat window and would be subtracted away.
        lm = log_magnitude(roi)
        n = s
        fx, fy, r = _freq_grids(n)
        ang = np.degrees(np.arctan2(fy, fx))
        dist_axis = np.minimum(
            np.abs((ang + 90.0) % 180.0 - 90.0), np.abs((ang + 180.0) % 180.0 - 90.0)
        )
        band = (r >= cfg.band_lo) & (r < cfg.band_hi) & (dist_axis <= cfg.axis_tol_deg)
        if not band.any():
            continue
        med = float(np.median(lm[band]))
        m = lm.copy()
        m[~band] = -np.inf
        j, i = np.unravel_index(int(np.argmax(m)), m.shape)
        if float(m[j, i]) < med + 1.0:  # must stand out of the LF floor
            continue
        rr = float(r[j, i])
        axis = "v" if abs(abs(float(ang[j, i])) - 90.0) < 45.0 else "h"
        votes.append((rr, axis))
    # cluster with tolerance: the broad LF hump jitters between ROIs
    bands: list[LFBand] = []
    unused = votes[:]
    while unused:
        seed_r, seed_ax = unused.pop(0)
        group = [(seed_r, seed_ax)]
        rest: list[tuple[float, str]] = []
        for v in unused:
            if v[1] == seed_ax and abs(v[0] - seed_r) < 0.0025:
                group.append(v)
            else:
                rest.append(v)
        unused = rest
        if len(group) >= cfg.min_rois:
            mean_r = sum(g[0] for g in group) / len(group)
            bands.append(
                LFBand(
                    r=mean_r,
                    angle_deg=90.0 if seed_ax == "v" else 0.0,
                    axis=seed_ax,
                )
            )
    return bands


def subtract_lf_rgb(img: np.ndarray, bands: list[LFBand], cfg: LFConfig) -> np.ndarray:
    """Narrow directional notch per band, all channels."""
    if not bands:
        return img.copy()
    h, w, _ = img.shape
    mask = notch_mask(
        h,
        w,
        [(b.r, b.angle_deg) for b in bands],
        cfg.notch_r,
        cfg.notch_order,
    )
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = apply_notch_channel(img[:, :, c], mask)
    return out
