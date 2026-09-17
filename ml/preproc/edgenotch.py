"""E1: edge-adaptive notch.

Global notch kills the stationary screen carrier but leaves sidebands around
contrast transitions (AM screen x content = amplitude modulation). Fix: split
into overlapping patches, scale the notch radius by local contrast
(radius = base * (1 + edge_gain * contrast01)), overlap-add with Hann window.
Flat areas keep the narrow notch (content safe), edges get a wide one.
"""

from __future__ import annotations

import numpy as np

from .config import EdgeNotchConfig
from .notch import apply_notch_channel, notch_mask
from .spectrum import luma


def local_contrast_map(gray: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Mean gradient magnitude per patch (grid covering the frame)."""
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gm = np.sqrt(gx[1:-1, :] ** 2 + gy[:, 1:-1] ** 2)
    h, w = gray.shape
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] + patch < h:
        ys.append(h - patch)
    if xs[-1] + patch < w:
        xs.append(w - patch)
    cmap = np.zeros((len(ys), len(xs)))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            cmap[j, i] = gm[y : y + patch, x : x + patch].mean()
    return cmap


def smooth_grid(c01: np.ndarray, sigma_cells: float = 1.0) -> np.ndarray:
    """Blur the coarse patch grid so neighboring patches transition gradually
    (otherwise radius steps show as fragment boundaries in texture)."""
    from scipy.ndimage import gaussian_filter as _gf

    return _gf(c01, sigma_cells, mode="nearest").astype(np.float32)


def edge01(gray: np.ndarray, patch: int, stride: int, lo_q: float, hi_q: float) -> np.ndarray:
    """Quantile-normalized local-contrast map in [0, 1] (full-frame pixels).

    Bilinear upscale + blur: nearest upscale leaves block seams between
    patches with different pre/net texture.
    """
    from scipy.ndimage import gaussian_filter as _gf
    from scipy.ndimage import zoom as _zoom

    cmap = local_contrast_map(gray, patch, stride)
    lo, hi = np.quantile(cmap, [lo_q, hi_q])
    c01 = np.clip((cmap - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    h, w = gray.shape
    up = _zoom(c01, (h / c01.shape[0], w / c01.shape[1]), order=1)
    up = up[:h, :w]
    if up.shape != (h, w):
        pad = np.zeros((h, w), dtype=np.float32)
        pad[: up.shape[0], : up.shape[1]] = up
        up = pad
    return _gf(up, stride / 4.0, mode="nearest").astype(np.float32)


def adaptive_notch_rgb(
    img: np.ndarray,
    peaks_per_channel: list[list[tuple[float, float]]],
    cfg: EdgeNotchConfig,
    radius_ref_roi: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Patch-wise notch with contrast-scaled radius. Returns (image, cmap01)."""
    from .notch import radius_px_for_frame  # noqa: PLC0415 (keeps import graph flat)

    h, w, _ = img.shape
    gray = luma(img)
    cmap = local_contrast_map(gray, cfg.patch, cfg.stride)
    lo, hi = np.quantile(cmap, [cfg.lo_q, cfg.hi_q])
    cmap01 = np.clip((cmap - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    p, s = cfg.patch, cfg.stride
    ys = list(range(0, max(1, h - p + 1), s))
    xs = list(range(0, max(1, w - p + 1), s))
    if ys[-1] + p < h:
        ys.append(h - p)
    if xs[-1] + p < w:
        xs.append(w - p)
    gain_map = smooth_grid(cmap01)
    t = 2.0 * np.pi * np.arange(p) / (p - 1.0)
    win1 = 0.5 * (1.0 - np.cos(t))
    win2 = (win1[:, None] * win1[None, :]).astype(np.float64)

    acc = np.zeros_like(img, dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    n_done = 0
    n_total = len(ys) * len(xs)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            gain = 1.0 + cfg.edge_gain * float(gain_map[j, i])
            r_roi = cfg.base_radius_roi_px * gain
            r_px = radius_px_for_frame(r_roi, radius_ref_roi, p)
            radius_r = r_px / p
            for c in range(3):
                patch = img[y : y + p, x : x + p, c].astype(np.float64)
                mask = notch_mask(
                    p, p, peaks_per_channel[c], radius_r, cfg.order, cfg.notch_shape
                )
                acc[y : y + p, x : x + p, c] += apply_notch_channel(patch, mask) * win2
            weight[y : y + p, x : x + p] += win2
            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                print(f"  anotch patch {n_done}/{n_total}", flush=True)
    out = acc / np.maximum(weight[:, :, None], 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32), cmap01.astype(np.float32)
