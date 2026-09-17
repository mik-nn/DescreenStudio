"""Butterworth notch prefilter (v3.7 formula, numpy port).

H(d) = 1 / (1 + (R / max(d, eps)) ^ (2*order)), applied at a peak and its
Hermitian mirror; masks multiply. Radius scales with image size so that the
notch width is constant in cycles/px.
"""

from __future__ import annotations

import numpy as np

from .config import NotchConfig


def notch_mask(
    h: int,
    w: int,
    peaks_r_angle: list[tuple[float, float]],
    radius_r: float,
    order: int,
    shape: str = "butterworth",
) -> np.ndarray:
    """Multiplicative mask in [0, 1] for full-frame HxW spectrum.

    butterworth: H = 1/(1+(R/d)^2n) — steep but rings near edges.
    gaussian: H = 1-exp(-d^2/2s^2), s matched to the same -3dB width
    (s = R/sqrt(2 ln 2)) — no sidelobes, no ringing.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    fx = (xx - w / 2) / w
    fy = (yy - h / 2) / h
    mask = np.ones((h, w))
    for r, ang in peaks_r_angle:
        ax, ay = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for sgn in (1.0, -1.0):
            d = np.sqrt((fx - sgn * r * ax) ** 2 + (fy - sgn * r * ay) ** 2)
            if shape == "gaussian":
                s = radius_r / np.sqrt(2.0 * np.log(2.0))
                hh = 1.0 - np.exp(-(d**2) / (2.0 * s * s))
            else:
                with np.errstate(divide="ignore"):
                    q = radius_r / np.maximum(d, 1e-9)
                    q = np.where(d < 1e-12, np.inf, q)
                hh = 1.0 / (1.0 + q ** (2 * order))
            mask *= hh
    return mask


def apply_notch_channel(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(f * mask))) + gray.mean()
    return out


def radius_px_for_frame(radius_roi_px: float, roi_size: int, frame_min: int) -> float:
    return radius_roi_px / roi_size * frame_min


def auto_notch_rgb(
    img: np.ndarray,
    peaks_per_channel: list[list[tuple[float, float]]],
    cfg: NotchConfig,
    roi_size: int = 256,
) -> np.ndarray:
    """Apply per-channel notch masks built from (r, angle) peak lists."""
    h, w, _ = img.shape
    r_px = radius_px_for_frame(cfg.radius_roi_px, roi_size, min(h, w))
    radius_r = r_px / min(h, w)
    out = np.empty_like(img)
    for c in range(3):
        mask = notch_mask(h, w, peaks_per_channel[c], radius_r, cfg.order)
        out[:, :, c] = apply_notch_channel(img[:, :, c], mask)
    return out
