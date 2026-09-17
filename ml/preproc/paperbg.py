"""Paper background split & return.

Idea: the network mishandles tone (brightens +0.10) and soaps flat paper.
Decompose deterministically and let the net own ONLY the detail layer:

    bg   = gaussian_lowpass(img, sigma)      # paper tone, illumination
    det  = img - bg                           # zero-mean detail (raster+content)

    final = clip(bg_src + (net_out - bg_net))

Background comes back bit-exact from the source (paper never touched);
the net contributes detail only, so it cannot shift tone or blur paper.
LF banding (r~0.01) stays in the detail path (sigma default keeps bg
cutoff ~0.005) where the LF-notch owns it.
"""

from __future__ import annotations

import numpy as np

from .config import P0Config


def gaussian_lowpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """FFT Gaussian blur, per channel. img: HxWx3 float."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    fx = (xx - w / 2) / w
    fy = (yy - h / 2) / h
    g = np.exp(-0.5 * ((fx * fx + fy * fy) * (2.0 * np.pi * sigma) ** 2))
    f = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(f * g[:, :, None], axes=(0, 1)), axes=(0, 1)))
    return out


def split_bg_detail(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    bg = gaussian_lowpass(img, sigma)
    return bg, img - bg


def return_background(
    src: np.ndarray, net_out: np.ndarray, sigma: float
) -> np.ndarray:
    """final = bg(src) + detail(net_out). Tone/paper bit-preserved."""
    bg_src, _ = split_bg_detail(src, sigma)
    bg_net, _ = split_bg_detail(net_out, sigma)
    return np.clip(bg_src + (net_out - bg_net), 0.0, 1.0).astype(np.float32)


def linear_tone(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Diet tone fix: per-channel mean/std match (no CDF stretch)."""
    res = np.empty_like(out)
    for c in range(3):
        o, r = out[:, :, c], ref[:, :, c]
        a = r.std() / (o.std() + 1e-12)
        res[:, :, c] = (o - o.mean()) * a + r.mean()
    return np.clip(res, 0.0, 1.0)


def default_sigma(_cfg: P0Config) -> float:
    return 32.0
