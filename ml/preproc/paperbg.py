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
from scipy.ndimage import uniform_filter

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
    """Diet tone fix: per-channel mean match + LF-contrast match (2.52).

    Contrast scale is measured on the low-frequency part (box 33..129 px), not on
    the whole frame. After descreening the std of the output is legitimately below
    the scan's, because the raster produced most of it; matching full-frame std
    scaled the whole frame up and inflated paper texture and 16-128 px blotches by
    4-5% (measured on `1821146u`: bands 1.05x with the old rule, 1.00x without it).
    Tone and illumination live in the LF, so that is what gets matched.
    """
    h, w = out.shape[0], out.shape[1]
    win = max(33, min(129, min(h, w) // 4))
    if win % 2 == 0:
        win += 1
    res = np.empty_like(out)
    for c in range(3):
        o, r = out[:, :, c], ref[:, :, c]
        lo = uniform_filter(o.astype(np.float32), win, mode="nearest")
        lr = uniform_filter(r.astype(np.float32), win, mode="nearest")
        a = float(np.clip(lr.std() / (lo.std() + 1e-12), 0.9, 1.1)) if lo.std() > 1e-6 else 1.0
        res[:, :, c] = (o - o.mean()) * a + r.mean()
    return np.clip(res, 0.0, 1.0)


def butter_lowpass_channel(plane: np.ndarray, t_cut: float, order: float) -> np.ndarray:
    """Mirror of `prefilter::butter_lowpass_channel` (replicate pad 2*t_cut)."""
    h, w = plane.shape
    pad = int(max(16.0, np.ceil(2.0 * t_cut)))
    big = np.pad(plane.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    hp, wp = big.shape
    fy = np.fft.fftfreq(hp)[:, None]
    fx = np.fft.fftfreq(wp)[None, :]
    f = np.hypot(fy, fx) * t_cut
    gain = 1.0 / (1.0 + np.power(np.maximum(f, 1e-12), 2.0 * order))
    out = np.real(np.fft.ifft2(np.fft.fft2(big) * gain))
    return out[pad:pad + h, pad:pad + w].astype(np.float32)


def local_tone_guard(
    net: np.ndarray,
    src: np.ndarray,
    t_cut: float,
    order: float,
    t_chroma: float | None = None,
) -> np.ndarray:
    """Mirror of `prefilter::local_tone_guard` (2.50-2.53).

    out = net + LP_luma(src - net) + box2_chroma((src - net) - luma_part)

    Tone and illumination are shared by the channels, so only the luma part of
    the scan's LF comes back (cut above the halftone supercell, `t_cut`). The
    chromatic part of `src - net` comes back only at the very lowest frequency
    (`t_chroma`, default 4*t_cut): the four-colour rosette lives at 8-32 px in
    u/v and must stay out, otherwise the guard refills the colour lattice the
    net has already removed (2.53).
    """
    h, w = net.shape[0], net.shape[1]
    t_chroma = 4.0 * t_cut if t_chroma is None else t_chroma
    t_chroma = float(np.clip(t_chroma, 32.0, 160.0))
    c = net.shape[2]
    d = (src.astype(np.float32) - net.astype(np.float32))
    if c >= 3:
        luma = 0.299 * d[:, :, 0] + 0.587 * d[:, :, 1] + 0.114 * d[:, :, 2]
    else:
        luma = d[:, :, 0]
    lp_luma = butter_lowpass_channel(luma, t_cut, order)
    win = int(max(3, int(t_chroma)) | 1)
    res = np.empty_like(d)
    for ch in range(c):
        q = d[:, :, ch] - luma
        q = uniform_filter(q, win, mode="nearest")
        q = uniform_filter(q, win, mode="nearest")
        res[:, :, ch] = net[:, :, ch] + lp_luma + q
    return np.clip(res, 0.0, 1.0).astype(np.float32)


def default_sigma(_cfg: P0Config) -> float:
    return 32.0
