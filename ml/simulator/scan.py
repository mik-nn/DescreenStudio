from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ml.simulator.config import PrintConfig, ScanConfig


def sample_scan_params(cfg: ScanConfig, rng: np.random.Generator) -> dict:
    return {
        "angle": float(rng.uniform(cfg.rotation_min_deg, cfg.rotation_max_deg)),
        "psf_sigma": float(rng.uniform(cfg.psf_sigma_min, cfg.psf_sigma_max)),
        "gauss_sigma": float(rng.uniform(cfg.gauss_sigma_min, cfg.gauss_sigma_max)),
        "poisson_peak": float(rng.uniform(cfg.poisson_peak_min, cfg.poisson_peak_max)),
        "dust_lines": int(rng.integers(0, cfg.dust_lines_max + 1)),
        "dust_dots": int(rng.integers(0, cfg.dust_dots_max + 1)),
        "band_lpi": float(rng.uniform(cfg.band_lpi_min, cfg.band_lpi_max)),
        "band_amp": float(rng.uniform(cfg.band_amp_min, cfg.band_amp_max)),
        "band_axis": int(rng.integers(0, 2)),  # 0: vary along x, 1: vary along y
        "band_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "show_alpha": 0.0,  # set by caller when a ghost (back side) is provided
    }


def _apply_banding(scan: np.ndarray, scan_dpi: int, params: dict) -> np.ndarray:
    amp = params["band_amp"]
    if amp <= 0:
        return scan
    h, w, _ = scan.shape
    period = scan_dpi / max(params["band_lpi"], 1e-6)
    coord = np.arange(w, dtype=np.float32) if params["band_axis"] == 0 else np.arange(h, dtype=np.float32)
    wave = 1.0 + amp * np.sin(2.0 * np.pi * coord / period + params["band_phase"])
    mult = wave[None, :, None] if params["band_axis"] == 0 else wave[:, None, None]
    return (scan * mult).astype(np.float32)


def rotate_rgb(img: np.ndarray, angle_deg: float) -> np.ndarray:
    u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8).rotate(angle_deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return np.asarray(im, dtype=np.float32) / 255.0


def _add_dust(img: np.ndarray, n_lines: int, n_dots: int, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = img.shape
    u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8)
    draw = ImageDraw.Draw(im)
    for _ in range(n_lines):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(min(h, w) * 0.1, min(h, w) * 0.8)
        x1, y1 = x0 + length * np.cos(ang), y0 + length * np.sin(ang)
        val = 255 if rng.random() < 0.5 else 0
        draw.line([(x0, y0), (x1, y1)], fill=(val, val, val), width=1)
    for _ in range(n_dots):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = int(rng.integers(1, 3))
        val = 255 if rng.random() < 0.5 else 0
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(val, val, val))
    return np.asarray(im, dtype=np.float32) / 255.0


def simulate_scan(
    print_img: np.ndarray,
    pcfg: PrintConfig,
    scfg: ScanConfig,
    params: dict | None = None,
    ghost: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Scan cascade. `ghost` (same HxW as the downsampled scan) is the duplex
    back side, blended at params['show_alpha'] before noise."""
    rng = np.random.default_rng(scfg.seed)
    if params is None:
        params = sample_scan_params(scfg, rng)
    else:
        params = dict(params)
    hp, wp, _ = print_img.shape
    s = pcfg.oversample
    rotated = rotate_rgb(print_img, params["angle"])
    if params["psf_sigma"] > 0:
        from scipy.ndimage import gaussian_filter

        rotated = gaussian_filter(rotated, sigma=(params["psf_sigma"] * s, params["psf_sigma"] * s, 0)).astype(np.float32)
    im = Image.fromarray((np.clip(rotated, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    scan = np.asarray(im.resize((wp // s, hp // s), Image.BOX), dtype=np.float32) / 255.0
    alpha = float(params.get("show_alpha", 0.0))
    if ghost is not None and alpha > 0:
        g = np.clip(np.asarray(ghost, dtype=np.float32), 0.0, 1.0)
        scan = np.clip(scan * (1.0 - alpha) + g * alpha, 0.0, 1.0).astype(np.float32)
    scan = _apply_banding(scan, pcfg.scan_dpi, params)
    scan = scan + rng.normal(0.0, params["gauss_sigma"], scan.shape).astype(np.float32)
    peak = params["poisson_peak"]
    scan = rng.poisson(np.clip(scan, 0.0, 1.0) * peak).astype(np.float32) / peak
    scan = np.clip(scan, 0.0, 1.0).astype(np.float32)
    if params["dust_lines"] or params["dust_dots"]:
        scan = _add_dust(scan, params["dust_lines"], params["dust_dots"], rng)
    return scan, params
