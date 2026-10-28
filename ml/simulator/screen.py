from __future__ import annotations

import math

import numpy as np
from PIL import Image

from ml.simulator.config import PrintConfig


def apply_tvi(tone: np.ndarray, gain: float) -> np.ndarray:
    tone = np.clip(tone, 0.0, 1.0)
    return np.clip(tone + gain * tone * (1.0 - tone), 0.0, 1.0).astype(np.float32)


def _radius(coverage: np.ndarray, shape: str, ellipse_ratio: float) -> np.ndarray:
    if shape == "square":
        return np.sqrt(np.clip(coverage, 0.0, 1.0)) * 0.5
    if shape == "ellipse":
        return np.sqrt(np.clip(coverage, 0.0, 1.0) / (math.pi * ellipse_ratio))
    return np.sqrt(np.clip(coverage, 0.0, 1.0) / math.pi)


def _warp_field(
    h: int, w: int, amp_px: float, corr_px: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Медленное поле смещения плиты (приводка + коробление бумаги), px печати.

    Сглаженное случайное поле с амплитудой amp_px и корреляционной длиной
    corr_px. Физический смысл: каждая краска ложится со своей погрешностью
    приводки и растяжения бумаги. Дрейф медленный относительно клетки растра,
    поэтому несущая остаётся узкой, но относительная фаза двух красок гуляет —
    межкрасочные биения теряют когерентность (в реальных сканах они на 0.22–0.28
    заметно слабее несущей, у идеальной решётки — на уровне несущей).
    """
    if amp_px <= 0:
        return None
    corr = max(8, int(corr_px))
    gh = max(2, int(np.ceil(h / corr)) + 2)
    gw = max(2, int(np.ceil(w / corr)) + 2)
    fx = rng.normal(0.0, amp_px, (gh, gw)).astype(np.float32)
    fy = rng.normal(0.0, amp_px, (gh, gw)).astype(np.float32)

    def up(a: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(a, mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32
        )

    return up(fx), up(fy)


def halftone_channel(
    tone: np.ndarray, cfg: PrintConfig, angle_deg: float, rng: np.random.Generator
) -> np.ndarray:
    hp, wp = tone.shape
    cell = cfg.cell_px
    theta = math.radians(angle_deg + rng.uniform(-cfg.angle_jitter_deg, cfg.angle_jitter_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Постоянная приводка плиты: у реальных красок начало решётки своё.
    # При нуле (legacy) rng не трогаем — прежние датасеты воспроизводятся бит-в-бит.
    if cfg.registration_px_max > 0:
        off_x = rng.uniform(-cfg.registration_px_max, cfg.registration_px_max)
        off_y = rng.uniform(-cfg.registration_px_max, cfg.registration_px_max)
    else:
        off_x = off_y = 0.0
    warp = _warp_field(hp, wp, cfg.warp_amp_px, cfg.warp_corr_px, rng)
    out = np.empty((hp, wp), dtype=np.float32)
    for y0 in range(0, hp, cfg.chunk_rows):
        y1 = min(y0 + cfg.chunk_rows, hp)
        yy, xx = np.meshgrid(
            np.arange(y0, y1, dtype=np.float32),
            np.arange(wp, dtype=np.float32),
            indexing="ij",
        )
        if warp is not None:
            xx = xx + warp[0][y0:y1] + off_x
            yy = yy + warp[1][y0:y1] + off_y
        elif off_x or off_y:
            xx = xx + off_x
            yy = yy + off_y
        xr = xx * cos_t + yy * sin_t
        yr = -xx * sin_t + yy * cos_t
        px = (np.mod(xr, cell) / cell) - 0.5
        py = (np.mod(yr, cell) / cell) - 0.5
        if cfg.dot_shape == "square":
            dist = np.maximum(np.abs(px), np.abs(py))
        elif cfg.dot_shape == "ellipse":
            dist = np.sqrt(px * px + (py / cfg.ellipse_ratio) ** 2)
        else:
            dist = np.sqrt(px * px + py * py)
        cov = apply_tvi(tone[y0:y1], cfg.tvi_gain)
        rad = _radius(cov, cfg.dot_shape, cfg.ellipse_ratio)
        chunk = (dist < rad).astype(np.float32)
        chunk[cov >= 0.999] = 1.0
        chunk[cov <= 0.001] = 0.0
        out[y0:y1] = chunk
    if cfg.optical_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, sigma=cfg.optical_blur_sigma).astype(np.float32)
    return out
