from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def gradient(size: int = 512) -> np.ndarray:
    ramp = np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32), (size, 1))
    return np.stack([ramp, ramp, ramp], axis=-1)


def zone_plate(size: int = 512, cycles: float = 64.0) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax)
    r2 = xx * xx + yy * yy
    wave = 0.5 + 0.5 * np.cos(np.pi * cycles * r2).astype(np.float32)
    return np.stack([wave, wave, wave], axis=-1)


def gray_steps(size: int = 512, steps: int = 10) -> np.ndarray:
    idx = (np.arange(size, dtype=np.float32) * steps / size).astype(np.int32)
    vals = (idx / (steps - 1)).astype(np.float32)
    field = np.tile(vals, (size, 1))
    return np.stack([field, field, field], axis=-1)


def text_target(size: int = 512) -> np.ndarray:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(0, size, 24):
        draw.text((8, i), "Ag 0123456789 Proof print %d" % i, fill=(10, 10, 10))
    for i in range(0, size, 96):
        draw.rectangle([size - 90, i, size - 10, i + 64], fill=(200, 30, 30))
        draw.ellipse([10, i, 74, i + 64], fill=(30, 60, 200))
    return np.asarray(img, dtype=np.float32) / 255.0


def flat(size: int = 512, value: float = 0.5) -> np.ndarray:
    field = np.full((size, size), np.float32(value), dtype=np.float32)
    return np.stack([field, field, field], axis=-1)


TARGETS: dict[str, object] = {
    "flat": flat,
    "gradient": gradient,
    "zone_plate": zone_plate,
    "gray_steps": gray_steps,
    "text": text_target,
}
