from __future__ import annotations

import numpy as np
from PIL import Image

from ml.simulator.color import rgb_to_cmyk
from ml.simulator.config import CHANNELS, PrintConfig
from ml.simulator.screen import halftone_channel


def _paper_texture(h: int, w: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return np.ones((h, w), dtype=np.float32)
    small = rng.normal(0.0, 1.0, (max(h // 8, 1), max(w // 8, 1))).astype(np.float32)
    tex = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR), dtype=np.float32)
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    return np.clip(1.0 + strength * tex, 0.5, 1.5).astype(np.float32)


def simulate_print(gt: np.ndarray, cfg: PrintConfig) -> np.ndarray:
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    h, w, _ = gt.shape
    s = cfg.oversample
    big = np.asarray(
        Image.fromarray((gt * 255.0 + 0.5).astype(np.uint8)).resize(
            (w * s, h * s), Image.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0
    inks = rgb_to_cmyk(big)
    rng = np.random.default_rng(cfg.seed)
    masks = {}
    for ch in CHANNELS:
        masks[ch] = halftone_channel(inks[ch], cfg, cfg.angles_deg[ch], rng)
    paper = _paper_texture(h, w, cfg.paper_texture, rng)
    paper_big = np.asarray(
        Image.fromarray((paper * 128).astype(np.uint8)).resize((w * s, h * s), Image.BILINEAR),
        dtype=np.float32,
    ) / 128.0
    one = np.ones_like(masks["k"])
    r = paper_big * (one - masks["c"]) * (one - masks["k"])
    g = paper_big * (one - masks["m"]) * (one - masks["k"])
    b = paper_big * (one - masks["y"]) * (one - masks["k"])
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return rgb.astype(np.float32)
