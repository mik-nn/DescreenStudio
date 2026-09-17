from __future__ import annotations

import numpy as np
from PIL import Image

from ml.simulator.color import separate_rgb
from ml.simulator.config import PrintConfig
from ml.simulator.screen import halftone_channel


def _paper_texture(h: int, w: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return np.ones((h, w), dtype=np.float32)
    small = rng.normal(0.0, 1.0, (max(h // 8, 1), max(w // 8, 1))).astype(np.float32)
    tex = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR), dtype=np.float32)
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    return np.clip(1.0 + strength * tex, 0.5, 1.5).astype(np.float32)


_FALLBACK_ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0, "s": 75.0}


def _tinted_paper(
    h: int, w: int, s: int, strength: float, tint: tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    paper = _paper_texture(h, w, strength, rng)
    big = np.asarray(
        Image.fromarray((paper * 128).astype(np.uint8)).resize((w * s, h * s), Image.BILINEAR),
        dtype=np.float32,
    ) / 128.0
    return np.clip(big[..., None] * np.asarray(tint, dtype=np.float32), 0.0, 1.5)


def _separate(gt_big: np.ndarray, cfg: PrintConfig) -> dict[str, np.ndarray]:
    """Ink separation honoring the ink set. Returns {plate: tone}."""
    if cfg.ink_set == "k":
        lum = (
            0.299 * gt_big[..., 0] + 0.587 * gt_big[..., 1] + 0.114 * gt_big[..., 2]
        ).astype(np.float32)
        return {"k": np.clip(1.0 - lum, 0.0, 1.0)}
    if cfg.ink_set == "duotone":
        lum = (
            0.299 * gt_big[..., 0] + 0.587 * gt_big[..., 1] + 0.114 * gt_big[..., 2]
        ).astype(np.float32)
        dark = np.clip(1.0 - lum, 0.0, 1.0)
        return {
            "k": np.clip(dark * 1.1, 0.0, 1.0).astype(np.float32),
            "s": np.clip(dark * 0.55, 0.0, 1.0).astype(np.float32),
        }
    if cfg.ink_set != "cmyk":
        raise ValueError(f"unknown ink_set {cfg.ink_set!r}")
    return separate_rgb(gt_big, cfg.icc_profile, cfg.icc_intent)


def print_masks(gt: np.ndarray, cfg: PrintConfig) -> dict[str, np.ndarray]:
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    h, w, _ = gt.shape
    s = cfg.oversample
    big = np.asarray(
        Image.fromarray((gt * 255.0 + 0.5).astype(np.uint8)).resize(
            (w * s, h * s), Image.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0
    inks = _separate(big, cfg)
    rng = np.random.default_rng(cfg.seed)
    out = {}
    for ch in inks:
        ang = cfg.angles_deg[ch] if ch in cfg.angles_deg else _FALLBACK_ANGLES[ch]
        out[ch] = halftone_channel(inks[ch], cfg, ang, rng)
    return out


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
    inks = _separate(big, cfg)
    rng = np.random.default_rng(cfg.seed)
    masks = {}
    for ch in inks:
        ang = cfg.angles_deg[ch] if ch in cfg.angles_deg else _FALLBACK_ANGLES[ch]
        masks[ch] = halftone_channel(inks[ch], cfg, ang, rng)
    paper_rgb = _tinted_paper(h, w, s, cfg.paper_texture, cfg.paper_tint, rng)
    one = np.ones_like(masks["k"])
    if cfg.ink_set == "k":
        gray = paper_rgb[..., 0] * (one - masks["k"])
        rgb = np.stack([gray, gray, gray], axis=-1)
    elif cfg.ink_set == "duotone":
        v = paper_rgb[..., 0] * (one - masks["k"]) * (one - 0.5 * masks["s"])
        rgb = np.stack([v, v, v], axis=-1)
    else:
        r = paper_rgb[..., 0] * (one - masks["c"]) * (one - masks["k"])
        g = paper_rgb[..., 1] * (one - masks["m"]) * (one - masks["k"])
        b = paper_rgb[..., 2] * (one - masks["y"]) * (one - masks["k"])
        rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)
