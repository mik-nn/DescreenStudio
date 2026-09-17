"""Spatially-varying AI blend: protect highlights (the net crushes their
contrast) and deep shadows (the net grains them) by lowering the net weight
there. w = base * (1-hi) * (1-sh), smoothsteps on source luma."""

from __future__ import annotations

import numpy as np

from .config import BlendConfig
from .spectrum import luma


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend_map(
    img: np.ndarray, cfg: BlendConfig, edge01_map: np.ndarray | None = None
) -> np.ndarray:
    """Per-pixel net weight in [0, base], from source luma (+ optional edge map).

    NOTE: the edge map must be patch-averaged (dot-averaging); a pixel-precise
    gradient map is poisoned by the raster dots themselves (measured worse).
    """
    from .edgenotch import edge01 as _edge01

    lum = luma(img)
    hi = cfg.hi_strength * smoothstep(cfg.hi_lo, cfg.hi_hi, lum)
    sh = cfg.sh_strength * (1.0 - smoothstep(cfg.sh_hi, cfg.sh_lo, lum))
    w = cfg.base * (1.0 - hi) * (1.0 - sh)
    if edge01_map is None and cfg.edge_guard > 0:
        edge01_map = _edge01(
            lum, cfg.edge_patch, cfg.edge_stride, cfg.edge_lo_q, cfg.edge_hi_q
        )
    if edge01_map is not None:
        w = w * (1.0 - cfg.edge_guard * edge01_map)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def adaptive_blend(
    pre: np.ndarray, net: np.ndarray, src: np.ndarray, cfg: BlendConfig
) -> np.ndarray:
    w = blend_map(src, cfg)[..., None]
    return np.clip(pre * (1.0 - w) + net * w, 0.0, 1.0).astype(np.float32)
