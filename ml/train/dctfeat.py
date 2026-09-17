"""Block-DCT input features (run4): explicit full-res frequency info.

CONTRACT (mirrored exactly in src-tauri/src/prefilter.rs `dct_channels`):
- input: luma in [0, 1], HxW; edge-replicate pad to multiples of 8.
- non-overlapping 8x8 blocks, orthonormal DCT-II (scipy dctn norm='ortho').
- bin frequency r_uv = sqrt(u^2+v^2)/8 cycles/px.
- screen band S: 0.08 <= r < 0.28, minus DC (0,0).
  covers print screens r~0.12-0.18: bins (1,0),(0,1),(1,1),(2,0),(0,2).
- hi band Hf: r >= 0.28.
- E_S = ln(1 + sum(c^2 over S)); E_H = ln(1 + sum(c^2 over Hf)).
- upscale: nearest-neighbor replicate over each 8x8 block -> full res.
- output: (H, W, 2) float32 appended AFTER RGB: [R,G,B,E_S,E_H].
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn

BLK = 8
BAND_S_LO = 0.08
BAND_S_HI = 0.28


def _band_masks() -> tuple[np.ndarray, np.ndarray]:
    uu, vv = np.mgrid[0:BLK, 0:BLK]
    r = np.sqrt(uu * uu + vv * vv) / BLK
    s = (r >= BAND_S_LO) & (r < BAND_S_HI)
    s[0, 0] = False
    h = r >= BAND_S_HI
    return s, h


MASK_S, MASK_H = _band_masks()


def dct_channels(luma01: np.ndarray) -> np.ndarray:
    """(H, W, 2) float32 DCT band-energy channels for luma in [0, 1]."""
    h, w = luma01.shape
    ph = (-h) % BLK
    pw = (-w) % BLK
    padded = np.pad(luma01.astype(np.float64), ((0, ph), (0, pw)), mode="edge")
    hh, ww = padded.shape
    blocks = (
        padded.reshape(hh // BLK, BLK, ww // BLK, BLK)
        .transpose(0, 2, 1, 3)
        .reshape(-1, BLK, BLK)
    )
    d = dctn(blocks, type=2, norm="ortho", axes=(1, 2))
    sq = d * d
    es = np.log1p(sq[:, MASK_S].sum(axis=1)).reshape(hh // BLK, ww // BLK)
    eh = np.log1p(sq[:, MASK_H].sum(axis=1)).reshape(hh // BLK, ww // BLK)
    es = np.repeat(np.repeat(es, BLK, axis=0), BLK, axis=1)[:h, :w]
    eh = np.repeat(np.repeat(eh, BLK, axis=0), BLK, axis=1)[:h, :w]
    return np.stack([es, eh], axis=-1).astype(np.float32)
