from __future__ import annotations

import numpy as np


def rgb_to_cmyk(rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    k = 1.0 - rgb.max(axis=-1)
    denom = np.clip(1.0 - k, 1e-6, 1.0)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return {
        "c": np.clip(c, 0.0, 1.0).astype(np.float32),
        "m": np.clip(m, 0.0, 1.0).astype(np.float32),
        "y": np.clip(y, 0.0, 1.0).astype(np.float32),
        "k": np.clip(k, 0.0, 1.0).astype(np.float32),
    }
