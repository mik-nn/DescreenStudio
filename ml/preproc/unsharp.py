"""Адаптивный unsharp-mask (рычаг 3, 2026-09-21).

Факт из сетки postprocess_grid.py: unsharp amount∈[0.3,0.8], sigma=1.0
монотонно улучшает И resid, И sharp на всех трёх контрольных сканах.
Но фиксированный amount не годится: baseline sharp сильно разный
(0.40 Magazine … 1.68 Offset1). Решение — feedback-правило без GT:

  1. Измеряем sharp_ratio нейросетевого результата.
  2. Если sharp >= target (0.95) — unsharp не нужен (amount=0).
  3. Иначе итеративно: amount *= 1.4, пока sharp не войдёт в [0.95, 1.05],
     потолок amount = 1.5 (защита от овершарпа/усиления растра).

Цель — держать резкость в приёмочном коридоре 0.9–1.15 на любом скане.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ml.preproc.metrics import evaluate
from ml.preproc.spectrum import SpectrumConfig

SHARP_TARGET = 1.0
SHARP_TOL = 0.05
AMOUNT_MAX = 1.5
AMOUNT_STEP = 1.4
SIGMA = 1.0


def unsharp(img: np.ndarray, amount: float, sigma: float = SIGMA) -> np.ndarray:
    if amount <= 0:
        return img
    blur = np.stack([gaussian_filter(img[..., c], sigma) for c in range(3)], axis=-1)
    return np.clip(img + amount * (img - blur), 0.0, 1.0).astype(np.float32)


def adaptive_unsharp(img: np.ndarray, src: np.ndarray, cfg: SpectrumConfig,
                     target: float = SHARP_TARGET) -> tuple[np.ndarray, float, float]:
    """Возвращает (out, amount, sharp_ratio). Максимум 6 итераций оценки."""
    rep0 = evaluate(src, img, cfg)
    if rep0.sharp_ratio >= target - SHARP_TOL:
        return img, 0.0, rep0.sharp_ratio
    amount = 0.5
    best, best_d = img, abs(rep0.sharp_ratio - target)
    for _ in range(6):
        out = unsharp(img, amount)
        rep = evaluate(src, out, cfg)
        d = abs(rep.sharp_ratio - target)
        if d < best_d:
            best, best_d = out, d
        if d <= SHARP_TOL:
            return out, amount, rep.sharp_ratio
        if rep.sharp_ratio < target:
            amount = min(AMOUNT_MAX, amount * AMOUNT_STEP)
        else:
            amount /= AMOUNT_STEP
        if amount >= AMOUNT_MAX and rep.sharp_ratio < target:
            break
    return best, amount, evaluate(src, best, cfg).sharp_ratio
