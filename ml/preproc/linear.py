"""Линейный (денситометрический) домен скана печати.

Скан в sRGB — это гамма-кодированная величина с показателем ~2.2. Физически
оттиск ближе к аддитивному смешению в коэффициенте отражения: бумага умножается
на (1 − покрытие), т.е. в линейном свете. В sRGB-домене это же соотношение
превращается в нелинейное, из-за чего:

* амплитуда растра зависит от тона нефизически (гармоники усиливаются в тенях);
* гармоники растра и межкрасочные биения получают лишний вес относительно
  несущей — детектор A0 видит «ложные» пики;
* амплитуда НЧ-муара читается неверно (муар — мультипликативный эффект).

Поэтому анализ и вычитание растра корректнее делать в линейном коэффициенте
отражения, а на выходе возвращаться в sRGB для отображения.
"""
from __future__ import annotations

import numpy as np

# Порог перехода sRGB (IEC 61966-2-1).
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_GAMMA_THRESHOLD = 0.0031308


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> линейный коэффициент отражения [0,1]."""
    c = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    lo = c / 12.92
    hi = np.power((c + 0.055) / 1.055, 2.4)
    return np.where(c <= _SRGB_LINEAR_THRESHOLD, lo, hi).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Линейный коэффициент отражения [0,1] -> sRGB [0,1]."""
    c = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    lo = c * 12.92
    hi = 1.055 * np.power(c, 1.0 / 2.4) - 0.055
    return np.where(c <= _SRGB_GAMMA_THRESHOLD, lo, hi).astype(np.float32)


def density(rgb: np.ndarray, paper: np.ndarray | None = None) -> np.ndarray:
    """Оптическая плотность D = -log10(R/R_paper) — аддитивная мера краски.

    `paper` — коэффициент отражения бумаги (по умолчанию 1.0, т.е. нормировка
    на идеально белую подложку). Для реального скана полезно передавать
    локальный уровень бумаги: тогда тинт и неравномерность освещения уходят
    в аддитивный сдвиг, а не в модуляцию амплитуды.
    """
    r = np.clip(np.asarray(rgb, dtype=np.float32), 1e-4, 1.0)
    p = 1.0 if paper is None else np.clip(np.asarray(paper, dtype=np.float32), 1e-4, 1.0)
    return (-np.log10(r / p)).astype(np.float32)


def undensity(d: np.ndarray, paper: np.ndarray | None = None) -> np.ndarray:
    """Обратное к `density`: R = R_paper · 10^(-D)."""
    p = 1.0 if paper is None else np.clip(np.asarray(paper, dtype=np.float32), 1e-4, 1.0)
    return np.clip(p * np.power(10.0, -np.asarray(d, dtype=np.float32)), 0.0, 1.0).astype(np.float32)
