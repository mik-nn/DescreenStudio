"""V1: параметрическое вычитание растра (Python-эталон для датасета).

Зеркало `src-tauri/src/v1.rs` (продукт, JOURNAL 2.32–2.34). Нужен, чтобы вход
обучения совпадал с тем, что реально видит сеть на инференсе:

    plan() → V1 (нативное разрешение) → Найквист-даунсэмпл → A0 → notch
    (пропускается при сработавшем V1) → LF (гейт по пиковости) → сеть

Демодуляция выполняется в НАТИВНОМ разрешении: на chain-кадре (период растра
2 px) `cos(2π·0.5·n+φ) = (−1)ⁿcosφ` — боковые полосы АМ складываются, и
демодуляция вырождена.

Алгоритм на одну несущую:

    xac   = x − НЧ(x, σ_dc)         σ_dc ≈ 1.5/f  (снять фон: иначе DC попадает
                                     в комплексную огибающую и глушит гейт)
    z     = xac · exp(−i2π f·x)
    z_lp  = НЧ(z, σ)                огибающая (a/2)·exp(iφ)
    s     = 2·Re{ z_lp · exp(+i2π f·x) }
    out   = x − w·s,  w = smoothstep(gate_lo, gate_hi, coh)
    coh   = 2|z_lp| / НЧ(|xac|)     1 у чистой синусоиды, ~0 у широкополосного

Несущие: фундаменты в кольце ±r_tol от опорного r (до `max_fundamentals`) плюс
их диагонали √2·r под ±45° (до `max_carriers` всего). После вычитания известных
несущих пик ищется заново в остатке (`residual_iters`) — часть плит видна
только после удаления сильных.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import P0Config
from .spectrum import _freq_grids, detect_peaks, log_magnitude, luma, tophat_contrast


@dataclass(frozen=True)
class V1Config:
    """Параметры V1. Значения — паритет с `v1::V1Cfg::default()` (Rust)."""

    sigma: float = 16.0
    gate_lo: float = 0.0
    gate_hi: float = 0.5
    max_carriers: int = 4
    max_fundamentals: int = 2
    r_tol: float = 0.02
    min_coh: float = 0.15
    harmonics: int = 1
    residual_iters: int = 3
    residual_min_ratio: float = 2.0


def _lowpass(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    if np.iscomplexobj(a):
        return (gaussian_filter(a.real, sigma, mode="reflect")
                + 1j * gaussian_filter(a.imag, sigma, mode="reflect"))
    return gaussian_filter(a, sigma, mode="reflect")


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def prepare(plane: np.ndarray, f_mag: float, sigma: float = 16.0) -> tuple[np.ndarray, np.ndarray]:
    """(xac, НЧ(|xac|)) — не зависят от несущей, считаются раз на канал."""
    sigma_dc = max(1.5 / max(f_mag, 1e-6), 2.0)
    xac = plane.astype(np.float64) - _lowpass(plane.astype(np.float64), sigma_dc)
    return xac, _lowpass(np.abs(xac), sigma)


def subtract_carrier(
    plane: np.ndarray,
    xac: np.ndarray,
    abs_lp: np.ndarray,
    fx: float,
    fy: float,
    cfg: V1Config,
    ox: int = 0,
    oy: int = 0,
) -> tuple[np.ndarray, float]:
    """Вычитание одной несущей (и гармоник k=1..harmonics). Возвращает (out, coh)."""
    h, w = plane.shape
    out = plane.astype(np.float64).copy()
    xs = (np.arange(w) + ox).astype(np.float64)
    ys = (np.arange(h) + oy).astype(np.float64)
    coh_mean = 0.0
    for k in range(1, cfg.harmonics + 1):
        ex = np.exp(-2j * np.pi * k * fx * xs)
        ey = np.exp(-2j * np.pi * k * fy * ys)
        ph = ey[:, None] * ex[None, :]
        z = xac * ph
        zlp = _lowpass(z, cfg.sigma)
        coh = 2.0 * np.abs(zlp) / (abs_lp + 1e-9)
        wgt = _smoothstep(cfg.gate_lo, cfg.gate_hi, coh)
        synth = np.real(2.0 * zlp * np.conj(ph))
        out -= wgt * synth
        coh_mean = max(coh_mean, float(np.mean(coh)))
    return out, coh_mean


def channel_carriers(
    plane: np.ndarray, r_ref: float, cfg: V1Config, sp
) -> list[tuple[float, float]]:
    """Фундаменты (кольцо ±r_tol) + их диагонали √2·r под ±45°."""
    h, w = plane.shape
    n = min(sp.measure_roi, h, w)
    y0, x0 = h // 2 - n // 2, w // 2 - n // 2
    roi = plane[y0 : y0 + n, x0 : x0 + n].astype(np.float64)
    diff = tophat_contrast(log_magnitude(roi), sp.tophat_win)
    peaks, _ = detect_peaks(diff, sp)
    funds: list[tuple[float, float]] = []
    for p in peaks:
        if abs(p.r - r_ref) > cfg.r_tol:
            continue
        if any(abs(((p.angle_deg - a + 180.0) % 360.0) - 180.0) < 5.0 for _, a in funds):
            continue
        funds.append((p.r, p.angle_deg))
        if len(funds) >= cfg.max_fundamentals:
            break
    s2 = math.sqrt(2.0)
    out: list[tuple[float, float]] = []
    for r, a in funds:
        for rr, aa in ((r, a), (r * s2, a + 45.0), (r * s2, a - 45.0)):
            if rr >= 0.45:
                continue
            dup = any(
                abs(rr - r0) < 0.006
                and abs(((aa - a0 + 180.0) % 360.0) - 180.0) < 5.0
                for r0, a0 in out
            )
            if not dup:
                out.append((rr, aa))
            if len(out) >= cfg.max_carriers:
                break
        if len(out) >= cfg.max_carriers:
            break
    return out


def _residual_peak(
    plane: np.ndarray, r_ref: float, done: list[tuple[float, float]], cfg: V1Config, sp
):
    """Сильнейший пик остатка в диапазоне [0.85r, 1.6r], ещё не вычтенный."""
    h, w = plane.shape
    n = min(sp.measure_roi, h, w)
    y0, x0 = h // 2 - n // 2, w // 2 - n // 2
    roi = plane[y0 : y0 + n, x0 : x0 + n].astype(np.float64)
    diff = tophat_contrast(log_magnitude(roi), sp.tophat_win)
    peaks, thr = detect_peaks(diff, sp)
    best = None
    for p in peaks:
        if p.r < 0.85 * r_ref or p.r > 1.60 * r_ref:
            continue
        if p.strength < thr * cfg.residual_min_ratio:
            continue
        if any(
            abs(p.r - r0) < 0.004
            and abs(((p.angle_deg - a0 + 180.0) % 360.0) - 180.0) < 6.0
            for r0, a0 in done
        ):
            continue
        if best is None or p.strength > best[2]:
            best = (p.r, p.angle_deg, p.strength)
    return best


def apply_v1(
    img: np.ndarray, r_ref: float, cfg: V1Config | None = None, p0: P0Config | None = None
) -> tuple[np.ndarray, dict]:
    """Применить V1 к изображению (нативное разрешение). Возвращает (img, report)."""
    cfg = cfg or V1Config()
    p0 = p0 or P0Config()
    sp = p0.spectrum
    out = img.astype(np.float64).copy()
    report = {"r_ref": r_ref, "carriers": [], "coh": 0.0, "acted": False}
    if r_ref <= 0.0:
        return img, report
    cohs: list[float] = []
    for ch in range(3):
        plane = out[:, :, ch]
        carr = channel_carriers(plane, r_ref, cfg, sp)
        report["carriers"].append([(round(r, 4), round(a, 1)) for r, a in carr])
        if not carr:
            continue
        f_mag = max(carr[0][0], 1e-6)
        xac, abs_lp = prepare(plane, f_mag, cfg.sigma)
        ch_coh = 0.0
        for r, a in carr:
            th = math.radians(a)
            plane, c = subtract_carrier(
                plane, xac, abs_lp, r * math.cos(th), r * math.sin(th), cfg
            )
            ch_coh = max(ch_coh, c)
        done = list(carr)
        for _ in range(cfg.residual_iters):
            best = _residual_peak(plane, r_ref, done, cfg, sp)
            if best is None:
                break
            r, a, _s = best
            xac2, abs_lp2 = prepare(plane, max(r, 1e-6), cfg.sigma)
            th = math.radians(a)
            plane, c = subtract_carrier(
                plane, xac2, abs_lp2, r * math.cos(th), r * math.sin(th), cfg
            )
            ch_coh = max(ch_coh, c)
            done.append((r, a))
        out[:, :, ch] = plane
        cohs.append(ch_coh)
    report["coh"] = float(np.mean(cohs)) if cohs else 0.0
    report["acted"] = bool(cohs) and report["coh"] >= cfg.min_coh
    return np.clip(out, 0.0, 1.0).astype(np.float32), report


def axis_peakiness(gray: np.ndarray, lo: float = 0.006, hi: float = 0.05) -> float:
    """Пиковость осевой НЧ-полосы: max на осях / медиана вне осей.

    Паритет с `prefilter::axis_peakiness` (Rust). Настоящая полоса протяжки
    даёт резкий осевой пик; контент — широкий спектр без выделенного пика.
    """
    n = min(1024, *gray.shape)
    y0, x0 = gray.shape[0] // 2 - n // 2, gray.shape[1] // 2 - n // 2
    roi = gray[y0 : y0 + n, x0 : x0 + n].astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(roi - roi.mean()))
    p = np.abs(f) ** 2
    fx, fy, r = _freq_grids(n)
    band = (r >= lo) & (r < hi)
    on = band & ((np.abs(fx) < 0.004) | (np.abs(fy) < 0.004))
    off = band & ~((np.abs(fx) < 0.004) | (np.abs(fy) < 0.004))
    if not on.any() or not off.any():
        return 0.0
    return float(p[on].max() / (np.median(p[off]) + 1e-30))


def band_energy_native(img: np.ndarray, r_ref: float, half_lo: float = 0.80,
                       half_hi: float = 1.02) -> float:
    """Абсолютная энергия кольца вокруг несущей (паритет с band_energy_abs).

    Абсолютная, а не доля: после удаления растра доля бессмысленна —
    знаменатель сжимается вместе с сигналом.
    """
    g = luma(img)
    n = min(1024, *g.shape)
    y0, x0 = g.shape[0] // 2 - n // 2, g.shape[1] // 2 - n // 2
    roi = g[y0 : y0 + n, x0 : x0 + n].astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(roi - roi.mean()))
    p = np.abs(f) ** 2
    _, _, r = _freq_grids(n)
    return float(p[(r >= half_lo * r_ref) & (r < half_hi * r_ref)].sum())
