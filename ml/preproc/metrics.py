"""No-GT quality metrics for real scans.

Legacy (v1, поля сохранены для сравнимости):
- residual_peak: strongest symmetric peak above the raster band (raster leftovers)
- lf_ratio: axial LF-band energy out/src (banding amplified or kept?)
- sharp_ratio: Laplacian-variance out/src (detail eaten or kept?)
- dc_shift: per-channel mean(out-src) (tone darkening/brightening)

v2 (2026-09-25, анализ хвоста):
- resid_top4_sum: суммарная энергия 4 сильнейших пиков (один пик пропускает
  мультиэкран: Magazine-2 имеет 169+87 LPI, v1 видел только доминанту)
- resid_lf_moire: пик в кольце НЧ-муара r 0.008-0.05 (v1 обнулял всё r<0.05)
- resid_max_quad: худший остаточный пик по 4 квадрантам (центр-ROI слеп
  к пустым полям: Calendar2/Offset4 var~0.001 — пустые патчи пропускаются)
- edge_hf_ratio / flat_hf_ratio: ВЧ-энергия out/src по блокам с краями / плоским
  (возврат метрики E1 «край/плоскость»; ловит рингинг и зерно сети у границ)
- de_mean / de_p90: CIE76 в Lab (PIL) — цветовой сдвиг, которого DC не видит
  (отбеливание бумаги, tint)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import SpectrumConfig
from .spectrum import _freq_grids, log_magnitude, luma, tophat_contrast


@dataclass(frozen=True)
class NoGTReport:
    residual_peak_xmean: float
    residual_peak_r: float
    lf_ratio: float
    sharp_ratio: float
    dc_shift: tuple[float, float, float]
    # v2 (NaN/0.0 = не посчитано для старых отчётов)
    resid_top4_sum: float = 0.0
    resid_lf_moire: float = 0.0
    resid_max_quad: float = 0.0
    edge_hf_ratio: float = 0.0
    flat_hf_ratio: float = 0.0
    de_mean: float = 0.0
    de_p90: float = 0.0


def _lapvar(g: np.ndarray) -> float:
    lap = g[:-2, :-2] + g[:-2, 2:] + g[2:, :-2] + g[2:, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def center_roi(a: np.ndarray, s: int = 1024) -> np.ndarray:
    h, w = a.shape[:2]
    s = min(s, h, w)
    return a[h // 2 - s // 2 : h // 2 + s // 2, w // 2 - s // 2 : w // 2 + s // 2]


def residual_peak(gray: np.ndarray, r_min: float = 0.05) -> tuple[float, float]:
    g = center_roi(gray)
    n = g.shape[0]
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(f)
    mean = mag.mean()
    _, _, r = _freq_grids(n)
    m = mag.copy()
    m[r < r_min] = 0.0
    j, i = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(m[j, i] / mean), float(r[j, i])


def residual_peaks_multi(gray: np.ndarray, k: int = 4, r_min: float = 0.05,
                         excl: float = 0.02) -> tuple[float, list[tuple[float, float]]]:
    """Top-k пиков с дисковым исключением (та же нормировка xmean, что в v1).

    Возвращает (сумма xmean, [(xmean, r), ...]). Симметричный двойник пика
    давится автоматически: спектр симметричен, диск накрывает обе стороны
    только если пик на оси симметрии — поэтому исключаем и зеркальную точку.
    """
    g = center_roi(gray)
    n = g.shape[0]
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(f)
    mean = mag.mean()
    fx, fy, r = _freq_grids(n)
    m = mag.copy()
    m[r < r_min] = 0.0
    peaks: list[tuple[float, float]] = []
    for _ in range(k):
        j, i = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[j, i] <= 0:
            break
        peaks.append((float(m[j, i] / mean), float(r[j, i])))
        d2 = (fx - fx[j, i]) ** 2 + (fy - fy[j, i]) ** 2
        d2m = (fx + fx[j, i]) ** 2 + (fy + fy[j, i]) ** 2  # зеркальный двойник
        m[(d2 < excl ** 2) | (d2m < excl ** 2)] = 0.0
    return float(sum(p[0] for p in peaks)), peaks


def residual_lf_moire(gray: np.ndarray, lo: float = 0.008, hi: float = 0.05) -> float:
    """Пик в кольце НЧ-муара (вторичный муар и биения; v1 эту зону обнулял)."""
    g = center_roi(gray)
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    mag = np.abs(f)
    _, _, r = _freq_grids(g.shape[0])
    m = mag.copy()
    m[(r < lo) | (r >= hi)] = 0.0
    return float(m.max() / (mag.mean() + 1e-12))


def residual_quad_max(gray: np.ndarray, s: int = 512, var_min: float = 1e-6) -> float:
    """Худший остаточный пик по центрам 4 квадрантов (пустые пропускаются)."""
    h, w = gray.shape[:2]
    hy, hx = h // 2, w // 2
    quads = [gray[:hy, :hx], gray[:hy, hx:], gray[hy:, :hx], gray[hy:, hx:]]
    vals = []
    for q in quads:
        qh, qw = q.shape[:2]
        s0 = min(s, qh, qw)
        if s0 < 64:
            continue
        roi = q[qh // 2 - s0 // 2: qh // 2 + s0 // 2, qw // 2 - s0 // 2: qw // 2 + s0 // 2]
        if float(roi.var()) < var_min:
            continue  # пустое поле — пики там мусорные (урок Calendar2/Offset4)
        f = np.fft.fftshift(np.fft.fft2(roi - roi.mean()))
        mag = np.abs(f)
        # span среза = 2*floor(s0/2): при нечётном s0 roi на 1px меньше
        nq = roi.shape[0]
        _, _, r = _freq_grids(nq)
        m = mag.copy()
        m[r < 0.05] = 0.0
        vals.append(float(m.max() / (mag.mean() + 1e-12)))
    return float(max(vals)) if vals else 0.0


def zonal_hf_ratios(src: np.ndarray, out: np.ndarray, block: int = 128,
                    hf_min: float = 0.30) -> tuple[float, float]:
    """ВЧ-энергия out/src по блокам: edge (верхний квартиль градиента src)
    и flat (нижний квартиль). Возвращает (edge_hf, flat_hf)."""
    g = center_roi(src)
    o = center_roi(out)
    n = g.shape[0]
    gy, gx = np.gradient(g)
    gm = np.hypot(gy, gx)
    nb = n // block
    stats = []
    for by in range(nb):
        for bx in range(nb):
            sb = g[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            ob = o[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            gb = gm[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            if float(sb.var()) < 1e-6:
                continue
            _, _, r = _freq_grids(block)
            sel = r >= hf_min
            es = float((np.abs(np.fft.fft2(sb - sb.mean()))[sel] ** 2).sum())
            eo = float((np.abs(np.fft.fft2(ob - ob.mean()))[sel] ** 2).sum())
            stats.append((float(gb.mean()), eo / (es + 1e-12)))
    if not stats:
        return 0.0, 0.0
    grads = np.array([s[0] for s in stats])
    q_lo, q_hi = np.quantile(grads, [0.25, 0.75])
    flat = [v for gr, v in stats if gr <= q_lo]
    edge = [v for gr, v in stats if gr >= q_hi]
    return (float(np.mean(edge)) if edge else 0.0, float(np.mean(flat)) if flat else 0.0)


def delta_e_stats(src_rgb: np.ndarray, out_rgb: np.ndarray, s: int = 256) -> tuple[float, float]:
    """CIE76 mean/p90 в Lab (PIL). Входы float32 [0,1]."""
    def to_lab(a: np.ndarray) -> np.ndarray:
        # Pillow хранит a/b как SIGNED байты (серый = 0/0, отриц. значения
        # завёрнуты: uint8 -> int8), L — 0..255 от L* 0..100. Проверено:
        # (128,128,128)->(137,0,0), red->(138,81,70), blue->(75,68,144).
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("RGB")
        im = im.resize((s, s), Image.BILINEAR).convert("LAB")
        lab = np.asarray(im, dtype=np.int16)
        out = np.empty_like(lab, dtype=np.float32)
        out[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
        ab = lab[:, :, 1:]
        ab = np.where(ab > 127, ab - 256, ab).astype(np.float32)
        out[:, :, 1:] = ab
        return out
    d = np.linalg.norm(to_lab(src_rgb) - to_lab(out_rgb), axis=-1)
    return float(d.mean()), float(np.quantile(d, 0.90))


def lf_energy_ratio(
    src: np.ndarray,
    out: np.ndarray,
    lo: float = 0.006,
    hi: float = 0.05,
    axis_tol: float = 8.0,
) -> float:
    """Absolute LF-band energy out/src (NOT normalized by total energy, so
    that HF removal elsewhere does not inflate the ratio)."""

    def energy(g: np.ndarray) -> float:
        g = center_roi(g)
        n = g.shape[0]
        f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
        fx, fy, r = _freq_grids(n)
        ang = np.degrees(np.arctan2(fy, fx))
        d_axis = np.minimum(
            np.abs((ang + 90.0) % 180.0 - 90.0), np.abs((ang + 180.0) % 180.0 - 90.0)
        )
        sel = (r >= lo) & (r < hi) & (d_axis <= axis_tol)
        p = np.abs(f) ** 2
        return float(p[sel].sum())

    return energy(out) / (energy(src) + 1e-12)


def evaluate(src_rgb: np.ndarray, out_rgb: np.ndarray, cfg: SpectrumConfig) -> NoGTReport:
    _ = cfg
    gs, go = luma(src_rgb), luma(out_rgb)
    xm, rr = residual_peak(go)
    top4, _ = residual_peaks_multi(go)
    edge_hf, flat_hf = zonal_hf_ratios(gs, go)
    de_m, de_90 = delta_e_stats(src_rgb, out_rgb)
    return NoGTReport(
        residual_peak_xmean=xm,
        residual_peak_r=rr,
        lf_ratio=lf_energy_ratio(gs, go),
        sharp_ratio=_lapvar(center_roi(go)) / (_lapvar(center_roi(gs)) + 1e-12),
        dc_shift=tuple(float((out_rgb[:, :, c] - src_rgb[:, :, c]).mean()) for c in range(3)),
        resid_top4_sum=top4,
        resid_lf_moire=residual_lf_moire(go),
        resid_max_quad=residual_quad_max(go),
        edge_hf_ratio=edge_hf,
        flat_hf_ratio=flat_hf,
        de_mean=de_m,
        de_p90=de_90,
    )
