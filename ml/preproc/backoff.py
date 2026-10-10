"""Backoff коррекции сети в плоских зонах (кейс 1821146u, JOURNAL 2026-10-10).

Проблема: на чистом префильтре сеть добавляет периодику (top4 113→182 @ r=0.211).
Абляция zero-cond (панель 11: дельты ±3) показала, что cond-FiLM не причастна —
вредит сама коррекция net−pre, поэтому backoff применяется к поправке, а не к
вектору cond.

Механика: изображение режется на блоки; в плоских блоках (низкий градиент
префильтра — там и живёт растр) сравнивается периодическая энергия
(пик/среднее спектра при r ≥ r_min) до и после сети. Если сеть увеличила
периодику больше чем на eps — вес коррекции блока прижимается к wmin.
В текстурных блоках (текст, контуры) и в блоках, где сеть периодику снизила,
вес = 1 — резкость и выигрыши не трогаем.
Карта весов сглаживается bilinear перед применением (без квадратной мозаики).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ml.preproc.spectrum import luma


def _peak_ratio(g: np.ndarray, r_min: float) -> float:
    n = g.shape[0]
    f = np.fft.fft2(g - float(g.mean()))
    mag = np.abs(f)
    yy, xx = np.mgrid[0:n, 0:n]
    fx = (xx - n // 2) / n
    fy = (yy - n // 2) / n
    r = np.hypot(fx, fy)
    m = np.where(r >= r_min, mag, 0.0)
    return float(m.max() / (float(mag.mean()) + 1e-12))


def apply_global(pre: np.ndarray, net_out: np.ndarray, w: float) -> np.ndarray:
    """Скалярный backoff: коррекция net−pre умножается на w по всему кадру."""
    if w >= 1.0:
        return net_out
    return (pre + float(w) * (net_out - pre)).astype(np.float32)


def flat_periodicity_backoff(
    pre: np.ndarray,
    net_out: np.ndarray,
    *,
    block: int = 128,
    r_min: float = 0.05,
    wmin: float = 0.25,
    eps: float = 0.10,
    flat_q: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Прижимает коррекцию net_out→pre в плоских блоках с выросшей периодикой.

    Возвращает (net_adj, info), где info — сводка по блокам (для отчёта).
    """
    h, w = pre.shape[:2]
    ny, nx = h // block, w // block
    if ny == 0 or nx == 0:
        return net_out, {"applied": False}

    gy, gx = np.gradient(luma(pre))
    gm = np.hypot(gy, gx)

    grad_means = np.empty((ny, nx), dtype=np.float64)
    e_pre = np.empty((ny, nx), dtype=np.float64)
    e_net = np.empty((ny, nx), dtype=np.float64)
    for by in range(ny):
        for bx in range(nx):
            sb = slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block)
            grad_means[by, bx] = float(gm[sb].mean())
            e_pre[by, bx] = _peak_ratio(luma(pre[sb]), r_min)
            e_net[by, bx] = _peak_ratio(luma(net_out[sb]), r_min)

    thr = float(np.quantile(grad_means, flat_q))
    flat = grad_means <= thr
    added = e_net > e_pre * (1.0 + eps)
    hit = flat & added

    weights = np.ones((ny, nx), dtype=np.float32)
    weights[hit] = np.float32(wmin)

    # bilinear-сглаживание карты весов до размера покрытой области
    wm = Image.fromarray((weights * 255).astype(np.uint8), mode="L")
    wm = wm.resize((nx * block, ny * block), Image.BILINEAR)
    wmap = np.asarray(wm, dtype=np.float32) / 255.0

    out = net_out.copy()
    region = out[: ny * block, : nx * block]
    base = pre[: ny * block, : nx * block]
    region[:] = base + wmap[..., None] * (region - base)

    info = {
        "applied": True,
        "blocks": int(ny * nx),
        "flat": int(flat.sum()),
        "backoff": int(hit.sum()),
        "wmin": float(wmin),
        "eps": float(eps),
        "grad_thr": round(thr, 6),
    }
    return out, info
