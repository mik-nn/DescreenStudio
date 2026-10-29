"""Chain-geometry dataset construction (Nyquist-normalised domain).

Зачем: продукт обрабатывает скан в разрешении Найквиста–Котельникова
(`src-tauri/src/nyquist.rs`: scale = 2·r, период растра ровно 2 px), а сеть
работает тайлом 512 на этом же chain-разрешении. Значит патч обучения обязан
покрывать ту же физическую площадь: 512 chain px = 512/(2·LPI) дюймов.

Геометрия (всё выводится из линиатуры, не из «магических» констант):

    chain_dpi = 2 * LPI                 # период растра 2 px
    inches    = chain_px / chain_dpi
    scan_px   = inches * scan_dpi       # нативный скан, r = LPI/scan_dpi
    print_px  = inches * print_dpi      # «виртуальная печать» перед ресемплингом

Пара «вход/эталон»:

    scan_chain = box_down(scan_native, 2r)      # ровно прод-пре-стейдж
    gt_chain   = box_down(contone_native, 2r)   # контонный оттиск без растра

Контонный оттиск — тот же физический тракт (сепарация, TVI, бумага, PSF,
поворот), но без растровой маски (см. `compose.simulate_print_pair`). Поэтому
интегрирующий даунсэмпл обоих даёт одну низкочастотную основу: различается
только растр (и шум сканера, которого в эталоне нет).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ml.simulator.compose import simulate_print_pair
from ml.simulator.config import PrintConfig, ScanConfig
from ml.simulator.scan import rotate_rgb, sample_scan_params, simulate_scan


@dataclass(frozen=True)
class ChainGeometry:
    """Физическая геометрия одного патча (кадр может быть не квадратным).

    `r_out` — целевая частота растра в рабочем кадре (цикл/px, Найквист 0.5).
    0.5 — предел Котельникова (период 2 px), 0.3 — полоса шире в 1.67 раза.
    """

    lpi: float
    scan_dpi: int
    print_dpi: int
    chain_h: int
    chain_w: int
    r_out: float = 0.5

    @property
    def r(self) -> float:
        """Частота растра в цикл/px нативного скана (Nyquist = 0.5)."""
        return self.lpi / self.scan_dpi

    @property
    def scale(self) -> float:
        """Линейный масштаб нормализации (scale = r / r_out)."""
        return self.r / self.r_out

    @property
    def chain_dpi(self) -> float:
        """Разрешение рабочего кадра: период растра = 1/r_out px."""
        return self.lpi / self.r_out

    @property
    def inches_h(self) -> float:
        return self.chain_h / self.chain_dpi

    @property
    def inches_w(self) -> float:
        return self.chain_w / self.chain_dpi

    @property
    def scan_h(self) -> int:
        return int(round(self.inches_h * self.scan_dpi))

    @property
    def scan_w(self) -> int:
        return int(round(self.inches_w * self.scan_dpi))

    @property
    def print_h(self) -> int:
        return int(round(self.inches_h * self.print_dpi))

    @property
    def print_w(self) -> int:
        return int(round(self.inches_w * self.print_dpi))

    def validate(self) -> None:
        if not 0.05 <= self.r <= 0.24:
            raise ValueError(
                f"r={self.r:.4f} вне рабочего диапазона 0.05..0.24 "
                f"(реальные сканы 0.099..0.146, выше — уже почти Найквист)"
            )
        if not 0.15 <= self.r_out <= 0.5:
            raise ValueError(f"r_out={self.r_out:.3f} вне 0.15..0.5")
        for name, n_scan, n_chain in (("h", self.scan_h, self.chain_h),
                                      ("w", self.scan_w, self.chain_w)):
            if abs(n_scan * self.scale - n_chain) > 1.5:
                raise ValueError(
                    f"геометрия не сходится по {name}: scan*scale={n_scan * self.scale:.1f} "
                    f"vs chain={n_chain}"
                )


def chain_geometry(lpi: float, scan_dpi: int, print_dpi: int, chain_h: int,
                   chain_w: int | None = None, r_out: float = 0.5) -> ChainGeometry:
    g = ChainGeometry(lpi=float(lpi), scan_dpi=int(scan_dpi), print_dpi=int(print_dpi),
                      chain_h=int(chain_h),
                      chain_w=int(chain_w if chain_w is not None else chain_h),
                      r_out=float(r_out))
    g.validate()
    return g


def box_weights(n_in: int, n_out: int) -> np.ndarray:
    """Матрица (n_out × n_in) фракционных box-окон — 1:1 с `nyquist::resize_box`
    (src-tauri/src/nyquist.rs): вес = площадь пересечения окна с пикселем."""
    a = n_in / max(n_out, 1)
    W = np.zeros((n_out, n_in), dtype=np.float32)
    x0s = np.arange(n_out) * a
    x1s = x0s + a
    lo = np.floor(x0s).astype(np.int64)
    hi = np.maximum(np.ceil(x1s).astype(np.int64), lo + 1)
    hi = np.minimum(hi, n_in)
    for i in range(n_out):
        ks = np.arange(lo[i], hi[i])
        wl = np.minimum(ks + 1.0, x1s[i]) - np.maximum(ks.astype(np.float64), x0s[i])
        wl = np.clip(wl, 0.0, None)
        s = wl > 0.0
        W[i, ks[s]] = wl[s] / np.sum(wl[s])
    return W


def box_down_rgb(img: np.ndarray, w2: int, h2: int) -> np.ndarray:
    """Интегрирующий даунсэмпл в точные размеры: out = Wy @ im @ Wx.T (BLAS)."""
    h, w = img.shape[:2]
    if h == h2 and w == w2:
        return img.astype(np.float32)
    Wx = box_weights(w, w2)
    Wy = box_weights(h, h2)
    im = np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32))
    out = Wy @ (im @ Wx.T)
    return np.clip(out.transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


def _psf(rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return rgb
    from scipy.ndimage import gaussian_filter

    s = sigma
    return gaussian_filter(rgb, sigma=(s, s, 0)).astype(np.float32)


def build_chain_pair(
    gt_crop: np.ndarray,
    geom: ChainGeometry,
    pcfg: PrintConfig,
    scfg: ScanConfig,
    sparams: dict,
    ghost: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Собрать пару для chain-домена.

    `gt_crop` — контент в разрешении скана (geom.scan_px), RGB float [0,1].

    Возвращает (scan_native, gt_native, scan_chain, gt_chain):
      scan_native — скан с растром/шумом/дефектами на geom.scan_px
      gt_native   — контонный оттиск (без шума и дефектов) на geom.scan_px
      scan_chain  — box_down(scan_native, 2r)  — вход продукта до пре-фильтра
      gt_chain    — box_down(gt_native, 2r)    — эталон для сети
    """
    s = pcfg.oversample
    printed, contone = simulate_print_pair(gt_crop, pcfg)
    scan_native, _ = simulate_scan(printed, pcfg, scfg, params=sparams, ghost=ghost)
    del printed

    # Эталон идёт через тот же поворот и ту же PSF, что и скан (на разрешении
    # печати), затем тот же интегрирующий даунсэмпл до нативного.
    rot = rotate_rgb(contone, sparams["angle"])
    del contone
    rot = _psf(rot, sparams["psf_sigma"] * s)
    if s > 1:
        hh, ww = rot.shape[0] // s, rot.shape[1] // s
        im = Image.fromarray((np.clip(rot, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        gt_native = np.asarray(im.resize((ww, hh), Image.BOX), dtype=np.float32) / 255.0
        del rot, im
    else:
        gt_native = rot

    scan_chain = box_down_rgb(scan_native, geom.chain_w, geom.chain_h)
    gt_chain = box_down_rgb(gt_native, geom.chain_w, geom.chain_h)
    return scan_native, gt_native, scan_chain, gt_chain


def angle_band_tolerance(lpi: float, scan_dpi: int, chain_px: int) -> float:
    """Частотное разрешение chain-кадра в цикл/px (бин FFT)."""
    return 1.0 / chain_px


def expected_chain_r(lpi: float, scan_dpi: int, r_out: float = 0.5) -> float:
    """r растра в chain-домене после нормализации scale = r/r_out."""
    return lpi / scan_dpi / (lpi / scan_dpi / r_out)


def lpi_for_r(r: float, scan_dpi: int) -> float:
    return r * scan_dpi


def geometry_table(scan_dpi: int, chain_h: int, chain_w: int, lpis: list[float],
                   print_dpi: int, r_out: float = 0.5) -> list[dict]:
    rows = []
    for lpi in lpis:
        g = chain_geometry(lpi, scan_dpi, print_dpi, chain_h, chain_w, r_out=r_out)
        rows.append({
            "lpi": round(lpi, 1),
            "r": round(g.r, 4),
            "r_out": round(g.r_out, 4),
            "scale": round(g.scale, 4),
            "inches": [round(g.inches_h, 3), round(g.inches_w, 3)],
            "scan_px": [g.scan_h, g.scan_w],
            "print_px": [g.print_h, g.print_w],
            "chain_px": [g.chain_h, g.chain_w],
            "fft_bin": round(1.0 / g.chain_h, 5),
        })
    return rows
