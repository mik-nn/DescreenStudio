from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


CHANNELS: tuple[str, ...] = ("c", "m", "y", "k")

STANDARD_ANGLES_DEG: dict[str, float] = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}


@dataclass(frozen=True)
class PrintConfig:
    print_dpi: int = 2400
    scan_dpi: int = 300
    oversample: int = 8
    lpi: float = 150.0
    angles_deg: Mapping[str, float] = field(
        default_factory=lambda: dict(STANDARD_ANGLES_DEG)
    )
    angle_jitter_deg: float = 3.0
    dot_shape: str = "circle"
    ellipse_ratio: float = 0.7
    tvi_gain: float = 0.6
    optical_blur_sigma: float = 0.6
    paper_texture: float = 0.06
    chunk_rows: int = 512
    seed: int = 0

    @property
    def cell_px(self) -> float:
        return self.print_dpi / self.lpi

    @property
    def fundamental_freq(self) -> float:
        return self.lpi / self.print_dpi

    def with_overrides(self, **kwargs: object) -> PrintConfig:
        return replace(self, **kwargs)

    def for_scan_dpi(self, scan_dpi: int) -> PrintConfig:
        return replace(self, scan_dpi=scan_dpi, oversample=self.print_dpi // scan_dpi)


PRESETS: dict[str, dict] = {
    "newspaper": {"lpi": 95.0, "tvi_gain": 0.9, "optical_blur_sigma": 0.8},
    "magazine": {"lpi": 133.0, "tvi_gain": 0.6, "optical_blur_sigma": 0.6},
    "fine": {"lpi": 165.0, "tvi_gain": 0.45, "optical_blur_sigma": 0.5},
}


def preset(name: str, **kwargs: object) -> PrintConfig:
    cfg = PrintConfig(**PRESETS[name])
    if kwargs:
        cfg = cfg.with_overrides(**kwargs)
    return cfg


@dataclass(frozen=True)
class ScanConfig:
    scan_dpi: int = 300
    rotation_min_deg: float = 0.1
    rotation_max_deg: float = 2.0
    psf_sigma_min: float = 0.3
    psf_sigma_max: float = 1.2
    gauss_sigma_min: float = 0.004
    gauss_sigma_max: float = 0.012
    poisson_peak_min: float = 40.0
    poisson_peak_max: float = 120.0
    dust_lines_max: int = 3
    dust_dots_max: int = 30
    seed: int = 0

    def with_overrides(self, **kwargs: object) -> ScanConfig:
        return replace(self, **kwargs)


def expected_fundamentals(cfg: PrintConfig) -> dict[str, list[tuple[float, float]]]:
    import math

    out: dict[str, list[tuple[float, float]]] = {}
    for ch in CHANNELS:
        theta = math.radians(cfg.angles_deg[ch])
        f = cfg.fundamental_freq
        out[ch] = [
            (f * math.cos(theta), f * math.sin(theta)),
            (-f * math.sin(theta), f * math.cos(theta)),
        ]
    return out
