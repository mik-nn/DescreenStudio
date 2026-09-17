from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


CHANNELS: tuple[str, ...] = ("c", "m", "y", "k")

STANDARD_ANGLES_DEG: dict[str, float] = {
    "c": 15.0,
    "m": 75.0,
    "y": 0.0,
    "k": 45.0,
    "s": 75.0,  # spot gray plate for duotone jobs
}


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
    paper_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # aged-paper RGB multiplier
    ink_set: str = "cmyk"  # cmyk | k (mono) | duotone (k + spot gray)
    icc_profile: str | None = None  # ECI profile filename under ml/simulator/icc/
    icc_intent: int = 0  # ImageCms Intent.PERCEPTUAL
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


PRESET_ICC: dict[str, str] = {
    # Context-appropriate ECI offset profiles (see ml/simulator/icc/manifest.json).
    # Uncoated extras (PSO_Uncoated_ISO12647, ISOuncoatedyellowish) available
    # for explicit icc_profile overrides.
    "newspaper": "PSO_SNP_Paper_eci.icc",
    "magazine": "ISOcoated_v2_eci.icc",
    "fine": "ISOcoated_v2_eci.icc",
}


PRESETS: dict[str, dict] = {
    "newspaper": {
        "lpi": 95.0,
        "tvi_gain": 0.9,
        "optical_blur_sigma": 0.8,
        "icc_profile": PRESET_ICC["newspaper"],
    },
    "magazine": {
        "lpi": 133.0,
        "tvi_gain": 0.6,
        "optical_blur_sigma": 0.6,
        "icc_profile": PRESET_ICC["magazine"],
    },
    "fine": {
        "lpi": 165.0,
        "tvi_gain": 0.45,
        "optical_blur_sigma": 0.5,
        "icc_profile": PRESET_ICC["fine"],
    },
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
    band_lpi_min: float = 6.0  # paper/scanner axial banding (measured r~0.01)
    band_lpi_max: float = 10.0
    band_amp_min: float = 0.0
    band_amp_max: float = 0.02
    show_alpha_min: float = 0.005  # duplex show-through strength
    show_alpha_max: float = 0.04
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
