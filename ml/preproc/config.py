"""Explicit configs for the P0 classical wrapper around frozen run3.

All tunable numbers live here; functions take their parameters explicitly
(no magic constants inside algorithms). Mirrors v3.7 detector defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpectrumConfig:
    """Contrast-enhanced spectrum + peak detection (v3.7 parity)."""

    roi_size: int = 256  # detection ROI, matches v3.7 FFT_SIZE
    dc_radius_px: int = 12  # v3.7 DC_RADIUS
    tophat_win: int = 7  # v3.7 7x7 box top-hat
    sensitivity: float = 2.5  # thr = mean + std*sensitivity (v3.7 default)
    sym_ratio: float = 0.5  # Hermitian pair must exceed thr*sym_ratio
    max_peaks: int = 12
    # ROI used for the single-shot estimate / notch detection (finer freq grid).
    measure_roi: int = 1024


@dataclass(frozen=True)
class NotchConfig:
    """Butterworth notch (v3.7 formula). Radius is given at roi_size scale."""

    radius_roi_px: float = 6.0  # narrower than v3.7 UI default: prefilter
    order: int = 3  # v3.7 order-slider default
    max_peaks: int = 12


@dataclass(frozen=True)
class EdgeNotchConfig:
    """Patch-wise adaptive notch: radius grows with local contrast.

    radius(patch) = base_radius_roi_px * (1 + edge_gain * contrast01).
    Patches overlap-add with Hann window (same pattern as tiled inference).
    """

    patch: int = 256
    stride: int = 128
    base_radius_roi_px: float = 6.0
    radius_ref_roi: int = 1024  # base radius expressed at this ROI scale
    notch_shape: str = "gaussian"  # gaussian (no ringing) | butterworth
    edge_gain: float = 1.0  # radius max = base*(1+gain); capped (ringing!)
    order: int = 3  # butterworth only
    lo_q: float = 0.5  # contrast quantile mapped to 0
    hi_q: float = 0.95  # contrast quantile mapped to 1


@dataclass(frozen=True)
class BlendConfig:
    """Spatially-varying net weight: protect highlights (net crushes them)
    and deep shadows (net grains them). w = base*(1-hi)*(1-sh)."""

    base: float = 1.0
    hi_lo: float = 0.75
    hi_hi: float = 0.95
    hi_strength: float = 1.0
    sh_lo: float = 0.25
    sh_hi: float = 0.05
    sh_strength: float = 0.5
    edge_guard: float = 0.7  # lower net weight on contrast transitions
    edge_patch: int = 128  # finer than the notch grid: precise, still dot-averaging
    edge_stride: int = 64
    edge_lo_q: float = 0.5
    edge_hi_q: float = 0.95


@dataclass(frozen=True)
class LFConfig:
    """Global axial low-frequency banding (paper / scanner bands)."""

    band_lo: float = 0.006
    band_hi: float = 0.05
    axis_tol_deg: float = 8.0  # near-horizontal / near-vertical only
    min_rois: int = 3  # band must repeat in >= this many ROIs to be global
    roi_size: int = 1024
    notch_r: float = 0.004  # notch radius in cycles/px for band subtraction
    notch_order: int = 4


@dataclass(frozen=True)
class ToneConfig:
    bins: int = 64


@dataclass(frozen=True)
class TileConfig:
    tile: int = 512
    overlap: int = 64


@dataclass(frozen=True)
class P0Config:
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    notch: NotchConfig = field(default_factory=NotchConfig)
    edge_notch: EdgeNotchConfig = field(default_factory=EdgeNotchConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    lf: LFConfig = field(default_factory=LFConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    tile: TileConfig = field(default_factory=TileConfig)
    seed: int = 0
