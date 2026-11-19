"""Smoke-tests for the Python mirror of `prefilter::linear_tone` (2.52).

The Rust stage matches per channel: mean of the scan + LF-contrast scale
(box window 33..129 px, clamped to [0.9, 1.1]). The Python copy is used by
`ml/preproc/run_p0.py`, so it must behave the same way: the raster in the
scan's full-frame std is NOT tone and must not be scaled back into the output.
"""

from __future__ import annotations

import numpy as np

from ml.preproc.paperbg import linear_tone

H = W = 256


def _lf(shape: tuple[int, int, int] = (H, W, 3)) -> np.ndarray:
    h, w, c = shape
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.55 + 0.10 * xx / w + 0.06 * yy / h
    return np.stack([base + 0.02 * ch for ch in range(c)], axis=-1).astype(np.float32)


def _checker(period: int = 3, amp: float = 0.03) -> np.ndarray:
    """Raster-like high-frequency carrier, zero mean."""
    yy, xx = np.mgrid[0:H, 0:W]
    sign = ((yy // period) + (xx // period)) % 2 * 2.0 - 1.0
    return (amp * sign)[..., None].astype(np.float32)


def test_mean_comes_from_scan() -> None:
    src = np.clip(_lf() - 0.02 + _checker(), 0.0, 1.0)
    out = np.clip(_lf() + 0.04, 0.0, 1.0)  # net drifts bright
    res = linear_tone(out, src)
    for ch in range(3):
        assert abs(res[:, :, ch].mean() - src[:, :, ch].mean()) < 1e-4


def test_family_raster_in_scan_does_not_inflate_output() -> None:
    """Scan = LF + raster; net output = same LF without raster.

    Old rule (full-frame std) scaled everything by ~1.04 to return the raster's
    variance. New rule measures contrast on the LF part only and must stay ~1.
    """
    src = np.clip(_lf() + _checker(amp=0.06), 0.0, 1.0)
    out = _lf()
    res = linear_tone(out, src)
    for ch in range(3):
        hf_src = (src[:, :, ch] - _box(src[:, :, ch])).std()
        hf_out = (out[:, :, ch] - _box(out[:, :, ch])).std()
        hf_res = (res[:, :, ch] - _box(res[:, :, ch])).std()
        # Output's own texture is preserved (~1.0), not the scan's raster.
        assert abs(hf_res / hf_out - 1.0) < 0.05, hf_res / hf_out
        # Sanity: the scan really did carry the raster.
        assert hf_src > 2.0 * hf_out


def test_lf_contrast_is_matched() -> None:
    """Low-frequency tone of the net output is pulled to the scan's."""
    src = _lf()
    # Net flattens the tone by 7% (inside the [0.9, 1.1] clamp, so it is fully
    # recovered; a 15% loss would legitimately hit the clamp).
    out = (src - src.mean()) * 0.93 + src.mean()
    res = linear_tone(out, src)
    for ch in range(3):
        lo_out = _box(out[:, :, ch])
        lo_res = _box(res[:, :, ch])
        lo_src = _box(src[:, :, ch])
        assert abs(lo_res.std() / lo_src.std() - 1.0) < 0.03


def _box(plane: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    return uniform_filter(plane.astype(np.float32), 33, mode="nearest")
