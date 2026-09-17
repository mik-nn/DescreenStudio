from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageCms

ICC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icc")


def rgb_to_cmyk(rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    k = 1.0 - rgb.max(axis=-1)
    denom = np.clip(1.0 - k, 1e-6, 1.0)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return {
        "c": np.clip(c, 0.0, 1.0).astype(np.float32),
        "m": np.clip(m, 0.0, 1.0).astype(np.float32),
        "y": np.clip(y, 0.0, 1.0).astype(np.float32),
        "k": np.clip(k, 0.0, 1.0).astype(np.float32),
    }


def rgb_to_cmyk_icc(
    rgb: np.ndarray, profile_name: str, intent: int = ImageCms.Intent.PERCEPTUAL
) -> dict[str, np.ndarray]:
    """Context-appropriate separation via ECI offset ICC profile.

    Source is assumed sRGB. Raises FileNotFoundError naming the fetch script
    when the profile binary is absent (gitignored, see manifest.json).
    """
    path = os.path.join(ICC_DIR, profile_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ICC profile missing: {path} — run scripts/fetch_icc_profiles.ps1"
        )
    rgb8 = (np.clip(rgb.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(rgb8, mode="RGB")
    src_prof = ImageCms.createProfile("sRGB")
    dst_prof = ImageCms.getOpenProfile(path)
    transform = ImageCms.buildTransform(
        src_prof, dst_prof, "RGB", "CMYK", renderingIntent=intent
    )
    out = transform.apply(pil)
    arr = np.asarray(out, dtype=np.float32) / 255.0
    return {
        "c": arr[..., 0].astype(np.float32),
        "m": arr[..., 1].astype(np.float32),
        "y": arr[..., 2].astype(np.float32),
        "k": arr[..., 3].astype(np.float32),
    }


def separate_rgb(
    rgb: np.ndarray, icc_profile: str | None, intent: int = 0
) -> dict[str, np.ndarray]:
    """Dispatcher: ICC separation when a profile is configured, else naive."""
    if icc_profile:
        return rgb_to_cmyk_icc(rgb, icc_profile, intent)
    return rgb_to_cmyk(rgb)
