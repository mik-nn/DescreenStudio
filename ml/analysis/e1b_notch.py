from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ml.simulator.color import rgb_to_cmyk
from ml.simulator.config import ScanConfig
from ml.simulator.preview import center_crop
from ml.simulator.scan import sample_scan_params

ROOT = Path(__file__).resolve().parents[2]
ANGLES = {"c": 15.0, "m": 75.0, "y": 0.0, "k": 45.0}
LPI = 133.0
PRINT_DPI = 2400
SCAN_DPI = 300


def cmyk_to_rgb(masks: dict[str, np.ndarray]) -> np.ndarray:
    c, m, y, k = (np.clip(masks[ch], 0, 1) for ch in ("c", "m", "y", "k"))
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)


def butterworth_notch(shape: tuple[int, int], peaks: list[tuple[float, float]], d0: float, order: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    nx, ny = (xx - w // 2) / w, (yy - h // 2) / h
    mask = np.ones((h, w), dtype=np.float64)
    for fx, fy in peaks:
        for sx, sy in ((fx, fy), (-fx, -fy)):
            d2 = (nx - sx) ** 2 + (ny - sy) ** 2
            mask *= 1.0 / (1.0 + (d0 * d0 / np.maximum(d2, 1e-12)) ** order)
    return mask


def screen_peaks(angle_deg: float, harmonics: int = 2) -> list[tuple[float, float]]:
    f = LPI / PRINT_DPI * (PRINT_DPI / SCAN_DPI)
    th = math.radians(angle_deg)
    out = []
    for h in range(1, harmonics + 1):
        out.append((h * f * math.cos(th), h * f * math.sin(th)))
    return out


def notch_channel(gray: np.ndarray, peaks: list[tuple[float, float]], d0: float, order: int) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft2(gray))
    out = spec * butterworth_notch(gray.shape, peaks, d0, order)
    return np.real(np.fft.ifft2(np.fft.ifftshift(out))).astype(np.float32)


def residual_db(before: np.ndarray, after: np.ndarray, peaks: list[tuple[float, float]], band: float = 0.02) -> float:
    def band_energy(img: np.ndarray) -> float:
        spec = np.fft.fftshift(np.fft.fft2(img - img.mean()))
        mag2 = np.abs(spec).astype(np.float64) ** 2
        h, w = img.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        nx, ny = (xx - w // 2) / w, (yy - h // 2) / h
        m = np.zeros_like(mag2, dtype=bool)
        for fx, fy in peaks:
            m |= ((nx - fx) ** 2 + (ny - fy) ** 2 < band**2) | ((nx + fx) ** 2 + (ny + fy) ** 2 < band**2)
        return float(mag2[m].sum())

    return round(float(10 * np.log10(band_energy(after) / (band_energy(before) + 1e-12))), 2)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return round(10 * math.log10(1 / max(mse, 1e-12)), 2)


def process_scan(scan: np.ndarray, gt: np.ndarray, rot: float, d0: float, order: int) -> dict:
    all_peaks = [p for ch in ANGLES for p in screen_peaks(ANGLES[ch] + rot)]
    rgb_out = np.stack([notch_channel(scan[..., i], all_peaks, d0, order) for i in range(3)], axis=-1)
    masks = rgb_to_cmyk(np.clip(scan, 0, 1))
    cmyk_out = {}
    for ch, ang in ANGLES.items():
        cmyk_out[ch] = notch_channel(masks[ch], screen_peaks(ang + rot), d0, order)
    cmyk_rgb = cmyk_to_rgb(cmyk_out)
    g0 = scan.mean(axis=-1)
    return {
        "rgb_resid_db": residual_db(g0, np.clip(rgb_out, 0, 1).mean(axis=-1), all_peaks),
        "cmyk_resid_db": residual_db(g0, np.clip(cmyk_rgb, 0, 1).mean(axis=-1), all_peaks),
        "rgb_psnr": psnr(np.clip(rgb_out, 0, 1), gt),
        "cmyk_psnr": psnr(np.clip(cmyk_rgb, 0, 1), gt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="ml/analysis/_plates")
    ap.add_argument("--d0", default="6,10")
    ap.add_argument("--order", default="2,3")
    ap.add_argument("--out", default="ml/analysis/_e1b.json")
    args = ap.parse_args()
    root = ROOT / args.plates
    names = sorted(p.stem for p in (root / "scan").glob("*.png"))
    results = []
    for i, name in enumerate(names):
        scan = np.asarray(Image.open(root / "scan" / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0
        src = next((ROOT / "Data" / "gt" / "dev").glob(f"{name}.jpg"))
        gt0 = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
        gt0 = center_crop(gt0, scan.shape[0])
        rng = np.random.default_rng(200 + i)
        rot = sample_scan_params(ScanConfig(seed=200 + i), rng)["angle"]
        from ml.simulator.scan import rotate_rgb

        gt = rotate_rgb(gt0, rot)
        for d0 in [float(x) / 512 for x in args.d0.split(",")]:
            for order in [int(x) for x in args.order.split(",")]:
                r = process_scan(scan, gt, rot, d0, order)
                r.update({"id": name, "d0": d0 * 512, "order": order})
                results.append(r)
        print(f"{i + 1}/{len(names)} {name}", flush=True)
    (ROOT / args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for d0 in args.d0.split(","):
        for order in args.order.split(","):
            sub = [r for r in results if r["d0"] == float(d0) and r["order"] == int(order)]
            rr = sum(r["rgb_resid_db"] for r in sub) / len(sub)
            cr = sum(r["cmyk_resid_db"] for r in sub) / len(sub)
            rp = sum(r["rgb_psnr"] for r in sub) / len(sub)
            cp = sum(r["cmyk_psnr"] for r in sub) / len(sub)
            print(f"d0={d0} order={order} resid rgb={rr:.2f} cmyk={cr:.2f} | psnr rgb={rp:.2f} cmyk={cp:.2f}")


if __name__ == "__main__":
    main()
