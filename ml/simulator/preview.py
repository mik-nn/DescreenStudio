from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ml.simulator.compose import simulate_print
from ml.simulator.config import PrintConfig, expected_fundamentals, preset
from ml.simulator.targets import TARGETS


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w, _ = arr.shape
    y0, x0 = max((h - size) // 2, 0), max((w - size) // 2, 0)
    return arr[y0 : y0 + size, x0 : x0 + size]


def top_peaks(gray: np.ndarray, k: int = 8, dc_radius: int = 3) -> list[tuple[float, float, float]]:
    h, w = gray.shape
    spec = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.log1p(np.abs(spec)).astype(np.float64)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mag[(xx - w // 2) ** 2 + (yy - h // 2) ** 2 < dc_radius**2] = -1.0
    flat = mag.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    out = []
    for i in idx:
        y, x = int(i // w), int(i % w)
        out.append(((x - w // 2) / w, (y - h // 2) / h, float(mag[y, x])))
    return sorted(out, key=lambda t: -t[2])


def match_distance(peaks: list[tuple[float, float, float]], expected: list[tuple[float, float]]) -> float:
    dists = []
    for fx, fy, _ in peaks:
        d = min(math.hypot(fx - ex, fy - ey) for ex, ey in expected)
        dists.append(d)
    return float(np.median(dists))


def scan_view(print_img: np.ndarray, s: int) -> np.ndarray:
    h, w, _ = print_img.shape
    im = Image.fromarray((np.clip(print_img, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((w // s, h // s), Image.BOX), dtype=np.float32) / 255.0


def spectrum_distance(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> float:
    da = a.mean(axis=-1) - ref.mean(axis=-1)
    db = b.mean(axis=-1) - ref.mean(axis=-1)
    ma = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(da))))
    mb = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(db))))
    return float(np.linalg.norm(ma - mb) / (np.linalg.norm(mb) + 1e-6))


def save_spectrum(gray: np.ndarray, path: Path, size: int = 512) -> None:
    spec = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.log1p(np.abs(spec)).astype(np.float64)
    mag = (255 * (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)).astype(np.uint8)
    Image.fromarray(mag).resize((size, size), Image.BILINEAR).save(path)


def tone_curve(view: np.ndarray, steps: int = 10) -> list[float]:
    g = view.mean(axis=-1)
    h, _ = g.shape
    return [round(float(g[:, int((i + 0.5) * h / steps)].mean()), 4) for i in range(steps)]


def run_case(gt: np.ndarray, name: str, preset_name: str, factors: list[int], out_dir: Path, seed: int) -> dict:
    rec: dict = {"name": name, "preset": preset_name, "factors": {}}
    views: dict[int, np.ndarray] = {}
    for s in factors:
        cfg = preset(preset_name, oversample=s, seed=seed)
        t0 = time.perf_counter()
        printed = simulate_print(gt, cfg)
        dt = time.perf_counter() - t0
        view = scan_view(printed, s)
        views[s] = view
        Image.fromarray((view * 255 + 0.5).astype(np.uint8)).save(out_dir / f"{name}_{preset_name}_x{s}.png")
        hp, wp, _ = printed.shape
        cy, cx = hp // 2, wp // 2
        crop = printed[max(cy - 128, 0) : cy + 128, max(cx - 128, 0) : cx + 128]
        Image.fromarray((crop * 255 + 0.5).astype(np.uint8)).save(out_dir / f"{name}_{preset_name}_x{s}_crop.png")
        peaks = top_peaks(printed.mean(axis=-1))
        exp = [p for v in expected_fundamentals(cfg).values() for p in v]
        rec["factors"][s] = {
            "seconds": round(dt, 2),
            "peak_match_cycpx": round(match_distance(peaks, exp), 5),
            "peaks": [(round(fx, 4), round(fy, 4)) for fx, fy, _ in peaks[:4]],
        }
        if s == factors[0]:
            save_spectrum(printed.mean(axis=-1).astype(np.float64), out_dir / f"{name}_{preset_name}_spectrum.png")
        if name == "gray_steps":
            rec["factors"][s]["tone_curve"] = tone_curve(view)
    if len(views) == 2:
        s0, s1 = factors[0], factors[1]
        rec["spectrum_rel_dist"] = round(spectrum_distance(views[s0], views[s1], gt), 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="magazine")
    ap.add_argument("--factors", default="4,8")
    ap.add_argument("--gt-size", type=int, default=512)
    ap.add_argument("--photos", type=int, default=3)
    ap.add_argument("--out", default="ml/simulator/_preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / args.out / args.preset
    out.mkdir(parents=True, exist_ok=True)
    factors = [int(x) for x in args.factors.split(",")]
    cases: list[tuple[str, np.ndarray]] = [(n, fn(args.gt_size)) for n, fn in TARGETS.items()]
    dev = sorted((root / "Data" / "gt" / "dev").glob("*.jpg"))[: args.photos]
    for f in dev:
        im = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        cases.append((f.stem, center_crop(im, args.gt_size)))
    report = [run_case(gt, n, args.preset, factors, out, args.seed) for n, gt in cases]
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for r in report:
        f = r["factors"]
        line = r["name"] + " " + " ".join(f"{s}: {v['seconds']}s match={v['peak_match_cycpx']}" for s, v in f.items())
        print(line, "specdist=" + str(r.get("spectrum_rel_dist")))


if __name__ == "__main__":
    main()
