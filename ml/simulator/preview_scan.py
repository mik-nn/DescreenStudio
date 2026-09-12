from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ml.simulator.compose import simulate_print
from ml.simulator.config import ScanConfig, preset
from ml.simulator.preview import center_crop, save_spectrum, top_peaks
from ml.simulator.scan import simulate_scan
from ml.simulator.targets import TARGETS, gray_steps


def lowband_ratio(gray: np.ndarray, band: float = 0.08) -> float:
    h, w = gray.shape
    spec = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag2 = np.abs(spec).astype(np.float64) ** 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt(((xx - w // 2) / w) ** 2 + ((yy - h // 2) / h) ** 2)
    mask = (r < band) & (r > 0.005)
    return float(mag2[mask].sum() / (mag2.sum() + 1e-9))


def run_angle(gt: np.ndarray, angle: float, printed: np.ndarray, scfg: ScanConfig, out: Path, tag: str) -> dict:
    params = {
        "angle": angle,
        "psf_sigma": 0.6,
        "gauss_sigma": 0.008,
        "poisson_peak": 80.0,
        "dust_lines": 0,
        "dust_dots": 0,
    }
    scan, used = simulate_scan(printed, preset("magazine", oversample=8), scfg, params)
    Image.fromarray((scan * 255 + 0.5).astype(np.uint8)).save(out / f"scan_{tag}.png")
    gray = scan.mean(axis=-1).astype(np.float64)
    save_spectrum(gray, out / f"scan_{tag}_spectrum.png")
    peaks = top_peaks(gray, k=12)
    return {
        "angle": angle,
        "lowband_ratio": round(lowband_ratio(gray), 5),
        "peaks": [(round(float(fx), 4), round(float(fy), 4)) for fx, fy, _ in peaks[:6]],
        "used": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in used.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="magazine")
    ap.add_argument("--gt-size", type=int, default=256)
    ap.add_argument("--photo", default="")
    ap.add_argument("--target", default="gray")
    ap.add_argument("--angles", default="0.0,0.5,1.5")
    ap.add_argument("--out", default="ml/simulator/_scan")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / args.out / args.preset
    out.mkdir(parents=True, exist_ok=True)
    if args.photo:
        im = np.asarray(Image.open(root / args.photo).convert("RGB"), dtype=np.float32) / 255.0
        gt = center_crop(im, args.gt_size)
        tag = Path(args.photo).stem
    else:
        gt = TARGETS[args.target](args.gt_size)
        tag = args.target
    out_tag = out / tag
    out_tag.mkdir(parents=True, exist_ok=True)
    pcfg = preset(args.preset, oversample=8, seed=0)
    printed = simulate_print(gt, pcfg)
    save_spectrum(printed.mean(axis=-1).astype(np.float64), out_tag / "print_spectrum.png")
    scfg = ScanConfig(seed=0)
    report = [run_angle(gt, float(a), printed, scfg, out_tag, f"a{a}") for a in args.angles.split(",")]
    (out_tag / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for r in report:
        print(f"angle={r['angle']} lowband={r['lowband_ratio']} peaks={r['peaks'][:4]}")


if __name__ == "__main__":
    main()
