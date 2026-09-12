from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "DESCAN-18K"


def read_scan(zip_path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(name)
    im = Image.open(io.BytesIO(raw))
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def list_scans(zip_path: Path, split: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n for n in z.namelist() if n.startswith(f"{split}/scan/") and n.endswith(".tif"))


def spectrum_stats(gray: np.ndarray, k: int = 10) -> dict:
    h, w = gray.shape
    spec = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    mag = np.abs(spec).astype(np.float64)
    mean = mag.mean()
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    nx, ny = (xx - w // 2) / w, (yy - h // 2) / h
    r = np.sqrt(nx * nx + ny * ny)
    work = mag.copy()
    work[r < 0.01] = 0.0
    flat = work.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    peaks = []
    for i in idx:
        y, x = int(i // w), int(i % w)
        fx, fy = (x - w // 2) / w, (y - h // 2) / h
        peaks.append(
            {
                "fx": round(float(fx), 4),
                "fy": round(float(fy), 4),
                "r": round(float(math.hypot(fx, fy)), 4),
                "deg": round(float(math.degrees(math.atan2(fy, fx))), 1),
                "xmean": round(float(mag[y, x] / (mean + 1e-9)), 1),
            }
        )
    peaks.sort(key=lambda p: -p["xmean"])
    hf = mag[r > 0.45]
    return {
        "peaks": peaks,
        "hf_floor_xmean": round(float(np.median(hf) / (mean + 1e-9)), 3),
        "dc_share": round(float(mag[r < 0.01].sum() / (mag.sum() + 1e-9)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="Valid,Test")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ml/simulator/_refs")
    args = ap.parse_args()
    import random

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    all_recs = []
    for split in args.splits.split(","):
        names = list_scans(DATA / f"{split}.zip", split)
        chosen = rng.sample(names, min(args.count, len(names)))
        for n in chosen:
            img = read_scan(DATA / f"{split}.zip", n)
            st = spectrum_stats(img.mean(axis=-1).astype(np.float64))
            rec = {"split": split, "file": n, **st}
            all_recs.append(rec)
            print(f"{split}/{Path(n).name} top_r={[p['r'] for p in st['peaks'][:4]]} hf={st['hf_floor_xmean']}", flush=True)
    (out / "refs.json").write_text(json.dumps(all_recs, indent=1), encoding="utf-8")
    print(f"images={len(all_recs)} -> {out}/refs.json")


if __name__ == "__main__":
    main()
