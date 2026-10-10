#!/usr/bin/env python
"""Верификация метрик v2 на сохранённых выходах _eval0921_run8 (CPU, без GPU).

src = центральный кроп 2048 исходного TIF (как в eval_realscan --crop 2048),
out = PNG из outdir. Сверяет v1-поля с metrics.json + считает v2-поля.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.benchmark.eval_realscan import fmt  # noqa: E402

CROP = 2048
EVAL = ROOT / "Data" / "RealScan" / "_eval0921_run8"
SCANS = {
    "1821145u_1": "Data/RealScan/1821145u_1.tif",
    "Offset1": "Data/RealScan/Offset1.tif",
    "Magazine": "Data/RealScan/Magazine.tif",
}
OUTS = ["prefilter", "run7", "run7+unsharp", "run8", "run8+unsharp"]


def load_rgb(p: Path) -> np.ndarray:
    im = Image.open(p)
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    cfg = P0Config()
    old = json.loads((EVAL / "metrics.json").read_text(encoding="utf-8"))
    for stem, rel in SCANS.items():
        src = load_rgb(ROOT / rel)
        s = min(CROP, src.shape[0], src.shape[1])
        y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
        src = src[y:y + s, x:x + s]
        key = stem.replace(" ", "_")
        print(f"=== {stem} ===")
        for o in OUTS:
            fn = "prefilter.png" if o == "prefilter" else f"{o.replace('+', '_')}.png"
            p = EVAL / f"{key}_{fn}"
            if not p.exists():
                print(f"  {o}: no file {p.name}")
                continue
            out = load_rgb(p)
            r = evaluate(src, out, cfg.spectrum)
            print(f"  {o}: {fmt(r)}".encode("ascii", "replace").decode(), flush=True)
        print(f"  [v1 was] run7={old[stem].get('run7')} | run8={old[stem].get('run8')}".encode("ascii", "replace").decode(), flush=True)


if __name__ == "__main__":
    main()
