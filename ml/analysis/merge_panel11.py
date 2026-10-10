#!/usr/bin/env python
"""Сводная таблица панели 11 сканов из сохранённых PNG (CPU, без GPU).

Источники: _eval0925_panel (6 сканов, без metrics.json — прогон был убит),
_eval0925_panel2/metrics.json (3 скана), _eval0921_run8 (Offset1, Magazine).
src = центральный кроп 2048 исходного TIF. Выход: _panel11.json + печать.
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
P1 = ROOT / "Data" / "RealScan" / "_eval0925_panel"
P2 = ROOT / "Data" / "RealScan" / "_eval0925_panel2"
P0 = ROOT / "Data" / "RealScan" / "_eval0921_run8"

SCANS = {  # stem -> (tif, [(outdir, tag)])
    "1821145u_1": ("Data/RealScan/1821145u_1.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "1821146u": ("Data/RealScan/1821146u.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "book1": ("Data/RealScan/book1.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "Offset1": ("Data/RealScan/Offset1.tif", [(P0, "run7"), (P0, "run8")]),
    "Offset2": ("Data/RealScan/Offset2.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "Calendar1": ("Data/RealScan/Calendar1.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "BookCOver": ("Data/RealScan/BookCOver.tif", [(P1, "run7"), (P1, "run7_unsharp")]),
    "Magazine": ("Data/RealScan/Magazine.tif", [(P0, "run7"), (P0, "run8")]),
    "Magazine_(2)": ("Data/RealScan/Magazine (2).tif", [(P2, "run7"), (P2, "run7_unsharp")]),
    "Pack": ("Data/RealScan/Pack.tif", [(P2, "run7"), (P2, "run7_unsharp")]),
    "OldBook": ("Data/RealScan/OldBook.tif", [(P2, "run7"), (P2, "run7_unsharp")]),
}


def load_rgb(p: Path) -> np.ndarray:
    im = Image.open(p)
    im.load()
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    cfg = P0Config()
    table: dict = {}
    for stem, (tif, outs) in SCANS.items():
        src = load_rgb(ROOT / tif)
        s = min(CROP, src.shape[0], src.shape[1])
        y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
        src = src[y:y + s, x:x + s]
        pre = outs[0][0] / f"{stem}_prefilter.png"
        if pre.exists():
            table.setdefault(stem, {})["prefilter"] = fmt(evaluate(src, load_rgb(pre), cfg.spectrum))
        for outdir, tag in outs:
            p = outdir / f"{stem}_{tag}.png"
            if not p.exists():
                table.setdefault(stem, {})[tag] = "MISSING"
                continue
            table.setdefault(stem, {})[tag] = fmt(evaluate(src, load_rgb(p), cfg.spectrum))
        print(f"=== {stem} ===", flush=True)
        for k, v in table[stem].items():
            print(f"  {k}: {v}".encode("ascii", "replace").decode(), flush=True)
    outp = ROOT / "Data" / "RealScan" / "_panel11.json"
    outp.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("saved:", outp)


if __name__ == "__main__":
    main()
