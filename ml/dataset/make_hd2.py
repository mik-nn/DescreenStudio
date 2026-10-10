#!/usr/bin/env python
"""Data/synth_hd2 — HD-v1 фикс для run9 (2026-09-25, анализ хвоста в JOURNAL).

Отличия от synth_hd (v1, run8):
1. Только ветвь hd800 (кроп 384 -> x4/3 Lanczos -> 512, r_new = r_old*0.75).
   hd1200 (256 -> x2) УДАЛЕНА: GT в ней мягче исходника в 5-50x (LapVar 28/480,
   0/355...), сеть учится размывать (sharp-коллапс run8: 0.02-0.20).
   hd800 держит 11-38% резкости — приемлемо.
2. Третий источник Data/synth_gap (LPI 100-120 @600DPI, seed 49): закрывает
   межпресетную щель — r_gap800 = lpi/800 = 0.125-0.150, точное накрытие
   реал-полосы 0.128-0.149, где v1 давал 0.0% (hd800) / 11.5% (в целом).
3. Исправлен комментарий v1: реал-полосу закрывает hd800 (+gap), НЕ hd1200.

Выход: Data/synth_hd2/{train,val}/{scan,gt} 512x512 + manifest + dataset.json.
Схема манифеста = v1 + emulated_dpi/freq_mul/source_root.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRCS = ["Data/synth", "Data/synth_run4", "Data/synth_gap"]
OUT = ROOT / "Data" / "synth_hd2"

# (суффикс, кроп, целевой размер, множитель частоты, эмулируемый DPI)
VARIANTS = [
    ("hd800", 384, 512, 0.75, 800),
]


def main() -> None:
    stats = {}
    for split in ("train", "val"):
        out_scan = OUT / split / "scan"
        out_gt = OUT / split / "gt"
        out_scan.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)
        man_out = []
        n = 0
        for src_root in SRCS:
            root = ROOT / src_root
            man_path = root / split / "manifest.jsonl"
            if not man_path.exists():
                print(f"{split}: {src_root} no manifest, skip", flush=True)
                continue
            man = [json.loads(l) for l in open(man_path, encoding="utf-8")]
            for row in man:
                if row["scan_dpi"] != 600:
                    continue
                sid = row["id"]
                for suffix, crop, target, fmul, edpi in VARIANTS:
                    uid = f"{Path(src_root).name}_{suffix}_{sid}"
                    rec = dict(row)
                    rec["id"] = uid
                    rec["emulated_dpi"] = edpi
                    rec["freq_mul"] = fmul
                    rec["source_root"] = src_root
                    man_out.append(rec)
                    if (out_scan / f"{uid}.png").exists():
                        n += 1
                        continue
                    lr = random.Random(hash(uid) & 0xFFFF)
                    for sub, out_dir in (("scan", out_scan), ("gt", out_gt)):
                        im = Image.open(root / split / sub / f"{sid}.png")
                        w, h = im.size
                        x0 = lr.randint(0, w - crop)
                        y0 = lr.randint(0, h - crop)
                        im = im.crop((x0, y0, x0 + crop, y0 + crop))
                        im = im.resize((target, target), Image.LANCZOS)
                        im.save(out_dir / f"{uid}.png")
                    n += 1
            print(f"{split}: {src_root} ok", flush=True)
        (OUT / split / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in man_out) + "\n", encoding="utf-8")
        stats[split] = n
        print(f"{split}: {n} pairs", flush=True)
    (OUT / "dataset.json").write_text(json.dumps({
        "train": stats["train"], "val": stats["val"], "seed": "post-hoc-v3-hd800-only",
        "note": "hd800-only (384->512 x4/3); sources synth+synth_run4+synth_gap; hd1200 dropped (GT blur 5-50x)",
        "variants": [[v[0], v[4]] for v in VARIANTS],
    }, indent=1), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
