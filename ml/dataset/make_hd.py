#!/usr/bin/env python
"""Data/synth_hd — эмуляция высокодипиальных сканов (800/1200 DPI) для run8+.

Контекст (2026-09-21): реальные продакшн-сканы — 800–1200 DPI, r растра
0.128–0.149 cyc/px; обучающий конверт 300/600 DPI — r 0.158–0.55. Все три
RealScan-скана ЛЕВЕЕ конверта; Offset1 (r=0.128) сеть не дожимает
(E в кольце 110 -> 148 вместо очистки).

Частотная арифметика (исправлена ошибка первой версии!): cyc/px при даунсэмпле
РАСТЁТ (r_new = r_old / scale). Эмулировать более высокий DPI =
БОЛЬШИЙ период растра в px = кроп 600-DPI изображения с АПСЭМПЛОМ обратно:
  - 1200-DPI-экв: кроп 256 -> x2 Lanczos -> 512, r_new = r_old/2  (0.079..0.275)
  - 800-DPI-экв:  кроп 384 -> x4/3 Lanczos -> 512, r_new = r_old*0.75 (0.119..0.41)
Покрытие: Offset1 0.128 (hd800), 1821145u 0.149 (hd800), Magazine 0.142 (hd800).
Ограничение эмуляции: детальность контента на уровне 600 DPI (реальный
1200-DPI скан детальнее) — растровая геометрия верна, контент мягче.

Выход: Data/synth_hd/{train,val}/{scan,gt} 512x512 + manifest + dataset.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRCS = ["Data/synth", "Data/synth_run4"]
OUT = ROOT / "Data" / "synth_hd"

# (суффикс, кроп, целевой размер, множитель частоты, эмулируемый DPI)
VARIANTS = [
    ("hd1200", 256, 512, 0.5, 1200),
    ("hd800", 384, 512, 0.75, 800),
]


def main() -> None:
    rng = random.Random(7)
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
            man = [json.loads(l) for l in open(root / split / "manifest.jsonl", encoding="utf-8")]
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
                    # центрированный кроп со случайным сдвигом (детерминир. по uid)
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
        (OUT / split / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in man_out) + "\n", encoding="utf-8")
        stats[split] = n
        print(f"{split}: {n} пар", flush=True)
    (OUT / "dataset.json").write_text(json.dumps({
        "train": stats["train"], "val": stats["val"], "seed": "post-hoc-v2",
        "note": "600-DPI crop upsampled: hd1200=256->512 (r/2), hd800=384->512 (r*0.75)",
        "variants": [[v[0], v[4]] for v in VARIANTS],
    }, indent=1), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
