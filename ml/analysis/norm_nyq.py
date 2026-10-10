#!/usr/bin/env python
"""Пилот нормализации по Найквисту (zero-shot, без обучения).

Идея владельца: даунсемпл каждого скана к каноническим 2px/ячейку (r=0.5) —
без потерь тона по Найквисту, геометрия растра константа для сети.
Выход апсемплится обратно (чистый контент апсемплить безопасно).
Сравнение попарно против baseline на тех же сэмплах, GT всегда нативный.

Протокол чистоты (уроки провалов):
- один процесс, lockfile train.py не трогаем (только чтение);
- batch IDs и r фиксируются в выходной JSON;
- размеры снапятся к кратным 8 (ограничение UNet L3).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.benchmark.eval_models import load_model  # noqa: E402

R_TARGET = 0.5
R_LO, R_HI = 0.10, 0.50
SETS = ["Data/synth", "Data/synth_run4", "Data/synth_1200"]
N_PER_SET = 20


def snap8(x: int) -> int:
    return max(64, round(x / 8) * 8)


def load_pair(root: Path, sid: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.asarray(Image.open(root / "val" / "scan" / f"{sid}.png"), dtype=np.float32) / 255.0
    y = np.asarray(Image.open(root / "val" / "gt" / f"{sid}.png"), dtype=np.float32) / 255.0
    to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)
    return to_t(x), to_t(y)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def theory_map(root: Path) -> dict[str, float]:
    out = {}
    man = root / "val" / "manifest.jsonl"
    if man.exists():
        for line in open(man, encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r["lpi"] / r["scan_dpi"] * r.get("freq_mul", 1.0)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="run10")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    models = {}
    for name in args.models.split(","):
        name = name.strip()
        m, cfg, has_cond = load_model(f"ml/train/_{name}")
        models[name] = (m.to(device).eval(), has_cond)
    table: dict = {}
    for s in SETS:
        root = ROOT / "Data" / s
        tmap = theory_map(root)
        scans = sorted((root / "val" / "scan").glob("*.png"))
        # детерминированный отбор: первые N в полосе (воспроизводимо!)
        cand = [p for p in scans if tmap.get(p.stem) is not None
                and R_LO <= tmap[p.stem] <= R_HI][:N_PER_SET]
        cond_map: dict[str, list[float]] = {}
        a0p = root / "val" / "a0.json"
        if a0p.exists():
            from ml.model.film import build_cond_vec
            for k, v in json.loads(a0p.read_text(encoding="utf-8")).items():
                cond_map[k] = build_cond_vec(*v)
        rows = []
        for p in cand:
            r = tmap[p.stem]
            x, y = load_pair(root, p.stem)
            h, w = x.shape[1:]
            k = R_TARGET / r
            nh, nw = snap8(round(h * k)), snap8(round(w * k))
            xu = torch.nn.functional.interpolate(
                x[None], size=(nh, nw), mode="bicubic",
                align_corners=False, antialias=True)[0]
            rec: dict = {"id": p.stem, "r": round(r, 4), "scale": round(nh / h, 3),
                         "canon": [nh, nw]}
            for name, (m, has_cond) in models.items():
                kw = {}
                if has_cond:
                    kw["cond"] = torch.tensor(
                        [cond_map.get(p.stem, [0.0, 0.0, 1.0, 0.0])],
                        dtype=torch.float32).to(device)
                with torch.amp.autocast("cuda", enabled=False):
                    pb = m(x[None].to(device).float(), **kw)[0].cpu()
                    pu = m(xu[None].to(device).float(), **kw)[0].cpu()
                back = torch.nn.functional.interpolate(
                    pu[None], size=(h, w), mode="bicubic",
                    align_corners=False, antialias=True)[0].clamp(0, 1)
                rec[f"{name}_base"] = round(psnr(pb.clamp(0, 1), y), 2)
                rec[f"{name}_norm"] = round(psnr(back, y), 2)
            rows.append(rec)
        nb = np.mean([r[f"{list(models)[0]}_base"] for r in rows])
        nn = np.mean([r[f"{list(models)[0]}_norm"] for r in rows])
        print(f"{s}: n={len(rows)} base={nb:.2f} norm={nn:.2f} delta={nn - nb:+.2f}",
              flush=True)
        table[s] = rows
    outp = ROOT / "ml" / "analysis" / "_norm_nyq.json"
    outp.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("saved:", outp)


if __name__ == "__main__":
    main()
