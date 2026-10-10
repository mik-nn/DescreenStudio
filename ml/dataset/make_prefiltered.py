#!/usr/bin/env python
"""Prefiltered dataset for the hybrid track (2.22): classical physics up front,
small cleanup net behind.

Вход сети = scan после ДЕШЁВОГО глобального notch (A0-замер + harmonic family +
Gaussian-маска, один FFT на канал, БЕЗ patch-адаптации и LF — их делает полный
прод-пайплайн на инференсе; чуть более сильные остатки в трейне = робастность).
GT — хардлинки исходников (нуль места; fallback — копия).

Выход: Data/prefiltered/<rootname>/{train,val}/{scan,gt} + manifest.jsonl
(копия исходного + prefilter-тег) + dataset.json.
Схема: те же корни, что рецепт run10 (synth, synth_run4, synth_1200).

Использование:
    python ml/dataset/make_prefiltered.py --workers 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.notch import apply_notch_channel, notch_mask, radius_px_for_frame  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists  # noqa: E402
from ml.preproc.spectrum import estimate_raster, luma  # noqa: E402

SRCS = ["Data/synth", "Data/synth_run4", "Data/synth_1200"]
NOTCH_RADIUS_ROI_PX = 6.0
NOTCH_SHAPE = "butterworth"  # как prod-стадия 1 (auto_notch_rgb без shape)

_cfg = None


def _cfg_once() -> P0Config:
    global _cfg
    if _cfg is None:
        _cfg = P0Config()
    return _cfg


def cheap_prefilter(img: np.ndarray) -> np.ndarray:
    """A0 + per-channel harmonic notch, single FFT per channel."""
    cfg = _cfg_once()
    _ = estimate_raster(luma(img), cfg.spectrum)  # замер ради конф/an: списки строит build
    ch_lists = build_notch_lists(img, cfg, cfg.spectrum.measure_roi)
    h, w, _ = img.shape
    r_px = radius_px_for_frame(NOTCH_RADIUS_ROI_PX, cfg.spectrum.measure_roi, min(h, w))
    radius_r = r_px / min(h, w)
    out = np.empty_like(img)
    for c in range(3):
        m = notch_mask(h, w, ch_lists[c], radius_r, cfg.notch.order, shape=NOTCH_SHAPE)
        out[:, :, c] = apply_notch_channel(img[:, :, c].astype(np.float64), m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _work(args: tuple[str, str, str]) -> dict:
    src_root, split, sid = args
    scan_p = ROOT / src_root / split / "scan" / f"{sid}.png"
    im = Image.open(scan_p).convert("RGB")
    im.load()
    img = np.asarray(im, dtype=np.float32) / 255.0
    pre = cheap_prefilter(img)
    name = Path(src_root).name
    out_scan = ROOT / "Data/prefiltered" / name / split / "scan" / f"{sid}.png"
    out_scan.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pre * 255 + 0.5).astype(np.uint8)).save(out_scan)
    # GT — хардлинк (тот же файл, нуль места)
    out_gt = ROOT / "Data/prefiltered" / name / split / "gt" / f"{sid}.png"
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    if not out_gt.exists():
        try:
            os.link(ROOT / src_root / split / "gt" / f"{sid}.png", out_gt)
        except OSError:
            import shutil
            shutil.copy2(ROOT / src_root / split / "gt" / f"{sid}.png", out_gt)
    return {"id": sid, "source_root": src_root}


def run_split(src_root: str, split: str, workers: int) -> None:
    man_path = ROOT / src_root / split / "manifest.jsonl"
    man = [json.loads(l) for l in open(man_path, encoding="utf-8")]
    name = Path(src_root).name
    out_man = ROOT / "Data" / "prefiltered" / name / split / "manifest.jsonl"
    out_man.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_man.exists():
        with io.open(out_man, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in man if r["id"] not in done]
    print(f"{name}/{split}: total={len(man)} todo={len(todo)} workers={workers}", flush=True)
    by_id = {r["id"]: r for r in man}
    with io.open(out_man, "a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        args = [(src_root, split, r["id"]) for r in todo]
        for i, res in enumerate(ex.map(_work, args)):
            rec = dict(by_id[res["id"]])
            rec["prefilter"] = f"cheap-global-{NOTCH_SHAPE}"
            rec["prefilter_radius_roi_px"] = NOTCH_RADIUS_ROI_PX
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 500 == 0:
                print(f"{name}/{split}: {i + 1}/{len(todo)}", flush=True)
    print(f"{name}/{split}: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--roots", default=",".join(SRCS))
    args = ap.parse_args()
    for root in args.roots.split(","):
        for split in ("train", "val"):
            run_split(root.strip(), split, args.workers)
    (ROOT / "Data" / "prefiltered" / "dataset.json").write_text(json.dumps({
        "sources": args.roots.split(","),
        "prefilter": "cheap-global-notch-butterworth-6px",
        "radius_roi_px": NOTCH_RADIUS_ROI_PX,
        "note": "inputs for hybrid cleanup net (2.22); GT hardlinked",
    }, indent=1), encoding="utf-8")
    # edge_q: notch почти не меняет средний градиент — копируем как есть
    # (иначе hard_boost слепнет к краям на этих корнях)
    import shutil
    for root in args.roots.split(","):
        src_eq = ROOT / root.strip() / "train" / "edge_q.json"
        dst_eq = ROOT / "Data" / "prefiltered" / Path(root.strip()).name / "train" / "edge_q.json"
        if src_eq.exists() and not dst_eq.exists():
            try:
                os.link(src_eq, dst_eq)
            except OSError:
                shutil.copy2(src_eq, dst_eq)


if __name__ == "__main__":
    main()
