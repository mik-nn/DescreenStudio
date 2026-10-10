#!/usr/bin/env python
"""Oracle-эксперимент: есть ли запас качества в детекторе пиков/настройке notch?

Part A (синтетика): A0-детектор против GT r=lpi/scan_dpi из манифеста
  -> ошибка по частоте, доля пропусков фундаментала, лишние пики.

Part B (RealScan-кропы, prefilter-only): три варианта notch-списков
  - prod:   production build_notch_lists (A0)
  - wide2:  prod-список, радиус x2 (та же геометрия пиков, шире вырез)
  - ang4:   фундаментальная r от A0 + полный набор углов экрана
            15/75/0/45 (+гармоническое семейство каждого)
  Метрики: resid / sharp (NoGTReport). Без сети — изолируем префильтр.

Использование:
    python ml/benchmark/oracle_peaks.py --n 200
    python ml/benchmark/oracle_peaks.py --scans Data/RealScan/Offset1.tif --crop 2048
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.preproc.config import P0Config  # noqa: E402
from ml.preproc.edgenotch import adaptive_notch_rgb  # noqa: E402
from ml.preproc.lfband import detect_lf_bands, subtract_lf_rgb  # noqa: E402
from ml.preproc.metrics import evaluate  # noqa: E402
from ml.preproc.notch import auto_notch_rgb  # noqa: E402
from ml.preproc.run_p0 import build_notch_lists, load_rgb  # noqa: E402
from ml.preproc.spectrum import harmonic_family, luma  # noqa: E402


def part_a(n: int) -> dict:
    from ml.preproc.spectrum import estimate_raster

    man = [json.loads(l) for l in open(ROOT / "Data/synth_run4/val/manifest.jsonl", encoding="utf-8")][:n]
    cfg = P0Config()
    errs, missed, extra, confs = [], 0, 0, []
    for rec in man:
        scan = load_rgb(str(ROOT / "Data/synth_run4/val/scan" / f"{rec['id']}.png"))
        gt_r = rec["lpi"] / rec["scan_dpi"]
        est = estimate_raster(luma(scan), cfg.spectrum)
        confs.append(est.confidence)
        if est.confidence < 0.3 or est.r <= 0.05:
            missed += 1
            continue
        errs.append(abs(est.r - gt_r) / gt_r)
        lists = build_notch_lists(scan, cfg, min(cfg.spectrum.measure_roi, 512))
        fund_found = any(abs(r - gt_r) < 0.01 for ch in lists for r, _a in ch)
        if not fund_found:
            missed += 1
        extra += max(0, sum(len(ch) for ch in lists) - 4)
    return dict(
        n=len(man), missed=missed,
        freq_rel_err_median=float(np.median(errs)) if errs else None,
        freq_rel_err_p90=float(np.percentile(errs, 90)) if errs else None,
        extra_peaks_avg=extra / max(len(man), 1),
        conf_median=float(np.median(confs)),
    )


SCREEN_ANGLES = (15.0, 75.0, 0.0, 45.0)


def ang4_lists(prod_lists: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Фундаментальная r продакшен-списка + все 4 угла экрана + гармоники."""
    out = []
    for ch in prod_lists:
        if not ch:
            out.append(ch)
            continue
        r0 = ch[0][0]
        lst: list[tuple[float, float]] = []
        for a in SCREEN_ANGLES:
            lst.extend(harmonic_family(r0, a))
        out.append(lst[:12])
    return out


def part_b(scans: list[str], crop: int) -> dict:
    cfg = P0Config()
    table: dict[str, dict[str, str]] = {}
    for scan_path in scans:
        src = load_rgb(str(ROOT / scan_path))
        if crop:
            s = min(crop, src.shape[0], src.shape[1])
            y, x = (src.shape[0] - s) // 2, (src.shape[1] - s) // 2
            src = src[y : y + s, x : x + s]
        stem = Path(scan_path).stem.replace(" ", "_")
        prod = build_notch_lists(src, cfg, cfg.spectrum.measure_roi)
        variants = {
            "prod": prod,
            "wide2": prod,
            "ang4": ang4_lists(prod),
        }
        for name, lists in variants.items():
            ncfg = cfg.notch
            if name == "wide2":
                ncfg = replace(ncfg, radius_roi_px=ncfg.radius_roi_px * 2)
            cur = auto_notch_rgb(src, lists, ncfg, cfg.spectrum.measure_roi)
            cur, _ = adaptive_notch_rgb(cur, lists, cfg.edge_notch, cfg.edge_notch.radius_ref_roi)
            bands = detect_lf_bands(luma(cur), cfg.lf)
            cur = subtract_lf_rgb(cur, bands, cfg.lf)
            rep = evaluate(src, cur, cfg.spectrum)
            table.setdefault(stem, {})[name] = (
                f"resid={rep.residual_peak_xmean:7.1f} sharp={rep.sharp_ratio:.2f} LF={rep.lf_ratio:.2f}"
            )
            print(f"{stem:14s} {name:6s} {table[stem][name]}", flush=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--scans", nargs="*", default=[])
    ap.add_argument("--crop", type=int, default=2048)
    args = ap.parse_args()

    out = {}
    if not args.scans:
        out["detector_A_vs_GT"] = part_a(args.n)
        print(json.dumps(out["detector_A_vs_GT"], indent=1))
    if args.scans:
        out["notch_variants"] = part_b(args.scans, args.crop)
    p = ROOT / "ml/benchmark/_out/oracle_peaks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved:", p)


if __name__ == "__main__":
    main()
