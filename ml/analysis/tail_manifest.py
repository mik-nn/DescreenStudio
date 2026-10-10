#!/usr/bin/env python
"""Шаг 2a: анализ манифестов synth/synth_run4/synth_hd (только чтение, CPU).

Что считает:
1. Состав train/val: preset x ink_set x scan_dpi x kind, ghost/show-through, banding.
2. Покрытие растровых частот r = lpi/scan_dpi * freq_mul против RealScan-репера 0.128-0.149.
3. GT-QC: распределение gt_top_peak (грязный GT как предиктор тяжёлых сэмплов).
4. HD-разрез: emulated_dpi x source_root.

Выход: таблицы в stdout + JSON-отчёт.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ["Data/synth", "Data/synth_run4", "Data/synth_hd"]
REAL_R = (0.128, 0.149)  # Offset1..1821145u, реальные сканы 800-1200 DPI


def load(root: str, split: str) -> list[dict]:
    p = ROOT / root / split / "manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


def report_set(root: str) -> dict:
    out: dict = {"root": root}
    for split in ("train", "val"):
        rows = load(root, split)
        out[split] = {"n": len(rows)}
        if not rows:
            continue
        out[split]["preset"] = dict(Counter(r.get("preset") for r in rows))
        out[split]["kind"] = dict(Counter(r.get("kind") for r in rows))
        out[split]["scan_dpi"] = dict(Counter(r.get("scan_dpi") for r in rows))
        inks = Counter(r.get("ink_set", "n/a") for r in rows)
        out[split]["ink_set"] = dict(inks)
        ghosts = [r for r in rows if r.get("ghost_src")]
        shows = [r.get("scan_params", {}).get("show_alpha", 0) or 0 for r in rows]
        bands = [r.get("scan_params", {}).get("band_amp", 0) or 0 for r in rows]
        out[split]["ghost_frac"] = round(len(ghosts) / len(rows), 4)
        out[split]["show_alpha>0_frac"] = round(sum(1 for s in shows if s > 0) / len(rows), 4)
        out[split]["band_amp_mean"] = round(sum(bands) / len(bands), 5)
        # частоты
        rs = [r["lpi"] / r["scan_dpi"] * r.get("freq_mul", 1.0) for r in rows]
        out[split]["r_min"] = round(min(rs), 4)
        out[split]["r_max"] = round(max(rs), 4)
        out[split]["r_p10"] = round(quantile(rs, 0.10), 4)
        out[split]["r_p50"] = round(quantile(rs, 0.50), 4)
        out[split]["r_p90"] = round(quantile(rs, 0.90), 4)
        in_real = sum(1 for x in rs if REAL_R[0] <= x <= REAL_R[1])
        out[split]["r_in_real_frac"] = round(in_real / len(rs), 4)
        # GT-QC
        peaks = [r.get("gt_top_peak", 0) or 0 for r in rows]
        out[split]["gt_peak_mean"] = round(sum(peaks) / len(peaks), 1)
        out[split]["gt_peak_p50"] = round(quantile(peaks, 0.50), 1)
        out[split]["gt_peak_p90"] = round(quantile(peaks, 0.90), 1)
        out[split]["gt_peak_max"] = round(max(peaks), 1)
        out[split]["gt_peak>100_frac"] = round(sum(1 for p in peaks if p > 100) / len(peaks), 4)
        if root == "Data/synth_hd":
            out[split]["emulated_dpi"] = dict(Counter(r.get("emulated_dpi") for r in rows))
            out[split]["source_root"] = dict(Counter(r.get("source_root") for r in rows))
    return out


def main() -> None:
    full = [report_set(s) for s in SETS]
    outp = ROOT / "ml" / "analysis" / "_tail_manifest.json"
    outp.write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    for rep in full:
        print(f"===== {rep['root']} =====")
        for split in ("train", "val"):
            d = rep[split]
            print(f"  [{split}] n={d['n']}")
            if not d["n"]:
                continue
            print(f"    preset={d['preset']} kind={d['kind']} dpi={d['scan_dpi']}")
            print(f"    ink={d['ink_set']} ghost={d['ghost_frac']} show>0={d['show_alpha>0_frac']} band_mean={d['band_amp_mean']}")
            print(f"    r: min={d['r_min']} p10={d['r_p10']} p50={d['r_p50']} p90={d['r_p90']} max={d['r_max']} in_real[0.128-0.149]={d['r_in_real_frac']}")
            print(f"    gt_peak: mean={d['gt_peak_mean']} p50={d['gt_peak_p50']} p90={d['gt_peak_p90']} max={d['gt_peak_max']} >100={d['gt_peak>100_frac']}")
            if "emulated_dpi" in d:
                print(f"    emulated={d['emulated_dpi']} src={d['source_root']}")
    print("saved:", outp)


if __name__ == "__main__":
    sys.exit(main())
