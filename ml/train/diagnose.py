from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml.model import FourierUNet
from ml.train.data import DescreenDataset

ROOT = Path(__file__).resolve().parents[2]


def evaluate(ckpt: Path, split: str = "val") -> list[dict]:
    m = FourierUNet().cuda()
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    ds = DescreenDataset(ROOT / "Data" / "synth", split)
    dl = DataLoader(ds, batch_size=1, num_workers=2)
    man = [json.loads(l) for l in open(ROOT / "Data" / "synth" / split / "manifest.jsonl", encoding="utf-8")]
    by_id = {r["id"]: r for r in man}
    recs = []
    with torch.no_grad():
        for (xs, ys), path in zip(dl, ds.scans):
            pid = path.stem
            pr = m(xs.cuda()).float()
            mse = torch.mean((pr - ys.cuda()) ** 2).item()
            recs.append({"id": pid, "psnr": round(10 * math.log10(1 / max(mse, 1e-12)), 2), **{k: by_id[pid][k] for k in ("preset", "scan_dpi", "kind", "lpi")}})
    return recs


def save_triples(ckpt: Path, ids: list[str], out: Path, split: str = "val") -> None:
    m = FourierUNet().cuda()
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    ds = DescreenDataset(ROOT / "Data" / "synth", split)
    by_stem = {p.stem: p for p in ds.scans}
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for pid in ids:
            scan = np.asarray(Image.open(by_stem[pid]).convert("RGB"))
            gt = np.asarray(Image.open(str(by_stem[pid]).replace("/scan/", "/gt/")).convert("RGB"))
            x = torch.from_numpy(np.asarray(scan, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            pr = m(x).float()[0].permute(1, 2, 0).cpu().numpy()
            pr = (np.clip(pr, 0, 1) * 255 + 0.5).astype(np.uint8)
            triple = np.concatenate([scan, pr, gt], axis=1)
            Image.fromarray(triple).save(out / f"{pid}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/train/_run1/best.pt")
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--out", default="ml/train/_run1/worst")
    args = ap.parse_args()
    recs = evaluate(ROOT / args.ckpt)
    recs.sort(key=lambda r: r["psnr"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranking.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    worst = [r["id"] for r in recs[: args.worst]]
    save_triples(ROOT / args.ckpt, worst, out)
    import collections
    print("mean:", round(sum(r["psnr"] for r in recs) / len(recs), 2))
    print("worst20 preset:", collections.Counter(r["preset"] for r in recs[:20]))
    print("worst20 dpi:", collections.Counter(r["scan_dpi"] for r in recs[:20]))
    print("worst20 kind:", collections.Counter(r["kind"] for r in recs[:20]))
    print("bottom5:", [(r["id"], r["psnr"], r["preset"], r["scan_dpi"], r["kind"]) for r in recs[:5]])


if __name__ == "__main__":
    main()
