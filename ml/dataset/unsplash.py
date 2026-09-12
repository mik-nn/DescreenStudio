from __future__ import annotations

import argparse
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ARCHIVE = Path(__file__).resolve().parents[2] / "Data" / "unsplash-research-dataset-lite-latest.zip"
IMAGE_PARAMS = "?w=2560&q=95&fm=jpg&fit=max"


def load_candidates(min_side: int = 2000) -> list[dict]:
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read("photos.tsv000").decode("utf-8", "replace").splitlines()
    hdr = raw[0].split("\t")
    col = {name: hdr.index(name) for name in ("photo_id", "photo_image_url", "photo_width", "photo_height", "photo_featured", "stats_views")}
    out: list[dict] = []
    for line in raw[1:]:
        p = line.split("\t")
        try:
            w, h = int(p[col["photo_width"]]), int(p[col["photo_height"]])
        except ValueError:
            continue
        if p[col["photo_featured"]] != "t" or min(w, h) < min_side:
            continue
        try:
            views = int(p[col["stats_views"]] or 0)
        except ValueError:
            views = 0
        out.append({"id": p[col["photo_id"]], "url": p[col["photo_image_url"]] + IMAGE_PARAMS, "w": w, "h": h, "views": views})
    out.sort(key=lambda d: (-d["views"], d["id"]))
    return out


def fetch_one(session: requests.Session, item: dict, out_dir: Path, retries: int = 3) -> dict | None:
    dest = out_dir / f"{item['id']}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return {"id": item["id"], "file": dest.name, "skipped": True, **{k: item[k] for k in ("url", "w", "h")}}
    for _ in range(retries):
        try:
            r = session.get(item["url"], timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return {"id": item["id"], "file": dest.name, "skipped": False, **{k: item[k] for k in ("url", "w", "h")}}
        except requests.RequestException:
            continue
    return None


def download(count: int, out_dir: Path, seed: int = 7, workers: int = 4) -> list[dict]:
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_candidates()
    head = pool[: max(count * 10, 200)]
    rnd = random.Random(seed)
    rnd.shuffle(head)
    chosen = head[:count]
    results: list[dict] = []
    with requests.Session() as session, ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(lambda it: fetch_one(session, it, out_dir), chosen):
            if rec is not None:
                results.append(rec)
    manifest = out_dir / "manifest.jsonl"
    with io.open(manifest, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", type=str, default="Data/gt/dev")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    recs = download(args.count, root / args.out, seed=args.seed, workers=args.workers)
    print(f"downloaded={sum(1 for r in recs if not r['skipped'])} skipped={sum(1 for r in recs if r['skipped'])} failed={args.count - len(recs)} -> {args.out}")


if __name__ == "__main__":
    main()
