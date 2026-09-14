from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import NMF, PCA

from ml.simulator.color import rgb_to_cmyk

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ["c", "m", "y", "k"]
EPS = 1e-6


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_density(scan: np.ndarray) -> np.ndarray:
    lin = np.clip(srgb_to_linear(scan), EPS, 1.0)
    return (-np.log(lin)).astype(np.float32)


def corr_matrix(sep: np.ndarray, plates: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    h, w, cs = sep.shape
    out = {}
    for i in range(cs):
        x = sep[..., i].ravel()
        x = x - x.mean()
        row = {}
        for ch in CHANNELS:
            y = plates[ch].ravel().astype(np.float64)
            y = y - y.mean()
            row[ch] = round(float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)), 4)
        out[f"ch{i}"] = row
    return out


def isolation_db(corr: dict[str, dict[str, float]]) -> dict[str, float]:
    res = {}
    for ch in CHANNELS:
        vals = sorted((abs(corr[c][ch]), c) for c in corr)
        best, rest = vals[-1][0], [v for v, _ in vals[:-1]]
        num = best * best + 1e-6
        den = sum(v * v for v in rest) / max(len(rest), 1) + 1e-6
        res[ch] = round(float(10 * np.log10(num / den)), 2)
    return res


def fit_pca_global(dens: list[np.ndarray], rng: np.random.Generator) -> PCA:
    pool = []
    for d in dens:
        h, w, _ = d.shape
        idx = rng.choice(h * w, size=min(20000, h * w), replace=False)
        pool.append(d.reshape(-1, 3)[idx])
    pca = PCA(n_components=3, random_state=0)
    pca.fit(np.concatenate(pool, axis=0).astype(np.float64))
    return pca


def nmf_per_image(d: np.ndarray, k: int) -> np.ndarray:
    h, w, _ = d.shape
    m = NMF(n_components=k, init="nndsvda", max_iter=300, random_state=0)
    wgt = m.fit_transform(np.clip(d.reshape(-1, 3), 1e-6, None))
    comps = m.components_
    order = np.argsort(-wgt.mean(axis=0))
    return (wgt[:, order].reshape(h, w, k) / (wgt.max() + 1e-9)).astype(np.float32), comps[order]


def stability(h_list: list[np.ndarray]) -> float:
    base = h_list[0].reshape(-1, h_list[0].shape[-1])
    base = (base - base.mean(0)) / (base.std(0) + 1e-9)
    scores = []
    for h in h_list[1:]:
        cur = h.reshape(-1, h.shape[-1])
        cur = (cur - cur.mean(0)) / (cur.std(0) + 1e-9)
        c = np.abs(base.T @ cur) / base.shape[0]
        best = 0.0
        for perm in itertools.permutations(range(c.shape[1])):
            best = max(best, float(np.mean([c[i, perm[i]] for i in range(c.shape[1])])))
        scores.append(best)
    return round(float(np.mean(scores)), 3)


def isolation_report(plates_dir: Path, scans: list[np.ndarray], names: list[str]) -> dict:
    rng = np.random.default_rng(0)
    dens = [to_density(s) for s in scans]
    pca = fit_pca_global(dens, rng)
    report = {"images": {}, "nmf_stability": {}}
    h3 = []
    for s, d, name in zip(scans, dens, names):
        plates = {ch: np.load(plates_dir / f"{name}_{ch}.npy") for ch in CHANNELS}
        naive = np.stack([rgb_to_cmyk(s)[ch] for ch in CHANNELS], axis=-1)
        pc = pca.transform(d.reshape(-1, 3).astype(np.float64)).reshape(d.shape).astype(np.float32)
        w3, _ = nmf_per_image(d, 3)
        h3.append(w3)
        c0 = corr_matrix(s, plates)
        cn = corr_matrix(naive, plates)
        cp = corr_matrix(pc, plates)
        c3 = corr_matrix(w3, plates)
        report["images"][name] = {
            "rgb": isolation_db(c0),
            "naive_cmyk": isolation_db(cn),
            "pca": isolation_db(cp),
            "nmf3": isolation_db(c3),
        }
    report["nmf_stability"] = {"k3": stability(h3)}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="ml/analysis/_plates")
    ap.add_argument("--out", default="ml/analysis/_e0.json")
    args = ap.parse_args()
    root = ROOT / args.plates
    names = sorted(p.stem for p in (root / "scan").glob("*.png"))
    scans = [np.asarray(Image.open(root / "scan" / f"{n}.png").convert("RGB"), dtype=np.float32) / 255.0 for n in names]
    rep = isolation_report(root / "plates", scans, names)
    (ROOT / args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    methods = ["rgb", "naive_cmyk", "pca", "nmf3"]
    agg: dict[str, list[float]] = {m: [] for m in methods}
    for img in rep["images"].values():
        for m in methods:
            agg[m].append(sum(img[m].values()) / 4)
    for m in methods:
        print(m, "mean_isolation_dB:", round(sum(agg[m]) / len(agg[m]), 2))
    print("stability:", rep["nmf_stability"])


if __name__ == "__main__":
    main()
