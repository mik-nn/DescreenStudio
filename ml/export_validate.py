from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from ml.export_onnx import build_champion
from ml.train.data import DescreenDataset
from pytorch_msssim import ssim as pt_ssim

ROOT = Path(__file__).resolve().parents[1]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float((a - b).square().mean())
    return 10 * np.log10(1 / max(mse, 1e-12))


def run_ort(path: Path, scan, providers):
    sess = ort.InferenceSession(str(path), providers=providers)
    out = sess.run(["output"], {"input": scan.numpy()})[0]
    return torch.from_numpy(np.clip(out, 0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="ml/export/fourier_descreen_unet.onnx")
    ap.add_argument("--onnx-fp16", default="ml/export/fourier_descreen_unet.fp16.onnx")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    ds = DescreenDataset(ROOT / "Data" / "synth", "val")
    indices = np.random.default_rng(0).choice(len(ds), size=min(args.n, len(ds)), replace=False)
    model = build_champion("cuda").cuda()

    rows = []
    for k in indices:
        scan, gt, stem = ds[k]
        x, g = scan.unsqueeze(0).cuda(), gt.unsqueeze(0).cuda()
        with torch.no_grad():
            t = model(x)
        torch_cuda = torch.clip(t, 0, 1).cpu()
        ort_cpu = run_ort(ROOT / args.onnx, scan.unsqueeze(0), ["CPUExecutionProvider"])
        row = {"id": stem}
        for name, pred in (("torch_cuda", torch_cuda), ("ort_fp32_cpu", ort_cpu)):
            row[f"{name}_psnr"] = round(psnr(pred, gt.unsqueeze(0)), 2)
            row[f"{name}_ssim"] = round(float(pt_ssim(pred, gt.unsqueeze(0), data_range=1.0, size_average=True)), 4)
        rows.append(row)
        print(f"{stem} torch cuda psnr={row['torch_cuda_psnr']} ssim={row['torch_cuda_ssim']} | "
              f"ort fp32 psnr={row['ort_fp32_cpu_psnr']} ssim={row['ort_fp32_cpu_ssim']}", flush=True)

    for metric in ("psnr", "ssim"):
        for col in ("torch_cuda", "ort_fp32_cpu"):
            vals = [r[f"{col}_{metric}"] for r in rows]
            print(f"mean {metric} {col}: {sum(vals) / len(vals):.4f}")
    print("diff psnr ort-torch:", round(sum(r['ort_fp32_cpu_psnr'] - r['torch_cuda_psnr'] for r in rows) / len(rows), 3))


if __name__ == "__main__":
    main()