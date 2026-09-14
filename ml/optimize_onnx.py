from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer

from ml.export_onnx import build_champion

ROOT = Path(__file__).resolve().parents[1]


def count_nodes(model: onnx.ModelProto) -> int:
    return len(model.graph.node)


def verify_equiv(orig: Path, opt: Path) -> dict:
    import numpy as np
    o1 = ort.InferenceSession(str(orig), providers=["CPUExecutionProvider"])
    o2 = ort.InferenceSession(str(opt), providers=["CPUExecutionProvider"])
    errs = []
    rng = np.random.default_rng(1)
    for s in (224, 512, 768):
        x = rng.random((1, 3, s, s), dtype=np.float32)
        a = o1.run(["output"], {"input": x})[0]
        b = o2.run(["output"], {"input": x})[0]
        errs.append(float(np.abs(a - b).max()))
    return {s: e for s, e in zip((224, 512, 768), errs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="ml/export/fourier_descreen_unet.onnx")
    args = ap.parse_args()
    src = ROOT / args.onnx
    dst = src.with_name(src.stem + ".opt.onnx")
    before = onnx.load(str(src))
    opt = optimizer.optimize_model(
        str(src), model_type="bert",
        num_heads=0, hidden_size=0,
        opt_level=99,
        use_gpu=False, only_onnxruntime=False,
        provider="CPUExecutionProvider",
        verbose=False,
    )
    opt.save_model_to_file(str(dst))
    after = onnx.load(str(dst))
    print(f"nodes: {count_nodes(before)} -> {count_nodes(after)}")
    print(f"size: {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    ep = sess.get_providers()
    print("opt loads on:", ep)
    errs = verify_equiv(src, dst)
    for s, e in errs.items():
        print(f"max_abs_err {s}: {e:.2e}")


if __name__ == "__main__":
    main()