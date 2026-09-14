from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

from ml.model import FourierUNet

ROOT = Path(__file__).resolve().parents[1]


def build_champion(device: str = "cpu") -> FourierUNet:
    m = FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                    dec_blocks=(1, 1), spectral=(False, True, True))
    ck = torch.load(ROOT / "ml" / "train" / "_run3" / "best.pt", map_location="cpu", weights_only=False)
    src = ck["ema"] if ck.get("ema") else ck["model"]
    m.load_state_dict({k: v.float() for k, v in src.items()})
    return m.eval().to(device)


def export(out: Path, opset: int = 17) -> Path:
    m = build_champion("cpu")
    dummy = torch.rand(1, 3, 512, 512)
    torch.onnx.export(
        m, dummy, str(out), export_params=True, opset_version=opset,
        do_constant_folding=True, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
    )
    model = onnx.load(str(out))
    onnx.save(model, str(out))
    print("exported fp32:", out, round(out.stat().st_size / 1e6, 1), "MB")
    return out


def to_fp16(src: Path, dst: Path) -> Path:
    model = onnx.load(str(src))
    fp16 = float16.convert_float_to_float16(
        model, keep_io_types=True,
        op_block_list=["DFT", "DFT2D"] if any(n.op_type in ("DFT", "DFT2D") for n in model.graph.node) else None,
    )
    onnx.save(fp16, str(dst))
    print("exported fp16:", dst, round(dst.stat().st_size / 1e6, 1), "MB")
    return dst


def verify(path: Path, providers: list[str], sizes: tuple[int, ...] = (512, 256)) -> dict:
    m = build_champion("cuda" if "CUDAExecutionProvider" in providers else "cpu")
    sess = ort.InferenceSession(str(path), providers=providers)
    in_type = sess.get_inputs()[0].type
    rep = {}
    with torch.no_grad():
        for s in sizes:
            x = torch.rand(1, 3, s, s)
            ref = m(x.cuda() if "CUDAExecutionProvider" in providers else x).cpu().numpy()
            feed = x.numpy().astype(np.float16 if "float16" in in_type else np.float32)
            out = sess.run(["output"], {"input": feed})[0].astype(np.float32)
            err = float(np.abs(ref - out).max())
            rep[s] = round(err, 6)
            print(f"{path.name} {s}x{s} max_abs_err={err:.2e}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/export/fourier_descreen_unet.onnx")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = export(out)
    print("verify fp32/cpu:", verify(fp32, ["CPUExecutionProvider"], (512,)))
    print("verify fp32/cuda:", verify(fp32, ["CUDAExecutionProvider", "CPUExecutionProvider"], (512,)))
    fp16 = to_fp16(fp32, out.with_suffix(".fp16.onnx"))
    print("verify fp16/cuda:", verify(fp16, ["CUDAExecutionProvider", "CPUExecutionProvider"]))


if __name__ == "__main__":
    main()
