from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

from ml.model import FourierUNet
from torch.export import Dim as _Dim

ROOT = Path(__file__).resolve().parents[1]


def build_champion(device: str = "cpu") -> FourierUNet:
    m = FourierUNet(base=96, levels=3, enc_blocks=(1, 1), bottleneck_blocks=3,
                    dec_blocks=(1, 1), spectral=(False, True, True))
    ck = torch.load(ROOT / "ml" / "train" / "_run3" / "best.pt", map_location="cpu", weights_only=False)
    src = ck["ema"] if ck.get("ema") else ck["model"]
    m.load_state_dict({k: v.float() for k, v in src.items()})
    return m.eval().to(device)


def fix_inverse_dft(model: onnx.ModelProto) -> int:
    fixed = 0
    for n in model.graph.node:
        if n.op_type != "DFT":
            continue
        attrs = {a.name: a for a in n.attribute}
        inv, one = attrs.get("inverse"), attrs.get("onesided")
        if inv and int(inv.i) == 1 and one and int(one.i) == 1:
            one.i = 0
            fixed += 1
    return fixed


def export(out: Path, opset: int = 18) -> Path:
    m = build_champion("cpu")
    dummy = torch.rand(1, 3, 512, 512)
    hdim = _Dim("height", min=16, max=16384)
    wdim = _Dim("width", min=16, max=16384)
    dynamic_shapes = [{0: None, 2: hdim, 3: wdim}]
    torch.onnx.export(
        m, dummy, str(out), dynamo=True, export_params=True, opset_version=opset,
        do_constant_folding=True, input_names=["input"], output_names=["output"],
        dynamic_shapes=dynamic_shapes,
    )
    model = onnx.load(str(out))
    fixed = fix_inverse_dft(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    print(f"exported fp32: {out} {out.stat().st_size / 1e6:.1f} MB (inverse-DFT fixed: {fixed})")
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
    ap.add_argument("--fp16", action="store_true", help="also emit half-precision artifact")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = export(out)
    print("verify fp32/cpu:", verify(fp32, ["CPUExecutionProvider"], (512,)))
    if args.fp16:
        fp16 = to_fp16(fp32, out.with_suffix(".fp16.onnx"))
        print("verify fp16/cuda:", verify(fp16, ["CUDAExecutionProvider", "CPUExecutionProvider"]))


if __name__ == "__main__":
    main()
