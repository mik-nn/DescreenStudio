"""Полная numpy-эмуляция FourierUNet (конфиг из чекпоинта: base=96,
spectral=(False,True,True), bottleneck x3 complex) для сверки candle vs ONNX.
"""
import json
import struct
import sys

import numpy as np
from PIL import Image
from scipy.special import erf

ST = r"F:\DescreenStudioPro\ml\train\_run3\best.safetensors"
IMG = r"F:\DescreenStudioPro\ml\benchmark\_out\scan_a1.5_96.png"
OUT = r"F:\DescreenStudioPro\ml\benchmark\_out\numpy_a1.5_96.png"
ONNX_CMP = r"F:\DescreenStudioPro\ml\benchmark\_out\onnx_a1.5_96.png"
CANDLE_CMP = r"F:\DescreenStudioPro\ml\benchmark\_out\candle_a1.5_96.png"

if len(sys.argv) > 1:
    IMG = sys.argv[1]
    OUT = sys.argv[2]
    ONNX_CMP = sys.argv[3]
    CANDLE_CMP = sys.argv[4]


def load_st(path):
    buf = open(path, "rb").read()
    hlen = struct.unpack_from("<Q", buf, 0)[0]
    header = json.loads(buf[8 : 8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        shape = meta["shape"]
        out[name] = (
            np.frombuffer(buf, dtype="<f4", count=int(np.prod(shape)), offset=base + s)
            .astype(np.float32)
            .reshape(shape)
        )
    return out


def gelu(x):
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)


def conv(x, w, b, stride=1, pad=1):
    """x [C,H,W], w [O,C,k,k]."""
    C, H, W = x.shape
    O = w.shape[0]
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    out = np.zeros((O, Ho, Wo), dtype=np.float32)
    for ki in range(k):
        for kj in range(k):
            sl = xp[:, ki : ki + (Ho - 1) * stride + 1 : stride, kj : kj + (Wo - 1) * stride + 1 : stride]
            out += np.einsum("oc,cij->oij", w[:, :, ki, kj].astype(np.float32), sl.astype(np.float32))
    return out + b[:, None, None]


def convT(x, w, b, stride=2, pad=1):
    """Transposed conv: x [C,H,W], w [C,O,k,k] (torch layout)."""
    C, H, W = x.shape
    O = w.shape[1]
    k = w.shape[2]
    Ho, Wo = (H - 1) * stride - 2 * pad + k, (W - 1) * stride - 2 * pad + k
    y = np.zeros((O, Ho, Wo), dtype=np.float32)
    ii, jj = np.arange(Ho), np.arange(Wo)
    o_idx = np.arange(O)
    # Определение transposed conv: y[n] += x[(n-k+p)/s]·w[k] при (n-k+p)%s==0.
    for ki in range(k):
        si = ii - ki + pad
        mi = np.where((si % stride == 0) & (si // stride >= 0) & (si // stride < H))[0]
        for kj in range(k):
            sj = jj - kj + pad
            mj = np.where((sj % stride == 0) & (sj // stride >= 0) & (sj // stride < W))[0]
            if len(mi) == 0 or len(mj) == 0:
                continue
            xs = x[:, si[mi] // stride][:, :, sj[mj] // stride]
            contrib = np.einsum("co,cij->oij", w[:, :, ki, kj].astype(np.float32), xs.astype(np.float32))
            y[np.ix_(o_idx, mi, mj)] += contrib
    return y + b[:, None, None]


def conv1x1(x, w, b):
    return np.einsum("chw,oc->ohw", x, w[:, :, 0, 0].astype(np.float32)) + b[:, None, None]


def ffc(x, st, pfx, complex_):
    s = conv(x, st[f"{pfx}.spatial.weight"], st[f"{pfx}.spatial.bias"])
    if not complex_:
        f = conv(x, st[f"{pfx}.spectral.weight"], st[f"{pfx}.spectral.bias"])
    else:
        h, w = x.shape[1], x.shape[2]
        m = w // 2 + 1
        spec = np.fft.rfft2(x, norm="ortho")
        re, im = spec.real.astype(np.float32), spec.imag.astype(np.float32)

        def branch(v, tag):
            a = gelu(conv1x1(v, st[f"{pfx}.spectral.conv_{tag}.0.weight"], st[f"{pfx}.spectral.conv_{tag}.0.bias"]))
            return conv1x1(a, st[f"{pfx}.spectral.conv_{tag}.2.weight"], st[f"{pfx}.spectral.conv_{tag}.2.bias"])

        out_re = branch(re, "real") - branch(im, "imag")
        out_im = branch(im, "real") + branch(re, "imag")
        src_re, src_im = out_re[..., 1 : m - 1], out_im[..., 1 : m - 1]
        l = src_re.shape[-1]
        iw = (l - 1) - np.arange(l)
        ih = (h - np.arange(h)) % h
        mir_re = src_re[:, :, iw][:, ih, :]
        mir_im = (-src_im)[:, :, iw][:, ih, :]
        full = np.concatenate([out_re, mir_re], -1) + 1j * np.concatenate([out_im, mir_im], -1)
        f = np.fft.ifft2(full, norm="ortho").real.astype(np.float32)
    cat = np.concatenate([s, f], 0)
    return gelu(conv1x1(cat, st[f"{pfx}.fuse.0.weight"], st[f"{pfx}.fuse.0.bias"])) + x


def main():
    st = load_st(ST)
    img = np.asarray(Image.open(IMG).convert("RGB"), dtype=np.float32) / 255.0
    x = img.transpose(2, 0, 1)  # [3,96,96]
    stages = {}

    e = gelu(conv(x, st["inc.weight"], st["inc.bias"]))
    stages["inc"] = e
    e = ffc(e, st, "encs.0.0", complex_=False)
    stages["enc0"] = e
    skip0 = e
    e = gelu(conv(e, st["downs.0.op.0.weight"], st["downs.0.op.0.bias"], stride=2, pad=1))
    e = ffc(e, st, "encs.1.0", complex_=True)
    stages["enc1"] = e
    skip1 = e
    e = gelu(conv(e, st["downs.1.op.0.weight"], st["downs.1.op.0.bias"], stride=2, pad=1))
    for i in range(3):
        e = ffc(e, st, f"bottleneck.{i}", complex_=True)
    stages["bottleneck"] = e
    d = gelu(convT(e, st["ups.0.op.0.weight"], st["ups.0.op.0.bias"]))
    d = ffc(d + skip1, st, "decs.0.0", complex_=True)
    stages["dec0"] = d
    d = gelu(convT(d, st["ups.1.op.0.weight"], st["ups.1.op.0.bias"]))
    d = ffc(d + skip0, st, "decs.1.0", complex_=False)
    stages["dec1"] = d
    residual = conv(d, st["outc.weight"], st["outc.bias"])
    stages["residual"] = residual
    out = np.clip(x - residual, 0.0, 1.0)

    np.save(r"F:\DescreenStudioPro\ml\benchmark\_out\numpy_stages.npy", np.array([0.0]))
    for k, v in stages.items():
        print(f"stage {k}: mean={v.mean():.5f} std={v.std():.5f} first4={np.round(v.flatten()[:4],5).tolist()}")
    print(f"numpy out: mean={out.mean():.5f} std={out.std():.5f} first4={np.round(out.flatten()[:4],5).tolist()}")
    Image.fromarray((out.transpose(1, 2, 0) * 255).astype(np.uint8)).save(OUT)

    onnx = np.asarray(Image.open(ONNX_CMP).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    candle = np.asarray(Image.open(CANDLE_CMP).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    print(f"mean|numpy-onnx|   = {np.abs(out-onnx).mean():.5f}")
    print(f"mean|numpy-candle| = {np.abs(out-candle).mean():.5f}")
    print(f"mean|onnx-candle|  = {np.abs(onnx-candle).mean():.5f}")


if __name__ == "__main__":
    main()
