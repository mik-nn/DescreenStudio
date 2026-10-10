"""Numpy-эмуляция FFCBlock(bottleneck.0) для сверки с Rust (block_test.rs).

Воспроизводит ровно то же вычисление, что и PyTorch-эталон ffblock.py:
  s = conv3x3(x); f = spectral(x); out = gelu(conv1x1(cat[s,f])) + x
Эталон Rust: mean=2.142724 std=2.178113
"""
import json
import struct

import numpy as np

ST = r"F:\DescreenStudioPro\ml\train\_run3\best.safetensors"


def load_safetensors(path):
    with open(path, "rb") as f:
        buf = f.read()
    hlen = struct.unpack_from("<Q", buf, 0)[0]
    header = json.loads(buf[8 : 8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        shape = meta["shape"]
        n = int(np.prod(shape))
        data = np.frombuffer(buf, dtype="<f4", count=n, offset=base + s).astype(np.float64)
        out[name] = data.reshape(shape)
    return out


def gelu(x):
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def conv1x1(x, w, b):
    # x: [C,H,W], w: [Cout,Cin,1,1]
    return np.einsum("chw,oc->ohw", x, w[:, :, 0, 0]) + b[:, None, None]


def conv3x3(x, w, b):
    # padding=1
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    H, W = x.shape[1], x.shape[2]
    out = np.empty((w.shape[0], H, W))
    for i in range(H):
        for j in range(W):
            patch = xp[:, i : i + 3, j : j + 3]  # [C,3,3]
            out[:, i, j] = np.einsum("ckl,ockl->o", patch, w) + b
    return out


def spectral(x, st, pfx):
    h, w = x.shape[1], x.shape[2]
    spec = np.fft.rfft2(x, norm="ortho")  # [C,h,w//2+1]
    re, im = spec.real, spec.imag

    def branch(re_or_im, tag):
        a = conv1x1(re_or_im, st[f"{pfx}.spectral.conv_{tag}.0.weight"], st[f"{pfx}.spectral.conv_{tag}.0.bias"])
        a = gelu(a)
        return conv1x1(a, st[f"{pfx}.spectral.conv_{tag}.2.weight"], st[f"{pfx}.spectral.conv_{tag}.2.bias"])

    out_re = branch(re, "real") - branch(im, "imag")
    out_im = branch(im, "real") + branch(re, "imag")

    m = out_re.shape[-1]
    if w % 2 == 0:
        src_re, src_im = out_re[..., 1 : m - 1], out_im[..., 1 : m - 1]
    else:
        src_re, src_im = out_re[..., 1:], out_im[..., 1:]
    l = src_re.shape[-1]
    iw = (l - 1) - np.arange(l)
    ih = h - np.arange(h)
    ih = np.where(ih == h, 0, ih)
    mir_re = src_re[:, :, iw][:, ih, :]
    mir_im = (-src_im)[:, :, iw][:, ih, :]
    full_re = np.concatenate([out_re, mir_re], axis=-1)
    full_im = np.concatenate([out_im, mir_im], axis=-1)
    full = full_re + 1j * full_im
    return np.fft.ifft2(full, norm="ortho").real


def main():
    st = load_safetensors(ST)
    pfx = "bottleneck.0"

    # Тот же LCG, что в block_test.rs: seed>>33 / u32::MAX * 2 - 1, flat-заполнение
    seed = 0x12345678
    vals = np.empty(384 * 16 * 16)
    for i in range(vals.size):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        vals[i] = ((seed >> 33) / 4294967295.0) * 2.0 - 1.0
    x = vals.reshape(1, 384, 16, 16)[0]  # [C,H,W]

    f = spectral(x, st, pfx)
    s = conv3x3(x, st[f"{pfx}.spatial.weight"], st[f"{pfx}.spatial.bias"])
    cat = np.concatenate([s, f], axis=0)
    out = conv1x1(cat, st[f"{pfx}.fuse.0.weight"], st[f"{pfx}.fuse.0.bias"])
    out = gelu(out) + x

    print(f"numpy  out: mean={out.mean():.6f} std={out.std():.6f} first16={np.round(out.flatten()[:16], 7).tolist()}")
    print("эталон Rust: mean=2.142724 std=2.178113 first16=[-0.23884526, 0.5265521, -0.6964287, ...]")


if __name__ == "__main__":
    main()
