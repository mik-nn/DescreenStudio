from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierSpectralBlock(nn.Module):
    def __init__(self, channels: int, zero_init: bool = False):
        super().__init__()
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        # zero-init (как в Res/Norm-вариантах): spectral-ветвь стартует с нуля,
        # gain нарастает постепенно вместо взрыва на случайной инициализации
        if zero_init:
            _zero_init_last(self.conv_real)
            _zero_init_last(self.conv_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            spec = torch.fft.rfft2(xf, norm="ortho")
            real, imag = spec.real, spec.imag
            out_real = self.conv_real(real) - self.conv_imag(imag)
            out_imag = self.conv_real(imag) + self.conv_imag(real)
            m = out_real.shape[-1]  # w//2 + 1
            # Full-length 2D Hermitian spectrum in real domain; identical to the
            # training-time onesided irfft2 (conjugation mirrors BOTH axes, with
            # the DC row as its own partner), but valid for ONNX inverse DFT.
            # Mirrors use index_select (positive dynamic indices) so DirectML
            # can execute them (Slice with negative steps is CPU-only).
            if w % 2 == 0:
                src_re, src_im = out_real[..., 1:m - 1], out_imag[..., 1:m - 1]
            else:
                src_re, src_im = out_real[..., 1:], out_imag[..., 1:]
            l = src_re.shape[-1]
            iw = (l - 1) - torch.arange(l, device=src_re.device)
            ih = h - torch.arange(h, device=src_re.device)
            ih = torch.where(ih == h, torch.zeros_like(ih), ih)
            mir_re = src_re.index_select(-1, iw).index_select(-2, ih)
            mir_im = (-src_im).index_select(-1, iw).index_select(-2, ih)
            full_re = torch.cat([out_real, mir_re], dim=-1)
            full_im = torch.cat([out_imag, mir_im], dim=-1)
            full = torch.complex(full_re, full_im)
            out = torch.fft.ifft2(full, norm="ortho").real
        return out.to(x.dtype)


class FFCBlock(nn.Module):
    def __init__(self, channels: int, use_spectral: bool = True):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.spectral = FourierSpectralBlock(channels) if use_spectral else nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        f = self.spectral(x)
        return self.fuse(torch.cat([s, f], dim=1)) + x


def _zero_init_last(conv_seq: nn.Sequential) -> None:
    """Identity at init: residual branch starts at zero."""
    last = conv_seq[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class FourierSpectralBlockRes(nn.Module):
    """Residual spectral block (zero-init): out = spec + conv_stack(spec).

    Ответ на вопрос «Residual-блоки в спектре»: ветвь стартует с тождества,
    сеть учит только поправку спектра. Пары свёрток общие с plain-вариантом.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        _zero_init_last(self.conv_real)
        _zero_init_last(self.conv_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            spec = torch.fft.rfft2(xf, norm="ortho")
            real, imag = spec.real, spec.imag
            out_real = real + (self.conv_real(real) - self.conv_imag(imag))
            out_imag = imag + (self.conv_real(imag) + self.conv_imag(real))
            out = _hermitian_ifft(out_real, out_imag, h, w)
        return out.to(x.dtype)


class FourierSpectralBlockNorm(nn.Module):
    """Spectral block с контрастной предобработкой: real/imag стандартизуются
    по частотной решётке (per-sample, per-channel), затем residual-поправка
    (zero-init). Слабые пики получают тот же динамический диапазон, что и
    сильные — аналог top-hat контраста детектора A0, но обучаемый.
    Глобальная энергия спектра проходит через тождественный остаток.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        _zero_init_last(self.conv_real)
        _zero_init_last(self.conv_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            spec = torch.fft.rfft2(xf, norm="ortho")
            real, imag = spec.real, spec.imag
            mu_r = real.mean(dim=(-2, -1), keepdim=True)
            sd_r = real.std(dim=(-2, -1), keepdim=True)
            mu_i = imag.mean(dim=(-2, -1), keepdim=True)
            sd_i = imag.std(dim=(-2, -1), keepdim=True)
            re = (real - mu_r) / (sd_r + self.eps)
            im = (imag - mu_i) / (sd_i + self.eps)
            out_real = real + (self.conv_real(re) - self.conv_imag(im))
            out_imag = imag + (self.conv_real(im) + self.conv_imag(re))
            out = _hermitian_ifft(out_real, out_imag, h, w)
        return out.to(x.dtype)


def _hermitian_ifft(out_real: torch.Tensor, out_imag: torch.Tensor,
                    h: int, w: int) -> torch.Tensor:
    """Полный эрмитов спектр -> вещественная область (ONNX-валидно)."""
    m = out_real.shape[-1]  # w//2 + 1
    if w % 2 == 0:
        src_re, src_im = out_real[..., 1:m - 1], out_imag[..., 1:m - 1]
    else:
        src_re, src_im = out_real[..., 1:], out_imag[..., 1:]
    l = src_re.shape[-1]
    iw = (l - 1) - torch.arange(l, device=src_re.device)
    ih = h - torch.arange(h, device=src_re.device)
    ih = torch.where(ih == h, torch.zeros_like(ih), ih)
    mir_re = src_re.index_select(-1, iw).index_select(-2, ih)
    mir_im = (-src_im).index_select(-1, iw).index_select(-2, ih)
    full_re = torch.cat([out_real, mir_re], dim=-1)
    full_im = torch.cat([out_imag, mir_im], dim=-1)
    full = torch.complex(full_re, full_im)
    return torch.fft.ifft2(full, norm="ortho").real


class LayerNorm2d(nn.Module):
    """LayerNorm по каналам на пиксель (как в NAFNet)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet-блок (Chen et al., ICCV'23) — SOTA-эффективная реставрация
    без тяжёлого attention. dw-conv + SimpleGate + SCA + FFN, оба residual
    с zero-init beta/gamma. Кандидат на замену spatial-части FFCBlock."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw, kernel_size=1)
        self.conv2 = nn.Conv2d(dw, dw, kernel_size=3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw // 2, channels, kernel_size=1)
        self.norm2 = LayerNorm2d(channels)
        ff = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ff, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ff // 2, channels, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        y = self.norm2(x)
        y = self.conv5(self.sg2(self.conv4(y)))
        return x + y * self.gamma


class FFCBlockV2(nn.Module):
    """FFCBlock с выбираемой spectral-ветвью (plain/res/norm/local)."""
    def __init__(self, channels: int, use_spectral: bool = True,
                 spectral_mode: str = "plain", ff_norm: bool = False,
                 ff_zero_init: bool = False):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if not use_spectral:
            self.spectral: nn.Module = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        elif spectral_mode == "plain":
            self.spectral = FourierSpectralBlock(channels, zero_init=ff_zero_init)
        elif spectral_mode == "res":
            self.spectral = FourierSpectralBlockRes(channels)
        elif spectral_mode == "norm":
            self.spectral = FourierSpectralBlockNorm(channels)
        elif spectral_mode == "local":
            self.spectral = FourierSpectralBlockLocal(channels)
        else:
            raise ValueError(spectral_mode)
        # Пропущенная нормировка (2026-10-02, урок run12): без неё pile-up
        # в spectral-ветви (bottleneck maxabs 110 vs 766, ups.1 25 vs 2610)
        # роняет выход при сдвиге весов 0.023. LayerNorm структурно капает
        # дисперсию независимо от фаз; выкл по умолчанию (обратная совместимость).
        self.norm_f: nn.Module = LayerNorm2d(channels) if ff_norm else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        f = self.norm_f(self.spectral(x))
        return self.fuse(torch.cat([s, f], dim=1)) + x


class MDTA(nn.Module):
    """Multi-Dconv Transposed Attention (Restormer, CVPR'22) — «щепотка ViT».

    Attention поперёк каналов: q,k,v (n, heads, t, HW) -> attn-map (t, t),
    а не (HW, HW) — поэтому на 128² bottleneck (16384 позиций) это дёшево
    (~5 GFLOP), в отличие от обычного spatial self-attention.
    Масштаб — обучаемая температура per-head.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.dw = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                            padding=1, groups=channels * 3)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            qkv = self.dw(self.qkv(xf)).reshape(n, 3, self.heads, c // self.heads, h * w)
            q, k, v = qkv.unbind(1)  # (n, heads, t, HW)
            # fp32 + кламп: без клампа логиты ~±230 при случайной инициализации
            # -> one-hot softmax -> градиент температуры переполняется (NaN в AMP)
            logits = ((q @ k.transpose(-2, -1)) / (h * w) ** 0.5).clamp(-30.0, 30.0)
            attn = torch.softmax(logits * self.temperature, dim=-1)
            out = (attn @ v).reshape(n, c, h, w)
            out = self.proj(out)
        return out.to(x.dtype)


class TransformerBottle(nn.Module):
    """Пара MDTA + GDFN (gated dwconv FFN) поверх bottleneck-фичей."""

    def __init__(self, channels: int, heads: int = 4, ffn_expand: int = 2):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, heads=heads)
        self.norm2 = LayerNorm2d(channels)
        ff = channels * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ff, kernel_size=1),
            nn.Conv2d(ff, ff, kernel_size=3, padding=1, groups=ff),
            SimpleGate(),
            nn.Conv2d(ff // 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class FourierSpectralBlockLocal(nn.Module):
    """Локальный адаптивный контраст в спектре: real/imag делятся на локальное
    среднее магнитуды (avg_pool 9x9), а не на глобальную std. Аналог top-hat
    детектора A0, но плотный и обучаемый; слабые боковые полосы растра рядом
    с сильным фундаменталом получают контраст. Residual + zero-init.
    """

    def __init__(self, channels: int, win: int = 9, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.pad = win // 2
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        _zero_init_last(self.conv_real)
        _zero_init_last(self.conv_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()
            spec = torch.fft.rfft2(xf, norm="ortho")
            real, imag = spec.real, spec.imag
            mag = torch.sqrt(real * real + imag * imag + self.eps)
            loc = F.avg_pool2d(mag, kernel_size=2 * self.pad + 1, stride=1, padding=self.pad)
            re = real / (loc + self.eps)
            im = imag / (loc + self.eps)
            out_real = real + (self.conv_real(re) - self.conv_imag(im))
            out_imag = imag + (self.conv_real(im) + self.conv_imag(re))
            out = _hermitian_ifft(out_real, out_imag, h, w)
        return out.to(x.dtype)
