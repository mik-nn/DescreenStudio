from __future__ import annotations

from dataclasses import dataclass

import lpips
import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM


@dataclass(frozen=True)
class LossConfig:
    w_l1: float = 1.0
    w_msssim: float = 0.5
    w_fourier: float = 0.1
    w_lpips: float = 0.1
    lpips_net: str = "alex"  # "alex" | "vgg" (LPIPS-VGG: более тяжёлый perceptual)
    w_edge: float = 0.0  # edge-weighted L1 (run4: residual screen lives on edges)
    edge_alpha: float = 2.0  # weight = 1 + alpha * |grad_target| / mean(|grad|)
    w_tone: float = 0.0  # per-channel mean preservation (run4: fixes +0.10 brightening)
    l1_type: str = "l1"  # "l1" | "charbonnier" (sqrt(x^2+eps^2): гладкий в нуле, робастный)
    charb_eps: float = 1e-3
    focal_gamma: float = 0.0  # focal-frequency: вес |fp-ft|^gamma (0 = обычный Fourier_Mag)
    w_aux: float = 0.0  # deep supervision: вес aux-головы 1/2 разрешения (0 = выкл)
    w_grad: float = 0.0  # Sobel-градиент L1: привязка ГРАНИЦ pred к GT (против извилистых контуров)


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 2:] - x[:, :, :, :-2]
    dy = x[:, :, 2:, :] - x[:, :, :-2, :]
    g = torch.sqrt(dx[:, :, 1:-1, :] ** 2 + dy[:, :, :, 1:-1] ** 2 + 1e-12)
    pad = torch.nn.functional.pad(g, (1, 1, 1, 1), mode="replicate")
    return pad.mean(dim=1, keepdim=True)


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """Вектор градиента (Sobel 3x3, norm=1) по каждому каналу: (B,C,H,W)."""
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.t().contiguous()
    c = x.shape[1]
    kx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = torch.nn.functional.conv2d(x, kx, padding=1, groups=c)
    gy = torch.nn.functional.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


class CompositeLoss(nn.Module):
    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = lpips.LPIPS(net=self.cfg.lpips_net)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = pred - target
        if self.cfg.l1_type == "charbonnier":
            return torch.mean(torch.sqrt(d * d + self.cfg.charb_eps ** 2))
        return torch.mean(torch.abs(d))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                aux: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        loss_l1 = self.pixel_loss(pred, target)
        loss_ms = 1.0 - self.ms_ssim(pred, target)
        fp = torch.abs(torch.fft.rfft2(pred, norm="ortho"))
        ft = torch.abs(torch.fft.rfft2(target, norm="ortho"))
        diff = torch.abs(fp - ft)
        if self.cfg.focal_gamma > 0:
            with torch.no_grad():
                w = (diff / (diff.max().detach() + 1e-6)) ** self.cfg.focal_gamma
            loss_fourier = torch.mean(w * diff / (ft.mean() + 1e-6))
        else:
            loss_fourier = torch.mean(diff / (ft.mean() + 1e-6))
        if self.cfg.w_lpips > 0:
            loss_lpips = self.lpips(2.0 * pred - 1.0, 2.0 * target - 1.0).mean()
        else:
            loss_lpips = torch.zeros((), device=pred.device)
        if self.cfg.w_edge > 0:
            g = _grad_mag(target.detach())
            w = 1.0 + self.cfg.edge_alpha * g / (g.mean() + 1e-6)
            loss_edge = (w * torch.abs(pred - target)).mean() / w.mean()
        else:
            loss_edge = torch.zeros((), device=pred.device)
        if self.cfg.w_tone > 0:
            loss_tone = torch.mean(
                torch.abs(pred.mean(dim=(0, 2, 3)) - target.mean(dim=(0, 2, 3)))
            )
        else:
            loss_tone = torch.zeros((), device=pred.device)
        if self.cfg.w_grad > 0:
            loss_grad = torch.mean(torch.abs(
                _sobel_mag(pred) - _sobel_mag(target.detach())
            ))
        else:
            loss_grad = torch.zeros((), device=pred.device)
        total = (
            self.cfg.w_l1 * loss_l1
            + self.cfg.w_msssim * loss_ms
            + self.cfg.w_fourier * loss_fourier
            + self.cfg.w_lpips * loss_lpips
            + self.cfg.w_edge * loss_edge
            + self.cfg.w_tone * loss_tone
            + self.cfg.w_grad * loss_grad
        )
        aux_out = torch.zeros((), device=pred.device)
        if aux is not None and self.cfg.w_aux > 0:
            t_half = torch.nn.functional.avg_pool2d(target.detach(), 2)
            aux_out = self.pixel_loss(aux, t_half)
            total = total + self.cfg.w_aux * aux_out
        return {
            "total": total,
            "l1": loss_l1.detach(),
            "ms_ssim": loss_ms.detach(),
            "fourier": loss_fourier.detach(),
            "lpips": loss_lpips.detach(),
            "edge": loss_edge.detach(),
            "tone": loss_tone.detach(),
            "grad": loss_grad.detach(),
            "aux": aux_out.detach(),
        }
