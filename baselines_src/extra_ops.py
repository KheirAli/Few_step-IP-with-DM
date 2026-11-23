"""Operators added to the DPS and DAPS clones so they can be run on the tasks their repositories do not ship (2026-11-18).
All take and return images in [-1, 1] (as the two code bases do).  Non-differentiable steps get a surrogate gradient, because both
methods need a gradient through the operator: rounding passes the gradient straight through, the sign is replaced by a smooth tanh."""
import sys, torch
import torch.nn.functional as F
PNP = "/usr/scratch/akheirandish3/PNP/PnP-CM"
_C = {}


def jpeg5(x):                                                  # JPEG at quality 5, differentiable codec from the PnP-CM repo
    if PNP not in sys.path:
        sys.path.append(PNP)
    import functions.diff_jpeg as dj
    q = torch.tensor([5.0], device=x.device).expand(x.shape[0])
    return dj.diff_jpeg_coding(image_rgb=(x + 1) / 2 * 255, jpeg_quality=q) / 255 * 2 - 1


def quant8(x):                                                 # 3-bit quantisation, straight-through gradient
    q = (torch.floor((x.clamp(-1.0, 1.0 - 1e-6) + 1) * 4) + 0.5) / 4 - 1
    return x + (q - x).detach()


def onebit(x):                                                 # one bit per pixel against a known random threshold
    if "tau" not in _C:
        _C["tau"] = 2 * torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(8000)) - 1
    tau = _C["tau"].to(x.device)
    return torch.tanh(5 * (x - tau)) if x.requires_grad else torch.sign(x - tau)


def hdr(x):
    return torch.clip(2 * x, -1, 1)


def pr_masked(x):                                              # PnP-CM's phase retrieval: random +-1 mask keeping 1 pixel in 4, 512x512 frame
    if "mod" not in _C:
        g = torch.randint(0, 8, (256, 256), generator=torch.Generator().manual_seed(4321)); _C["mod"] = (-(g == 0).long() + (g == 7).long()).float()
    p = F.pad(x * _C["mod"].to(x.device), (128, 128, 128, 128))
    Fx = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(p, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    return (Fx.real ** 2 + Fx.imag ** 2 + 1e-12).sqrt()


# ---- added 2026-11-20: four more separable operators, identical to mode_probe/unified.py ----
import math as _math


def _gblur(x):                                                 # circular Gaussian blur, std 3 pixels, in the Fourier domain
    k = ("H", x.shape[-1], str(x.device))
    if k not in _C:
        f = torch.fft.fftfreq(x.shape[-1], device=x.device); _C[k] = torch.exp(-2 * _math.pi ** 2 * 9.0 * (f[:, None] ** 2 + f[None, :] ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * _C[k]).real


def gamma04(x):                                                # tone curve: intensity in [0,1] to the power 0.4
    return 2 * ((x.clamp(-1.0, 1.0) + 1) / 2 + 1e-4) ** 0.4 - 1


def wrap2(x):                                                  # wrap-around sensor
    return torch.remainder(2 * x + 1, 2.0) - 1


def blurclip(x):
    return (2 * _gblur(x)).clamp(-1.0, 1.0)


def blurq(x):
    b = _gblur(x); q = (torch.floor((b.clamp(-1.0, 1.0 - 1e-6) + 1) * 4) + 0.5) / 4 - 1
    return b + (q - b).detach()


def _radon(x, K):                                              # sparse-view parallel-beam CT, identical to mode_probe/unified.py (any batch size)
    out = []
    for xb in x:
        xp = F.pad(xb[None], (54, 54, 54, 54)).expand(K, -1, -1, -1); th = torch.arange(K, device=x.device).float() * _math.pi / K
        M = torch.stack([torch.stack([th.cos(), -th.sin(), torch.zeros_like(th)], 1), torch.stack([th.sin(), th.cos(), torch.zeros_like(th)], 1)], 1)
        rot = F.grid_sample(xp, F.affine_grid(M, list(xp.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
        out.append(rot.sum(dim=2).permute(1, 0, 2) / 16.0)
    return torch.stack(out)


def ct60(x):
    return _radon(x, 60)


def ct30(x):
    return _radon(x, 30)


def ctnl60(x):                                                  # nonlinear CT (Beer-Lambert), identical to mode_probe/unified.py NEWOP=ctnl: gain 10, alpha 0.25
    return 10.0 * (1.0 - torch.exp(-0.25 * _radon((x + 1) / 2, 60)))


def ctnl30(x):
    return 10.0 * (1.0 - torch.exp(-0.25 * _radon((x + 1) / 2, 30)))


OPS = {"jpeg5": jpeg5, "quant8": quant8, "onebit": onebit, "hdr2": hdr, "pr_masked": pr_masked,
       "gamma04": gamma04, "wrap2": wrap2, "blurclip": blurclip, "blurq": blurq, "ct60": ct60, "ct30": ct30, "ctnl60": ctnl60, "ctnl30": ctnl30}
