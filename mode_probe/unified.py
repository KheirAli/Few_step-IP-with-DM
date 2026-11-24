"""One data-fit solver + one restart rule for every PnP-CM task (linear and nonlinear).

Outer loop: PnP-CM's own (ADMM-style split, one consistency-model call per step, repo schedules).
Data-fit subproblem at every step:  min_z  D(z) + (rho/2)||z - v||^2,   D = 0.5||A(z) - y||^2
  -> damped Gauss-Newton with matrix-free CG (rho is the damping).  No learning rate.
     CG stops at ||r|| <= ETA ||g||  (inexact-Newton forcing term), GN stops at ||g|| <= TOLG ||g_0||.
     The same rule and caps are used for all operators; for a linear A this is CG on the normal equations.
Restart rule (no threshold): after the pass, compare how far the output moved from the first image
  the sampler denoised (z_1) with the noise the sampler injected at that step (sigma_1).
  movement_rms > sigma_1  ->  the start was out of the noise's reach: run one more pass from the output.

MODE=unified | base        (base = the repo's solver, timed, same images/noise)
Run from the PnP-CM repo root:
  MODE=unified SHARD=0:25 python ../mode_probe/unified.py <variant> --deg=<task>
"""
import os, sys, time, json, math
import torch
import torch.nn as nn
torch.set_num_threads(4)  # avoid CPU oversubscription when many jobs share the machine

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("bkse"))
variant = sys.argv[1]
MODE = os.environ.get("MODE", "unified")
ETA = float(os.environ.get("ETA", "0.3")); TOLG = float(os.environ.get("TOLG", "0.1"))
KMAX = int(os.environ.get("KMAX", "10")); CGMAX = int(os.environ.get("CGMAX", "20"))
RESTART = os.environ.get("RESTART", "rule")        # rule | never | always
EXACT = os.environ.get("EXACT", "0") == "1"         # use the operator's closed-form proximal solve when it has one
HALF = os.environ.get("HALF", "")                   # "" | fl (first half, then last half) | il (interleaved halves)
RULE = os.environ.get("RULE", "noise")              # restart rule: noise | r2
EW = os.environ.get("EW", "0") == "1"               # Eisenstat-Walker adaptive forcing term instead of fixed ETA
STEPSTOP = float(os.environ.get("STEPSTOP", "0.05"))  # stop GN when the update (RMS) < STEPSTOP * noise injected next
SHARD = os.environ.get("SHARD", "0:10"); S0, S1E = (int(t) for t in SHARD.split(":"))
DATA = os.environ.get("DATA", "")
deg = [a.split("=")[1] for a in sys.argv if a.startswith("--deg=")][0]
LINEAR = deg in ("sr_bicubic", "deblur_gauss", "inpainting")

exp = os.path.abspath(f"../mode_probe/exp_{variant}_s{S0}")
os.makedirs(exp, exist_ok=True)
for sub, src in [("logs", "exp/logs"), ("datasets", DATA or "exp/datasets"), ("inpainting_masks", "exp/inpainting_masks")]:
    if not os.path.exists(f"{exp}/{sub}"):
        os.symlink(os.path.abspath(src), f"{exp}/{sub}")
# DSET=bedroom : LSUN bedroom (the consistency model PnP-CM uses for natural scenes); default CelebA-HQ
DSET = os.environ.get("DSET", "celeba"); DSET_PATH, DSET_CFG, DSET_CKPT = {"celeba": ("celeba_hq", "celeba_hq_256.yml", ""), "bedroom": ("bedroom", "lsun_bedroom_256.yml", "lsun_bedroom/cd_bedroom256_lpips.pt")}[DSET]
sys.argv = ["main.py"] + sys.argv[2:] + [f"--subset_start={S0}", f"--subset_end={S1E}", f"--path_y={DSET_PATH}",
                                         "--sigma_y=0.05", f"--config={DSET_CFG}", "--save_y", f"--exp={exp}"] + ([f"--model_ckpt={DSET_CKPT}"] if DSET_CKPT else [])

import functions.diff_jpeg as dj


class DiffJPEGAdapter:  # our reconstruction of the unreleased wrapper (see run_nl.py)
    def __init__(self, jpeg_qf, device):
        self.q = torch.tensor([float(jpeg_qf)], device=device)

    def A(self, x, **kwargs):
        return dj.diff_jpeg_coding(image_rgb=(x + 1) / 2 * 255, jpeg_quality=self.q.expand(x.shape[0])) / 255 * 2 - 1


dj.DiffJPEGCoding = DiffJPEGAdapter

# ---- NEWOP (2026-11-17): extra nonlinear tasks the repo does not ship.  Run with --deg=deblur_nl (its settings are borrowed for both
# the baseline and ours - nothing is tuned for the new task); the blur operator class is swapped for the one named here.
NEWOP = os.environ.get("NEWOP", "")


class _HDROp:
    """High dynamic range, as in DPS/DAPS: the image is scaled by 2 and clipped to the sensor range."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return (2 * x).clamp(-1.0, 1.0)
    forward = A


def _q8(x):
    return (torch.floor((x.clamp(-1.0, 1.0 - 1e-6) + 1) * 4) + 0.5) / 4 - 1


class _QuantOp:
    """3-bit quantisation: every pixel value replaced by the centre of one of 8 equal bins.  The gradient is passed straight
    through (so the repo's gradient loop can run at all); the VALUE is the true quantised image."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return x + (_q8(x) - x).detach()
    forward = A


def _dither(idx, like):
    g = torch.Generator().manual_seed(8000 + idx)
    return (2 * torch.rand(like.shape, generator=g) - 1).to(like.device)


class _OneBitOp:
    """One-bit sensing with a known random threshold per pixel (dithering): y = sign(x - tau), tau uniform in [-1, 1].
    Every pixel is reduced to ONE bit.  When a gradient is requested the sign is replaced by a smooth tanh, so the repo's
    gradient loop has something to follow; the measurement itself is always the true sign."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity(); self.idx = None

    def A(self, x, **k):
        if not x.requires_grad:
            self.tau = _dither(state["img"] + 1, x)
            return torch.sign(x - self.tau)
        return torch.tanh(5 * (x - self.tau))
    forward = A


def _smooth_field(idx, dev):
    g = torch.Generator().manual_seed(7000 + idx)
    f = torch.nn.functional.interpolate(torch.rand(1, 1, 5, 5, generator=g), size=(256, 256), mode="bicubic", align_corners=True)
    f = (f - f.min()) / (f.max() - f.min())
    return (0.25 + 0.65 * f).to(dev)


class _HazeOp:
    """Haze: y = t * x + (1 - t) * a, with a smooth transmission map t in [0.25, 0.9] and bright airlight a - neither told to the solver."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        self.t = _smooth_field(state["img"] + 1, x.device); self.a = torch.tensor([0.85, 0.85, 0.9], device=x.device).view(1, 3, 1, 1)
        return self.t * x + (1 - self.t) * self.a
    forward = A


def _blind_kernel(idx, dev):
    g = torch.Generator().manual_seed(9000 + idx)
    s1 = 1.5 + 2.0 * torch.rand(1, generator=g).item(); s2 = 0.7 + 0.8 * torch.rand(1, generator=g).item(); th = 3.14159 * torch.rand(1, generator=g).item()
    r = torch.arange(-10, 11).float(); yy, xx = torch.meshgrid(r, r, indexing="ij")
    a = xx * torch.cos(torch.tensor(th)) + yy * torch.sin(torch.tensor(th)); b = -xx * torch.sin(torch.tensor(th)) + yy * torch.cos(torch.tensor(th))
    k = torch.exp(-0.5 * (a / s1) ** 2 - 0.5 * (b / s2) ** 2); return (k / k.sum()).to(dev)


def _k2f(k):                                                   # 21x21 kernel -> 256x256 transfer function (circular)
    pad = torch.zeros(256, 256, device=k.device); pad[:21, :21] = k
    return torch.fft.fft2(torch.roll(pad, (-10, -10), dims=(0, 1)))


class _BlindOp:
    """Blind deblurring: each image is blurred by its own random elongated Gaussian kernel, which the solver is NOT told."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        self.k = _blind_kernel(state["img"] + 1, x.device)
        return torch.fft.ifft2(torch.fft.fft2(x) * _k2f(self.k)).real
    forward = A


class _PRDapsOp:
    """Phase retrieval exactly as in DPS/DAPS: image shifted to [0, 1], zero-padded by 64 px (384x384 frame), Fourier magnitudes,
    NO mask.  Returned x2 so that the wrapper's noise (0.1) equals their noise of 0.05 on the magnitudes."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        import functions.fft_util as _FU
        return 2 * _FU.fft2_m(torch.nn.functional.pad(x * 0.5 + 0.5, (64, 64, 64, 64))).abs()
    forward = A


PR_MASKS = int(os.environ.get("PR_MASKS", "2")); _PRM = {}


def _pr_masks(dev):
    """K fixed random phase masks (unit modulus).  T = stack_k F D_k / sqrt(K) satisfies T'T = I."""
    if dev not in _PRM:
        g_ = torch.Generator(device="cpu").manual_seed(777); _PRM[dev] = torch.exp(2j * math.pi * torch.rand(PR_MASKS, 1, 256, 256, generator=g_)).to(dev)
    return _PRM[dev]


class _PRSqOp:
    """Fourier magnitudes of the UNPADDED image (square unitary transform): the per-coefficient formula is the exact data-fit."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return 2 * torch.fft.fft2(x * 0.5 + 0.5, norm="ortho").abs()
    forward = A


class _PRCdpOp:
    """Coded diffraction: magnitudes of F(D_k x) for PR_MASKS random phase masks, placed side by side along the width."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        D = _pr_masks(x.device); return 2 * torch.cat([torch.fft.fft2(D[k_] * (x * 0.5 + 0.5), norm="ortho").abs() for k_ in range(PR_MASKS)], -1)
    forward = A


def priso_prox(y, coded):
    """Amplitude data-fit on an isometry.  Per coefficient: keep the phase, magnitude = (a + r|c|)/(1 + r) (exact minimiser of
    (1/2)(|u| - a)^2 + (r/2)|u - c|^2), then back with T' and the real part.  Exact image-space data-fit for the square transform."""
    amp = y / 2

    def prox(v, rho):
        g = v * 0.5 + 0.5; r4 = 4.0 * float(rho)
        if not coded:
            c = torch.fft.fft2(g, norm="ortho"); u = c / (c.abs() + 1e-8) * (amp + r4 * c.abs()) / (1 + r4)
            return torch.fft.ifft2(u, norm="ortho").real.clamp(0, 1) * 2 - 1
        D = _pr_masks(v.device); out = 0
        for k_ in range(PR_MASKS):
            c = torch.fft.fft2(D[k_] * g, norm="ortho"); a_ = amp[..., 256 * k_:256 * k_ + 256]; u = c / (c.abs() + 1e-8) * (a_ + r4 * c.abs()) / (1 + r4)
            out = out + (D[k_].conj() * torch.fft.ifft2(u, norm="ortho")).real
        return (out / PR_MASKS).clamp(0, 1) * 2 - 1
    return prox


def prd_setup(y):
    """Classical tools for unmasked phase retrieval.  Everything is FFTs: 'project' replaces the Fourier magnitudes of an image by
    the measured ones (keeping its phases) and then re-imposes what is known about the image: zero outside the 256x256 window
    and values in [0, 1].  hio() is Fienup's algorithm from a random start; refine() is a few plain projections from a given image."""
    import functions.fft_util as _FU
    amp = y / 2; S = torch.zeros(1, 1, 384, 384, device=y.device, dtype=torch.bool); S[..., 64:320, 64:320] = True

    def mag(g):
        G = _FU.fft2_m(g); return _FU.ifft2_m(amp * G / (G.abs() + 1e-8)).real

    def refine(img, k):                                        # img in [-1, 1], 256x256
        g = torch.nn.functional.pad((img * 0.5 + 0.5).clamp(0, 1), (64, 64, 64, 64))
        for _ in range(k):
            g = mag(g).clamp(0, 1) * S
        return g[..., 64:320, 64:320] * 2 - 1

    def hio(n_hio, n_er, beta=0.9):
        g = torch.rand_like(amp) * S
        for i in range(n_hio + n_er):
            gp = mag(g); ok = S & (gp >= 0)
            g = torch.where(ok, gp, g - beta * gp) if i < n_hio else gp.clamp(0, 1) * S
        res = ((_FU.fft2_m(g).abs() - amp) ** 2).sum()
        return g[..., 64:320, 64:320] * 2 - 1, float(res)
    def mean_init():
        # warm start from the PRIOR, not from the measurement alone: measured magnitudes combined with the Fourier phases of the
        # average face of 100 other images.  This fixes the orientation and gives the right coarse layout.
        m = torch.load(os.path.abspath("../mode_probe/" + os.environ.get("MEANFACE", "mean_face_gen.pt"))).to(y.device)
        Fm = _FU.fft2_m(torch.nn.functional.pad(m, (64, 64, 64, 64)))
        return (_FU.ifft2_m(amp * torch.exp(1j * torch.angle(Fm))).real.clamp(0, 1) * S)[..., 64:320, 64:320] * 2 - 1

    fr = torch.fft.fftshift(torch.fft.fftfreq(384, device=y.device)); rad = (fr[:, None] ** 2 + fr[None, :] ** 2).sqrt()

    def band_fit(img, cut, k):
        # coarse-to-fine: inside the band |f| < cut the measured magnitude is imposed with the image's current phase (a LINEAR
        # constraint once the phase is fixed); outside the band the image is left as the prior produced it.
        g = torch.nn.functional.pad((img * 0.5 + 0.5).clamp(0, 1), (64, 64, 64, 64)); inb = rad < cut
        for _ in range(k):
            G = _FU.fft2_m(g); G = torch.where(inb, amp * G / (G.abs() + 1e-8), G); g = _FU.ifft2_m(G).real.clamp(0, 1) * S
        return g[..., 64:320, 64:320] * 2 - 1
    def pca_init(kmax=256):
        # eigenface warm start: the image is restricted to mean + k directions of a face subspace built from samples of the
        # consistency model itself; the k coefficients are fitted to the measured magnitudes (L-BFGS, k grown 8 -> kmax).
        Bp = torch.load(os.path.abspath("../mode_probe/pca_cm.pt")); mu_, U_, var_ = Bp["mu"].to(y.device), Bp["U"].to(y.device), Bp["var"].to(y.device)
        c = torch.zeros(0, device=y.device)
        with torch.enable_grad():
            k = 8
            while k <= kmax:
                Uk = U_[:k].flatten(1); c = torch.cat([c, torch.zeros(k - c.numel(), device=y.device)]).detach().requires_grad_(True)
                opt = torch.optim.LBFGS([c], max_iter=150, history_size=30, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(); img = mu_ + (c @ Uk).view(1, 3, 256, 256)
                    loss = ((_FU.fft2_m(torch.nn.functional.pad(img, (64, 64, 64, 64))).abs() - amp) ** 2).mean() + 1e-4 * (c ** 2 / var_[:k]).mean()
                    loss.backward(); return loss
                opt.step(closure); k *= 2
        return (mu_ + (c.detach() @ U_[:c.numel()].flatten(1)).view(1, 3, 256, 256)).clamp(0, 1) * 2 - 1
    _NT = {}

    def soft_fit(img, rho):
        # noise-aware data-fit: the EXACT minimiser, per Fourier coefficient, of (1/2)(|c| - y)^2 + (rho'/2)|c - u|^2.
        # The magnitude moves only part of the way to the (noisy) measurement; the phase is kept.  One FFT pair.
        g = torch.nn.functional.pad((img * 0.5 + 0.5), (64, 64, 64, 64)); G = _FU.fft2_m(g); r4 = 4.0 * float(rho)
        if _NT.get("c") is not None and float(os.environ.get("PRD_KAPPA", "0")) > 0:
            # NON-ITERATIVE mapping to a partly LINEAR problem: inside a low-frequency disc the predictor's phases turn the measured magnitudes into
            # complex coefficients c (a linear measurement).  Per coefficient, exact minimiser of
            #   (1/2)(|u| - a)^2 + (r/2)|u - z|^2 + (kappa/2)|u - c|^2   =  the magnitude formula with anchor (r z + kappa c)/(r + kappa) and weight r + kappa.
            # kappa decays over the steps, so predicted phases are never locked in.
            kap = float(os.environ["PRD_KAPPA"]) * r4 * float(os.environ.get("PRD_KDECAY", "0.5")) ** state.get("step", (0, 1))[0] * _NT["M"]
            Gz = (r4 * G + kap * _NT["c"]) / (r4 + kap); re_ = r4 + kap; G = Gz / (Gz.abs() + 1e-8) * (amp + re_ * Gz.abs()) / (1 + re_)
            return (_FU.ifft2_m(G).real * S)[..., 64:320, 64:320].clamp(0, 1) * 2 - 1
        n_ad = int(os.environ.get("PRD_ADMM", "0"))
        if n_ad:
            # The step below ("fit every coefficient, transform back, crop") is ONE splitting step, not the exact data-fit: the padded transform is
            # redundant.  PRD_ADMM=n runs n inner ADMM iterations on the true problem  min_x (1/2)||Tx| - a|^2 + (r/2)|x - v|^2  (u = Tx split;
            # coefficient update = the same closed form, image update = a weighted average, dual update).  No model calls.
            ra = r4 * float(os.environ.get("PRD_RMUL", "1")); be = ra * float(os.environ.get("PRD_BETA", "1")); gx = g.clone(); d_ = torch.zeros_like(G)
            Jf = lambda q: float(0.5 * ((_FU.fft2_m(q).abs() - amp) ** 2).sum() + 0.5 * ra * ((q - g) ** 2).sum())
            if os.environ.get("PRD_JLOG", "0") == "1":
                u1 = G / (G.abs() + 1e-8) * (amp + ra * G.abs()) / (1 + ra); j1 = Jf(_FU.ifft2_m(u1).real * S)
            for _ in range(n_ad):
                w_ = _FU.fft2_m(gx) - d_; u_ = w_ / (w_.abs() + 1e-8) * (amp + be * w_.abs()) / (1 + be)
                gx = (be * (_FU.ifft2_m(u_ + d_).real * S) + ra * g) / (be + ra); d_ = d_ + u_ - _FU.fft2_m(gx)
            if os.environ.get("PRD_JLOG", "0") == "1":
                state.setdefault("prdJ", []).append([Jf(g), j1, Jf(gx)])
            return gx[..., 64:320, 64:320].clamp(0, 1) * 2 - 1
        mag = (amp + r4 * G.abs()) / (1 + r4)
        ev_ = os.environ.get("PRD_EV", "")
        if ev_:
            # expected-value magnitude fit: the model's coefficient a is uncertain (std s per pixel), the measurement fixes only |c|.
            # Given |c| = m, the phase follows a von Mises law around the phase of a with concentration 2 m |a| / s^2, so the posterior
            # mean is  m * I1(k)/I0(k) * exp(i phase(a)) : coefficients whose phase is uncertain are shrunk instead of fully imposed.
            ev_ = ev_.split(","); sd = float(ev_[min(state.get("step", (0, 1))[0], len(ev_) - 1)]) * state.get("sig", 0.5) / 2
            kap = (2 * mag.clamp(min=0) * G.abs() / sd ** 2).clamp(max=1e6); mag = mag * torch.special.i1e(kap) / torch.special.i0e(kap).clamp(min=1e-30)
        G = G / (G.abs() + 1e-8) * mag
        return (_FU.ifft2_m(G).real * S)[..., 64:320, 64:320].clamp(0, 1) * 2 - 1
    OFF = torch.load(os.path.abspath("../mode_probe/face_bank_8192.pt"))["rbar"].to(y.device) if os.environ.get("PRD_OFFSET", "0") == "1" else 0.0

    def _fit_from(c, mu_, U_, var_):
        with torch.enable_grad():
            for k in (64, 128, 256, 512):
                ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(); im_ = mu_ + (ck @ U_[:k]).view(1, 3, 256, 256)
                    loss = ((_FU.fft2_m(torch.nn.functional.pad(im_, (64, 64, 64, 64))).abs() + OFF - amp) ** 2).mean() + 1e-4 * (ck ** 2 / var_[:k]).mean()
                    loss.backward(); return loss
                opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
        return c

    def _fit_batch(C, mu_, U_, var_):
        # the same fit for all starts at once: the starts are independent, the loss is their sum
        with torch.enable_grad():
            for k in (64, 128, 256, 512):
                ck = C[:, :k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=int(os.environ.get("PRD_BIT", "250")), history_size=50, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(); im_ = mu_ + (ck @ U_[:k]).view(-1, 3, 256, 256)
                    loss = ((_FU.fft2_m(torch.nn.functional.pad(im_, (64, 64, 64, 64))).abs() + OFF - amp) ** 2).mean(dim=(1, 2, 3)).sum() + 1e-4 * (ck ** 2 / var_[:k]).mean(1).sum()
                    loss.backward(); return loss
                opt.step(closure); C = torch.cat([ck.detach(), C[:, k:]], 1)
        return C

    def tv_loop(img, n, lam=0.05, r=1.0, band=None):
        # plug-and-play coarse refinement with a classical prior: noise-aware magnitude fit, then a total-variation denoiser (FFTs only)
        x = (img * 0.5 + 0.5).clamp(0, 1)
        for _ in range(n):
            G = _FU.fft2_m(torch.nn.functional.pad(x, (64, 64, 64, 64))); Gn = G / (G.abs() + 1e-8) * (amp + r * G.abs()) / (1 + r)
            if band is not None:                               # inside the band: complex targets (measured magnitude, borrowed phase), soft
                Gn = torch.where(band[0], (band[1] + r * G) / (1 + r), Gn)
            x = (_FU.ifft2_m(Gn).real * S)[..., 64:320, 64:320].clamp(0, 1)
            px = torch.zeros_like(x); py = torch.zeros_like(x)
            for _ in range(30):
                u = x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2)); gx = torch.roll(u, -1, 3) - u; gy = torch.roll(u, -1, 2) - u
                nrm = 1 + 0.25 / lam * (gx ** 2 + gy ** 2).sqrt(); px = (px + 0.25 / lam * gx) / nrm; py = (py + 0.25 / lam * gy) / nrm
            x = (x + lam * (px - torch.roll(px, 1, 3) + py - torch.roll(py, 1, 2))).clamp(0, 1)
        return x * 2 - 1
    refine.tv_loop = tv_loop; refine.amp = amp

    def netfit_init(extra_reg=0):
        # learned low-frequency phases as SOFT linear constraints.  A small network (trained only on consistency-model samples) maps the
        # magnitudes to K low-resolution face hypotheses; for each, the eigenface fit is run with the hypothesis' phases inside a
        # low-frequency disc as complex targets, then refitted freely on the magnitudes (phases are never locked).  Best magnitude fit wins.
        import torch.nn as nn
        tag = os.environ.get("PRD_NET", "k8biglong"); Kh = int(tag[1]) if tag[0] == "k" else 8; Cn = torch.load(os.path.abspath(f"../mode_probe/phase_net_{tag}.pt"), map_location=y.device)
        ch = [3, 64, 128, 256, 256, 256]; L = []
        for i in range(5): L += [nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.GroupNorm(8, ch[i + 1]), nn.SiLU()]
        if not tag.startswith("v2"):
            net = nn.Module(); net.enc = nn.Sequential(*L); net.fc = nn.Sequential(nn.Flatten(), nn.Linear(4096, 2048), nn.SiLU(), nn.Linear(2048, Kh * 3 * 32 * 32)); net.load_state_dict(Cn["sd"]); net.to(y.device).eval()
            H = torch.sigmoid(net.fc(net.enc((torch.log1p(amp.clamp(min=0))[..., 128:256, 128:256] - Cn["fm"]) / Cn["fs"]))).view(Kh, 3, 32, 32)
        if tag.startswith("v2"):                               # stronger predictor: autocorrelation + magnitudes in, K faces at 64x64 out
            from phnet_arch import Net2, feat2
            Kh = int(tag.split("k")[1]); net = Net2(Kh).to(y.device); net.load_state_dict(Cn["sd"]); net.eval(); H = net((feat2(amp, _FU) - Cn["fm"]) / Cn["fs"])[0]
        H = torch.nn.functional.interpolate(H, size=256, mode="bicubic", align_corners=False).clamp(0, 1)
        Bp = torch.load(os.path.abspath("../mode_probe/pca_cm.pt")); mu_, U_, var_ = Bp["mu"].to(y.device), Bp["U"].to(y.device).flatten(1), Bp["var"].to(y.device)
        Rg = torch.load(os.path.abspath("../mode_probe/reg_best.pt")); rs = torch.load(os.path.abspath("../mode_probe/reg_resstd.pt")).to(y.device)
        creg = ((torch.log1p(amp.clamp(min=0))[..., 160:224, 160:224].flatten(1) - Rg["m"].to(y.device)) / Rg["s"].to(y.device) @ Rg["W"].to(y.device))[0]
        ii = torch.arange(384, device=y.device).float() - 192; M = ((ii[:, None] ** 2 + ii[None, :] ** 2).sqrt() < float(os.environ.get("PRD_NETR", "16"))).expand(1, 3, 384, 384)
        Afp = lambda im: _FU.fft2_m(torch.nn.functional.pad(im, (64, 64, 64, 64)))

        def bandfit(c, bO):
            with torch.enable_grad():
                for k in (64, 128, 256, 512):
                    ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")

                    def closure():
                        opt.zero_grad(); G = Afp(mu_ + (ck @ U_[:k]).view(1, 3, 256, 256))
                        loss = torch.where(M, (G - bO).abs() ** 2, (G.abs() + OFF - amp) ** 2).mean() + 1e-4 * (ck ** 2 / var_[:k]).mean()
                        loss.backward(); return loss
                    opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
            return c
        best = None; cands = [_fit_from(bandfit(creg, amp * torch.exp(1j * Afp(H[k:k + 1]).angle())), mu_, U_, var_) for k in range(Kh)]
        cands += [_fit_from(creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs, mu_, U_, var_) for j in range(extra_reg)]
        for cj in cands:
            lj = float(((Afp(mu_ + (cj @ U_).view(1, 3, 256, 256)).abs() + OFF - amp) ** 2).mean())
            if best is None or lj < best[0]:
                best = (lj, cj)
        return (mu_ + (best[1] @ U_).view(1, 3, 256, 256)).clamp(0, 1) * 2 - 1
    refine.netfit_init = netfit_init

    def netdirect_init():
        """NON-ITERATIVE start.  One pass of the phase predictor (about 1 ms) gives K low-resolution face hypotheses.  The one whose low-frequency
        magnitudes agree best with the measurement is kept; its phases, attached to the MEASURED magnitudes, give the coarse image in one inverse
        transform.  Its phases inside the disc PRD_NETR are stored as linear targets for the data step.  No fit, no search over starts, no loop."""
        from phnet_arch import Net2, feat2
        tag = os.environ.get("PRD_NET", "v2k8"); Cn = torch.load(os.path.abspath(f"../mode_probe/phase_net_{tag}.pt"), map_location=y.device); Kh = int(tag.split("k")[1])
        net = Net2(Kh).to(y.device); net.load_state_dict(Cn["sd"]); net.eval()
        with torch.no_grad(): H = net((feat2(amp, _FU) - Cn["fm"]) / Cn["fs"])[0]
        H = torch.nn.functional.interpolate(H, size=256, mode="bicubic", align_corners=False).clamp(0, 1)
        ii = torch.arange(384, device=y.device).float() - 192; rad = (ii[:, None] ** 2 + ii[None, :] ** 2).sqrt(); M = (rad < float(os.environ.get("PRD_NETR", "32"))).expand(1, 3, 384, 384)
        GH = _FU.fft2_m(torch.nn.functional.pad(H, (64, 64, 64, 64))); k = int((((GH.abs() - amp) ** 2) * M).flatten(1).mean(1).argmin())
        ph = torch.exp(1j * GH[k:k + 1].angle()); _NT["c"] = amp * ph; _NT["M"] = M.float(); refine.net_face = H[k:k + 1] * 2 - 1
        return (_FU.ifft2_m(amp * ph).real * S)[..., 64:320, 64:320].clamp(0, 1) * 2 - 1
    refine.netdirect_init = netdirect_init

    def regfit_init(nstart=1):
        # calibrated inverse map + fit: (1) ridge regression, trained only on samples of the consistency model, predicts the 512
        # eigenface coefficients from the low-frequency log-magnitudes; (2) the coefficients are then refined on the measured
        # magnitudes (L-BFGS, 64 -> 512 components) starting from that prediction.  No ground truth, no model calls.
        Bp = torch.load(os.path.abspath("../mode_probe/pca_cm.pt")); mu_, U_, var_ = Bp["mu"].to(y.device), Bp["U"].to(y.device).flatten(1), Bp["var"].to(y.device)
        Rg = torch.load(os.path.abspath("../mode_probe/reg_best.pt"))
        creg = ((torch.log1p(amp.clamp(min=0))[..., 160:224, 160:224].flatten(1) - Rg["m"].to(y.device)) / Rg["s"].to(y.device) @ Rg["W"].to(y.device))[0]
        if nstart > 1:                                         # several starts around the regression prediction, keep the best magnitude fit
            rs = torch.load(os.path.abspath("../mode_probe/reg_resstd.pt")).to(y.device); best = None
            if os.environ.get("PRD_BATCH", "0") == "1" or int(os.environ.get("PRD_TOPK", "1")) > 1:
                C0 = torch.stack([creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs for j in range(nstart)])
                Cf = _fit_batch(C0, mu_, U_, var_) if os.environ.get("PRD_BATCH", "0") == "1" else torch.stack([_fit_from(c_, mu_, U_, var_) for c_ in C0])
                ims = (mu_ + (Cf @ U_).view(-1, 3, 256, 256)); ls = ((_FU.fft2_m(torch.nn.functional.pad(ims, (64, 64, 64, 64))).abs() + OFF - amp) ** 2).mean(dim=(1, 2, 3))
                order = torch.argsort(ls); refine.cands = [ims[int(j)][None].clamp(0, 1) * 2 - 1 for j in order]
                return refine.cands[0]
            for j in range(nstart):
                cj = _fit_from(creg + (0 if j == 0 else 1.0) * torch.randn_like(creg) * rs, mu_, U_, var_)
                lj = float(((_FU.fft2_m(torch.nn.functional.pad(mu_ + (cj @ U_).view(1, 3, 256, 256), (64, 64, 64, 64))).abs() + OFF - amp) ** 2).mean())
                if best is None or lj < best[0]:
                    best = (lj, cj)
            return (mu_ + (best[1] @ U_).view(1, 3, 256, 256)).clamp(0, 1) * 2 - 1
        c = creg
        with torch.enable_grad():
            for k in (64, 128, 256, 512):
                ck = c[:k].detach().clone().requires_grad_(True); opt = torch.optim.LBFGS([ck], max_iter=150, history_size=30, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(); im_ = mu_ + (ck @ U_[:k]).view(1, 3, 256, 256)
                    loss = ((_FU.fft2_m(torch.nn.functional.pad(im_, (64, 64, 64, 64))).abs() - amp) ** 2).mean() + 1e-4 * (ck ** 2 / var_[:k]).mean()
                    loss.backward(); return loss
                opt.step(closure); c = torch.cat([ck.detach(), c[k:]])
        return (mu_ + (c @ U_).view(1, 3, 256, 256)).clamp(0, 1) * 2 - 1
    refine.mean_init = mean_init; refine.band_fit = band_fit; refine.pca_init = pca_init; refine.soft_fit = soft_fit; refine.regfit_init = regfit_init
    return refine, hio


# ---- more separable nonlinear operators (2026-11-20): pointwise, and "linear filter then pointwise" ----
_GAM = 0.4


class _GammaOp:
    """Tone curve: intensity (in [0,1]) raised to the power 0.4, as a camera response; noise is added AFTER the curve."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return 2 * ((x.clamp(-1.0, 1.0) + 1) / 2 + 1e-4) ** _GAM - 1
    forward = A


class _ModOp:
    """Modulo (wrap-around) sensor: the image is scaled by 2 and WRAPPED into the sensor range instead of clipped."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return torch.remainder(2 * x + 1, 2.0) - 1
    forward = A


_HB = {}


def _gblur(x, conj=False):
    """Circular Gaussian blur (std 3 pixels), applied in the Fourier domain; returns the blurred image."""
    k = (x.shape[-1], x.device)
    if k not in _HB:
        f = torch.fft.fftfreq(x.shape[-1], device=x.device); _HB[k] = torch.exp(-2 * math.pi ** 2 * 9.0 * (f[:, None] ** 2 + f[None, :] ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * _HB[k]).real


class _BlurClipOp:
    """Saturated blur: Gaussian blur, then scaled by 2 and clipped (linear optics followed by a saturating sensor)."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return (2 * _gblur(x)).clamp(-1.0, 1.0)
    forward = A


class _BlurQOp:
    """Blur then 3-bit quantisation (linear optics followed by a coarse analogue-to-digital converter)."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        b = _gblur(x); return b + (_q8(b) - b).detach()
    forward = A


def unwrap_ls(y):
    """Classical least-squares unwrapping of a wrap-around measurement (period 2), per colour channel.  For an image whose neighbouring
    values differ by less than half a period, the WRAPPED differences of the measurement equal the true differences, so the image is the
    solution of a LINEAR (Poisson) problem; the remaining constant is fixed by agreement with the measurement modulo the period.
    Returns the measurement with the wrap count of every pixel restored: an ordinary noisy image.  FFTs only, no model call."""
    Wp = lambda a: torch.remainder(a + 1, 2.0) - 1; H, Wd = y.shape[-2:]
    dx = Wp(y[..., :, 1:] - y[..., :, :-1]); dy = Wp(y[..., 1:, :] - y[..., :-1, :]); rho = torch.zeros_like(y)
    rho[..., :, :-1] += dx; rho[..., :, 1:] -= dx; rho[..., :-1, :] += dy; rho[..., 1:, :] -= dy
    ext = torch.cat([rho, rho.flip(-1)], -1); ext = torch.cat([ext, ext.flip(-2)], -2)                     # even extension = Neumann boundary
    ky = torch.arange(2 * H, device=y.device).float(); kx = torch.arange(2 * Wd, device=y.device).float()
    eig = (2 * torch.cos(math.pi * ky / H) - 2)[:, None] + (2 * torch.cos(math.pi * kx / Wd) - 2)[None, :]; eig[0, 0] = 1.0
    Fq = torch.fft.fft2(ext) / eig; Fq[..., 0, 0] = 0; phi = torch.fft.ifft2(Fq).real[..., :H, :Wd]
    ang = math.pi * (y - phi); c = torch.atan2(torch.sin(ang).mean(dim=(2, 3), keepdim=True), torch.cos(ang).mean(dim=(2, 3), keepdim=True)) / math.pi
    cand = torch.stack([phi + c + 2 * j for j in (-1, 0, 1)]); bad = (cand.abs() - 2).clamp(min=0).mean(dim=(3, 4), keepdim=True)    # true values lie in [-2, 2]
    full = torch.gather(cand, 0, bad.argmin(0, keepdim=True).expand(1, *cand.shape[1:]))[0]
    k = torch.round((full - y) / 2); return y + 2 * k


def gamma_prox(y):
    """Per-pixel minimiser of (1/2)(g(z) - y)^2 + (rho/2)(z - v)^2 for the tone curve g: exact up to a 257-point grid in z."""
    zc = torch.linspace(-1, 1, 257, device=y.device).view(257, 1, 1, 1, 1); gc = 2 * ((zc + 1) / 2 + 1e-4) ** _GAM - 1

    def prox(v, rho):
        return torch.gather(zc.expand(257, *v.shape), 0, (0.5 * (gc - y[None]) ** 2 + 0.5 * float(rho) * (zc - v[None]) ** 2).argmin(0, keepdim=True))[0]
    return prox


def mod_prox(y):
    """Exact per-pixel minimiser for the wrap-around sensor: one quadratic per branch (k = -1, 0, 1), keep the best."""
    kk = torch.tensor([-1.0, 0.0, 1.0], device=y.device).view(3, 1, 1, 1, 1)

    def prox(v, rho):
        rho = float(rho); z = ((2 * (y[None] + 2 * kk) + rho * v[None]) / (4 + rho)); z = torch.maximum(torch.minimum(z, kk + 0.5 - 1e-6), kk - 0.5).clamp(-1.0, 1.0)
        return torch.gather(z, 0, (0.5 * (2 * z - 2 * kk - y[None]) ** 2 + 0.5 * rho * (z - v[None]) ** 2).argmin(0, keepdim=True))[0]
    return prox


def blackbox_pointwise_prox(A, y, K=257):
    """GENERIC data-fit for ANY operator that acts on each pixel separately - no formula for the operator is needed.
    The operator is evaluated once on K constant images (grey levels c_1..c_K); that gives, for every pixel, the table f_pixel(c_k).
    The data-fit is then the exact minimiser over the grid of (1/2)(f(c) - y)^2 + (rho/2)(c - v)^2, pixel by pixel (non-convex or
    discontinuous f included).  GENERIC=mean returns the mean of exp(-objective / sigma_y^2) over the grid instead of the minimiser."""
    c = torch.linspace(-1.0, 1.0, K, device=y.device); tab = torch.stack([A(torch.full_like(y, float(v_))) for v_ in c]); cg = c.view(K, 1, 1, 1, 1)

    def prox(v, rho):
        ob = 0.5 * (tab - y[None]) ** 2 + 0.5 * float(rho) * (cg - v[None]) ** 2
        if os.environ.get("GENERIC", "") == "mean":
            w = torch.exp(-(ob - ob.min(0, keepdim=True)[0]) / SIGY ** 2); return (w * cg).sum(0) / w.sum(0)
        return torch.gather(cg.expand(K, *v.shape), 0, ob.argmin(0, keepdim=True))[0]
    return prox


def ev_fit(f, y, lo=None):
    """STATISTICAL scalar update for 'linear, then a per-coordinate function f':  instead of the minimiser of
    (1/2)(f(s) - y)^2 + (mu/2)(s - p)^2  it returns the MEAN of the density exp(-objective / T), T = SPLIT_EV * sigma_y^2  (the posterior mean of s
    under the belief N(p, T/mu) and the measurement y).  One number per coordinate, by quadrature on 41 points around p; still fully separable.
    This is the one-bit expected-value fit extended to any f: where the measurement is one-sided (clipped) the mean moves off the boundary."""
    T = SPLIT_EV * SIGY ** 2; d = torch.linspace(-4.0, 4.0, 41, device=y.device).view(41, 1, 1, 1, 1)

    def pw(p, mu):
        mu = float(mu); c = p[None] + d * (T / mu) ** 0.5
        if lo is not None:
            c = c.clamp(min=lo)
        ob = (0.5 * (f(c) - y[None]) ** 2 + 0.5 * mu * (c - p[None]) ** 2) / T; w = torch.exp(-(ob - ob.min(0, keepdim=True)[0]))
        return (w * c).sum(0) / w.sum(0)
    return pw


def split_prox(pw, n=8):
    """Data-fit for 'Gaussian blur, then a pointwise nonlinearity':  min (1/2)|f(Hz) - y|^2 + (rho/2)|z - v|^2.
    Alternates two CLOSED FORMS: the pointwise formula on w ~ Hz, and the linear (Fourier-diagonal) formula for z.  No model calls."""
    def prox(v, rho):
        rho = float(rho); z = v.clone(); H = None
        for mu in [0.5 * (64.0 ** (i_ / max(n - 1, 1))) for i_ in range(n)]:
            w = pw(_gblur(z), float(mu)); H = _HB[(v.shape[-1], v.device)]
            z = torch.fft.ifft2((mu * H * torch.fft.fft2(w) + rho * torch.fft.fft2(v)) / (mu * H ** 2 + rho)).real
        return z
    return prox


def blurclip_active(y, n_out=3, n_cg=25):
    """ACTIVE-SET data-fit for  y = clip(2 H x) + noise,  min (1/2)|clip(2Hz) - y|^2 + (rho/2)|z - v|^2.
    Once it is fixed which measurements are saturated, the problem is a QUADRATIC one: unsaturated pixels give equations 2(Hz)_i = y_i, saturated
    pixels give inequalities (2(Hz)_i >= 1 or <= -1) that cost nothing while they hold.  The set is predicted from the measurement and the current
    image, the quadratic problem is solved (conjugate gradients: mask and blur do not share a basis), the inequalities are checked, violated ones
    are made active (2(Hz)_i = +-1), satisfied active ones released, and the solve repeated (n_out rounds).  No model calls."""
    tau = 2 * SIGY; hi = y >= 1 - tau; lo = y <= -1 + tau; un = ~(hi | lo)

    def prox(v, rho):
        rho = float(rho); z = v.clone(); info = None
        for _ in range(n_out):
            b_ = 2 * _gblur(z); act_hi = hi & (b_ < 1); act_lo = lo & (b_ > -1); M = (un | act_hi | act_lo).to(v.dtype)
            t_ = torch.where(un, y, torch.where(act_hi, torch.ones_like(y), -torch.ones_like(y)))
            op = lambda q: 4 * _gblur(M * _gblur(q)) + rho * q; rhs = 2 * _gblur(M * t_) + rho * v
            r = rhs - op(z); p_ = r.clone(); rs = (r * r).sum()
            for _ in range(n_cg):
                Ap = op(p_); al = rs / (p_ * Ap).sum().clamp(min=1e-20); z = z + al * p_; r = r - al * Ap; rn = (r * r).sum()
                if float(rn) < 1e-10 * float((rhs * rhs).sum()):
                    break
                p_ = r + (rn / rs) * p_; rs = rn
            info = (float(act_hi.float().mean() + act_lo.float().mean()), float((hi | lo).float().mean()))
        state["bc_active"] = info
        return z
    return prox


def _bc_obj(y, z, v, rho):
    return float(0.5 * (((2 * _gblur(z)).clamp(-1.0, 1.0) - y) ** 2).sum() + 0.5 * float(rho) * ((z - v) ** 2).sum())


# ---- sparse-view computed tomography (2026-11-20): parallel-beam Radon transform, CT_K views over 180 degrees ----
CT_K = int(os.environ.get("CT_K", "60"))


def _radon(x):
    """Sinogram of x ([1,3,256,256] in [-1,1]): the image is padded to 364 so that no corner leaves the field of view, rotated to each
    view angle (bilinear) and summed along the rays.  Scaled by 1/16 so that one ray is a unit-norm functional of a 256-pixel line."""
    xp = torch.nn.functional.pad(x, (54, 54, 54, 54)).expand(CT_K, -1, -1, -1); th = torch.arange(CT_K, device=x.device).float() * math.pi / CT_K
    M = torch.stack([torch.stack([th.cos(), -th.sin(), torch.zeros_like(th)], 1), torch.stack([th.sin(), th.cos(), torch.zeros_like(th)], 1)], 1)
    rot = torch.nn.functional.grid_sample(xp, torch.nn.functional.affine_grid(M, list(xp.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    return rot.sum(dim=2).permute(1, 0, 2)[None] / 16.0          # [1, 3, views, 364]


class _CTOp:
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        return _radon(x)
    forward = A


_CTSVD = {}; _HDRREG = {}


def ct_prox(y, n_cg=None):
    """EXACT data-fit for the linear CT operator: (A'A + rho I) z = A'y + rho v by conjugate gradients (no learning rate).
    A'A is applied with one forward projection and one back-projection (automatic differentiation of the projector)."""
    n_cg = n_cg or int(os.environ.get("CT_CG", "25"))

    def At(r, like):
        with torch.enable_grad():
            z_ = torch.zeros_like(like).requires_grad_(True); return torch.autograd.grad((_radon(z_) * r).sum(), z_)[0]

    if os.environ.get("CT_SVD", "0") == "1":
        # closed form with the precomputed decomposition of the operator (ct_svd.py, once per geometry): no iterations.
        # (A'A + rho I) z = A'y + rho v   =>   in the basis V:  z_V = (V'A'y + rho v_V) / (s^2 + rho);  outside it z = v.
        if "V" not in _CTSVD:
            d_ = torch.load(os.path.abspath(f"../mode_probe/ct_svd_{CT_K}.pt")); _CTSVD["V"] = d_["V"].to(y.device); _CTSVD["s2"] = (d_["s"].to(y.device) ** 2)
        V_, s2_ = _CTSVD["V"], _CTSVD["s2"]; cy = At(y, torch.zeros(1, 3, 256, 256, device=y.device)).reshape(3, -1) @ V_      # [3, r]

        def prox_svd(v, rho):
            rho = float(rho); vv = v.reshape(3, -1).to(V_.dtype) @ V_
            return (v.reshape(3, -1) + ((cy + rho * vv) / (s2_ + rho) - vv) @ V_.T).reshape(v.shape).to(v.dtype)
        return prox_svd

    def prox(v, rho):
        rho = float(rho); b = At(y, v) + rho * v; op = lambda z_: At(_radon(z_), v) + rho * z_
        z = v.clone(); r = b - op(z); p_ = r.clone(); rs = (r * r).sum()
        for _ in range(n_cg):
            Ap = op(p_); al = rs / (p_ * Ap).sum().clamp(min=1e-20); z = z + al * p_; r = r - al * Ap; rn = (r * r).sum()
            if float(rn) < 1e-10 * float((b * b).sum()):
                break
            p_ = r + (rn / rs) * p_; rs = rn
        return z
    return prox


# ---- NONLINEAR CT (2026-11-23): raw transmission measurements  y = G (1 - exp(-s)),  s = alpha * Radon((x+1)/2)  (Beer-Lambert law;
# the model of Fridovich-Keil et al., "Gradient descent provably solves nonlinear tomographic reconstruction", arXiv 2310.03956).
# G only sets the noise level: the framework adds noise 0.1 to y, i.e. 0.1/G on the transmission scale (G=10 -> 0.01).
CT_ALPHA = float(os.environ.get("CT_ALPHA", "0.25")); CT_GAIN = float(os.environ.get("CT_GAIN", "10")); CTNL_MODE = os.environ.get("CTNL_MODE", "split")


CT_I0 = float(os.environ.get("CT_I0", "1000"))


class _CTNLOp(_CTOp):
    def A(self, x, **k):
        if CTNL_MODE in ("poisson", "poislog"):
            return CT_I0 * torch.exp(-CT_ALPHA * _radon((x + 1) / 2))      # mean photon count per ray; Poisson sampling is applied to the measurement
        return CT_GAIN * (1.0 - torch.exp(-CT_ALPHA * _radon((x + 1) / 2)))
    forward = A


def ctnl_ray_fit(y):
    """EXACT per-ray minimiser of (1/2)(G(1-e^{-s}) - y)^2 + (mu/2)(s - p)^2 over s >= 0.  The scalar problem is not convex, so the global
    minimiser is found the same way as for clipping / quantisation: evaluate candidates (a grid plus p itself), keep the best, polish with Newton."""
    grid = torch.linspace(0.0, 8.0, 65, device=y.device).view(65, 1, 1, 1, 1)

    def pw(p, mu):
        mu = float(mu); c = torch.cat([grid.expand(65, *p.shape), p.clamp(min=0)[None]], 0)
        obj = 0.5 * (CT_GAIN * (1 - torch.exp(-c)) - y[None]) ** 2 + 0.5 * mu * (c - p[None]) ** 2
        sb = torch.gather(c, 0, obj.argmin(0, keepdim=True))[0]
        for _ in range(6):
            e = torch.exp(-sb); g = CT_GAIN * e * (CT_GAIN * (1 - e) - y) + mu * (sb - p); h = CT_GAIN * e * (2 * CT_GAIN * e - CT_GAIN + y) + mu
            sb = (sb - (g / h.clamp(min=0.25 * mu)).clamp(-0.07, 0.07)).clamp(min=0.0)
        return sb
    return pw


def ctnl_prox(y, n=8):
    """Data-fit for 'Radon transform, then the exponential attenuation law':  min (1/2)|G(1-exp(-L z)) - y|^2 + (rho/2)|z - v|^2,  L z = alpha Radon((z+1)/2).
    Alternates two closed forms, as split_prox does for 'blur then pointwise': the per-ray formula above on w ~ L z, and the exact linear formula for z
    (precomputed decomposition of the Radon operator, ct_svd.py).  No model calls, no step size, no gradient of the nonlinearity.
    CTNL_MODE=log : the conventional alternative - undo the exponential by a logarithm once, then the LINEAR closed form (pre-processing baseline)."""
    if "V" not in _CTSVD:
        d_ = torch.load(os.path.abspath(f"../mode_probe/ct_svd_{CT_K}.pt")); _CTSVD["V"] = d_["V"].to(y.device); _CTSVD["s2"] = (d_["s"].to(y.device) ** 2)
    V_, s2_ = _CTSVD["V"], _CTSVD["s2"]; h = CT_ALPHA / 2; b2 = h * h * s2_; c0 = h * _radon(torch.ones(1, 3, 256, 256, device=y.device)); pw = ctnl_ray_fit(y)
    if SPLIT_EV > 0:
        pw = ev_fit(lambda s_: CT_GAIN * (1 - torch.exp(-s_)), y, lo=0.0)
    if CTNL_MODE == "poisson":
        # photon counts n ~ Poisson(I0 exp(-s)).  Per-ray update of  c (I0 e^{-s} + n s) + (mu/2)(s - p)^2  in CLOSED FORM with the Lambert W function
        # (c = sigma_y^2 puts the exact negative log-likelihood on the scale the penalty assumes):  s = p - c n/mu + W0((c I0/mu) exp(-p + c n/mu)).
        from scipy.special import lambertw as _lw; import numpy as _npx
        cc = SIGY ** 2; n_ = y

        def pw(p, mu):
            mu = float(mu); a_ = (p - cc * n_ / mu).double().cpu().numpy()
            return (torch.from_numpy(a_ + _lw((cc * CT_I0 / mu) * _npx.exp(-a_)).real).to(p.device).to(p.dtype)).clamp(min=0.0)

    def Atv(r):                                                   # V' A' r  (one back-projection)
        with torch.enable_grad():
            z_ = torch.zeros(1, 3, 256, 256, device=y.device, requires_grad=True); g_ = torch.autograd.grad((_radon(z_) * r).sum(), z_)[0]
        return g_.reshape(3, -1) @ V_

    def lin(w, v, mu, rho):                                       # argmin (mu/2)|h A z + c0 - w|^2 + (rho/2)|z - v|^2
        vv = v.reshape(3, -1).to(V_.dtype) @ V_
        return (v.reshape(3, -1) + ((mu * h * Atv(w - c0) + rho * vv) / (mu * b2 + rho) - vv) @ V_.T).reshape(v.shape).to(v.dtype)
    plog = -torch.log((1.0 - y / CT_GAIN).clamp(min=math.exp(-8.0), max=1.0))      # log pre-processing of the raw data
    if CTNL_MODE in ("poisson", "poislog"):
        plog = -torch.log((y.clamp(min=0.5) / CT_I0).clamp(max=1.0))

    def prox(v, rho):
        rho = float(rho)
        if CTNL_MODE == "log":
            return lin(plog, v, 1.0, rho / CT_GAIN ** 2)
        if CTNL_MODE == "poislog":                               # baseline: logarithm of the counts, then the linear closed form
            return lin(plog, v, 1.0, rho / CT_I0)
        z = v.clone(); n_ = int(os.environ.get("CTNL_N", n))
        # CTNL_N=1: NO alternation - one per-ray solve and one linear formula per step (coupling weight CTNL_MU); the outer steps do the rest
        for mu in ([float(os.environ.get("CTNL_MU", "2"))] if n_ == 1 else [0.5 * (64.0 ** (i_ / max(n_ - 1, 1))) for i_ in range(n_)]):
            z = lin(pw(CT_ALPHA * _radon((z + 1) / 2), mu), v, mu, rho)
        return z
    prox.start = lambda r0: lin(plog, torch.zeros(1, 3, 256, 256, device=y.device), 1.0, r0)
    return prox


# ---- accelerated MRI-style sampling (2026-11-20): single coil, Cartesian, 1-D random lines with a fully sampled centre ----
MRI_R = int(os.environ.get("MRI_R", "4")); _MM = {}


def _mri_mask(dev_):
    if dev_ not in _MM:
        g = torch.Generator().manual_seed(2024); n = 256; keep = torch.zeros(n, dtype=torch.bool); fr = torch.fft.fftfreq(n) * n          # line index, centre = 0
        keep[fr.abs() < 12] = True                                                                                                   # 24 centre lines
        w = torch.exp(-0.5 * (fr / (n / 6)) ** 2); w[keep] = 0; extra = n // MRI_R - int(keep.sum())
        keep[torch.multinomial(w, extra, replacement=False, generator=g)] = True                                                    # Gaussian-density random lines
        _MM[dev_] = keep.float().view(1, 1, 1, n).to(dev_)
    return _MM[dev_]


class _MRIOp:
    """k-space of each colour channel on 1/MRI_R of the lines.  Real and imaginary parts are returned side by side ([1,3,256,512])."""
    def __init__(self, *a, **k):
        self.blur_model = nn.Identity()

    def A(self, x, **k):
        X = torch.fft.fft2(x, norm="ortho") * _mri_mask(x.device); return torch.cat([X.real, X.imag], -1)
    forward = A


def mri_prox(y):
    """EXACT data-fit, one division per k-space coefficient: measured lines are blended with the model's k-space, the rest is kept."""
    Y = torch.complex(y[..., :256], y[..., 256:]); M = _mri_mask(y.device)

    def prox(v, rho):
        return torch.fft.ifft2((M * Y + float(rho) * torch.fft.fft2(v, norm="ortho")) / (M + float(rho)), norm="ortho").real
    return prox


_OPS = {"prdaps": _PRDapsOp, "hdr": _HDROp, "quant": _QuantOp, "onebit": _OneBitOp, "blind": _BlindOp, "haze": _HazeOp, "ct": _CTOp, "mri": _MRIOp, "ctnl": None, "prsq": _PRSqOp, "prcdp": _PRCdpOp,
        "gamma": _GammaOp, "mod": _ModOp, "blurclip": _BlurClipOp, "blurq": _BlurQOp}
if NEWOP:
    import functions.measurements as _FMM
    _FMM.NonlinearBlurOperator = _OPS[NEWOP] if NEWOP != "ctnl" else _CTNLOp


def quant_prox(y):
    """Exact per-pixel minimiser of (1/2)(Q(z) - y)^2 + (rho/2)(z - v)^2: try the nearest point of each of the 8 bins."""
    lo = (torch.arange(8, device=y.device).float() / 4 - 1).view(8, 1, 1, 1, 1); cen = lo + 0.125

    def prox(v, rho):
        if EVFIT:
            # posterior mean: belief N(v, s^2) about each pixel, the (noisy) measured level says how likely each of the 8 bins is
            sd = s_cal(); m_, Z_ = trunc_mean(v[None].expand(8, *v.shape), sd, lo.expand(8, *v.shape), (lo + 0.25).expand(8, *v.shape))
            w_ = Z_.float() * torch.exp(-0.5 * ((cen - y[None]) / SIGY) ** 2) + 1e-30
            return ((w_ * m_).sum(0) / w_.sum(0)).clamp(-1.0, 1.0)
        c = torch.maximum(torch.minimum(v[None], lo + 0.25 - 1e-6), lo)
        return torch.gather(c, 0, (0.5 * (cen - y[None]) ** 2 + 0.5 * float(rho) * (c - v[None]) ** 2).argmin(0, keepdim=True))[0]
    return prox


def onebit_prox(y, tau):
    """Exact per-pixel minimiser of (1/2)(sign(z - tau) - y)^2 + (rho/2)(z - v)^2: keep the pixel, or move it just across its
    threshold if agreeing with the measured bit is worth the move."""
    def prox(v, rho):
        if os.environ.get("OBMEAN", "0") != "0":
            # posterior-mean data-fit: the model's image is uncertain by about the step's noise level, the bit says on which side of
            # its threshold the pixel lies -> mean of that truncated Gaussian (closed form), not merely the nearest point of the half-line
            om_ = os.environ["OBMEAN"].split(","); sg = s_cal() if om_[0] == "cal" else float(om_[min(state.get("step", (0, 1))[0], len(om_) - 1)]) * state.get("sig", 0.5); u = y * (v - tau) / sg
            return (v + y * sg * torch.exp(-0.5 * u * u - 0.9189385332 - torch.special.log_ndtr(u))).clamp(-1.0, 1.0)
        sv = torch.sign(v - tau) + (v == tau).float()
        keep = 0.5 * (sv - y) ** 2; flip = 0.5 * (-sv - y) ** 2 + 0.5 * float(rho) * (v - tau) ** 2
        return torch.where(flip < keep, tau - 1e-3 * sv, v)
    return prox


def haze_setup(y, mode, A_funcs):
    """Dehazing reduced to a DIAGONAL linear problem.  est: transmission and airlight from the classical dark-channel prior
    (darkest value in a 15x15 patch across colours), smoothed; true: told both (ceiling); none: no haze model."""
    mp = lambda q, k: -torch.nn.functional.max_pool2d(-q, k, stride=1, padding=k // 2)
    if mode == "true":
        t, a = A_funcs.t, A_funcs.a
    elif mode == "none":
        t, a = torch.ones(1, 1, 256, 256, device=y.device), torch.zeros(1, 3, 1, 1, device=y.device)
    else:
        y01 = (y.clamp(-1, 1) + 1) / 2; dark = mp(y01.min(1, keepdim=True).values, 15)
        top = dark.flatten().topk(66).indices; a01 = y01.flatten(2)[:, :, top].mean(-1).view(1, 3, 1, 1).clamp(min=0.05)
        t = (1 - 0.95 * mp((y01 / a01).min(1, keepdim=True).values, 15)).clamp(0.1, 1.0)
        t = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(t, (20, 20, 20, 20), mode="replicate"), 41, stride=1); a = 2 * a01 - 1
    b = y - (1 - t) * a
    return (lambda v, rho: (float(rho) * v + t * b) / (float(rho) + t ** 2)), (b / t).clamp(-1.0, 1.0), t


def blind_prox(y, mode, A_funcs):
    """Blind deblurring with closed forms only.  At every step: (1) estimate one kernel from the current image and the measurement
    by least squares over all frequencies and colours, keep its central 21x21 part, clip negatives, normalise; (2) deconvolve with
    that kernel in closed form.  mode: est | true (told the kernel: the non-blind ceiling) | none (no kernel: plain denoising)."""
    Y = torch.fft.fft2(y); mem = {"n": 0, "K": None}

    def est(v):
        X = torch.fft.fft2(v); fx = torch.fft.fftfreq(256, device=y.device); w2 = fx[:, None] ** 2 + fx[None, :] ** 2    # fit on image gradients (edges)
        num = (X.conj() * Y).sum(1)[0] * w2; den = (X.abs() ** 2).sum(1)[0] * w2
        k = torch.roll(torch.fft.ifft2(num / (den + 1e-2 * den.mean())).real, (10, 10), dims=(0, 1))[:21, :21].clamp(min=0)
        return k / k.sum().clamp(min=1e-8)

    def prox(v, rho):
        if mode == "true":
            k = A_funcs.k
        elif mode == "none" or mem["n"] == 0:
            k = torch.zeros(21, 21, device=y.device); k[10, 10] = 1.0
        else:
            k = est(v)
        mem["n"] += 1; mem["K"] = k; K = _k2f(k)
        return torch.fft.ifft2((K.conj() * Y + float(rho) * torch.fft.fft2(v)) / (K.abs() ** 2 + float(rho))).real
    prox.mem = mem
    return prox


def hdr_prox(y):
    """Exact minimiser, pixel by pixel, of  (1/2)(clip(2z) - y)^2 + (rho/2)(z - v)^2  - the TRUE data-fit, no iterations."""
    def prox(v, rho):
        if EVFIT:
            # posterior mean under the belief N(v, s^2): three pieces (clipped low, unclipped, clipped high), each a truncated Gaussian
            sd = s_cal(); sy2 = SIGY ** 2; pr = 4 / sy2 + 1 / sd ** 2; mu = (2 * y / sy2 + v / sd ** 2) / pr; sm = pr ** -0.5
            m0, Z0 = trunc_mean(mu, sm, -0.5, 0.5); w0_ = Z0.float() * torch.exp(-0.5 * (y - 2 * v) ** 2 / (sy2 + 4 * sd ** 2)) / (sy2 + 4 * sd ** 2) ** 0.5
            m1, Z1 = trunc_mean(v, sd, 0.5, 1e3); w1_ = Z1.float() * torch.exp(-0.5 * (y - 1) ** 2 / sy2) / SIGY
            m2, Z2 = trunc_mean(v, sd, -1e3, -0.5); w2_ = Z2.float() * torch.exp(-0.5 * (y + 1) ** 2 / sy2) / SIGY
            return ((w0_ * m0 + w1_ * m1 + w2_ * m2) / (w0_ + w1_ + w2_ + 1e-30)).clamp(-1.0, 1.0)
        rho = float(rho); f = lambda z: 0.5 * ((2 * z).clamp(-1, 1) - y) ** 2 + 0.5 * rho * (z - v) ** 2
        c = torch.stack([((2 * y + rho * v) / (4 + rho)).clamp(-0.5, 0.5), v.clamp(min=0.5), v.clamp(max=-0.5)])
        return torch.gather(c, 0, torch.stack([f(k) for k in c]).argmin(0, keepdim=True))[0]
    return prox


import main as M
import wrappers.natural_nonlinear_wrapper as WN
import wrappers.natural_linear_wrapper as WL
from pnp_cm_scheme.natural_nonlinear_scheme import (
    get_scalings_for_boundary_condition, append_dims, get_next_alpha, compute_alpha, phase_retrieval_initialization)

state = {"img": S0 - 1, "stats": [], "gt": None, "speclog": {}}

# ---- frequency-shaped noise injection (idea from shaped_noise.py, applied to the unified sampler) ----
# NOISE=white : shipped white noise.   NOISE=calib : measure, on calibration images, the error power of the
# iterate r = z + u per spatial frequency at every (pass, step), shaping with the image's own spectrum.
# NOISE=shape : inject noise with power max(F^2 - P(f), 0), F^2 = sigma_t^2 + mean P, using the stored tables,
# so existing error + added noise is roughly white at the same total budget.  Steps with sigma_t < 0.05 unchanged.
NOISE = os.environ.get("NOISE", "white"); NB = 64
SPECFILE = os.path.abspath(f"../mode_probe/{os.environ.get('SPECDIR', 'spec_unified')}/{deg}.json")
SPEC = {}
if NOISE in ("shape", "fire") and os.path.exists(SPECFILE):
    SPEC = {k: torch.tensor(v) for k, v in json.load(open(SPECFILE)).items()}
_FB = {}


def _bands(dev):
    if dev not in _FB:
        fx = torch.fft.fftfreq(256, device=dev)
        rad = (fx[:, None] ** 2 + fx[None, :] ** 2).sqrt().flatten()
        fb = (torch.argsort(torch.argsort(rad)) * NB // rad.numel()).long()
        _FB[dev] = (fb, torch.bincount(fb, minlength=NB).float())
    return _FB[dev]


# ---- NOISE=fire (2026-11-24): coloured renoising in the OPERATOR's coordinates (after FIRE, Bendel et al. 2025) -------------------------
# The error of the model input r = z + u is far from white in the operator's coordinates (small where the measurement is informative).  Its power
# C_b is tabulated per coordinate group b on generated images (NOISE=firecal; groups: singular-value bins for linear operators, measurement-value
# bins for pointwise operators), and noise of variance lambda_b is injected so that error + noise is white:
#   FIRE_MODE=budget : lambda_b = max(sigma^2 + mean C - C_b, 0), rescaled to mean sigma^2 (same injected power as the shipped noise; told level unchanged)
#   FIRE_MODE=exact  : lambda_b = max(told^2 - C_b, 0)                (total = the level the model is told; FIRE's rule)
FIRE_MODE = os.environ.get("FIRE_MODE", "budget"); FIRE_NB = 24


def _fire_setup(P, like):
    if getattr(P, "_fire", None) is None:
        dev = like.device; NBf = FIRE_NB
        if NEWOP == "ct" and "V" in _CTSVD:
            sv = _CTSVD["s2"].sqrt(); b = 1 + (sv / sv.max() * (NBf - 1)).clamp(max=NBf - 1 - 1e-4).floor().long(); n0 = 65536 - sv.numel()
            cnt = torch.bincount(b, minlength=NBf).float() * 3; cnt[0] = 3 * n0; P._fire = ("ct", b, cnt)
        elif P.ATy is not None and hasattr(P.F, "singulars"):
            sv = _svd_parts(P)[0]; b = torch.where(sv <= 1e-6 * sv.max(), torch.zeros_like(sv), 1 + (sv / sv.max() * (NBf - 1)).clamp(max=NBf - 1 - 1e-4).floor()).long()
            P._fire = ("lin", b, torch.bincount(b, minlength=NBf).float())
        elif P.y is not None and P.y.shape == like.shape:
            b = ((P.y.clamp(-1.3, 1.3) + 1.3) / 2.6 * NBf).clamp(max=NBf - 1e-3).floor().long().flatten(); P._fire = ("pix", b, torch.bincount(b, minlength=NBf).float())
        else:
            P._fire = ("none", None, None); print("FIRE: no operator coordinates for this task - white noise", flush=True)
    return P._fire


def fire_spec(P, e):
    """Per-group SUM of squared error coefficients in the operator's coordinates."""
    kind, b, cnt = _fire_setup(P, e)
    if kind == "ct":
        V_ = _CTSVD["V"]; ef = e.reshape(3, -1).float(); c = ef @ V_; nul = ef - c @ V_.T; sp = torch.zeros(FIRE_NB, device=e.device).index_add_(0, b.repeat(3), (c ** 2).flatten()); sp[0] = (nul ** 2).sum(); return sp
    c = P.F.Vt(e)[0] if kind == "lin" else e.flatten()
    return torch.zeros(FIRE_NB, device=e.device).index_add_(0, b, c.float() ** 2)


def fire_inject(r, sig, key, w):
    P = state["P"]; kind, b, cnt = _fire_setup(P, r)
    if kind == "none":
        return w * sig
    if NOISE == "firecal":
        sm = fire_spec(P, r - state["gt"].to(r.device)); state["speclog"].setdefault(key, []).append(torch.cat([sm, cnt]).cpu().tolist())
    elif key in SPEC:
        sm = SPEC[key][:FIRE_NB].to(r.device) * cnt / SPEC[key][FIRE_NB:].to(r.device).clamp(min=1e-6)       # tabulated mean power per group x this image's group sizes
    else:
        return w * sig
    C = sm / cnt.clamp(min=1); wt = cnt / cnt.sum(); Cb = float((wt * C).sum())
    if FIRE_MODE == "exact":
        lam = (float(state.get("told", sig)) ** 2 - C).clamp(min=0)
    else:
        lam = (sig ** 2 + Cb - C).clamp(min=0); lam = lam * sig ** 2 / float((wt * lam).sum().clamp(min=1e-12))
    if state["img"] < S0 + 1 and NOISE == "fire":
        print(f"FIRE step {key}: sigma {sig:.3f} told {float(state.get('told', sig)):.3f} rms error of the input {Cb ** 0.5:.3f}; error rms per group min {float(C[cnt > 0].min()) ** 0.5:.3f} max {float(C[cnt > 0].max()) ** 0.5:.3f}; injected rms {float((wt * lam).sum()) ** 0.5:.3f}", flush=True)
    g = lam.sqrt()
    if kind == "ct":
        V_ = _CTSVD["V"]; wf = w.reshape(3, -1).float(); c = wf @ V_; return (g[0] * (wf - c @ V_.T) + (g[b][None] * c) @ V_.T).reshape(w.shape).to(w.dtype)
    if kind == "lin":
        return P.F.V((g[b] * P.F.Vt(w)[0])[None]).reshape(w.shape)
    return (g[b] * w.flatten()).reshape(w.shape)


def inject(r, sig, pas, step, pr):
    """Noise to add to r = z + u at this step."""
    w = torch.randn_like(r)
    sig = float(sig)
    if (NOISE == "white" and TILT == 0 and not NBG) or sig < 0.05:
        return w * sig
    if NOISE in ("fire", "firecal"):
        return fire_inject(r, sig, f"{pas}_{step}", w)
    if NOISE == "white" and NBG:                               # one power gain per octave band (the 8 bands of the combination), total power unchanged
        g2 = torch.tensor(NBG, device=r.device)[lle_band_index(256, len(NBG), r.device)]
        return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * (g2 / g2.mean()).sqrt() * sig, norm="ortho").real
    fb, cnt = _bands(r.device); key = f"{pas}_{step}"
    if NOISE == "white":                                       # white budget, tilted toward high (TILT > 0) or low (TILT < 0) frequencies
        return _tilted(w, torch.full((256, 256), sig, device=r.device), fb)
    if NOISE == "calib":
        x0 = state["gt"].to(r.device)
        if pr:
            x0 = x0 * torch.sign((r * x0).sum(dim=(2, 3), keepdim=True))
        p = (torch.fft.fft2(r - x0, norm="ortho").abs() ** 2).mean(dim=(0, 1)).flatten()
        sp = torch.bincount(fb, weights=p, minlength=NB) / cnt
        state["speclog"].setdefault(key, []).append(sp.cpu().tolist())
    elif key in SPEC:
        sp = SPEC[key].to(r.device)
    else:
        return w * sig                                         # no table for this (pass, step): shipped noise
    g = (sig ** 2 + float(sp.mean()) - sp[fb].view(256, 256)).clamp(min=0).sqrt()
    return _tilted(w, g, fb)


NBG = [float(v) for v in os.environ.get("NBG", "").split(",") if v]
TILT = float(os.environ.get("TILT", "0"))   # noise power multiplied by (frequency rank)^TILT, total power unchanged; 0 = off


def _tilted(w, g, fb):
    """Gaussian noise with per-frequency amplitude g, optionally tilted; the total power of g is preserved."""
    if TILT != 0:
        g2 = g ** 2 * (((fb.float() + 0.5) / NB) ** TILT).view(256, 256)
        g = (g2 * (g ** 2).mean() / g2.mean()).sqrt()
    return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * g, norm="ortho").real


def _seeded(orig):
    def f(config, X):
        torch.manual_seed(7000000 + state["img"] + 1 + 100000 * int(os.environ.get("MEAS_SEED", "0")))  # per-image seed: identical measurement noise in every run
        out = orig(config, X); state["gt"] = out
        return out
    return f


WN.data_transform = _seeded(WN.data_transform)
WL.data_transform = _seeded(WL.data_transform)
ORIG_NL, ORIG_LIN = WN.pnp_cm_natural_nonlinear_restoration, WL.pnp_cm_natural_linear_restoration


class Problem:
    """Data term D(z) with gradient and Gauss-Newton product; counts operator applications."""

    def __init__(self, A_funcs, y=None, ATy=None):
        self.A = A_funcs.A; self.At = getattr(A_funcs, "At", None); self.F = A_funcs
        self.y, self.ATy = y, ATy
        self.exact = getattr(A_funcs, "AtA_add_eta_inv", None) if ATy is not None else None
        self.n = {"A": 0, "At": 0, "jvp": 0}

    def val(self, z):
        self.n["A"] += 1
        if self.y is not None:
            r = self.A(z) - self.y
            return 0.5 * (r * r).sum()
        Az = self.A(z)
        return 0.5 * (Az * Az).sum() - (z * self.ATy).sum()   # = 0.5||Az - y||^2 up to a constant

    def grad(self, z):
        self.n["A"] += 1; self.n["At"] += 1
        if self.y is not None:
            with torch.enable_grad():
                zr = z.detach().requires_grad_(True)
                r = self.A(zr) - self.y
                f = 0.5 * (r * r).sum()
                g, = torch.autograd.grad(f, zr)
            return f.detach(), g
        Az = self.A(z)
        return 0.5 * (Az * Az).sum() - (z * self.ATy).sum(), self.At(Az).view_as(z) - self.ATy

    def gn(self, z, p):
        if self.y is None:
            self.n["A"] += 1; self.n["At"] += 1
            return self.At(self.A(p)).view_as(p)
        self.n["jvp"] += 1; self.n["A"] += 1; self.n["At"] += 1
        try:
            Jp = torch.func.jvp(self.A, (z,), (p,))[1]
        except Exception:
            eps = 1e-3 / (p.norm() + 1e-12); self.n["A"] += 1
            Jp = (self.A(z + eps * p) - self.A(z - eps * p)) / (2 * eps)
        with torch.enable_grad():
            zr = z.detach().requires_grad_(True)
            out, = torch.autograd.grad((self.A(zr) * Jp).sum(), zr)
        return out


REPEL_FROM = [q for q in os.environ.get("REPEL_FROM", "").split(",") if q]; REPEL_G = float(os.environ.get("REPEL_G", "0.3"))


def _repel_ref(dev_):
    # finished images of earlier particles for the image now being solved (same measurement, other sampler noise)
    key = state["img"] + 1
    if state.get("_ref", (None,))[0] != key:
        import glob as _g
        xs = [torch.load(f)["x_raw"].to(dev_).float() for pat in REPEL_FROM for f in _g.glob(os.path.abspath(f"../mode_probe/exp_{pat}_s*/raw_{key}.pt"))]
        state["_ref"] = (key, torch.stack(xs).mean(0) if xs else None)
    return state["_ref"][1]


FWMODE = os.environ.get("FW", "0"); FW = FWMODE == "1"; FWP = float(os.environ.get("FWP", "1.0")); FWK = int(os.environ.get("FWK", "6")); FWMIN = float(os.environ.get("FWMIN", "0.05"))
LASTTRUE = int(os.environ.get("LASTTRUE", "0")); FINTOLD = float(os.environ.get("FINTOLD", "1.0")); _FWC = {}


CALW = os.environ.get("CALW", ""); CALU = os.environ.get("CALU", ""); EVFIT = os.environ.get("EVFIT", "0") == "1"; NBC = 32; _CAL = {}
EVS = float(os.environ.get("EVS", "1.0"))


def _cal_bins(dev_):
    if "bins" not in _CAL:
        f = torch.fft.fftfreq(256, device=dev_); _CAL["bins"] = ((f[:, None] ** 2 + f[None, :] ** 2).sqrt() / 0.7072 * NBC).long().clamp(max=NBC - 1)
    return _CAL["bins"]


def _cal_record(v):
    # calibration run: error of the image handed to the data-fit (the model's image, dual removed) against the truth, per step
    g = state["gt"].to(v.device)
    if deg == "phase_retrieval":
        g = g * torch.sign((v * g).sum(dim=(2, 3), keepdim=True))
    e = v - g; pw = (torch.fft.fft2(e, norm="ortho").abs() ** 2).mean(dim=(0, 1)); b = _cal_bins(v.device)
    sp = torch.bincount(b.flatten(), weights=pw.flatten(), minlength=NBC) / torch.bincount(b.flatten(), minlength=NBC).clamp(min=1)
    k = str(state.get("step", (0, 1))[0]); acc = state.setdefault("calacc", {}).setdefault(k, [[0.0] * NBC, 0.0, 0])
    acc[0] = [a_ + float(c_) for a_, c_ in zip(acc[0], sp)]; acc[1] += float((e ** 2).mean()); acc[2] += 1


def _cal_table():
    if "tab" not in _CAL:
        _CAL["tab"] = json.load(open(os.path.abspath(f"../mode_probe/cal/{CALU}.json")))
    t = _CAL["tab"]; k = str(state.get("step", (0, 1))[0]); k = k if k in t else sorted(t, key=int)[-1]
    return t[k]


def s_cal():
    """Calibrated uncertainty (std per pixel) of the model's image at the current step; falls back to the step's noise level."""
    return EVS * (float(_cal_table()["mse"]) ** 0.5 if CALU else state.get("sig", 0.5))


def trunc_mean(v, sd, lo, hi):
    """Mean of N(v, sd^2) restricted to [lo, hi] (closed form); where the interval has ~no mass, the nearest point of the interval."""
    v64, sd64 = v.double(), (sd.double() if torch.is_tensor(sd) else float(sd))
    a = (lo.double() - v64) / sd64 if torch.is_tensor(lo) else (lo - v64) / sd64; b = (hi.double() - v64) / sd64 if torch.is_tensor(hi) else (hi - v64) / sd64
    Z = torch.special.ndtr(b) - torch.special.ndtr(a); ph = lambda t: torch.exp(-0.5 * t * t) * 0.3989422804014327
    m = v64 + sd64 * (ph(a) - ph(b)) / Z.clamp(min=1e-300)
    return torch.where(Z > 1e-10, m, torch.maximum(torch.minimum(v64, hi.double() if torch.is_tensor(hi) else torch.full_like(v64, hi)), lo.double() if torch.is_tensor(lo) else torch.full_like(v64, lo))).to(v.dtype), Z


def _fw_weights(v, rho):
    """Per-frequency penalty rho(f) from the calibrated error spectrum of the model's image.  cal: shipped level, calibrated shape
    (mean precision kept);  calfull: fully calibrated, rho(f) = sigma_y^2 / Sigma(f)."""
    sp = torch.tensor(_cal_table()["spec"], device=v.device, dtype=torch.float32).clamp(min=1e-8)[_cal_bins(v.device)]
    if FWMODE == "calfull":
        return (SIGY ** 2 / sp)
    w = 1.0 / sp; return float(rho) * w / w.mean()


def _fw_apply(a):
    # W a : per spatial frequency, the trust in the model's image relative to the full penalty (1 above 0.25 cycles/pixel, lower below)
    k = (a.shape[-2], a.shape[-1], a.device)
    if k not in _FWC:
        fy = torch.fft.fftfreq(a.shape[-2], device=a.device); fx = torch.fft.fftfreq(a.shape[-1], device=a.device)
        _FWC[k] = (((fy[:, None] ** 2 + fx[None, :] ** 2).sqrt() / float(os.environ.get("FWC", "0.25"))) ** FWP).clamp(FWMIN, 1.0)
    return torch.fft.ifft2(torch.fft.fft2(a) * _FWC[k]).real


def gauss_newton(P, z, v, rho, sig=None):
    """Data-fit of one step.  Optional: (FW) frequency-weighted trust in the model's image; (LASTTRUE) true operator in the last steps."""
    state["sig"] = float(sig) if sig is not None else 0.5
    rho = rho * float(os.environ.get("RHOX", "1"))            # control: plain rescaling of the penalty (same at every frequency)
    if CALW and state["gt"] is not None and v.dim() == 4 and v.shape[-1] == 256:
        _cal_record(v)
    if FWMODE in ("cal", "calfull") and CALU and v.dim() == 4 and v.shape[-1] == 256:
        rf = _fw_weights(v, rho); rmax = float(rf.max()); Wf = rf / rmax; z = _gn_inner(P, z, v, rmax, sig)
        for _ in range(FWK):                                    # same proximal-gradient reuse of the closed forms, with the calibrated weights
            z = _gn_inner(P, z, z - torch.fft.ifft2(torch.fft.fft2(z - v) * Wf).real, rmax, sig)
        return z
    z = _gn_inner(P, z, v, rho, sig)
    if FW and v.dim() == 4 and (not os.environ.get("FWSTEPS") or str(state.get("step", (0, 1))[0]) in os.environ["FWSTEPS"].split(",")):
        # minimise fit(z) + rho/2 (z - v)' W (z - v) with W <= I: proximal-gradient iterations, each ONE ordinary data-fit with a moved
        # anchor  u = z - W (z - v).  Every closed form is reused unchanged; no model calls.
        for _ in range(FWK):
            z = _gn_inner(P, z, z - _fw_apply(z - v), rho, sig)
    return z


SOLVER = os.environ.get("SOLVER", "")


def repo_solve(P, z, v, rho):
    """PnP-CM's own inner optimiser, copied literally from its nonlinear scheme (optimizer, learning rate, iteration count and decay
    from the repository's task settings; like the repository, gradients are never zeroed)."""
    import torch.optim as optim
    a_ = state["args"]
    with torch.enable_grad():
        zt_ = z.clone().requires_grad_(True); opt = getattr(optim, a_.opt_type)([zt_], lr=float(os.environ.get("REPO_LR", a_.opt_lr))); loss_pre = 0.0
        n_it_ = int(os.environ.get("REPO_N", a_.opt_num_iter))
        for _ in range(n_it_):
            res = P.A(zt_) - P.y; diff = zt_ - v
            loss = 0.5 * (res * res).sum() + 0.5 * rho * torch.sum(diff * diff).abs(); loss.backward(); opt.step()
            if loss.item() > loss_pre:
                for g_ in opt.param_groups:
                    g_["lr"] = g_["lr"] * a_.opt_decay_rate
            loss_pre = loss.item()
    P.n["A"] += a_.opt_num_iter
    return zt_.detach()


def _gn_inner(P, z, v, rho, sig=None):
    if SOLVER == "repo" and P.y is not None:
        return repo_solve(P, z, v, rho)
    if REPEL_FROM:
        # diversity term inside the data-fit: minimise  fit(z) + rho/2 |z - v|^2 - g/2 |z - ref|^2  with g = REPEL_G * rho.
        # Completing the square gives the SAME data-fit with a shifted anchor and a smaller penalty, so every closed form still applies;
        # the push away from earlier particles only acts where the measurement leaves the image free.
        ref = _repel_ref(v.device)
        if ref is not None:
            g_ = REPEL_G * rho; v = (rho * v - g_ * ref) / (rho - g_); rho = rho - g_
    st_ = state.get("step", (0, 99))
    if getattr(P, "prox", None) is not None and not (LASTTRUE and st_[0] >= st_[1] - LASTTRUE and P.y is not None):   # exact / surrogate data-fit
        P.n["exact"] = P.n.get("exact", 0) + 1
        return P.prox(v, rho)
    if EXACT and P.exact is not None:                          # operator offers an exact proximal solve: use it
        P.n["exact"] = P.n.get("exact", 0) + 1
        return P.exact(P.ATy + rho * v, eta=rho).reshape(*P.ATy.size())
    g0n = None; eta = 0.5 if EW else ETA; gprev = None
    for _ in range(KMAX):
        f, g = P.grad(z)
        f = f + 0.5 * rho * ((z - v) ** 2).sum(); g = g + rho * (z - v)
        gn = g.norm()
        if g0n is None:
            g0n = gn
        elif gn <= TOLG * g0n:
            break
        if EW and gprev is not None:                          # choice 2 of Eisenstat & Walker (gamma 0.9, alpha 2) with safeguard
            new = 0.9 * float(gn / gprev) ** 2
            if 0.9 * eta ** 2 > 0.1:
                new = max(new, 0.9 * eta ** 2)
            eta = min(max(new, 1e-3), 0.9)
        gprev = gn
        d = torch.zeros_like(z); r = -g.clone(); p = r.clone(); rs = (r * r).sum()
        for _ in range(CGMAX):
            Hp = P.gn(z, p) + rho * p
            a = rs / ((p * Hp).sum() + 1e-30)
            d = d + a * p; r = r - a * Hp
            rs_new = (r * r).sum()
            if rs_new.sqrt() <= eta * gn:
                break
            p = r + (rs_new / rs) * p; rs = rs_new
        slope = (g * d).sum(); a = 1.0; zn = z
        for _ in range(12):                                   # Armijo backtracking
            zc = z + a * d
            if P.val(zc) + 0.5 * rho * ((zc - v) ** 2).sum() <= f + 1e-4 * a * slope:
                zn = zc; break
            a *= 0.5
        if zn is z:
            break
        small = STEPSTOP > 0 and sig is not None and float(((zn - z) ** 2).mean().sqrt()) < STEPSTOP * float(sig)
        z = zn
        if small:                                             # further refinement is below the noise added next
            break
    return z


STEPLOG = []
SAVE_STEPS = os.environ.get("SAVE_STEPS", "0") == "1"


# ---- spatially targeted exploration (2026-11-17) ----------------------------------------------------
# Uncertainty map, free: m = smoothed per-pixel |z - x_prev|^2, the disagreement between the data-fit output and the
# previous denoised image.  Used from the second step of a pass on, and only when sigma_t >= 0.05.
#   EXPLORE=w : noise variance per pixel proportional to m (clamped to [1/4, 4] x mean), same total budget; CM told the shipped level
#   EXPLORE=g : control - noise XK x larger everywhere, CM told XK x the shipped level
#   EXPLORE=r : regional - denoise twice (shipped level, and XK x level with the same noise draw); keep the high-noise
#               result only where m is in its top 20% (soft ramp from the 60th percentile), the shipped one elsewhere
EXPLORE = os.environ.get("EXPLORE", ""); XK = float(os.environ.get("XK", "1.5"))
XLOG = []
import torch.nn.functional as Fn


# EQ: random transform before the model call, undone after it (equivariant plug-and-play, Terris et al. 2024), so that a structured
# input error does not line up with the model in the same way at every step.  EQ=flip | shift<S> (circular shift up to S pixels) | flipshift<S>
NREUSE = float(os.environ.get("NREUSE", "0")); SPLIT_EV = float(os.environ.get("SPLIT_EV", "0")); EQ = os.environ.get("EQ", ""); DIAG3 = os.environ.get("DIAG3", "0") == "1"


def cm(model, zin, level):
    c_skip, c_out, c_in = [append_dims(s, zin.ndim) for s in get_scalings_for_boundary_condition(level)]
    f = lambda z: torch.clamp(c_out * model(c_in * z, 1000 * 0.25 * torch.log(level + 1e-44)) + c_skip * z, -1.0, 1.0)
    if not EQ:
        return f(zin)
    fl = "flip" in EQ and bool(torch.rand(()) < 0.5); S = int(EQ.split("shift")[1]) if "shift" in EQ else 0
    dx, dy = (int(torch.randint(-S, S + 1, ())), int(torch.randint(-S, S + 1, ()))) if S else (0, 0)
    z = torch.flip(zin, dims=[-1]) if fl else zin; z = torch.roll(z, (dy, dx), dims=(-2, -1)) if S else z
    o = f(z); o = torch.roll(o, (-dy, -dx), dims=(-2, -1)) if S else o
    return torch.flip(o, dims=[-1]) if fl else o


def _rank(a):
    return torch.argsort(torch.argsort(a.flatten())).float() / a.numel()


def umap(zt, xt, pr):
    m = Fn.avg_pool2d(((zt - xt) ** 2).mean(1, keepdim=True), 15, stride=1, padding=7, count_include_pad=False)
    if state["gt"] is not None:                               # log only: does the map point at the true error of z?
        x0 = state["gt"].to(zt.device)
        if pr:
            x0 = x0 * torch.sign((zt * x0).sum(dim=(2, 3), keepdim=True))
        e = Fn.avg_pool2d(((zt - x0) ** 2).mean(1, keepdim=True), 8).flatten(); b = Fn.avg_pool2d(m, 8).flatten()
        XLOG.append(float(torch.corrcoef(torch.stack([_rank(b), _rank(e)]))[0, 1]))
    return m


def explore_step(model, zt, ut_m, xt, sig, lvl, pr):
    r = zt + ut_m; m = umap(zt, xt, pr); n = torch.randn_like(r); calls = 1
    if EXPLORE == "w":
        w2 = (m / m.mean()).clamp(0.25, 4.0); w2 = w2 / w2.mean()
        x = cm(model, r + sig * w2.sqrt() * n, lvl)
    elif EXPLORE == "g":
        x = cm(model, r + XK * sig * n, torch.clamp(XK * lvl, max=80.0))
    else:
        xs = cm(model, r + sig * n, lvl); xb = cm(model, r + XK * sig * n, torch.clamp(XK * lvl, max=80.0)); calls = 2
        M = ((_rank(m).view_as(m) - 0.6) / 0.2).clamp(0, 1)
        x = M * xb + (1 - M) * xs
    return x, calls


# ---- noise level told to the CM chosen from the measured uncertainty of the iterate (2026-11-17) ----
# LEVEL=match : told level = sqrt(sigma_t^2 + E^2), E = RMS error of the iterate at this (pass, step) from the calibration tables
# LEVEL=max   : the larger of that and the shipped level (only corrects steps where the shipped level under-reports)
LEVEL = os.environ.get("LEVEL", "")
SPECL = {k: float(sum(v) / len(v)) for k, v in json.load(open(SPECFILE)).items()} if LEVEL and os.path.exists(SPECFILE) else {}


def told_level(sig, lvl, pas, i):
    key = f"{pas}_{i}"
    if not LEVEL or key not in SPECL or float(sig) < 0.05:
        return lvl
    m = torch.full_like(lvl, (float(sig) ** 2 + SPECL[key]) ** 0.5).clamp(0.002, 80.0)
    return torch.maximum(m, lvl) if LEVEL == "max" else m


# ---- diagnostic (2026-11-17): how much does the NON-GAUSSIAN error of the CM input cost? ---------------
# At every step the CM input is  x0 + e + sigma*n  (e = z + u - x0, the error the iterate already carries).
# On the side (trajectory unchanged) the same CM call is made with e replaced by
#   phase : a random field with exactly e's power spectrum but random phases (Gaussian-like, independent of the image)
#   white : white Gaussian noise with the same total power as e + sigma*n      (told the shipped level / the true level)
DIAG = os.environ.get("DIAG", "0") == "1"
DLOG = []
_ps = lambda a, b: float(10 * torch.log10(1 / (((a - b) / 2) ** 2).mean()))


def diag_step(model, r, nz, sig, lvl, xt, pas, i, pr):
    x0 = state["gt"].to(r.device)
    if pr:
        x0 = x0 * torch.sign((r * x0).sum(dim=(2, 3), keepdim=True))
    e = r - x0; er = float((e ** 2).mean().sqrt()); tot = (er ** 2 + float(sig) ** 2) ** 0.5
    W = torch.fft.fft2(torch.randn_like(e)); ep = torch.fft.ifft2(torch.fft.fft2(e).abs() * W / (W.abs() + 1e-12)).real
    ep = ep * er / float((ep ** 2).mean().sqrt())
    g = torch.randn_like(e) * tot
    ec = e - e.mean(); xc = x0 - x0.mean()
    if os.environ.get("DUMP", "0") == "1":                     # tensors for a figure
        torch.save({k: v.detach().float().cpu() for k, v in dict(x0=x0, iterate=r, in_actual=r + nz, out_actual=xt, in_phase=x0 + ep + nz,
                    out_phase=cm(model, x0 + ep + nz, lvl), in_white=x0 + g, out_white=cm(model, x0 + g, lvl)).items()} | {"sigma": float(sig), "told": float(lvl), "err": er},
                   f"{exp}/dump_{state['img'] + 1}_{pas}_{i}.pt")
    DLOG.append(dict(pas=pas, step=i, err=er, sigma=float(sig), told=float(lvl),
                     kurt=float((ec ** 4).mean() / (ec ** 2).mean() ** 2 - 3), corr_x0=float((ec * xc).mean() / (ec.std() * xc.std())),
                     actual=_ps(xt, x0), phase=_ps(cm(model, x0 + ep + nz, lvl), x0),
                     white=_ps(cm(model, x0 + g, lvl), x0), white_true=_ps(cm(model, x0 + g, torch.full_like(lvl, tot)), x0),
                     no_err=_ps(cm(model, x0 + nz, lvl), x0)))


# ---- (2026-11-17) Gaussian-residual test (no ground truth needed) and one-step input-space trial -------
# Test: for an input the CM was trained on (image + white Gaussian noise at level L), the residual  input - CM(input)
#   is close to white noise of size ~L.  res_stats measures size/L, spectral flatness (1 = white), and correlation with the output.
# Trial (TRIAL=1, at the last step with sigma_t >= 0.05): optimise the CM INPUT r, starting from the shipped input, so that
#   the CM OUTPUT fits the measurement (normalised gradient steps with backtracking), optionally re-imposing after every step
#   that the residual r - CM(r) has the size (shell) / size and flat spectrum (flat) of a true Gaussian input at that level.
TRIAL = os.environ.get("TRIAL", "0") == "1"; TK = int(os.environ.get("TK", "5")); TETA = float(os.environ.get("TETA", "0.1"))
_rms = lambda a: float((a ** 2).mean().sqrt())


def f_data(P, x):
    if P.y is not None:
        rr = P.A(x) - P.y
        return 0.5 * (rr * rr).sum()
    Ax = P.A(x)
    return 0.5 * (Ax * Ax).sum() - (x * P.ATy).sum()


def _bandspec(res):
    fb, cnt = _bands(res.device)
    Fr = torch.fft.fft2(res, norm="ortho"); p = (Fr.abs() ** 2).mean(dim=(0, 1)).flatten()
    return Fr, torch.bincount(fb, weights=p, minlength=NB) / cnt, fb


def res_stats(res, out, L):
    _, sp, _ = _bandspec(res); oc = out - out.mean(); rc = res - res.mean()
    return dict(size=_rms(res) / float(L), flat=float(torch.exp(torch.log(sp + 1e-20).mean()) / sp.mean()),
                lowhigh=float(sp[:16].mean() / sp[48:].mean()), corr=float((rc * oc).mean() / (rc.std() * oc.std())))


def flatten_res(res, tgt):
    Fr, sp, fb = _bandspec(res)
    out = torch.fft.ifft2(Fr * (sp.mean() / (sp + 1e-20)).sqrt()[fb].view(256, 256), norm="ortho").real
    return out * tgt / _rms(out)


def trial(P, model, r0, L, tgt, data, shell, flat, eta=None):
    r = r0.clone(); Dr = cm(model, r, L); calls = 1
    for _ in range(TK):
        if data:
            with torch.enable_grad():
                rr = r.detach().requires_grad_(True); F = f_data(P, cm(model, rr, L)); g, = torch.autograd.grad(F, rr)
            calls += 3; gr = g / (_rms(g) + 1e-20); a = (eta or TETA) * float(L)
            for _ in range(4):
                rc = r - a * gr; oc = cm(model, rc, L); calls += 1
                if float(f_data(P, oc)) < float(F):
                    r, Dr = rc, oc; break
                a *= 0.5
        if shell or flat:
            res = r - Dr
            res = flatten_res(res, tgt) if flat else res * tgt / _rms(res)
            r = Dr + res; Dr = cm(model, r, L); calls += 1
    return r, Dr, calls


def trial_step(P, model, r_in, xt, sig, lvl, rho, x0, d):
    go = x0 + float(lvl) * torch.randn_like(x0); og = cm(model, go, lvl)          # a true Gaussian input at the told level
    d["res_actual"] = res_stats(r_in - xt, xt, lvl); d["res_gauss"] = res_stats(go - og, og, lvl)
    if not TRIAL:
        return
    tgt = _rms(go - og); f0 = float(f_data(P, xt)); ft = float(f_data(P, x0)); out = {}
    out["dc_image"] = _ps(gauss_newton(P, xt, xt, rho, None), x0)                 # control: data-fit of the output in image space
    for name, (da, sh, fl, et) in {"cons_only": (0, 0, 1, None), "data": (1, 0, 0, None), "data_shell": (1, 1, 0, None), "data_flat": (1, 0, 1, None),
                                   "data_small": (1, 0, 0, TETA / 5), "data_small_flat": (1, 0, 1, TETA / 5)}.items():
        r, Dr, calls = trial(P, model, r_in, lvl, tgt, da, sh, fl, et)
        out[name] = _ps(Dr, x0); out[name + "_calls"] = calls; out[name + "_res"] = res_stats(r - Dr, Dr, lvl)
        out[name + "_fit"] = float(f_data(P, Dr)) - f0
        if name in ("data", "data_small"):                     # why does it help or hurt?  (log only, uses the truth)
            out[name + "_excess"] = float(f_data(P, Dr)) - ft   # misfit minus the TRUE image's misfit: < 0 means fitting the measurement noise
            if P.ATy is not None:
                sv = P.F.add_zeros(P.F.singulars()[None])[0]; ms = sv > 0.3 * float(sv.max())
                for tag, img in (("before", xt), ("after", Dr)):
                    c = P.F.Vt(img - x0)[0] ** 2
                    out[f"{name}_err_meas_{tag}"] = float(c[ms].sum()); out[f"{name}_err_unmeas_{tag}"] = float(c[~ms].sum())
    out["excess_before"] = f0 - ft
    d["trial"] = out


# ---- INVIN=1 (linear tasks, 2026-11-17): build the CM input from the INVERTED measurement where that is possible ----
# In the operator's singular basis, component i of the measurement gives  y_i / s_i = x0_i + (sigma_y / s_i) * Gaussian.
# Where sigma_y / s_i <= sigma_t the input component is  y_i / s_i + sqrt(sigma_t^2 - sigma_y^2 / s_i^2) * eps   (exactly truth + white
# Gaussian noise at sigma_t, no dependence on the current estimate); elsewhere the shipped iterate component + sigma_t * eps.
INVIN = os.environ.get("INVIN", "0") == "1"; SIGY = 2 * 0.05
_SV = {}


def inv_input(P, r, sig):
    F_ = P.F
    _SV["s"] = F_.add_zeros(F_.singulars()[None])[0]; _SV["cy"] = F_.Vt(P.ATy)[0]     # per image: the measurement changes
    sv = _SV["s"]; sig = float(sig); ok = (sv > 0) & (SIGY / sv.clamp(min=1e-12) <= sig)
    cd = _SV["cy"] / sv.clamp(min=1e-12) ** 2; eps = torch.randn_like(sv)
    c = torch.where(ok, cd + (sig ** 2 - (SIGY / sv.clamp(min=1e-12)) ** 2).clamp(min=0).sqrt() * eps, F_.Vt(r)[0] + sig * eps)
    state["invfrac"] = float(ok.float().mean())
    if state["gt"] is not None and state.get("need_img") != state["img"]:
        # log only: per band of components (sorted by singular value), uncertainty = min(typical size of the image there, amplified
        # measurement noise).  The noise needed to cover what the measurement leaves open is the largest band value.
        g2 = F_.Vt(state["gt"].to(r.device))[0] ** 2; order = torch.argsort(sv, descending=True); nb = 64; m = (order.numel() // nb) * nb
        prior = g2[order][:m].view(nb, -1).mean(1).sqrt(); amp = SIGY / sv[order][:m].view(nb, -1).mean(1).clamp(min=1e-12)
        unc = torch.minimum(prior, amp); state["need"] = [float(unc.max()), float((unc ** 2).mean().sqrt()), float(prior.max())]; state["need_img"] = state["img"]
    return F_.V(c[None]).reshape(r.shape)


# ---- budget-compliant cold start (2026-11-17): same number of network calls as the baseline, no second pass -------------
# A start is "cold" when the starting image explains the measurement badly: ||A(z0) - y|| / ||y|| > COLDTHR.  Only then, and only
# for the first data-fit of the run, use the repo's accumulated-gradient loop (WARM_N iterations); every other solve is Gauss-Newton.
WARMUP = os.environ.get("WARMUP", "0") == "1"; COLDTHR = float(os.environ.get("COLDTHR", "0.5"))
WARM_N = int(os.environ.get("WARM_N", "1000")); WARM_LR = float(os.environ.get("WARM_LR", "0.4"))


def cold_start(P, z):
    P.n["A"] += 1
    rel = float((P.A(z) - P.y).norm() / (P.y.norm() + 1e-12)); state["relres"] = rel
    return rel > COLDTHR


def accum_solve(P, z, v, rho):
    lr = WARM_LR; f_pre = 0.0; acc = torch.zeros_like(z)
    for _ in range(WARM_N):
        f, g = P.grad(z)
        f = float(f + 0.5 * rho * ((z - v) ** 2).sum()); g = g + rho * (z - v)
        acc = acc + g; z = z - lr * acc
        if f > f_pre:
            lr *= 0.95
        f_pre = f
    state["warm"] = True
    return z


FITK = int(os.environ.get("FITK", "0")); FITPASS = int(os.environ.get("FITPASS", "1"))   # through-model fit at the last FITK steps of pass FITPASS
HFK = float(os.environ.get("HFK", "0")); HFSTEPS = os.environ.get("HFSTEPS", "1").split(",")
SKIPDC = set(os.environ.get("SKIPDC", "").split(",")) - {""}   # "pass:step" pairs (0-based) where the measurement fit is skipped


LLE_FILE = os.environ.get("LLE", ""); LLE_ON = bool(LLE_FILE)
LLE_C = ((torch.load(LLE_FILE) if LLE_FILE.endswith(".pt") else json.load(open(LLE_FILE))) if (LLE_ON and os.path.exists(LLE_FILE)) else {})


DISC = float(os.environ.get("DISC", "0")); DAMP = float(os.environ.get("DAMP", "0")); DAMPSTEPS = os.environ.get("DAMPSTEPS", "")
NULLNOISE = os.environ.get("NULLNOISE", ""); DIAG2 = os.environ.get("DIAG2", "0") == "1"


def _svd_parts(P):
    """Cached SVD quantities of a linear problem: singular values (full length), U'y on the measured directions, measured mask."""
    if not hasattr(P, "_svd"):
        sv = P.F.add_zeros(P.F.singulars()[None])[0]; c = P.F.Vt(P.ATy)[0]; meas = sv > 0.05 * float(sv.max())
        P._svd = (sv, torch.where(meas, c / sv.clamp(min=1e-8), torch.zeros_like(c)), meas)
    return P._svd


def meas_loss(P, x):
    """Squared measurement residual of image x and the number of scalar measurements (noise-normalised value = loss / (m sigma_y^2))."""
    if P.y is not None:
        return float(((P.A(x) - P.y) ** 2).sum()), P.y.numel()
    sv, uy, meas = _svd_parts(P); xv = P.F.Vt(x)[0]
    return float((((sv * xv - uy) ** 2) * meas).sum()), int(meas.sum())


def disc_prox(P):
    """Data-fit whose penalty is NOT a tuned number: for each image and step it is the one value for which the data-fit's residual equals
    the expected noise energy (discrepancy principle), found by bisection on closed forms.  Linear problems."""
    sv, uy, meas = _svd_parts(P); c = P.F.Vt(P.ATy)[0]; s2 = sv ** 2; tgt = DISC * float(meas.sum()) * SIGY ** 2

    def prox(v, rho):
        vv = P.F.Vt(v)[0]; f = lambda r: float((((sv * (c + r * vv) / (s2 + r) - uy) ** 2) * meas).sum())
        lo, hi = 1e-5, 1e5
        if f(hi) <= tgt:                                        # the model's image already explains the data to within the noise: keep it
            r = hi
        elif f(lo) >= tgt:
            r = lo
        else:
            for _ in range(40):
                mid = (lo * hi) ** 0.5
                if f(mid) > tgt: hi = mid
                else: lo = mid
            r = (lo * hi) ** 0.5
        state.setdefault("disc_rho", []).append(r)
        return P.F.V(((c + r * vv) / (s2 + r))[None]).reshape(v.shape)
    return prox


def lle_region_mask(P, like):
    """1 where a pixel is well measured, 0 elsewhere, read off the measurement itself (no ground truth): unclipped pixels for clipping tasks,
    kept pixels for inpainting; all ones for tasks without such a split."""
    one = torch.ones(1, 1, like.shape[-2], like.shape[-1], device=like.device)
    try:
        if NEWOP in ("hdr", "blurclip") and P.y is not None and P.y.shape[-2:] == like.shape[-2:]:
            return (P.y.abs() < 1.0 - 2 * SIGY).float()
        if deg == "inpainting" and getattr(P, "ATy", None) is not None:
            return (P.ATy.reshape(1, 3, like.shape[-2], like.shape[-1]).abs().sum(1, keepdim=True) > 0).float().expand(1, 3, -1, -1)
    except Exception:
        pass
    return one.expand(1, 3, -1, -1)


def lle_band_index(n, B, device):
    """Radial frequency band of every 2-D FFT coefficient: band 0 below 1.5 cycles per image, then octaves."""
    f = torch.fft.fftfreq(n, device=device) * n; r = (f[:, None] ** 2 + f[None, :] ** 2).sqrt()
    return torch.bucketize(r, torch.tensor([1.5 * 2 ** k for k in range(B - 1)], device=device))


def run_steps(P, model, st, sched, idx, nfe, pas=0):
    """PnP-CM outer loop over the schedule steps in idx; st carries (zt, xt, xt_m, ut, ut_m, z1)."""
    sig_t, sig_cm, rho, mom = sched
    zt, xt, xt_m, ut, ut_m, z1 = st
    for n_it, i in enumerate(idx):
        xt_1, ut_1 = xt.clone(), ut.clone()
        skip = f"{pas}:{i}" in SKIPDC                           # prior-only step: no measurement fit, no dual update
        if skip:
            zt = zt if n_it == 0 else xt.clone(); ut_m = torch.zeros_like(ut_m)
        elif WARMUP and pas == 0 and n_it == 0 and P.y is not None and cold_start(P, zt):
            zt = accum_solve(P, zt, xt_m - ut_m, rho[i])       # cold start: robust first-order solve for this one step
        else:
            state["step"] = (n_it, len(idx))
            zt = gauss_newton(P, zt, xt_m - ut_m, rho[i], sig_t[i])
        if z1 is None:
            z1 = zt.clone()
        if EXPLORE and n_it > 0 and float(sig_t[i]) >= 0.05:
            xt, calls = explore_step(model, zt, ut_m, xt, sig_t[i], sig_cm[i], deg == "phase_retrieval")
            nfe[0] += calls
        else:
            state["P"] = P; state["told"] = float(sig_cm[i])
            nz = inject(zt + ut_m, sig_t[i], pas, i, deg == "phase_retrieval")
            if NULLNOISE and P.ATy is not None:
                # noise only where the measurement says little: "hard" = none in the measured directions; "keep" = same, rescaled to the same total power
                sv_, _, meas_ = _svd_parts(P); nv = P.F.Vt(nz)[0] * (~meas_).float(); nn_ = P.F.V(nv[None]).reshape(nz.shape)
                nz = nn_ * (float(nz.norm() / nn_.norm().clamp(min=1e-8)) if NULLNOISE == "keep" else 1.0)
            if NREUSE > 0 and n_it > 0 and state.get("eps_prev") is not None:
                # deterministic (DDIM-like) re-noising: re-use the part of the previous model input that the model removed, instead of all-fresh noise
                ep_ = state["eps_prev"]; ep_ = ep_ * (nz.norm() / ep_.norm().clamp(min=1e-8)); nz = NREUSE * ep_ + (1.0 - NREUSE ** 2) ** 0.5 * nz
            z_noisy = zt + ut_m + nz
            if INVIN and P.ATy is not None and float(sig_t[i]) >= 0.05:
                z_noisy = inv_input(P, xt if (INVSRC == "x" and n_it > 0) else zt + ut_m, sig_t[i]); nz = z_noisy - (zt + ut_m)
                if pas == 0 and state["img"] < S0:
                    print("INVIN fraction of components taken from the inverted measurement:", round(state["invfrac"], 3), "at sigma", round(float(sig_t[i]), 3), flush=True)
            xt = cm(model, z_noisy, told_level(sig_t[i], sig_cm[i], pas, i) * (FINTOLD if n_it == len(idx) - 1 else 1.0))
            if NREUSE > 0:
                state["eps_prev"] = (z_noisy - xt).detach()
            if DIAG3 and state["gt"] is not None:
                # What makes the model's input error hard?  Same truth, same added noise, three versions of the error (diagnostic only, uses the truth):
                #   actual error | the same error with random Fourier phases (same spectrum, no structure) | white noise of the same power | no error
                g3 = state["gt"].to(zt.device).float(); e3 = (zt + ut_m).float() - g3; lv3 = told_level(sig_t[i], sig_cm[i], pas, i) * (FINTOLD if n_it == len(idx) - 1 else 1.0)
                E3 = torch.fft.fft2(e3, norm="ortho"); e_s = torch.fft.ifft2(E3.abs() * torch.exp(1j * torch.angle(torch.fft.fft2(torch.randn_like(e3)))), norm="ortho").real
                e_w = torch.randn_like(e3); e_w = e_w * e3.norm() / e_w.norm(); bi3 = lle_band_index(256, 8, zt.device)
                be3 = lambda a: [float(v) for v in torch.zeros(8, device=zt.device).index_add_(0, bi3.reshape(-1).repeat(3), (torch.fft.fft2(a.float(), norm="ortho").abs() ** 2).reshape(-1)) / a.numel()]
                with torch.no_grad():
                    outs3 = [xt] + [cm(model, (g3 + ee + nz).to(z_noisy.dtype), lv3) for ee in (e_s, e_w, torch.zeros_like(e3))]
                state.setdefault("diag3", []).append([be3(e3)] + [be3(o - g3) for o in outs3] + [be3(nz)])
            if DIAG2 or DAMP > 0:
                state["raw_x"] = xt.clone()
            if DAMP > 0 and (not DAMPSTEPS or str(n_it) in DAMPSTEPS.split(",")) and (P.y is not None or P.ATy is not None):
                # measurement-preserving damping: keep as much of the model's image as the measurement allows.  x(a) = z + a (model - z);
                # the largest a whose residual stays within DAMP x the expected noise energy (forward-operator evaluations only).
                _, m_ = meas_loss(P, zt); tol = DAMP * m_ * SIGY ** 2; best_a, best_l, best_a2 = 0.0, None, 0.0
                for a_ in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
                    l_ = meas_loss(P, zt + a_ * (xt - zt))[0]
                    if best_l is None or l_ < best_l: best_a2, best_l = a_, l_
                    if l_ <= tol: best_a = a_; break
                else:
                    best_a = best_a2
                state.setdefault("damp_a", []).append(best_a); xt = zt + best_a * (xt - zt)
            if FITK > 0 and pas == FITPASS and i in list(idx)[-FITK:]:
                # fit the measurement THROUGH the model: gradient steps on the CM input so that the CM output matches y
                _, xt, calls = trial(P, model, z_noisy, told_level(sig_t[i], sig_cm[i], pas, i), None, 1, 0, 0)
                nfe[0] += calls - 1; state["fitcalls"] = state.get("fitcalls", 0) + calls - 1
            if HFK > 0 and P.ATy is not None and float(sig_t[i]) >= 0.05 and str(i) in HFSTEPS:
                # sharper prior estimate: a second CM call at HFK x the noise (same noise draw) regenerates detail; keep from it only the
                # components the measurement does not pin down at this level (high frequencies for blur), the rest from the standard call
                x_hi = cm(model, zt + ut_m + HFK * nz, torch.clamp(HFK * sig_cm[i], max=80.0)); nfe[0] += 1
                sv = P.F.add_zeros(P.F.singulars()[None])[0]; okc = (sv > 0) & (SIGY / sv.clamp(min=1e-12) <= float(sig_t[i]))
                xt = P.F.V(torch.where(okc, P.F.Vt(xt)[0], P.F.Vt(x_hi)[0])[None]).reshape(xt.shape).clamp(-1.0, 1.0)
            if DIAG and state["gt"] is not None:
                diag_step(model, zt + ut_m, nz, sig_t[i], sig_cm[i], xt, pas, i, deg == "phase_retrieval")
                if float(sig_t[i]) >= 0.05:
                    last_hi = max(j for j in idx if float(sig_t[j]) >= 0.05)
                    x0g = state["gt"].to(zt.device)
                    if deg == "phase_retrieval":
                        x0g = x0g * torch.sign(((zt + ut_m) * x0g).sum(dim=(2, 3), keepdim=True))
                    if i == last_hi or not TRIAL:
                        trial_step(P, model, z_noisy, xt, sig_t[i], sig_cm[i], rho[i], x0g, DLOG[-1])
            nfe[0] += 1
        if DIAG2 and (P.y is not None or P.ATy is not None):
            g_ = state["gt"].to(zt.device) if state["gt"] is not None else None; rx = state.get("raw_x", xt)
            lz, m_ = meas_loss(P, zt); lx, _ = meas_loss(P, rx); nrm = m_ * SIGY ** 2; row = [lz / nrm, lx / nrm, float(((rx - zt) ** 2).mean().sqrt())]
            if g_ is not None:
                row.append(float(10 * torch.log10(1 / (((xt - g_) / 2) ** 2).mean())))
                if P.ATy is not None:
                    sv_, _, meas_ = _svd_parts(P); e_ = P.F.Vt(xt - g_)[0]; row += [float((e_ ** 2 * meas_).sum()) / e_.numel(), float((e_ ** 2 * (~meas_)).sum()) / e_.numel(), float(meas_.float().mean())]
            state.setdefault("diag2", []).append(row)
        if os.environ.get("RESLOG", "0") == "1":
            # per step: primal residual |z - x| (how far the data-fit had to move the model's image), change of the model's image from the
            # previous step, and (if the truth is known, for analysis only) the quality of the image at this step
            g_ = state["gt"].to(zt.device) if state["gt"] is not None else None
            if g_ is not None and deg == "phase_retrieval":
                g_ = g_ * torch.sign((xt * g_).sum(dim=(2, 3), keepdim=True))
            state.setdefault("reslog", []).append([float(((zt - xt) ** 2).mean().sqrt()), float(((xt - xt_1) ** 2).mean().sqrt()),
                                                   float(10 * torch.log10(1 / (((xt - g_) / 2) ** 2).mean())) if g_ is not None else float("nan")])
        if LLE_ON:
            # linear extrapolation over the run's own denoised images (after Zhang et al., "Learnable Linear Extrapolation", 2025):
            # the image passed on from step n is a learned linear combination of this step's output and the earlier passed-on images
            hist = state.setdefault("lle_hist", [])
            if n_it == 0:
                hist.clear()
            STEPLOG.append((pas, 100 + n_it, xt.detach().half().cpu())); STEPLOG.append((pas, 200 + n_it, zt.detach().half().cpu()))   # model output, data-fit image
            zh_ = state.setdefault("lle_zh", [])
            if n_it == 0:
                zh_.clear(); state["lle_mask"] = lle_region_mask(P, xt); STEPLOG.append((pas, 300, state["lle_mask"].detach().half().cpu()))
            zh_.append(zt.to(xt.dtype))
            cf = LLE_C.get(str(n_it))
            if cf is not None and isinstance(cf, dict) and "kind" in cf:
                # structured combinations.  terms = earlier passed-on images (+ this step's data-fit image if cf["z"]) + this step's model output
                stk = torch.cat(hist + (list(zh_) if cf.get("zall") else [zt.to(xt.dtype)] if cf.get("z") else []) + [xt], 0); Wc = cf["W"].to(xt.device).float()
                if cf["kind"] == "regband":     # weights per term, frequency band AND measurement region (well-measured pixels / the rest)
                    Fs_ = torch.fft.fft2(stk, norm="ortho"); bi_ = lle_band_index(xt.shape[-1], Wc.shape[2], xt.device); Mr_ = state["lle_mask"].to(xt.dtype)
                    cb_ = [torch.fft.ifft2((Wc[r_][:, bi_][:, None] * Fs_).sum(0, keepdim=True), norm="ortho").real for r_ in range(2)]
                    xt = (Mr_ * cb_[0] + (1 - Mr_) * cb_[1]).clamp(-1.0, 1.0)
                if cf["kind"] == "regband":
                    pass
                elif cf["kind"] == "band":        # one weight per term and radial frequency band
                    xt = torch.fft.ifft2((Wc[:, lle_band_index(xt.shape[-1], Wc.shape[1], xt.device)][:, None] * torch.fft.fft2(stk, norm="ortho")).sum(0, keepdim=True), norm="ortho").real.clamp(-1.0, 1.0)
                elif cf["kind"] == "gate":      # weights chosen per pixel by how much the model output moved since the previous step (nonlinear)
                    gi = torch.bucketize((xt - hist[-1]).abs().mean(1)[0], cf["thr"].to(xt.device))
                    xt = (Wc[:, gi][:, None] * stk).sum(0, keepdim=True).clamp(-1.0, 1.0)
            elif cf is not None and isinstance(cf, dict):         # weights per coordinate of a 30-dimensional embedding (+ one weight for the rest)
                Ub = cf["U"].to(xt.device); hs = torch.cat(hist + [xt], 0).flatten(1); co = hs @ Ub.T; rest = hs - co @ Ub
                xt = ((cf["A"].to(xt.device) * co).sum(0) @ Ub + (cf["s"].to(xt.device)[:, None] * rest).sum(0)).view_as(xt).clamp(-1.0, 1.0)
            elif cf is not None and torch.is_tensor(cf):          # weight MAPS: one weight per pixel (and colour) for every image in the combination
                xt = (cf.to(xt.device).float() * torch.cat(hist + [xt], 0)).sum(0, keepdim=True).clamp(-1.0, 1.0)
            elif cf is not None:
                xt = sum(float(c_) * h_ for c_, h_ in zip(cf, hist + [xt])).clamp(-1.0, 1.0)
            hist.append(xt)
        if os.environ.get("STEPDIAG", "0") == "1" and state["gt"] is not None and P.y is not None:
            g_ = state["gt"].to(zt.device); ps_ = lambda a: float(10 * torch.log10(1 / (((a.clamp(-1, 1) - g_) / 2) ** 2).mean()))
            ag_ = lambda a: float((torch.sign(P.A(a)) == torch.sign(P.y)).float().mean())
            state.setdefault("stepdiag", []).append([ps_(zt), ag_(zt), ps_(xt), ag_(xt)])
        STEPLOG.append((pas, i, xt.detach().half().cpu()))      # every denoised image of the run (for iterate averaging)
        ut = ut_1 if skip else ut_m + (zt - xt)
        if os.environ.get("NODUAL", "0") == "1":               # no dual variable: the data-fit is anchored on the model's image itself
            ut = torch.zeros_like(ut)
        xt_m = xt + mom[i] * (xt - xt_1); ut_m = ut + mom[i] * (ut - ut_1)
    return (zt, xt, xt_m, ut, ut_m, z1)


def init_state(z0, x0):
    return (z0.clone(), x0.clone(), x0.clone(), torch.zeros_like(z0), torch.zeros_like(z0), None)


def one_pass(P, model, z0, x0, sched, nfe, pas=0):
    st = run_steps(P, model, init_state(z0, x0), sched, range(len(sched[0])), nfe, pas)
    return st[1], st[5]


def schedule(betas, iN, gamma, deltas, rhos, mu_0, T, dev, sigma_min=0.002, sigma_max=80.0):
    rho = nn.Softplus()(torch.linspace(rhos[0], rhos[1], T)).to(dev)
    mom = (torch.tensor([mu_0]).repeat(T) / torch.arange(1, T + 1)).to(dev)
    al = [compute_alpha(betas, (torch.ones(1) * (iN + 1)).to(dev).long())]
    for _ in range(T - 1):
        al.append(get_next_alpha(al[-1], gamma).reshape(1))
    sig_t = [torch.clamp(torch.sqrt(1 - a), sigma_min, sigma_max) for a in al]
    sig_cm = [torch.clamp((1 + deltas[i]) * torch.sqrt(1 - al[i]), sigma_min, sigma_max) for i in range(T)]
    if SIGS:                                                   # custom noise schedule (2026-11-17): told level = SIGTOLD x sigma
        sv = [float(v) for v in SIGS.split(",")]; T = len(sv)
        if SIGPOS:
            # each new step is given a (fractional) position in the SHIPPED schedule; the level told to the model (as a ratio to the
            # injected noise) and the penalty weight are interpolated from the shipped values at that position, so that changing the
            # number of steps does not silently change those tuned quantities.
            import numpy as _np
            pos = [float(v) for v in SIGPOS.split(",")]; T0 = len(sig_t); xs = _np.arange(T0)
            ratio0 = [float(sig_cm[i] / sig_t[i]) for i in range(T0)]; rho0 = [float(v) for v in rho]
            rat = _np.interp(pos, xs, ratio0); rh = _np.interp(pos, xs, rho0)
            sig_cm = [sig_t[0] * 0 + v * r_ for v, r_ in zip(sv, rat)]; rho = torch.tensor(rh, dtype=torch.float32).to(dev)
        else:
            sig_cm = [sig_t[0] * 0 + v * (SIGTOLD if v >= 0.05 else 1.1) for v in sv]
            rho = nn.Softplus()(torch.linspace(rhos[0], rhos[1], T)).to(dev)
        sig_t = [sig_t[0] * 0 + v for v in sv]; mom = (torch.tensor([mu_0]).repeat(T) / torch.arange(1, T + 1)).to(dev)
    return sig_t, sig_cm, rho, mom


INVSRC = os.environ.get("INVSRC", "r")                    # where non-invertible components come from: r = iterate z+u, x = previous denoised image
SIGPOS = os.environ.get("SIGPOS", "")
SIGS = os.environ.get("SIGS", ""); SIGTOLD = float(os.environ.get("SIGTOLD", "1.2"))


def is_cold(x, z0, sig1):
    move0 = float(((x - z0) ** 2).mean().sqrt())
    spread = float(((x - x.mean(dim=(2, 3), keepdim=True)) ** 2).mean().sqrt())
    return (move0 >= spread) if RULE == "r2" else (move0 > sig1), move0, spread


def unified_half(P, model, z0, x0, sched):
    """Same number of model calls as the baseline: spend half the steps, then decide.
    cold -> restart from the current image and spend the other half as a fresh short pass;
    warm -> simply continue the remaining steps (identical to a normal single pass)."""
    nfe = [0]; T = len(sched[0]); h = T // 2
    first = list(range(h)) if HALF == "fl" else list(range(0, T, 2))
    rest = [i for i in range(T) if i not in first]
    torch.cuda.synchronize(); t0 = time.time()
    st = run_steps(P, model, init_state(z0, x0), sched, first, nfe)
    x = st[1]; cold, move0, spread = is_cold(x, z0, float(sched[0][0]))
    x_first = x; nfe1 = nfe[0]; ops1 = dict(P.n)
    if cold:
        st = run_steps(P, model, init_state(x, x), sched, rest, nfe, pas=1)
    else:
        st = run_steps(P, model, st, sched, rest, nfe)
    torch.cuda.synchronize()
    return st[1], {"time": time.time() - t0, "nfe": nfe[0], **P.n, "first": {"nfe": nfe1, **ops1}, "move": move0, "move0": move0,
                   "sigma1": float(sched[0][0]), "spread": spread, "second_pass": bool(cold)}, x_first


COMBO = os.environ.get("COMBO", "")


def unified_combo(P, model, z0, x0, sched):
    """Other ways to spend the same number of model calls (2026-11-20).
      avg / avgfit : the whole (short) schedule is run TWICE with different noise, the two outputs are averaged (avgfit: then one free data-fit)
      fork         : all steps but the last are run once, the LAST step twice with different noise, outputs averaged
      search       : the FIRST step is run twice, the two images averaged; the average is the start of the remaining steps
    Model calls are counted in nfe; data-fits are closed-form and free."""
    nfe = [0]; torch.cuda.synchronize(); t0 = time.time(); T = len(sched[0])
    if COMBO in ("avg", "avgfit"):
        x = torch.stack([one_pass(P, model, z0, x0, sched, nfe)[0] for _ in range(2)]).mean(0)
    elif COMBO in ("fork", "forkfit"):
        st = run_steps(P, model, init_state(z0, x0), sched, range(T - 1), nfe)
        x = torch.stack([run_steps(P, model, st, sched, [T - 1], nfe)[1] for _ in range(2)]).mean(0)
    else:
        xa = torch.stack([run_steps(P, model, init_state(z0, x0), sched, [0], nfe)[1] for _ in range(2)]).mean(0)
        x = run_steps(P, model, init_state(xa, xa), sched, range(1, T), nfe)[1]
    if COMBO.endswith("fit"):
        state["step"] = (T - 1, T); x = gauss_newton(P, x, x, sched[2][-1], sched[0][-1]).clamp(-1.0, 1.0)
    torch.cuda.synchronize()
    return x, {"time": time.time() - t0, "nfe": nfe[0], **P.n, "first": {}, "move": 0.0, "move0": 0.0, "sigma1": float(sched[0][0]), "spread": 0.0, "second_pass": False}, x


def unified(P, model, z0, x0, sched):
    if COMBO:
        return unified_combo(P, model, z0, x0, sched)
    if HALF:
        return unified_half(P, model, z0, x0, sched)
    nfe = [0]
    if state["img"] < S0:
        print("SCHED sigma_t", [round(float(v), 3) for v in sched[0]], "sigma_cm", [round(float(v), 3) for v in sched[1]], flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    x, z1 = one_pass(P, model, z0, x0, sched, nfe)
    move = float(((x - z1) ** 2).mean().sqrt()); sig1 = float(sched[0][0])
    move0 = float(((x - z0) ** 2).mean().sqrt())            # from the image the sampler was handed
    spread = float(((x - x.mean(dim=(2, 3), keepdim=True)) ** 2).mean().sqrt())   # RMS of the output about its own mean
    if RULE == "r2":      # start explains the output no better than a constant image -> the start carried no usable information
        cold = move0 >= spread
    else:                 # "noise": output moved further from the start than the first-step noise
        cold = move0 > sig1
    again = (RESTART == "always") or (RESTART == "rule" and cold)
    x_first = x; nfe1 = nfe[0]; ops1 = dict(P.n)
    if again:                                                  # start was out of the noise's reach: go again from x
        x, _ = one_pass(P, model, x, torch.zeros_like(x), sched, nfe, pas=1)
    torch.cuda.synchronize()
    return x, {"time": time.time() - t0, "nfe": nfe[0], **P.n, "first": {"nfe": nfe1, **ops1}, "move": move, "move0": move0, "sigma1": sig1, "spread": spread, "second_pass": bool(again)}, x_first


def finish(x, rec, x_first=None):
    state["img"] += 1
    rec["img"] = state["img"]
    if "hio" in state:
        rec["hio_psnr_kept"], rec["hio_relres"] = state.pop("hio")
    if "relres" in state:
        rec["relres"] = state.pop("relres"); rec["warm"] = bool(state.pop("warm", False))
    if state.get("stepdiag"):
        rec["stepdiag"] = state.pop("stepdiag")
    if "sign_ok" in state:
        rec["sign_ok"] = state.pop("sign_ok")
    if state.get("reslog"):
        rec["reslog"] = state.pop("reslog")
    for k_ in ("diag2", "damp_a", "disc_rho", "diag3", "prdJ", "bcJ", "bc_active"):
        if state.get(k_):
            rec[k_] = state.pop(k_)
    if CALW and state.get("calacc"):
        os.makedirs(os.path.abspath("../mode_probe/cal"), exist_ok=True)
        json.dump({k: {"spec": [a_ / c[2] for a_ in c[0]], "mse": c[1] / c[2], "n": c[2]} for k, c in state["calacc"].items()}, open(os.path.abspath(f"../mode_probe/cal/{CALW}.json"), "w"))
    if state.get("need"):
        rec["need"] = state["need"]
    if state.get("fitcalls"):
        rec["fitcalls"] = state["fitcalls"]; state["fitcalls"] = 0
    if XLOG:
        rec["map_corr"] = float(sum(XLOG) / len(XLOG)); rec["map_steps"] = len(XLOG); XLOG.clear()
    if DLOG:
        rec["diag"] = list(DLOG); DLOG.clear()
    state["stats"].append(rec)
    torch.save({"x_raw": x.cpu(), "x_first": None if x_first is None else x_first.cpu(),
                "steps": list(STEPLOG) if SAVE_STEPS else None}, f"{exp}/raw_{state['img']}.pt")
    STEPLOG.clear()
    json.dump(state["stats"], open(f"{exp}/stats.json", "w"))
    if NOISE in ("calib", "firecal"):                         # running mean table per (pass, step)
        os.makedirs(os.path.dirname(SPECFILE), exist_ok=True)
        import numpy as _np
        json.dump({k: _np.mean(v, 0).tolist() for k, v in state["speclog"].items()}, open(SPECFILE, "w"))
        json.dump({k: len(v) for k, v in state["speclog"].items()}, open(SPECFILE.replace(".json", "_counts.json"), "w"))
    print(f"[{variant}] img {state['img']} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
    return x


# ---- post-processing of samples that already exist -------------------------------------------------
# FUSE="v1,v2,..." : load those variants' outputs for this image and average them (phase retrieval: each
# sample is first sign-aligned per channel to the first one, using only the samples).
# FUSE_MODE = avg     : the plain average
#             dc      : one data-fit solve anchored on the average (re-impose the measurement), last-step rho
#             dcpolish: dc, then the schedule's last noise+denoise step, then nothing else
FUSE = [v for v in os.environ.get("FUSE", "").split(",") if v]
FUSE_MODE = os.environ.get("FUSE_MODE", "dc")


def fuse(P, model, sched, pr):
    import glob as _g
    idx = state["img"] + 1; xs = []
    for v in FUSE:
        f = _g.glob(f"../mode_probe/exp_fin_{v}_{deg}_s*/raw_{idx}.pt")
        xs.append(torch.load(f[0])["x_raw"].to(P.ATy.device if P.y is None else P.y.device))
    if pr:
        xs = [xs[0]] + [x * torch.sign((xs[0] * x).sum(dim=(2, 3), keepdim=True)) for x in xs[1:]]
    avg = torch.stack(xs).mean(0)
    torch.cuda.synchronize(); t0 = time.time(); nfe = 0
    x = avg
    if FUSE_MODE in ("dc", "dcpolish"):
        x = gauss_newton(P, avg, avg, sched[2][-1], None)
    if FUSE_MODE == "dcpolish":
        i = len(sched[0]) - 1
        zn = x + torch.randn_like(x) * sched[0][i]
        c_skip, c_out, c_in = [append_dims(s_, x.ndim) for s_ in get_scalings_for_boundary_condition(sched[1][i])]
        x = torch.clamp(c_out * model(c_in * zn, 1000 * 0.25 * torch.log(sched[1][i] + 1e-44)) + c_skip * zn, -1.0, 1.0); nfe = 1
    torch.cuda.synchronize()
    spread = float(torch.stack(xs).std(0).mean())
    return x, {"time": time.time() - t0, "nfe": nfe, **P.n, "sample_std": spread}


SAMPLE_SEED = os.environ.get("SAMPLE_SEED", "")     # re-seed AFTER the measurement is drawn: same y, different sampler noise


def _reseed():
    if SAMPLE_SEED:
        torch.manual_seed(int(SAMPLE_SEED) * 1000003 + state["img"] + 1)


_clamp = torch.clamp


def _clamp_rec(t, *a, **kw):
    out = _clamp(t, *a, **kw)
    if SAVE_STEPS and sys._getframe(1).f_code.co_name in ("pnp_cm_natural_nonlinear_restoration", "pnp_cm_natural_linear_restoration") \
            and out.ndim == 4 and out.shape[1] == 3:
        STEPLOG.append((0, len(STEPLOG), out.detach().half().cpu()))
    return out


torch.clamp = _clamp_rec


# ---- INIT=hio (2026-11-17): classical warm start for phase retrieval, FFTs only, no network calls ------------------------
# The operator keeps 1 pixel in 4 (random +-1 mask), zero-pads and takes Fourier magnitudes, so the masked image has a KNOWN
# sparse support.  Fienup's hybrid input-output with that support recovers the kept pixels up to one sign per colour channel;
# the other pixels are filled by local averaging.  What is left for the sampler is an inpainting-like problem from a warm start.
# ---- JLIN=1 (2026-11-17): JPEG treated as a LINEAR problem -----------------------------------------------------------------
# A compressed image is the true image plus a quantisation error that is (nearly) independent across 8x8 DCT coefficients, with a
# size that depends only on the coefficient.  So  y = x + e,  e ~ diagonal covariance in an orthonormal (colour x block-DCT) basis.
# The error variance per coefficient is measured once on the 8 calibration images.  The data-fit then has an exact closed form
# in that basis - no iterations and no operator calls:   z = (w y + rho v) / (w + rho),   w = sigma_y^2 / (var_quant + sigma_y^2).
JLIN = os.environ.get("JLIN", "0") == "1"; _JL = {}


def _jl_T(x, inv=False):
    dev = x.device
    if "C" not in _JL:
        k = torch.arange(8, device=dev).float()
        C = torch.cos((2 * k[None, :] + 1) * k[:, None] * torch.pi / 16) * (2 / 8) ** 0.5; C[0] = C[0] / 2 ** 0.5
        _JL["C"] = C
        _JL["M"] = torch.tensor([[1, 1, 1], [1, -1, 0], [1, 1, -2]], device=dev).float() / torch.tensor([3, 2, 6], device=dev).float().sqrt()[:, None]
    C, M = _JL["C"], _JL["M"]; B = x.shape[0]
    if not inv:
        x = torch.einsum("dc,bchw->bdhw", M, x).reshape(B, 3, 32, 8, 32, 8)
        return torch.einsum("ui,bcminj,vj->bcmunv", C, x, C)
    x = torch.einsum("ui,bcmunv,vj->bcminj", C, x, C).reshape(B, 3, 256, 256)
    return torch.einsum("dc,bdhw->bchw", M, x)


def jpeg_linear_prox(A_funcs, y):
    if "w" not in _JL:
        import glob as _g, numpy as _np
        from PIL import Image as _I
        var = 0
        for f in sorted(_g.glob(os.path.abspath("../mode_probe/datacal/celeba_hq/celeba_hq/*.png")))[:8]:
            xc = torch.from_numpy(_np.asarray(_I.open(f).convert("RGB")).astype(_np.float32) / 255).permute(2, 0, 1)[None].to(y.device) * 2 - 1
            var = var + (_jl_T(A_funcs.A(xc) - xc) ** 2).mean(dim=(0, 2, 4), keepdim=True) / 8      # per (colour, u, v)
        _JL["w"] = SIGY ** 2 / (var + SIGY ** 2)
        if os.environ.get("JLNORM", "1") == "1":               # best-measured coefficient gets weight 1, as in the unweighted misfit
            _JL["w"] = _JL["w"] / _JL["w"].max()
        print("JLIN weights: min %.4f median %.4f max %.4f | quantisation error rms per coefficient: min %.3f median %.3f max %.3f"
              % (float(_JL["w"].min()), float(_JL["w"].median()), float(_JL["w"].max()), float(var.min().sqrt()), float(var.median().sqrt()), float(var.max().sqrt())), flush=True)
    w = _JL["w"]; Ty = _jl_T(y)
    return lambda v, rho: _jl_T((w * Ty + rho * _jl_T(v)) / (w + rho), inv=True)


# ---- JPROJ=1 (2026-11-17): JPEG data-fit as an exact INTERVAL PROJECTION -------------------------------------------------------
# The compressed image tells, for every DCT coefficient, the quantisation interval the true coefficient lies in.  Re-encoding the
# measurement recovers those intervals (the added noise is far smaller than a quantisation step).  The data-fit is then: transform
# the anchor with the codec's own colour/DCT transform, clip every coefficient into its interval, transform the change back.
# Closed form, no gradients, no differentiable codec.
JSOFT = os.environ.get("JSOFT", "0") == "1"; JCOLD = os.environ.get("JCOLD", "0") == "1"
JEV = os.environ.get("JEV", ""); JEVFILE = os.environ.get("JEVFILE", ""); JEVS = float(os.environ.get("JEVS", "1")); _JE = {}
JPROJ = os.environ.get("JPROJ", "0") == "1"; JMARGIN = float(os.environ.get("JMARGIN", "0.5"))


def _jp_coef(x):
    import functions.jpeg_torch as JT
    x = (x + 1) / 2 * 255; yc = JT.torch_rgb2ycbcr(x); l, c = JT.chroma_subsample(yc); unf = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    l = unf(l).transpose(2, 1).reshape(-1, 8, 8) - 128; c = unf(c).transpose(2, 1).reshape(-1, 8, 8) - 128
    d = JT.LinearDCT(8, "dct", norm="ortho").to(x.device)
    return JT.apply_linear_2d(l, d).view(-1, 1, 8, 8), JT.apply_linear_2d(c, d).view(-1, 2, 8, 8)


def _jp_back(dl, dc, n=256):
    import functions.jpeg_torch as JT
    d = JT.LinearDCT(8, "idct", norm="ortho").to(dl.device)
    l = JT.apply_linear_2d(dl.reshape(-1, 8, 8), d).reshape(1, (n // 8) ** 2, 64).transpose(2, 1)
    c = JT.apply_linear_2d(dc.reshape(-1, 8, 8), d).reshape(1, (n // 16) ** 2, 128).transpose(2, 1)
    l = nn.Fold(output_size=(n, n), kernel_size=(8, 8), stride=(8, 8))(l)
    c = nn.Fold(output_size=(n // 2, n // 2), kernel_size=(8, 8), stride=(8, 8))(c).repeat_interleave(2, 2).repeat_interleave(2, 3)
    m = torch.tensor([[1.0, -3.68199903e-05, 1.40198758], [1.0, -3.44113281e-01, -7.14103821e-01], [1.0, 1.77197812, -1.34583413e-04]], device=dl.device)
    return torch.einsum("rc,bchw->brhw", m, torch.cat([l, c], 1)) / 255 * 2


def jpeg_interval_proj(y, qf):
    import functions.jpeg_torch as JT
    kl, kc = JT.jpeg_encode(y.clone(), qf); unf = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    kl = unf(kl).transpose(2, 1).reshape(-1, 1, 8, 8); kc = unf(kc).transpose(2, 1).reshape(-1, 2, 8, 8)
    q1, q2 = JT.quantization_matrix(qf); q1 = q1.to(y.device).view(1, 1, 8, 8); q2 = q2.to(y.device).view(1, 1, 8, 8)

    def proj(v, rho=None):
        cl, cc = _jp_coef(v); lo1, hi1, lo2, hi2 = (kl - JMARGIN) * q1, (kl + JMARGIN) * q1, (kc - JMARGIN) * q2, (kc + JMARGIN) * q2
        if JEV and rho is not None:
            # exact per-coefficient posterior mean: belief N(c, s_k^2) with s_k tabulated per DCT position and step on generated images, truncated to
            # the coefficient's quantisation interval (the intervals are known exactly, so the data term is a hard constraint, not a penalty)
            n_ = str(state.get("step", (0, 1))[0])
            if JEV == "cal":
                gl, gc = _jp_coef(state["gt"].to(v.device)); e_ = _JE.setdefault(n_, {"l": torch.zeros(1, 8, 8, device=v.device), "c": torch.zeros(2, 8, 8, device=v.device), "n": 0})
                e_["l"] += ((cl - gl) ** 2).mean(0); e_["c"] += ((cc - gc) ** 2).mean(0); e_["n"] += 1
                json.dump({k_: {"l": (d_["l"] / d_["n"]).tolist(), "c": (d_["c"] / d_["n"]).tolist()} for k_, d_ in _JE.items()}, open(JEVFILE, "w"))
            else:
                if not _JE: _JE.update(json.load(open(JEVFILE)))
                t_ = _JE.get(n_, _JE[max(_JE, key=int)]); sl = torch.tensor(t_["l"], device=v.device).sqrt() * JEVS; sc_ = torch.tensor(t_["c"], device=v.device).sqrt() * JEVS
                nl, _ = trunc_mean(cl, sl.expand_as(cl), lo1, hi1); nc, _ = trunc_mean(cc, sc_.expand_as(cc), lo2, hi2)
                return v + _jp_back(nl.to(cl.dtype) - cl, nc.to(cc.dtype) - cc)
        if EVFIT and rho is not None:
            # posterior mean per codec coefficient: belief N(c, s_c^2), the coefficient is known to lie in its quantisation interval
            sd = s_cal() * 127.5; nl, _ = trunc_mean(cl, sd * 0.669, (kl - JMARGIN) * q1, (kl + JMARGIN) * q1); nc, _ = trunc_mean(cc, sd * 0.311, (kc - JMARGIN) * q2, (kc + JMARGIN) * q2)
            return v + _jp_back(nl - cl, nc - cc)
        # JSOFT: move only part of the way into the intervals, 1 / (1 + rho) - the exact minimiser of
        # (1/2) dist(z, intervals)^2 + (rho/2) |z - v|^2 along the projection direction; rho is the step's own penalty weight
        a_ = 1.0 / (1.0 + float(rho)) if (JSOFT and rho is not None) else 1.0
        return v + a_ * _jp_back(torch.maximum(torch.minimum(cl, (kl + JMARGIN) * q1), (kl - JMARGIN) * q1) - cl,
                            torch.maximum(torch.minimum(cc, (kc + JMARGIN) * q2), (kc - JMARGIN) * q2) - cc)
    return proj


# ---- NLKER=1 (2026-11-17): the "nonlinear" blur replaced by ONE linear filter fitted on natural images -----------------------
# For every spatial frequency a 3x3 colour matrix K(f) is fitted by least squares so that K * x matches the operator's output on
# the 8 calibration images.  The data-fit then has an exact solve per frequency:  (K^H K + rho I) z = K^H y + rho v.
NLKER = os.environ.get("NLKER", "0") == "1"; _NK = {}


def nl_kernel(A_funcs, dev):
    if "K" not in _NK:
        import glob as _g, numpy as _np
        from PIL import Image as _I
        YX = 0; XX = 0
        for f in sorted(_g.glob(os.path.abspath("../mode_probe/datacal/celeba_hq/celeba_hq/*.png")))[:8]:
            xc = torch.from_numpy(_np.asarray(_I.open(f).convert("RGB")).astype(_np.float32) / 255).permute(2, 0, 1)[None].to(dev) * 2 - 1
            for xa in (xc, torch.flip(xc, dims=(3,))):
                X = torch.fft.fft2(xa)[0].permute(1, 2, 0); Y = torch.fft.fft2(A_funcs.A(xa))[0].permute(1, 2, 0)      # [H,W,3]
                YX = YX + Y[..., :, None] * X.conj()[..., None, :]; XX = XX + X[..., :, None] * X.conj()[..., None, :]
        eye = torch.eye(3, device=dev, dtype=XX.dtype)
        _NK["K"] = YX @ torch.linalg.inv(XX + 1e-3 * XX.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None] * eye)
        if os.environ.get("NLDIAG", "0") == "1" or os.environ.get("NLWIENER", "0") == "1":
            # Can the network blur be written as "linear filter, then a per-pixel curve" (Wiener model  y ~ g(K x))?  Fit g per colour channel
            # (monotone table, 65 knots) on the 8 fitting images, measure the unexplained share of the operator output on 8 OTHER images.
            fs_ = sorted(_g.glob(os.path.abspath("../mode_probe/datacal/celeba_hq/celeba_hq/*.png"))); ld_ = lambda f: torch.from_numpy(_np.asarray(_I.open(f).convert("RGB")).astype(_np.float32) / 255).permute(2, 0, 1)[None].to(dev) * 2 - 1
            Xa = torch.cat([ld_(f) for f in fs_[:8]]); Ya = torch.cat([A_funcs.A(x_[None]) for x_ in Xa]); Sa = torch.cat([nl_apply(_NK["K"], x_[None]) for x_ in Xa])
            kn = torch.linspace(-1.3, 1.3, 65, device=dev); gt_ = torch.zeros(3, 65, device=dev)
            for c_ in range(3):
                bi_ = torch.bucketize(Sa[:, c_].flatten(), (kn[1:] + kn[:-1]) / 2); sm_ = torch.zeros(65, device=dev).index_add_(0, bi_, Ya[:, c_].flatten()); cn_ = torch.zeros(65, device=dev).index_add_(0, bi_, torch.ones_like(Ya[:, c_].flatten()))
                gt_[c_] = torch.where(cn_ > 20, sm_ / cn_.clamp(min=1), kn); gt_[c_] = torch.cummax(gt_[c_], 0)[0]
            def g_apply(s_):
                t_ = ((s_.clamp(-1.3, 1.3) + 1.3) / 2.6 * 64); i0 = t_.floor().clamp(0, 63).long(); fr = t_ - i0
                return torch.stack([gt_[c_][i0[:, c_]] * (1 - fr[:, c_]) + gt_[c_][i0[:, c_] + 1] * fr[:, c_] for c_ in range(3)], 1)
            _NK["g"] = (kn, gt_, g_apply)
            Xb = torch.cat([ld_(f) for f in fs_[8:16]]); Yb = torch.cat([A_funcs.A(x_[None]) for x_ in Xb]); Sb = torch.cat([nl_apply(_NK["K"], x_[None]) for x_ in Xb])
            rr = lambda P_: float(((Yb - P_) ** 2).sum() / (Yb ** 2).sum()) ** 0.5
            print(f"NLDIAG unexplained share of the operator output on 8 held-out images: identity {rr(Xb):.4f} | linear filter {rr(Sb):.4f} | linear filter + per-pixel curve {rr(g_apply(Sb)):.4f} | curve departure from identity (max |g(s)-s|) {float((gt_ - kn).abs().max()):.4f}", flush=True)
    return _NK["K"]


def nl_apply(K, x):
    return torch.fft.ifft2((K @ torch.fft.fft2(x)[0].permute(1, 2, 0)[..., None])[..., 0].permute(2, 0, 1)[None]).real


MEC = os.environ.get("MEC", "")   # model-error correction for the linear filter: "" | once | step
# The filter K only approximates the true blur B.  Evaluate B once at a reference image x_ref (the anchor), measure the mismatch
# r = B(x_ref) - K x_ref, and solve the linear problem against the corrected measurement y - r.  The residual then comes from the
# TRUE operator while the solve stays a closed-form FFT.  once: r from the first anchor only (1 operator call); step: every step.


def nl_kernel_prox(K, y, A=None, P=None):
    KH = K.conj().transpose(-1, -2); KHK = KH @ K; KHy = (KH @ torch.fft.fft2(y)[0].permute(1, 2, 0)[..., None])
    eye = torch.eye(3, device=y.device, dtype=K.dtype); mem = {"r": None}

    def prox(v, rho):
        KHy_ = KHy
        if MEC and A is not None:
            if MEC == "step" or mem["r"] is None:
                xr = v.clamp(-1.0, 1.0); mem["r"] = A(xr) - nl_apply(K, xr); P.n["A"] += 1
            KHy_ = KH @ torch.fft.fft2(y - mem["r"])[0].permute(1, 2, 0)[..., None]
        rhs = KHy_ + float(rho) * torch.fft.fft2(v)[0].permute(1, 2, 0)[..., None]
        return torch.fft.ifft2(torch.linalg.solve(KHK + float(rho) * eye, rhs)[..., 0].permute(2, 0, 1)[None]).real
    return prox


# ---- LINEXACT=1 (2026-11-17): closed-form data-fit for any linear operator with a known SVD ---------------------------------
#   z = V [ (V^T A^T y + rho V^T v) / (s^2 + rho) ]      - no iterations.
LINEXACT = os.environ.get("LINEXACT", "0") == "1"; NOLOG = os.environ.get("NOLOG", "0") == "1"


def fire_loop(F_, ATy, model, levels, x_start=None):
    """Linear estimation + COLOURED renoising (after Bendel et al., "Iterative Colored Renoising", TMLR 2025), run with the consistency
    model in a handful of steps.  No dual variable and no hand-set penalty:
      * the uncertainty nu of the model's image is estimated from the measurement residual (no ground truth),
      * the data step is the exact minimum-mean-square estimate under that nu (closed form in the SVD basis),
      * the noise added before the next model call is shaped per singular direction so that the TOTAL error of the model's input is white
        at the level the model is told - the condition the model was trained under."""
    s = F_.add_zeros(F_.singulars()[None])[0]; s2 = s ** 2; c = F_.Vt(ATy)[0]; meas = s > float(os.environ.get("FIRE_STHR", "0.05")) * float(s.max()); m_ = float(meas.sum()); fro = float((s2 * meas).sum())
    sy2 = SIGY ** 2; uy = torch.where(meas, c / s.clamp(min=1e-6), torch.zeros_like(c))       # U'y on the measured directions
    def mmse(xbar_v, nu):
        return (c + (sy2 / nu) * xbar_v) / (s2 + sy2 / nu)
    def renoise(xhat_v, nu, sig):
        lam = (sig ** 2 - 1.0 / (s2 / sy2 + 1.0 / nu)).clamp(min=0); return xhat_v + lam.sqrt() * torch.randn_like(xhat_v)
    nu = float(os.environ.get("FIRE_NU0", "0.25")); xv = mmse(torch.zeros_like(c), nu); nfe = 0; log = []
    if x_start is not None and os.environ.get("FIRE_WARM", "0") == "1":      # start from the task's usual starting image, uncertainty from its residual
        x0v = F_.Vt(x_start)[0]; nu = max((float((((uy - s * x0v) ** 2) * meas).sum()) - m_ * sy2) / fro, 1e-3); xv = mmse(x0v, nu)
    for k, sig in enumerate(levels):
        sig = max(float(sig), nu ** 0.5 * float(os.environ.get("FIRE_FLOOR", "1.0"))) if k > 0 else float(sig)
        r = F_.V(renoise(xv, nu, sig)[None]).reshape(ATy.shape)
        xbar = cm(model, r, torch.full((1,), sig * float(os.environ.get("FIRE_TOLD", "1.0")), device=ATy.device)); nfe += 1
        xbv = F_.Vt(xbar)[0]; res = float((((uy - s * xbv) ** 2) * meas).sum())
        nu = max((res - m_ * sy2) / fro, float(os.environ.get("FIRE_NUMIN", "1e-4"))); xv = mmse(xbv, nu); log.append([sig, nu ** 0.5])
    xhat = F_.V(xv[None]).reshape(ATy.shape).clamp(-1.0, 1.0)
    return (xbar if os.environ.get("FIRE_OUT", "est") == "den" else xhat), nfe, log


def svd_prox(F_, ATy):
    s2 = F_.add_zeros(F_.singulars()[None])[0] ** 2; c = F_.Vt(ATy)[0]
    return lambda v, rho: F_.V(((c + float(rho) * F_.Vt(v)[0]) / (s2 + float(rho)))[None]).reshape(ATy.shape)


PRLIN = os.environ.get("PRLIN", "0") == "1"; _PRL = {}
INIT = os.environ.get("INIT", ""); HIO_N = int(os.environ.get("HIO_N", "300")); HIO_ER = int(os.environ.get("HIO_ER", "50")); HIO_K = int(os.environ.get("HIO_K", "1"))


def hio_init(A_funcs, y, beta=0.9):
    import functions.measurements as _FM
    import torchvision.transforms.functional as _TF
    pad = A_funcs.pad; mod = A_funcs.modulation.to(y.dtype)
    S = torch.nn.functional.pad((mod != 0).to(y.dtype), (pad, pad, pad, pad))[None, None] > 0
    best = None
    for _ in range(HIO_K):                                     # several random starts; keep the one that fits the magnitudes best
        g = torch.randn_like(y) * S
        for k in range(HIO_N + HIO_ER):
            G = _FM.fft2_m(g); gp = _FM.ifft2_m(y * G / (G.abs() + 1e-8)).real
            g = torch.where(S, gp, g - beta * gp) if k < HIO_N else gp * S
        res = ((_FM.fft2_m(g).abs() - y) ** 2).sum(dim=(2, 3), keepdim=True)     # per colour channel
        if best is None:
            best, bres = g, res
        else:
            best = torch.where(res < bres, g, best); bres = torch.minimum(res, bres)
    g = best
    xo = g[..., pad:-pad, pad:-pad] * mod; w = (mod != 0).to(y.dtype)[None, None].expand_as(xo)
    fill = _TF.gaussian_blur(xo, 9, 2.0) / _TF.gaussian_blur(w, 9, 2.0).clamp(min=1e-3)
    z = torch.where(w > 0, xo, fill).clamp(-1.0, 1.0)
    state["hio_xo"] = (xo, w, fill); state["hio_res"] = bres.flatten().tolist()
    if state["gt"] is not None and not NOLOG:                  # log only: quality of the classical solve on the kept pixels
        x0 = state["gt"].to(y.device); sg = torch.sign((xo * x0 * w).sum(dim=(2, 3), keepdim=True))
        state["hio"] = [float(10 * torch.log10(1 / ((((sg * xo - x0) / 2) ** 2 * w).sum() / w.sum()))),
                        float((A_funcs.A(z) - y).norm() / y.norm())]
    return z


def nl_entry(y, model, A_funcs, betas, iN, gamma, deltas, rhos, mu_0, classes, args, config, **kw):
    state["args"] = args
    dev = y.device; _reseed(); torch.cuda.synchronize(); t_entry = time.time()
    if MODE == "base":
        torch.cuda.synchronize(); t0 = time.time()
        x = ORIG_NL(y=y, model=model, A_funcs=A_funcs, betas=betas, iN=iN, gamma=gamma, deltas=deltas, rhos=rhos,
                    mu_0=mu_0, classes=classes, args=args, config=config)
        torch.cuda.synchronize()
        n = args.opt_num_iter * args.T_sampling
        rb = {"time": time.time() - t0, "nfe": args.T_sampling, "A": n, "At": n, "jvp": 0}
        if not NOLOG:
            rb["cons"] = float((A_funcs.A(x) - y).norm() / y.norm())
        if NEWOP == "hdr" and state["gt"] is not None:
            x0 = state["gt"].to(dev); sat = (x0.abs() > 0.5).float()
            rb["hdr"] = [float(sat.mean()), _ps((y / 2).clamp(-1, 1), x0), float(10 * torch.log10(1 / ((((x - x0) / 2) ** 2 * sat).sum() / sat.sum().clamp(min=1))))]
        return finish(x, rb)
    z0 = phase_retrieval_initialization(y, args.phase_ret_os) if args.deg == "phase_retrieval" else y.clone()
    if INIT == "hio" and args.deg == "phase_retrieval":
        z0 = hio_init(A_funcs, y)
    if PRLIN and args.deg == "phase_retrieval":
        # Diagnostic (2026-11-17): after the classical solve, phase retrieval is treated as plain INPAINTING of the kept pixels,
        # with the repo's inpainting settings (4 network calls).  The sign of each colour channel is taken from the ground truth
        # (ORACLE - the standard evaluation also assumes the sign known), so this measures what is left once the phase is solved.
        from functions.svd_operators import Inpainting
        xo, w, fill = state["hio_xo"]; x0 = state["gt"].to(dev)
        sg = torch.sign((xo * x0 * w).sum(dim=(2, 3), keepdim=True))
        if os.environ.get("PRSIGN", "oracle") == "none":       # no ground truth: keep whatever sign the classical solve returned
            sg = torch.ones_like(sg)
        state["sign_ok"] = float((torch.sign((xo * x0 * w).sum(dim=(2, 3))) > 0).float().mean())           # log only: share of channels whose sign is right as returned
        if os.environ.get("PRSIGN", "oracle") in ("prior", "corr"):
            # sign of each colour channel WITHOUT the truth.  "corr": natural images have positively correlated colour channels, so the
            # channels are flipped to agree with each other; the remaining global sign, and "prior" for every channel, is set by the
            # correlation with the mean face of the consistency model's own samples (kept pixels only).
            mf = torch.load(os.path.abspath("../mode_probe/" + os.environ.get("MEANFACE", "mean_face_gen.pt"))).to(dev).float(); mf = (mf if mf.dim() == 4 else mf[None]) * 2 - 1; mf = mf - mf.mean(dim=(2, 3), keepdim=True)
            xc = xo - (xo * w).sum(dim=(2, 3), keepdim=True) / w.sum(dim=(2, 3), keepdim=True) * 0          # (no centring: a flip also flips the mean)
            if os.environ["PRSIGN"] == "prior":
                sg = torch.sign((xc * mf * w).sum(dim=(2, 3), keepdim=True))
            else:
                rel = torch.sign((xc * xc[:, 1:2] * w).sum(dim=(2, 3), keepdim=True)); sg = rel * torch.sign(((xc * rel) * mf * w).sum())
            sg = torch.where(sg == 0, torch.ones_like(sg), sg)
            state["sign_ok"] = float((torch.sign(((xo * sg) * x0 * w).sum(dim=(2, 3))) > 0).float().mean())
        xo = xo * sg; fill = fill * sg
        if "op" not in _PRL:
            mask = (A_funcs.modulation != 0).reshape(-1); mr = torch.nonzero(mask == 0).long().reshape(-1) * 3
            _PRL["op"] = Inpainting(3, 256, torch.cat([mr, mr + 1, mr + 2]), dev)
        Aop = _PRL["op"]; ATy_ = Aop.At(Aop.A(xo)).view(1, 3, 256, 256)
        xi = torch.where(w > 0, xo, fill).clamp(-1.0, 1.0)
        sched = schedule(betas, 100, 0.1, [0.3, 0.2, 0.8, 0.8], [-6.0, 12.0], 0.05, 4, dev)
        Pl = Problem(Aop, ATy=ATy_)
        if LINEXACT:
            Pl.prox = svd_prox(Aop, ATy_)
        if os.environ.get("PR2", "0") == "1":
            # ATTEMPT 2 (2026-11-24): masked phase retrieval kept in the paper's operator class instead of being reduced to inpainting.
            # A(x) = |F P D x| : D = known +-1/0 mask (diagonal), P = zero padding, F = unitary FFT  ->  "linear, then coordinate-wise magnitude".
            # Data step = two exact pieces, applied once: per Fourier coefficient keep the phase and set the magnitude to (a + r|c|)/(1 + r);
            # then back to the image, which is DIAGONAL in pixels because (FPD)'(FPD) = D^2: kept pixels take mod * crop(F^-1 u), the rest keep v.
            import functions.measurements as _FM2
            pd_ = A_funcs.pad; md_ = A_funcs.modulation.to(y.dtype)[None, None]; wk_ = (md_ != 0); rs_ = float(os.environ.get("PR2_RS", "1"))

            def _prox_pr2(v, rho):
                G = _FM2.fft2_m(torch.nn.functional.pad(md_ * v, (pd_, pd_, pd_, pd_))); r = rs_ * float(rho)
                u = G / (G.abs() + 1e-8) * (y + r * G.abs()) / (1 + r)
                return torch.where(wk_.expand_as(v), _FM2.ifft2_m(u).real[..., pd_:-pd_, pd_:-pd_] * md_, v)
            Pl.prox = _prox_pr2
        x, rec, xf = unified(Pl, model, xi.clone(), xi.clone(), sched)
        KC = int(os.environ.get("PRCAND", "1"))
        if KC > 1:
            # several classical solutions, each completed by the sampler; scored WITHOUT the truth by how well the finished image
            # reproduces the measured magnitudes (per colour channel, since the channels are measured independently)
            import functions.measurements as _FM
            mis = lambda xx: (((A_funcs.A(xx) - y) ** 2).sum(dim=(2, 3)) / (y ** 2).sum(dim=(2, 3)))[0].sqrt().tolist()
            def psn(xx):
                c = xx * torch.sign((xx * x0).sum(dim=(2, 3), keepdim=True)); return [float(10 * torch.log10(1 / (((c[:, k] - x0[:, k]) / 2) ** 2).mean())) for k in range(3)]
            cands = [dict(x=x.cpu(), misfit=mis(x), hio=state["hio_res"], psnr_ch=psn(x), nfe=rec["nfe"])]
            for _ in range(KC - 1):
                hio_init(A_funcs, y); xo, w, fill = state["hio_xo"]; ATy_ = Aop.At(Aop.A(xo)).view(1, 3, 256, 256); xi = torch.where(w > 0, xo, fill).clamp(-1.0, 1.0)
                Pl = Problem(Aop, ATy=ATy_)
                if LINEXACT:
                    Pl.prox = svd_prox(Aop, ATy_)
                xk, rk, _ = unified(Pl, model, xi.clone(), xi.clone(), sched); cands.append(dict(x=xk.cpu(), misfit=mis(xk), hio=state["hio_res"], psnr_ch=psn(xk), nfe=rk["nfe"]))
            jb = min(range(KC), key=lambda j: sum(cands[j]["misfit"])); x = cands[jb]["x"].to(dev); rec["nfe"] = sum(c["nfe"] for c in cands)
            rec["cand"] = [dict(misfit=c["misfit"], hio=c["hio"], psnr_ch=c["psnr_ch"]) for c in cands]; rec["pick"] = jb
            torch.save([c["x"].half() for c in cands], f"{exp}/cand_{state['img'] + 1}.pt")
        torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
        return finish(x, rec, xf)
    sched = schedule(betas, iN, gamma, deltas, rhos, mu_0, args.T_sampling, dev)
    if FUSE:
        x, rec = fuse(Problem(A_funcs, y=y), model, sched, args.deg == "phase_retrieval")
        return finish(x, rec)
    Pb = Problem(A_funcs, y=y)
    if NEWOP == "prdaps":
        # Unmasked phase retrieval with our sampler.  Warm start = classical HIO (best of PRD_K random starts by magnitude misfit).
        # Data-fit at every step = PRD_ER plain magnitude/support/range projections started from the anchor (the prior's current
        # image supplies the phases), blended with the anchor by the step's penalty weight.  FFTs only.
        refine, hio = prd_setup(y); K_ = int(os.environ.get("PRD_K", "4")); ER_ = int(os.environ.get("PRD_ER", "30"))
        best = min((hio(int(os.environ.get("PRD_HIO", "500")), 50) for _ in range(K_)), key=lambda q: q[1]) if K_ > 0 else (torch.zeros(1, 3, 256, 256, device=dev), 0.0)
        w0 = best[0].clamp(-1.0, 1.0)
        Pb.prox = lambda v, rho: (refine(v, ER_) + float(rho) * v) / (1 + float(rho))
        if os.environ.get("PRD_SOFT", "0") == "1":
            Pb.prox = lambda v, rho: refine.soft_fit(v, float(rho) * float(os.environ.get("PRD_RS", "1")))
        if os.environ.get("PRD_INIT", "hio") == "mean":
            w0 = refine.mean_init().clamp(-1.0, 1.0)
        if os.environ.get("PRD_INIT", "hio") in ("oracle", "truth"):     # DIAGNOSTIC ONLY: start from the ground truth (or its eigenface projection)
            x0_ = (state["gt"].to(dev) + 1) / 2; Bp = torch.load(os.path.abspath("../mode_probe/pca_cm.pt"))
            Uo = Bp["U"][:int(os.environ.get("PRD_KMAX", "512"))].to(dev).flatten(1); mo = Bp["mu"].to(dev)
            w0 = x0_ if os.environ["PRD_INIT"] == "truth" else (mo + (((x0_ - mo).flatten() @ Uo.T) @ Uo).view(1, 3, 256, 256)).clamp(0, 1)
            w0 = w0 * 2 - 1
        if os.environ.get("PRD_INIT", "hio") in ("regfit", "regfit4"):
            w0 = refine.regfit_init(int(os.environ.get("PRD_NSTART", "4")) if os.environ["PRD_INIT"] == "regfit4" else 1).clamp(-1.0, 1.0)
        if os.environ.get("PRD_INIT", "hio") == "netdirect":
            w0 = refine.netdirect_init().clamp(-1.0, 1.0)
        if os.environ.get("PRD_INIT", "hio") == "netfit":
            w0 = refine.netfit_init(int(os.environ.get("PRD_NETREG", "0"))).clamp(-1.0, 1.0)
        if os.environ.get("PRD_INIT", "hio") == "pca":
            w0 = refine.pca_init(int(os.environ.get("PRD_KMAX", "256"))).clamp(-1.0, 1.0)
        if os.environ.get("PRD_BAND", "0") == "1":
            import numpy as _np
            cuts = list(_np.geomspace(float(os.environ.get("PRD_C0", "0.02")), 0.75, len(sched[0]))); cnt = {"i": 0}

            def _bp(v, rho):
                c = cuts[min(cnt["i"], len(cuts) - 1)]; cnt["i"] += 1
                return (refine.band_fit(v, c, ER_) + float(rho) * v) / (1 + float(rho))
            Pb.prox = _bp
        gt_ = state["gt"].to(dev) if state["gt"] is not None else None

        def _run_from(w0):
            r0 = _ps(w0, gt_) if gt_ is not None else float("nan"); ncm = 0
            if int(os.environ.get("PRD_TV", "0")) > 0:
                w0 = refine.tv_loop(w0, int(os.environ["PRD_TV"])).clamp(-1.0, 1.0)
            sc_ = float(os.environ.get("PRD_CMC", "0"))
            import functions.fft_util as _FU
            if sc_ > 0:      # ONE model call inside the coarse stage: its mid-frequency phases become soft linear targets, then the classical loop again
                xd = cm(model, w0 + sc_ * torch.randn_like(w0), torch.full((1,), sc_, device=dev)); ncm = 1
                ii = torch.arange(384, device=dev).float() - 192; Mb = ((ii[:, None] ** 2 + ii[None, :] ** 2).sqrt() < float(os.environ.get("PRD_CMR", "32"))).expand(1, 3, 384, 384)
                bO = refine.amp * torch.exp(1j * _FU.fft2_m(torch.nn.functional.pad(xd * 0.5 + 0.5, (64, 64, 64, 64))).angle())
                w0 = refine.tv_loop(xd if os.environ.get("PRD_CMFROM", "cm") == "cm" else w0, int(os.environ.get("PRD_CMTV", "20")), band=(Mb, bO)).clamp(-1.0, 1.0)
            x, rec, xf = unified(Pb, model, w0.clone(), w0.clone(), sched)
            if os.environ.get("PRD_FINAL", "0") == "1":
                x = refine(x, ER_).clamp(-1.0, 1.0)
            rec["nfe"] += ncm; rec["misfit"] = float((A_funcs.A(x) - y).norm() / y.norm()); rec["ref0"] = r0
            if gt_ is not None:
                rec["ref"] = _ps(w0, gt_); rec["ref_flip"] = _ps(torch.flip(w0, dims=(2, 3)), gt_); rec["psnr"] = _ps(x, gt_); rec["psnr_flip"] = _ps(torch.flip(x, dims=(2, 3)), gt_)
            return x, rec, xf
        K_top = int(os.environ.get("PRD_TOPK", "1")); cl = getattr(refine, "cands", None)
        starts = [w0] * K_top if (K_top == 1 or cl is None or os.environ.get("PRD_TOPSAME", "0") == "1") else [c_.clamp(-1.0, 1.0) for c_ in cl[:K_top]]
        runs = [_run_from(w_) for w_ in starts]; jb = min(range(len(runs)), key=lambda j: runs[j][1]["misfit"]); x, rec, xf = runs[jb]
        if len(runs) > 1:                                      # keep the finished image that fits the measured magnitudes best (no ground truth)
            rec["nfe"] = sum(r_[1]["nfe"] for r_ in runs); rec["pick"] = jb; rec["misfit_all"] = [r_[1]["misfit"] for r_ in runs]
            if gt_ is not None:
                rec["psnr_all"] = [r_[1]["psnr"] for r_ in runs]; rec["psnr_avg"] = _ps(torch.stack([r_[0] for r_ in runs]).mean(0), gt_)
        torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
        return finish(x, rec, xf)
    if NEWOP in ("quant", "onebit", "blind", "haze", "gamma", "mod", "blurclip", "blurq", "ct", "mri", "ctnl", "prsq", "prcdp") and (SOLVER != "repo" or NEWOP in ("ct", "ctnl", "prsq", "prcdp")):
        if NEWOP == "mri":
            Pb.prox = mri_prox(y); w0 = torch.fft.ifft2(_mri_mask(dev) * torch.complex(y[..., :256], y[..., 256:]), norm="ortho").real.clamp(-1.0, 1.0)   # start: zero-filled
        elif NEWOP == "ct":
            Pb.prox = ct_prox(y); w0 = ct_prox(y, 40)(torch.zeros(1, 3, 256, 256, device=dev), float(os.environ.get("CT_R0", "0.05"))).clamp(-1.0, 1.0)   # start: regularised least squares, no model
        elif NEWOP in ("prsq", "prcdp"):
            Pb.prox = priso_prox(y, NEWOP == "prcdp"); mf_ = torch.load(os.path.abspath("../mode_probe/" + os.environ.get("MEANFACE", "mean_face_gen.pt"))).to(dev).float(); mf_ = (mf_ if mf_.dim() == 4 else mf_[None]) * 2 - 1
            w0 = mf_                                                # start: the mean face of model samples (its phases); no search stage
            for _ in range(int(os.environ.get("PR_ISO_WARM", "0"))):   # optional classical warm-up: repeated magnitude fits (Gerchberg-Saxton style)
                w0 = Pb.prox(w0, 0.0)
        elif NEWOP == "ctnl":
            if CTNL_MODE in ("poisson", "poislog"):              # the wrapper added Gaussian noise of 0.1 counts (negligible); draw the photon counts
                gp_ = torch.Generator(device="cpu").manual_seed(1234 + int(state.get("img", 0))); y = torch.poisson(y.clamp(min=0.0).cpu(), generator=gp_).to(dev); Pb.y = y
            Pb.prox = ctnl_prox(y); w0 = Pb.prox.start(float(os.environ.get("CT_R0", "0.05"))).clamp(-1.0, 1.0)   # start: logarithm + regularised least squares
        elif NEWOP == "quant":
            Pb.prox = quant_prox(y); w0 = y.clamp(-1.0, 1.0)
            if os.environ.get("GENERIC", ""): Pb.prox = blackbox_pointwise_prox(A_funcs.A, y)
        elif NEWOP == "gamma":
            Pb.prox = blackbox_pointwise_prox(A_funcs.A, y) if os.environ.get("GENERIC", "") else gamma_prox(y); w0 = (2 * ((y.clamp(-1.0, 1.0) + 1) / 2) ** (1 / _GAM) - 1).clamp(-1.0, 1.0)     # start: the curve undone
        elif NEWOP == "mod":
            Pb.prox = mod_prox(y); w0 = (y / 2).clamp(-1.0, 1.0)
            if os.environ.get("MODUNWRAP", "0") == "1":       # reduction: wrap-around sensing -> unwrapping (linear Poisson solve) -> a denoising problem
                w0 = (unwrap_ls(y) / 2).clamp(-1.0, 1.0)
        elif NEWOP == "blurclip":
            Pb.prox = split_prox(ev_fit(lambda s_: (2 * s_).clamp(-1.0, 1.0), y) if SPLIT_EV > 0 else hdr_prox(y)); w0 = (y / 2).clamp(-1.0, 1.0)
            if os.environ.get("BC_MODE", "") == "active":
                Pb.prox = blurclip_active(y)
            if os.environ.get("BC_JLOG", "0") == "1":             # log the TRUE data-fit objective reached by whichever data-fit is used
                _p0 = Pb.prox

                def _plog(v_, rho_, _p0=_p0):
                    z_ = _p0(v_, rho_); state.setdefault("bcJ", []).append([_bc_obj(y, v_, v_, rho_), _bc_obj(y, z_, v_, rho_)]); return z_
                Pb.prox = _plog
        elif NEWOP == "blurq":
            Pb.prox = split_prox(quant_prox(y)); w0 = y.clamp(-1.0, 1.0)
        elif NEWOP == "onebit":
            Pb.prox = onebit_prox(torch.sign(y), A_funcs.tau); w0 = torch.zeros_like(y)
            if os.environ.get("GENERIC", ""): Pb.prox = blackbox_pointwise_prox(A_funcs.A, torch.sign(y))
            if os.environ.get("OBWARM", "0") == "1":             # warm start: the per-pixel solution of the first data-fit from a blank image
                w0 = Pb.prox(w0, float(sched[2][0])).clamp(-1.0, 1.0)
        elif NEWOP == "haze":
            Pb.prox, w0, t_ = haze_setup(y, os.environ.get("HAZEK", "est"), A_funcs)
            state["terr"] = float((t_ - A_funcs.t).abs().mean())
        else:
            Pb.prox = blind_prox(y, os.environ.get("BLINDK", "est"), A_funcs); w0 = y.clamp(-1.0, 1.0)
        x, rec, xf = unified(Pb, model, w0.clone(), w0.clone(), sched)
        torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
        if state["gt"] is not None:
            rec["ref"] = _ps(w0, state["gt"].to(dev))
            if NEWOP == "blind":
                kt, ke = A_funcs.k, Pb.prox.mem["K"]; rec["kcorr"] = float((kt * ke).sum() / (kt.norm() * ke.norm()))
            if "terr" in state:
                rec["terr"] = state.pop("terr")
        return finish(x, rec, xf)
    if NEWOP == "hdr" and SOLVER != "repo":
        z0 = (y / 2).clamp(-1.0, 1.0)                           # warm start: undo the scaling (exact on unsaturated pixels)
        if os.environ.get("HDRMID", "0") == "1":                # clipped pixels are known to lie in [0.5, 1]: start them mid-range, not at the edge
            z0 = torch.where(y.abs() >= 1.0 - 2 * SIGY, 0.75 * torch.sign(y), z0)
        if os.environ.get("HDR_REG", "0") == "1":
            # statistical start: eigenfaces of MODEL SAMPLES fitted (ridge, closed form) to the unclipped pixels only, used to fill the clipped pixels,
            # then moved into the range the measurement allows (clipped high -> at least 0.5, clipped low -> at most -0.5).  No model calls.
            if "U" not in _HDRREG:
                d_ = torch.load(os.path.abspath("../mode_probe/pca_cm.pt")); kk_ = int(os.environ.get("HDR_REGK", "256"))
                _HDRREG["U"] = d_["U"][:kk_].flatten(1).to(dev); _HDRREG["mu"] = d_["mu"].flatten().to(dev); _HDRREG["var"] = d_["var"][:kk_].to(dev)
            U_, mu_, var_ = _HDRREG["U"], _HDRREG["mu"], _HDRREG["var"]; unc = (y.abs() < 1.0 - 2 * SIGY).flatten().float(); t_ = ((y / 2).flatten() * 0.5 + 0.5 - mu_) * unc
            c_ = torch.linalg.solve((U_ * unc) @ U_.T + float(os.environ.get("HDR_REGL", "1.0")) * (SIGY / 4) ** 2 * torch.diag(1.0 / var_), U_ @ t_)
            pred = ((mu_ + c_ @ U_).view(1, 3, 256, 256) * 2 - 1); hi = y >= 1.0 - 2 * SIGY; lo_ = y <= -1.0 + 2 * SIGY
            z0 = torch.where(hi, pred.clamp(0.5, 1.0), torch.where(lo_, pred.clamp(-1.0, -0.5), z0))
        if os.environ.get("HDRX", "0") == "1":
            Pb.prox = hdr_prox(y)
        if os.environ.get("GENERIC", ""):
            Pb.prox = blackbox_pointwise_prox(A_funcs.A, y)
        x, rec, xf = unified(Pb, model, z0.clone(), z0.clone() if os.environ.get("HDRWARM", "1") == "1" else torch.zeros_like(z0), sched)
        torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
        if state["gt"] is not None:
            x0 = state["gt"].to(dev); sat = (x0.abs() > 0.5).float()
            rec["hdr"] = [float(sat.mean()), _ps(z0, x0), float(10 * torch.log10(1 / ((((x - x0) / 2) ** 2 * sat).sum() / sat.sum().clamp(min=1))))]
        return finish(x, rec, xf)
    if JLIN and args.deg == "jpeg":
        Pb.prox = jpeg_linear_prox(A_funcs, y)
    if JPROJ and args.deg == "jpeg":
        Pb.prox = jpeg_interval_proj(y, args.jpeg_qf)
        if state["gt"] is not None and not NOLOG:              # log only: is the truth inside the intervals, and what does projecting y give?
            x0 = state["gt"].to(dev); state["jp"] = [float(((Pb.prox(x0) - x0) ** 2).mean().sqrt()), _ps(y.clamp(-1, 1), x0), _ps(Pb.prox(y).clamp(-1, 1), x0)]
    if NLKER and args.deg == "deblur_nl":
        K = nl_kernel(A_funcs, dev); Pb.prox = nl_kernel_prox(K, y, A_funcs.A, Pb)
        if state["gt"] is not None and not NOLOG:              # log only: how well does the fitted filter reproduce the operator on this test image?
            x0 = state["gt"].to(dev); Ax = A_funcs.A(x0); pr_ = nl_apply(K, x0); c = (slice(None), slice(None), slice(24, 232), slice(24, 232))
            state["nk"] = [float((pr_ - Ax)[c].norm() / Ax[c].norm()), _ps(pr_[c], Ax[c]), float((pr_ - Ax).norm() / Ax.norm())]
    if os.environ.get("DIAGLIN", "0") == "1" and state["gt"] is not None:   # log only: how far is this operator from linear?
        x0 = state["gt"].to(dev); A = A_funcs.A; yb = y.clamp(-1, 1); n_ = lambda t: float(t.norm())
        Ay, Ax = A(yb), A(x0); J = torch.func.jvp(A, (yb,), (x0 - yb,))[1]
        xb = torch.roll(x0, 64, dims=3); zero = torch.zeros_like(x0); c = (slice(None), slice(None), slice(48, 208), slice(48, 208))
        A0 = A(zero); imp = zero.clone(); imp[:, :, 128, 128] = 1.0; kimg = A(imp) - A0        # response to one bright pixel
        K = torch.fft.fft2(torch.roll(kimg, (-128, -128), dims=(2, 3))); pred = torch.fft.ifft2(torch.fft.fft2(x0) * K).real + A0
        fd = (A(x0 + 1e-2 * (x0 - yb)) - Ax) / 1e-2                                               # finite-difference derivative
        state["conv"] = [n_((pred - Ax)[c]) / n_(Ax[c]), float(kimg.sum() / 3), n_(A0) / n_(Ax), n_(fd - (Ax - A(yb))) / n_(Ax - A(yb)),
                         float(10 * torch.log10(1 / ((((pred - Ax)[c]) / 2) ** 2).mean()))]
        state["lin"] = [n_(Ax - Ay - J) / n_(Ax - Ay), n_(A(0.5 * x0 + 0.5 * xb) - 0.5 * Ax - 0.5 * A(xb)) / n_(Ax - A(xb)),
                        n_((A(torch.roll(x0, 16, dims=3)) - torch.roll(Ax, 16, dims=3))[c]) / n_(Ax[c])]
    if os.environ.get("WARMX", "0") == "1" and args.deg in ("jpeg", "deblur_nl") or (os.environ.get("WARMX", "0") == "lin" and args.deg == "deblur_nl"):
        # warm start ONLY: a cheap linear estimate is the starting image and first anchor; every data-fit still uses the true operator
        if args.deg == "deblur_nl":
            wst = nl_kernel_prox(nl_kernel(A_funcs, dev), y)(torch.zeros_like(y), float(sched[2][0])).clamp(-1.0, 1.0)   # one deconvolution
        else:
            wst = y.clamp(-1.0, 1.0)
        x, rec, xf = unified(Pb, model, wst.clone(), wst.clone(), sched)
    elif (JLIN or JPROJ) and args.deg == "jpeg" and not JCOLD:  # warm start: the anchor of the first step is the measurement itself
        if os.environ.get("JLRHO", "ship") == "auto":           # penalty = sigma_y^2 / sigma_t^2 (the anchor is trusted to within the step's noise)
            sched = (sched[0], sched[1], torch.stack([(SIGY / s_.flatten()[0]) ** 2 for s_ in sched[0]]).to(dev), sched[3])
        x, rec, xf = unified(Pb, model, z0, y.clamp(-1.0, 1.0), sched)
    else:
        x, rec, xf = unified(Pb, model, z0, torch.zeros_like(z0), sched)
    if JPROJ and args.deg == "jpeg" and os.environ.get("JFINAL", "0") == "1":
        x = Pb.prox(x).clamp(-1.0, 1.0)                        # put the final image back inside the intervals (free)
    torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
    for k_ in ("jp", "nk"):
        if k_ in state:
            rec[k_] = state.pop(k_)
    if not NOLOG:
        rec["cons"] = float((A_funcs.A(x) - y).norm() / y.norm())   # consistency of the result with the measurement under the TRUE operator
    if "lin" in state:
        rec["lin"] = state.pop("lin"); rec["conv"] = state.pop("conv")
    return finish(x, rec, xf)


def lin_entry(ATy, x_init, model, A_funcs, betas, iN, gamma, deltas, rhos, mu_0, T_sampling, config, classes, **kw):
    _reseed(); torch.cuda.synchronize(); t_entry = time.time()
    if MODE == "base":
        torch.cuda.synchronize(); t0 = time.time()
        x = ORIG_LIN(ATy=ATy, x_init=x_init, model=model, A_funcs=A_funcs, betas=betas, iN=iN, gamma=gamma, deltas=deltas,
                     rhos=rhos, mu_0=mu_0, T_sampling=T_sampling, config=config, classes=classes)
        torch.cuda.synchronize()
        return finish(x, {"time": time.time() - t0, "nfe": T_sampling, "A": 0, "At": 0, "jvp": 0, "closed_form": T_sampling})
    sched = schedule(betas, iN, gamma, deltas, rhos, mu_0, T_sampling, ATy.device)
    if FUSE:
        x, rec = fuse(Problem(A_funcs, ATy=ATy), model, sched, False)
        return finish(x, rec)
    Pl = Problem(A_funcs, ATy=ATy)
    if LINEXACT:
        Pl.prox = svd_prox(A_funcs, ATy)
    if DISC > 0:
        Pl.prox = disc_prox(Pl)
    if os.environ.get("LINWARM", "0") == "1" and LINEXACT:      # explicit warm start: the regularised inverse of the measurement (one closed-form solve)
        x_init = Pl.prox(torch.zeros_like(x_init), float(sched[2][0]) * float(os.environ.get("LINWARM_R", "1"))).clamp(-1.0, 1.0)
    if os.environ.get("FIRE", "0") == "1":
        torch.cuda.synchronize(); t0 = time.time(); lv = [float(v) for v in os.environ["FIRE_SIGS"].split(",")]
        x, nfe_, flog = fire_loop(A_funcs, ATy, model, lv, x_init); torch.cuda.synchronize()
        return finish(x, {"time": time.time() - t0, "nfe": nfe_, "A": 0, "At": 0, "jvp": 0, "fire": flog, "time_total": time.time() - t_entry})
    x, rec, xf = unified(Pl, model, x_init.clone(), x_init.clone(), sched)
    torch.cuda.synchronize(); rec["time_total"] = time.time() - t_entry
    return finish(x, rec, xf)


WN.pnp_cm_natural_nonlinear_restoration = nl_entry
WL.pnp_cm_natural_linear_restoration = lin_entry
M.main()
