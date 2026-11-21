"""Where is the uncertainty, and does putting noise only where it is needed help?  (Gaussian deblur)

The blur operator has an SVD, A = U S V^T, so every image splits into components that the measurement
pins down well (large singular value s) and components it barely sees (small or zero s).

Part 1 (diagnostic): at each PnP-CM step, the true error of the CM input (z + u - ground truth) is split
into 8 equal-size bands of components ordered from best-measured to unmeasured.

Part 2 (variants of the noise step; rhos, momentum and the shipped levels stay as they are).
"Targeted" noise is added per component with variance max(F^2 - err_i^2, 0): components that already
carry error get less noise, so the total error entering the CM is close to white at level F.
  shipped      white noise sigma_t, CM told the tuned L = (1+delta)*sigma_t
  svd_fill     components = SVD of the operator;   F = L (the level the CM is told)
  svd_shape    components = SVD of the operator;   F = total error level of the shipped scheme
               (same noise budget as shipped, only its distribution changes; applied only in
               steps with sigma_t >= 0.05, the final polish steps are left as shipped)
  fft_fill     components = spatial frequencies;   F = L
  fft_shape    components = spatial frequencies;   F = total error level of the shipped scheme
The error spectra err_i are measured once, per step, on 8 calibration images and reused for every test
image, so no test ground truth is used.

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/uncertainty_probe.py
"""
import os, sys, json
import numpy as np, torch

TASK = sys.argv[1] if len(sys.argv) > 1 else "deblur_gauss"
sys.argv = ["random_steps.py", TASK, "--lib"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_steps as R

S, A, dev, T, net = R.S, R.A, R.dev, R.T, R.model
SIG_Y, NB = 0.1, 64  # measurement noise on [-1,1] images; bands used to smooth the error spectrum
sing = A.add_zeros(A.singulars()[None])[0]  # one singular value per SVD component (0 = unmeasured)
pen = torch.nn.Softplus()(torch.linspace(R.rhos[0], R.rhos[1], T)).to(dev)
mom = R.hp["mu_0"] / torch.arange(1, T + 1)
al = [S.compute_alpha(R.betas, torch.tensor([R.hp["iN"] + 1], device=dev))]
for _ in range(T - 1):
    al.append(S.get_next_alpha(al[-1], R.hp["gamma"]).reshape(1))
SIG_T = [float(torch.sqrt(1 - a).clamp(0.002, 80)) for a in al]
SIG_CM = [float(((1 + d) * torch.sqrt(1 - a)).clamp(0.002, 80)) for a, d in zip(al, R.ship_deltas)]


def cm(z, sigma):
    s = torch.tensor(float(np.clip(sigma, 0.002, 80.0)), device=dev)
    c_skip, c_out, c_in = S.get_scalings_for_boundary_condition(s)
    return (c_out * net(c_in * z, (250 * torch.log(s + 1e-44)).reshape(1)) + c_skip * z).clamp(-1, 1)


ORDER = torch.argsort(sing, descending=True)  # components from best measured to unmeasured
INV = torch.argsort(ORDER)


def band(v, n):  # mean over n equal-size bands of components ordered by singular value
    return v[ORDER].reshape(n, -1).mean(1)


def spread(b):  # band values -> one value per component
    return b.repeat_interleave(sing.numel() // b.numel())[INV]


# ---- 8 calibration images (same selection as calibrate.py), used for the ground-truth-free variant
import glob
from PIL import Image
_f = sorted(glob.glob("/usr/scratch/akheirandish3/diffgeo/data/celebahq256/*.png"))
_im = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).float().permute(2, 0, 1) / 255 for f in _f]).to(dev)
_t = torch.cat([(x0 + 1) / 2 for x0, _ in R.data])
_ok = np.where((torch.stack([((_im - t[None]) ** 2).mean(dim=(1, 2, 3)) for t in _t], 1).min(1).values > 1e-3).cpu().numpy())[0]
cal = R.make_data([_im[i][None] for i in np.random.default_rng(0).permutation(_ok)[:8]], 7000)
with torch.no_grad():
    CAL_INIT = [R.x_init_of(y) for _, y in cal]
    TEST_INIT = [R.x_init_of(y) for _, y in R.data]
CAL = {}  # (basis, mode) -> per-step banded error spectrum measured on the calibration images

# radial spatial-frequency bands (equal size) for the Fourier-basis variants
_fx = torch.fft.fftfreq(256, device=dev)
_rad = (_fx[:, None] ** 2 + _fx[None, :] ** 2).sqrt().flatten()
FB = (torch.argsort(torch.argsort(_rad)) * NB // _rad.numel()).long()
FCNT = torch.bincount(FB, minlength=NB).float()


def fft_band(p):  # p: [1,3,H,W] power per Fourier coefficient -> NB band means (low to high frequency)
    return torch.bincount(FB, weights=p.mean(dim=(0, 1)).flatten(), minlength=NB) / FCNT


@torch.no_grad()
def pnp(x0, y, variant, seed, log, xinit):
    torch.manual_seed(seed)
    shp = x0.shape
    ATy = A.At(y).view(shp)
    x = xinit.clone(); u = torch.zeros(shp, device=dev); xm = x.clone(); um = u.clone()
    for k in range(T):
        x1, u1 = x, u
        z = A.AtA_add_eta_inv(ATy + pen[k] * (xm - um), eta=pen[k]).reshape(shp)
        r = z + um
        e = r - x0  # true error of the CM input (used only for diagnostics and the oracle calibration pass)
        spec = {"svd": band(A.Vt(e)[0] ** 2, NB), "fft": fft_band(torch.fft.fft2(e, norm="ortho").abs() ** 2)}
        st, L = SIG_T[k], SIG_CM[k]
        if variant == "shipped" or (variant.split("_")[1] == "shape" and st < 0.05):
            noise = torch.randn_like(xm) * st  # "shape" leaves the near-noiseless polish steps untouched
        else:
            basis, mode, src = variant.split("_")
            sp = spec[basis] if src == "oracle" else CAL[(basis, mode)][k]
            F = L if mode == "fill" else float((st ** 2 + sp.mean()).sqrt())
            if basis == "svd":
                noise = A.V((torch.randn(sing.numel(), device=dev) * (F ** 2 - spread(sp)).clamp(min=0).sqrt())[None]).reshape(shp)
            else:
                g = (F ** 2 - sp[FB].view(256, 256)).clamp(min=0).sqrt()
                noise = torch.fft.ifft2(torch.fft.fft2(torch.randn_like(xm), norm="ortho") * g, norm="ortho").real
        x = cm(r + noise, L)
        u = um + (z - x)
        xm, um = x + mom[k] * (x - x1), u + mom[k] * (u - u1)
        log.append(dict(step=k, variant=variant, told=L, added=float(noise.pow(2).mean().sqrt()),
                        total=float((e + noise).pow(2).mean().sqrt()),
                        svd8=spec["svd"].reshape(8, -1).mean(1).sqrt().tolist(),
                        fft8=spec["fft"].reshape(8, -1).mean(1).sqrt().tolist(),
                        svd=spec["svd"].tolist(), fft=spec["fft"].tolist()))
    return x


# calibration pass: per-step error spectra, measured with ground truth on the 8 calibration images only
for basis in ("svd", "fft"):
    for mode in ("fill", "shape"):
        _cl = []
        for i, (x0, y) in enumerate(cal):
            pnp(x0, y, f"{basis}_{mode}_oracle", 100 + i, _cl, CAL_INIT[i])
        CAL[(basis, mode)] = [torch.tensor(np.mean([l[basis] for l in _cl if l["step"] == k], 0), device=dev,
                                           dtype=torch.float32) for k in range(T)]

VARS = ["shipped", "svd_fill_calib", "svd_shape_calib", "fft_fill_calib", "fft_shape_calib"]
res, logs = {v: [] for v in VARS}, []
for v in VARS:
    for i, (x0, y) in enumerate(R.data):
        for sd in range(3):
            x = pnp(x0, y, v, i + 1000 * sd, logs if sd == 0 else [], TEST_INIT[i])
            res[v].append((R.psnr(x, x0), float(R.lp(x, x0))))

with torch.no_grad():
    _ref = []
    for i, (x0, y) in enumerate(R.data):
        torch.manual_seed(i)
        xr = R.restore(R.configs_all[0], y)
        _ref.append((R.psnr(xr, x0), float(R.lp(xr, x0))))
print(f"\n################ {TASK}\nrepo's own sampler, shipped schedule: PSNR {np.mean([r[0] for r in _ref]):.2f} LPIPS {np.mean([r[1] for r in _ref]):.3f}"
      f"   (the 'shipped' row below is my re-implementation of the same loop and should match)")
sh = lambda k: [l for l in logs if l["variant"] == "shipped" and l["step"] == k]
print("\n=== Part 1: where the error of the CM input lives (shipped run, mean over 10 images; rms per band)")
print(f"(a) by operator SVD component, band 1 = best measured ... 8 = unmeasured; {float((sing == 0).float().mean()) * 100:.0f}% have singular value 0")
print("    mean singular value  " + "  ".join(f"{s:6.3f}" for s in band(sing, 8).tolist()) + " | white noise added")
for k in range(T):
    print(f"    step {k + 1} error rms    " + "  ".join(f"{np.mean([l['svd8'][b] for l in sh(k)]):6.3f}" for b in range(8)) + f" | {SIG_T[k]:.3f}")
print("(b) by spatial frequency, band 1 = lowest ... 8 = highest")
for k in range(T):
    print(f"    step {k + 1} error rms    " + "  ".join(f"{np.mean([l['fft8'][b] for l in sh(k)]):6.3f}" for b in range(8)) + f" | {SIG_T[k]:.3f}")

print("\n=== Part 2: results on the 10 test images, mean of 3 sampler seeds")
print(f"{'variant':16s} PSNR   LPIPS | level told to CM per step        | rms noise added per step         | total error entering CM")
for v in VARS:
    f = lambda key: np.round([float(np.mean([l[key] for l in logs if l["variant"] == v and l["step"] == k])) for k in range(T)], 3).tolist()
    print(f"{v:16s} {np.mean([r[0] for r in res[v]]):5.2f}  {np.mean([r[1] for r in res[v]]):.3f} | "
          f"{f('told')!s:32s} | {f('added')!s:32s} | {f('total')}")
json.dump(dict(res=res, logs=[{k: v for k, v in l.items() if k not in ("svd", "fft")} for l in logs]),
          open(os.path.abspath(f"../mode_probe/random_steps/uncertainty_probe_{TASK}.json"), "w"))
