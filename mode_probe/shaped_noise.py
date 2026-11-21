"""Frequency-shaped noise injection for PnP-CM, for ANY task (no linearity or SVD needed).

The repo's sampler is run unmodified with the shipped schedule.  The only intervention is on the one
noise draw per step (`torch.randn_like(...) * sigma_t`): a hook replaces the white noise with noise whose
power per spatial frequency is  max(F^2 - P_k(f), 0), where P_k(f) is the typical error power the
iterate r = z + u already carries at that frequency in step k (measured on 8 calibration images).
  shape : F^2 = sigma_t^2 + mean_f P_k(f)   (same noise budget as shipped, redistributed)
  fill  : F   = level told to the CM
Steps with sigma_t < 0.05 are left as shipped.  The level told to the CM is always the shipped one.

Phase retrieval: the error is measured against the ground truth with the per-channel sign that best
matches the iterate, because the measurement cannot distinguish the sign.

  python ../mode_probe/shaped_noise.py <task> calib <shipped|shape|fill>
  python ../mode_probe/shaped_noise.py <task> test  <shipped|shape|fill> <seed>
Run from the PnP-CM repo root.
"""
import os, sys, json, glob
import numpy as np, torch
from PIL import Image

TASK, PHASE, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
sys.argv = ["random_steps.py", TASK, "--lib"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_steps as R

dev, T, NB = R.dev, R.T, 64
out = os.path.abspath("../mode_probe/random_steps/shaped")
os.makedirs(out, exist_ok=True)

_fx = torch.fft.fftfreq(256, device=dev)
_rad = (_fx[:, None] ** 2 + _fx[None, :] ** 2).sqrt().flatten()
FB = (torch.argsort(torch.argsort(_rad)) * NB // _rad.numel()).long()
FCNT = torch.bincount(FB, minlength=NB).float()


def fft_band(p):  # [1,3,H,W] power per Fourier coefficient -> NB radial band means, low to high frequency
    return torch.bincount(FB, weights=p.mean(dim=(0, 1)).flatten(), minlength=NB) / FCNT


st = dict(n=0, x0=None, log=[], spec=None, oracle=False)
_randn_like = torch.randn_like
SCHEMES = ("pnp_cm_natural_nonlinear_restoration", "pnp_cm_natural_linear_restoration")


def hooked(t, *a, **kw):
    fr = sys._getframe(1)
    if fr.f_code.co_name not in SCHEMES:
        return _randn_like(t, *a, **kw)
    k = st["n"] % T
    st["n"] += 1
    w = _randn_like(t)
    loc = fr.f_locals
    sig, L = float(loc["sigma_t_i"].flatten()[0]), float(loc["sigma_cm_i"].flatten()[0])
    sp = None
    if st["x0"] is not None:  # calibration images only: measure the error spectrum of the iterate
        r, x0 = loc["zt"] + loc["ut_moment"], st["x0"]
        if TASK == "phase_retrieval":
            x0 = x0 * torch.sign((r * x0).sum(dim=(2, 3), keepdim=True))
        sp = fft_band(torch.fft.fft2(r - x0, norm="ortho").abs() ** 2)
        st["log"].append(dict(step=k, sigma_t=sig, told=L, spec=sp.tolist()))
    if MODE == "shipped" or sig < 0.05:
        return w
    sp = sp if st["oracle"] else st["spec"][k]
    F2 = L ** 2 if MODE == "fill" else sig ** 2 + float(sp.mean())
    g = (F2 - sp[FB].view(256, 256)).clamp(min=0).sqrt()
    return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * g, norm="ortho").real / sig


torch.randn_like = hooked
ship = R.configs_all[0]

if PHASE == "calib":
    f = sorted(glob.glob("/usr/scratch/akheirandish3/diffgeo/data/celebahq256/*.png"))
    im = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).float().permute(2, 0, 1) / 255 for p in f]).to(dev)
    tt = torch.cat([(x0 + 1) / 2 for x0, _ in R.data])
    ok = np.where((torch.stack([((im - t[None]) ** 2).mean(dim=(1, 2, 3)) for t in tt], 1).min(1).values > 1e-3).cpu().numpy())[0]
    cal = R.make_data([im[i][None] for i in np.random.default_rng(0).permutation(ok)[:8]], 7000)
    st["oracle"] = True
    for i, (x0, y) in enumerate(cal):
        st["x0"], st["n"] = x0, 0
        torch.manual_seed(100 + i)
        with torch.no_grad():
            R.restore(ship, y)
    spec = [np.mean([l["spec"] for l in st["log"] if l["step"] == k], 0).tolist() for k in range(T)]
    sig = [float(np.mean([l["sigma_t"] for l in st["log"] if l["step"] == k])) for k in range(T)]
    json.dump(dict(spec=spec, sigma_t=sig), open(f"{out}/{TASK}_cal_{MODE}.json", "w"))
    print(f"CALIB {TASK} {MODE}: error rms of the iterate by spatial frequency (8 bands low->high) | injected sigma")
    for k in range(T):
        print(f"  step {k + 1}: " + "  ".join(f"{v:.3f}" for v in np.sqrt(np.array(spec[k]).reshape(8, -1).mean(1))) + f" | {sig[k]:.3f}")
else:
    if MODE != "shipped":
        st["spec"] = [torch.tensor(s, device=dev, dtype=torch.float32)
                      for s in json.load(open(f"{out}/{TASK}_cal_{MODE}.json"))["spec"]]
    rows = R.run(dict(ship, seed=SEED), data=R.data)
    json.dump(rows, open(f"{out}/{TASK}_test_{MODE}_s{SEED}.json", "w"))
    print(f"TEST {TASK} {MODE} seed {SEED}: " + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in rows[0]), flush=True)
