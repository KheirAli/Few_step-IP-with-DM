"""PnP-CM's own MRI pipeline with ONE change: the injected noise is tilted towards high spatial frequencies (same total power).
usage (from PnP-CM root): TILT=0.5 python ../mode_probe/mri_tilt.py <main.py arguments>"""
import os, sys, torch
TILT = float(os.environ.get("TILT", "0.5")); _rl = torch.randn_like; _W = {}
def hooked(t, *a, **k):
    w = _rl(t, *a, **k)
    if sys._getframe(1).f_code.co_name != "pnp_cm_fastmri_restoration" or TILT == 0:
        return w
    key = (t.shape[-2], t.shape[-1], t.device)
    if key not in _W:
        fy = torch.fft.fftfreq(t.shape[-2], device=t.device); fx = torch.fft.fftfreq(t.shape[-1], device=t.device); r = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt().flatten()
        rank = torch.argsort(torch.argsort(r)).float(); p = ((rank + 0.5) / r.numel()) ** TILT; _W[key] = (p / p.mean()).sqrt().view(t.shape[-2], t.shape[-1])
    return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * _W[key], norm="ortho").real
torch.randn_like = hooked
sys.path.insert(0, os.getcwd()); sys.argv = ["main.py"] + sys.argv[1:]
import runpy; runpy.run_path("main.py", run_name="__main__")
