"""Our pipeline on REAL multi-coil MRI, inside PnP-CM's own MRI code (same model, operator, schedule, data step and metrics).
MRIMODE=gen  : write generated calibration data: 2-step samples of the consistency model, k-space simulated with the coil maps of the folder given
               by --path_y (subjects NOT in the test set) -> exp/datasets/fastMRI/<MRIGEN>/
MRIMODE=cal  : run on generated data, save the terms of step MRIK (earlier passed-on images, data-fit image, model output) and the truth
MRIMODE=use  : run with the fitted in-loop band combination (MRIW) and the noise tilt (TILT)
MRIMODE=fit  : (no model) least-squares fit of step MRIK's weights, 8 radial bands, from the saved terms
usage (from PnP-CM/): MRIMODE=... python ../mode_probe/mri_ours.py <main.py arguments>"""
import os, sys, glob, torch, numpy as np
MODE = os.environ.get("MRIMODE", "use"); TILT = float(os.environ.get("TILT", "0.5")); WF = os.environ.get("MRIW", ""); K_ = int(os.environ.get("MRIK", "0")); LOG = os.environ.get("MRILOG", "")
NB = 8; _C = {}
def bidx(n, dev):
    if (n, dev) not in _C:
        f = torch.fft.fftfreq(n, device=dev) * n; _C[(n, dev)] = torch.bucketize((f[:, None] ** 2 + f[None, :] ** 2).sqrt(), torch.tensor([1.5 * 2 ** k for k in range(NB - 1)], device=dev))
    return _C[(n, dev)]
cx = lambda t: torch.complex(t[0, 0], t[0, 1])                      # [1,2,H,W] real -> [H,W] complex
if MODE == "fit":
    T = K_ + 2; G = torch.zeros(NB, T, T, dtype=torch.float64); b = torch.zeros(NB, T, dtype=torch.float64); n = 0
    for f in sorted(glob.glob(f"{LOG}/*.pt")):
        d = torch.load(f); H = torch.fft.fft2(torch.stack([cx(t.float()) for t in d["terms"]]), norm="ortho"); g = torch.fft.fft2(d["gt"].to(torch.complex64), norm="ortho"); bi = bidx(H.shape[-1], "cpu"); n += 1
        for k in range(NB):
            A = H[:, bi == k].T.to(torch.complex128); G[k] += (A.conj().T @ A).real; b[k] += (A.conj().T @ g[bi == k].to(torch.complex128)).real
    W = torch.stack([torch.linalg.pinv(G[k] + 1e-6 * G[k].diagonal().mean() * torch.eye(T, dtype=torch.float64), hermitian=True) @ b[k] for k in range(NB)], 1).float()   # [T, NB]
    C = torch.load(WF) if os.path.exists(WF) else {}; C[str(K_)] = W; torch.save(C, WF)
    print(f"MRIFIT step {K_ + 1} n={n}: model output per band " + " ".join(f"{v:+.2f}" for v in W[-1].tolist()) + " | data-fit image per band " + " ".join(f"{v:+.2f}" for v in W[-2].tolist()), flush=True); sys.exit(0)
sys.path.insert(0, os.getcwd())
import wrappers.fastmri_wrapper as FW, pnp_cm_scheme.fastmri_scheme as S, torch.nn as nn, scipy.io as sio
ST = {"gt": None, "n": 0}; _nc = FW.normalize_complex
def _cap(x):
    o = _nc(x); ST["gt"] = o.detach().clone(); return o
FW.normalize_complex = _cap
def tilted(t, sig):
    w = torch.randn_like(t)
    if TILT == 0: return w * sig
    key = ("w", t.shape[-1], t.device)
    if key not in _C:
        f = torch.fft.fftfreq(t.shape[-1], device=t.device); r = (f[:, None] ** 2 + f[None, :] ** 2).sqrt().flatten(); rank = torch.argsort(torch.argsort(r)).float(); p = ((rank + 0.5) / r.numel()) ** TILT; _C[key] = (p / p.mean()).sqrt().view(t.shape[-2], t.shape[-1])
    return torch.fft.ifft2(torch.fft.fft2(w, norm="ortho") * _C[key], norm="ortho").real * sig
EXACT = os.environ.get("MRIEXACT", "0") == "1"; _X = {}
def exact_fit(A_funcs, ATy, v, zeta):
    """EXACT, non-iterative data step for Cartesian sampling that is undersampled along ONE axis (lines): the k-space mask depends only on the last
    axis, so after an inverse FFT along the fully sampled axis the problem splits into one independent system per image row h:
        ( sum_c diag(conj s_c[h]) K diag(s_c[h]) + zeta I ) z[h] = (zeta v + A'y)[h],     K = F' diag(mask) F  (W x W, the same for every row)
    H systems of size W, solved directly in one batched call.  Same minimiser as PnP-CM's conjugate gradients, without iterations."""
    s_ = A_funcs.sens[0]; key = id(A_funcs)
    if _X.get("key") != key:
        W_ = s_.shape[-1]; m_ = A_funcs.mask.reshape(-1, W_)[0].to(torch.complex64); fc = lambda t, inv: torch.fft.fftshift((torch.fft.ifft if inv else torch.fft.fft)(torch.fft.ifftshift(t, dim=0), dim=0, norm="ortho"), dim=0)
        K = fc(m_[:, None] * fc(torch.eye(W_, dtype=torch.complex64, device=s_.device), False), True)
        _X.update(key=key, N=torch.einsum("chi,chj->hij", s_.conj(), s_) * K[None], eye=torch.eye(W_, dtype=torch.complex64, device=s_.device))
    rhs = zeta * A_funcs.real_to_nchw_comp(v)[0, 0] + A_funcs.real_to_nchw_comp(ATy)[0, 0]
    z = torch.linalg.solve(_X["N"] + zeta * _X["eye"][None], rhs[..., None])[..., 0]
    if os.environ.get("MRICHECK") and not _X.get("chk"):
        _X["chk"] = 1; zc = A_funcs.real_to_nchw_comp(A_funcs.CG(ATy, v, zeta, cg_iter=300))[0, 0]; z10 = A_funcs.real_to_nchw_comp(A_funcs.CG(ATy, v, zeta, cg_iter=10))[0, 0]
        print(f"MRICHECK relative difference: exact vs 300 CG iterations {float((z - zc).norm() / zc.norm()):.2e} | 10 CG iterations vs 300: {float((z10 - zc).norm() / zc.norm()):.2e}", flush=True)
    return A_funcs.nchw_comp_to_real(z[None, None])


def restore(ATy, model, A_funcs, betas, iN, gamma, deltas, rhos, mu_0, T_sampling, config, classes=None, sigma_min=0.002, sigma_max=80.0):
    dev = ATy.device; den = lambda z, s: (lambda cs, co, ci: (co * model(ci * z, 1000 * 0.25 * torch.log(s + 1e-44)) + cs * z).clamp(-1.0, 1.0))(*[S.append_dims(v, z.ndim) for v in S.get_scalings_for_boundary_condition(s)])
    if MODE == "gen":                                               # generated calibration images with this slice's coil maps
        out = os.path.join("exp/datasets/fastMRI", os.environ["MRIGEN"]); os.makedirs(out, exist_ok=True); m0 = A_funcs.mask; A_funcs.mask = torch.ones_like(m0)
        for j in range(int(os.environ.get("MRINGEN", "4"))):
            s1 = torch.tensor([80.0], device=dev); s2 = torch.tensor([0.8], device=dev); x = den(torch.randn_like(ATy) * 80.0, s1); x = den(x + 0.8 * torch.randn_like(x), s2)
            k = A_funcs.A(A_funcs.real_to_nchw_comp(x)); sio.savemat(os.path.join(out, f"gen_{ST['n']:04d}_{j}.mat"), {"kspace": k[0].cpu().numpy().astype(np.complex64), "coils": A_funcs.sens[0].cpu().numpy().astype(np.complex64)})
        A_funcs.mask = m0; ST["n"] += 1; return torch.zeros_like(ATy)
    W = torch.load(WF) if (WF and os.path.exists(WF)) else {}
    pen = nn.Softplus()(torch.linspace(rhos[0], rhos[1], T_sampling)).to(dev) * float(os.environ.get("MRIRHOX", "1")); mom = (torch.tensor([mu_0]).repeat(T_sampling) / torch.arange(1, T_sampling + 1)).to(dev)
    xt = torch.zeros_like(ATy); xm = torch.zeros_like(ATy); ut = torch.zeros_like(ATy); um = torch.zeros_like(ATy)
    al = [S.compute_alpha(betas, (torch.ones(1) * (iN + 1)).to(dev).long())]
    for _ in range(T_sampling - 1): al.append(S.get_next_alpha(al[-1], gamma).reshape(1))
    hist = []
    for i, at in enumerate(al):
        sg = torch.sqrt(1 - at).clamp(sigma_min, sigma_max); sc = ((1 + deltas[i]) * torch.sqrt(1 - at)).clamp(sigma_min, sigma_max); x1 = xt.clone(); u1 = ut.clone()
        zt = exact_fit(A_funcs, ATy, (xm - um), pen[i]) if EXACT else A_funcs.CG(ATy, (xm - um), pen[i], cg_iter=10); xt = den(zt + um + tilted(xm, float(sg[0])), sc)
        if i >= 1:
            terms = hist + [zt, xt]
            if MODE == "cal" and i == K_:
                os.makedirs(LOG, exist_ok=True); torch.save({"terms": [t.detach().half().cpu() for t in terms], "gt": ST["gt"].reshape(ATy.shape[-2], ATy.shape[-1]).cpu()}, f"{LOG}/{ST['n']:04d}.pt")
            if str(i) in W:
                Wi = W[str(i)].to(dev); bi = bidx(ATy.shape[-1], dev); F = torch.fft.fft2(torch.stack([cx(t) for t in terms]), norm="ortho"); c = torch.fft.ifft2((Wi[:, bi] * F).sum(0), norm="ortho")
                xt = torch.stack([c.real, c.imag])[None].to(xt.dtype).clamp(-1.0, 1.0)
        hist.append(xt.clone()); ut = um + (zt - xt); xm = xt + mom[i] * (xt - x1); um = ut + mom[i] * (ut - u1)
    ST["n"] += 1; return xt
FW.pnp_cm_fastmri_restoration = restore
sys.argv = ["main.py"] + sys.argv[1:]
import runpy; runpy.run_path("main.py", run_name="__main__")
