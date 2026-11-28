"""RAM (Reconstruct Anything Model, Terris et al. 2025; public pretrained checkpoint, one forward pass, no dataset-specific training) on our LINEAR tasks,
with exactly our operators and noise level.  RAM takes linear operators only, so the nonlinear tasks have no RAM entry.
Our measurement is y = A(x) + 0.1 n with x in [-1,1]; for x01 = (x+1)/2 this is y01 = A(x01) + 0.05 n, which is what RAM is given.
usage (from PnP-CM/): /usr/scratch/akheirandish3/BAM/.venv/bin/python ../mode_probe/ram_baseline.py"""
import os, sys, glob, time, types, numpy as np, torch, lpips, deepinv as dinv
from PIL import Image
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "../baselines_src"); import extra_ops as EO
dev = torch.device("cuda"); L = "../mode_probe"; lp = lpips.LPIPS(net="vgg", verbose=False).to(dev); ram = dinv.models.RAM(pretrained=True).to(dev).eval(); NAME = "RAM"
if os.environ.get("RAMSTAR"):                                      # RAM fine-tuned on CelebA-HQ (BAM project); NOTE: its training set contains 104 of our 107 CelebA-HQ test images
    sys.path.insert(0, "/usr/scratch/akheirandish3/BAM"); from bam.model import BAMNet
    rs = BAMNet(pretrained=False).to(dev); rs.load_state_dict(torch.load(os.environ["RAMSTAR"], map_location=dev)["ema"]); ram = rs.ram.eval(); NAME = "RAM*"
src = open("wrappers/natural_linear_wrapper.py").read().split("\n"); i0 = next(i for i, l in enumerate(src) if l.strip() == "deg = args.deg"); i1 = next(i for i, l in enumerate(src) if i > i0 and l.startswith("    else:"))
body = "\n".join(l[4:] for l in src[i0:i1])
def pnp_op(deg):
    ns = dict(args=types.SimpleNamespace(deg=deg, deg_scale=4.0, operator_imp="SVD", exp="exp", inpainting_mask_path=MASK), config=types.SimpleNamespace(data=types.SimpleNamespace(channels=3, image_size=256)), device=dev, torch=torch, np=np, os=os)
    exec(body, ns); F = ns["A_funcs"]; return lambda x: F.A(x)
MASK = sorted(os.listdir("exp/inpainting_masks"))[0] if len(sys.argv) < 2 else sys.argv[1]
OPS = {"sr_bicubic": pnp_op("sr_bicubic"), "deblur_gauss": pnp_op("deblur_gauss"), "inpainting": pnp_op("inpainting"), "ct60": EO.ct60, "ct30": EO.ct30, "mri4": EO.mri4, "mri8": EO.mri8}
def opnorm(A):                                                      # largest singular value by power iteration (autograd adjoint)
    x = torch.randn(1, 3, 256, 256, device=dev)
    for _ in range(30):
        with torch.enable_grad():
            z = x.clone().requires_grad_(True); Az = A(z); g_ = torch.autograd.grad((Az * Az.detach()).sum(), z)[0]
        n_ = float(g_.norm()); x = g_ / n_
    return n_ ** 0.5
NORMALISE = os.environ.get("RAMNORM", "1") == "1"   # RAM expects operators of norm about 1: restate the SAME measurement as (y/c) = (A/c)(x) + (sigma/c) n
def physics(A, c=1.0):
    def At(y, **k):
        with torch.enable_grad():
            x = torch.zeros(y.shape[0], 3, 256, 256, device=dev, requires_grad=True); return torch.autograd.grad((A(x) * y).sum(), x)[0].detach()
    return dinv.physics.LinearPhysics(A=lambda x, **k: A(x), A_adjoint=At, noise_model=dinv.physics.GaussianNoise(sigma=0.05 / c))
ld = lambda f: torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((256, 256), Image.BICUBIC)).astype(np.float32) / 255).permute(2, 0, 1)
print("inpainting mask file:", MASK, flush=True)
for ds, folder in (("celeba", "data_paper"), ("bedroom", "data_bed")):
    X = torch.stack([ld(f) for f in sorted(glob.glob(f"{L}/{folder}/**/*.png", recursive=True))])
    for t, A in OPS.items():
        c = opnorm(A) if NORMALISE else 1.0; A0 = A; A = (lambda x, A0=A0, c=c: A0(x) / c); ph = physics(A, c); Ps = []; Ls = []; tt = []; g = torch.Generator().manual_seed(1)
        try:
            for b in range(0, len(X), 4):
                x = X[b:b + 4].to(dev); Ax = A(x); y = Ax + 0.05 / c * torch.randn(Ax.shape, generator=g).to(dev); torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad(): r = ram(y, ph).clamp(0, 1)
                torch.cuda.synchronize(); tt.append((time.time() - t0) / len(x))
                with torch.no_grad(): Ps += (10 * torch.log10(1 / ((r - x) ** 2).mean(dim=(1, 2, 3)))).tolist(); Ls += lp(r * 2 - 1, x * 2 - 1).flatten().tolist()
            print(f"{NAME} | {ds} | {t} | {np.mean(Ps):.2f}/{np.mean(Ls):.3f} (n={len(Ps)}) | {np.median(tt[1:]):.2f} s per image (shared GPU) | operator norm {c:.2f}", flush=True)
        except Exception as e:
            print(f"{NAME} | {ds} | {t} | failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
