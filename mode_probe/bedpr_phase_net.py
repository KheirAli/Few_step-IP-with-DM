"""Bedroom phase predictor (run from PnP-CM/): the training part of phase_net2.py (same architecture phnet_arch.Net2 with 8 heads, same input
features, loss, optimiser and augmentation), trained ONLY on samples of the bedroom consistency model (bank64k_bed/, made by bedpr_genbank.py).
The last 512 samples are held out for the printed validation numbers.  No real image is read.  Output: phase_net_v2k8_bed.pt
env: NPARTS (8 -> 65536 images), EP (16), BSZ (128).  Resumes from the per-epoch checkpoint if interrupted."""
import sys, os, time, torch, torch.nn as nn
sys.path.insert(0, "."); sys.path.insert(0, "../mode_probe"); import functions.fft_util as FU
from phnet_arch import Net2, feat2
dev = torch.device("cuda:0"); K = 8; EP = int(os.environ.get("EP", "16")); NP = int(os.environ.get("NPARTS", "8")); BS = int(os.environ.get("BSZ", "128")); ck = "../mode_probe/phase_net_v2k8_bed.pt"
if os.path.exists(ck): print("exists"); sys.exit(0)
pad = lambda x: torch.nn.functional.pad(x, (64, 64, 64, 64)); Af = lambda im: FU.fft2_m(pad(im)); feat = lambda y: feat2(y, FU); low = lambda x: torch.nn.functional.avg_pool2d(x, 4)
X = torch.cat([torch.load(f"../mode_probe/bank64k_bed/part_{i}.pt") for i in range(NP)]); NTR = X.shape[0] - 512; Xtr, Xva = X[:NTR], X[NTR:]; print("bank", tuple(X.shape), flush=True)
with torch.no_grad(): F0 = torch.cat([feat(Af(Xtr[i:i + 64].to(dev).float() / 255).abs()) for i in range(0, 2048, 64)]); fm, fs = F0.mean(0, keepdim=True), F0.std(0, keepdim=True) + 1e-3; del F0
net = Net2(K).to(dev); print("params (M)", sum(p.numel() for p in net.parameters()) / 1e6, flush=True)
opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-4, total_steps=EP * (NTR // BS), pct_start=0.1); t0 = time.time(); ep0 = 0; tprev = 0.0
if os.path.exists(ck + ".tmp"):
    C = torch.load(ck + ".tmp", map_location=dev)
    if C.get("total") == EP * (NTR // BS): net.load_state_dict(C["sd"]); opt.load_state_dict(C["opt"]); sched.load_state_dict(C["sched"]); ep0 = C["ep"]; tprev = C["time"]; print("resumed after epoch", ep0, flush=True)
    del C
def batch(xb):
    with torch.no_grad(): y = Af(xb).abs() + 0.05 * torch.randn(xb.shape[0], 3, 384, 384, device=dev); return (feat(y) - fm) / fs
def loss_fn(p, t):
    e = ((p - t[:, None]) ** 2).mean(dim=(2, 3, 4)); return e.min(1).values.mean() + 0.05 * e.mean(), e
for ep in range(ep0, EP):
    net.train(); torch.manual_seed(1000 + ep); perm = torch.randperm(NTR)
    for i in range(0, NTR - BS + 1, BS):
        xb = Xtr[perm[i:i + BS]].to(dev).float() / 255; fl = torch.rand(xb.shape[0], device=dev) < 0.5; xb = torch.where(fl.view(-1, 1, 1, 1), torch.flip(xb, dims=(3,)), xb); f = batch(xb)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = net(f)
        l, _ = loss_fn(p.float(), low(xb)); opt.zero_grad(set_to_none=True); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        e = []; e32 = []; torch.manual_seed(7)
        for i in range(0, 512, 64):
            xv = Xva[i:i + 64].to(dev).float() / 255; p = net(batch(xv)); e.append(loss_fn(p, low(xv))[1]); e32.append(((torch.nn.functional.avg_pool2d(p.flatten(0, 1), 2).view(-1, K, 3, 32, 32) - torch.nn.functional.avg_pool2d(xv, 8)[:, None]) ** 2).mean(dim=(2, 3, 4)))
        e = torch.cat(e); e32 = torch.cat(e32)
    el = tprev + time.time() - t0
    print(f"epoch {ep + 1}: train loss {float(l):.5f} | held-out model samples: 64px PSNR best head {float(-10 * torch.log10(e.min(1).values).mean()):.2f}, mean head {float(-10 * torch.log10(e).mean()):.2f} | 32px best head {float(-10 * torch.log10(e32.min(1).values).mean()):.2f} | {el:.0f}s", flush=True)
    torch.save(dict(sd=net.state_dict(), fm=fm, fs=fs, opt=opt.state_dict(), sched=sched.state_dict(), ep=ep + 1, time=el, total=EP * (NTR // BS)), ck + ".tmp2"); os.replace(ck + ".tmp2", ck + ".tmp")
torch.save(dict(sd=net.state_dict(), fm=fm, fs=fs), ck + ".fin"); os.replace(ck + ".fin", ck); os.remove(ck + ".tmp"); print("PHNET ALLDONE", flush=True)
