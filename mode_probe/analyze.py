"""Compare final sign-fix rules on the raw outputs of run_select.py variants.

Run from the PnP-CM repo root:  python ../mode_probe/analyze.py base sel0x ...
"""
import sys, glob, itertools, json, yaml
import numpy as np, torch, lpips, scipy.io as sio
from PIL import Image

sys.path.insert(0, ".")
from guided_diffusion.script_util import create_model
from pnp_cm_scheme.natural_nonlinear_scheme import get_scalings_for_boundary_condition

dev = torch.device("cuda:0")
cfg = yaml.safe_load(open("configs/celeba_hq_256.yml"))["model"]; cfg["image_size"] = 256
model = create_model(**cfg)
model.load_state_dict(torch.load("exp/logs/celeba_hq/ema_0.9999432189950708_1175000.pt", map_location=dev))
model.to(dev).eval()
lp = lpips.LPIPS(net="vgg").to(dev)
ref = torch.from_numpy(sio.loadmat("exp/phase_retrieval_ref_img/cm_out.mat")["cm_out"]).float().to(dev)
F = torch.tensor(list(itertools.product([1.0, -1.0], repeat=3)), device=dev).view(8, 3, 1, 1)


@torch.no_grad()
def cm_score(c, sigma=0.3, n=4, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    s = torch.tensor(sigma, device=dev)
    cs, co, ci = get_scalings_for_boundary_condition(s)
    t = (1000 * 0.25 * torch.log(s)).repeat(c.shape[0])
    acc = 0
    for _ in range(n):
        xn = c + s * torch.randn(c[:1].shape, device=dev, generator=g)
        acc = acc + (((co * model(ci * xn, t) + cs * xn).clamp(-1, 1) - c) ** 2).mean(dim=(1, 2, 3))
    return acc


def psnr(a, b):
    return float(10 * torch.log10(1 / (((a + 1) / 2 - (b + 1) / 2) ** 2).mean()))


out = {}
for v in sys.argv[1:]:
    rows = []
    for i in range(10):
        d = torch.load(f"../mode_probe/exp_{v}/raw_{i}.pt")
        x = d["x_raw"].to(dev)
        lab = glob.glob(f"../mode_probe/exp_{v}/image_samples/*/labels/orig_{i}.png")[0]
        gt = torch.from_numpy(np.asarray(Image.open(lab).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1).to(dev) * 2 - 1
        c = F * x
        with torch.no_grad():
            k_ref = int(lp(c, ref.expand(8, -1, -1, -1)).flatten().argmin())
            k_cm = int(cm_score(c).argmin())
            lp_all = lp(c, gt[None].expand(8, -1, -1, -1)).flatten()
        ps = [psnr(c[k], gt) for k in range(8)]
        k_gt = int(np.argmax(ps))
        rows.append(dict(img=i, raw_ok=(k_gt == 0), psnr_raw=ps[0], psnr_ref=ps[k_ref], psnr_cm=ps[k_cm],
                         psnr_best=ps[k_gt], ref_ok=(k_ref == k_gt), cm_ok=(k_cm == k_gt),
                         lpips_raw=float(lp_all[0]), lpips_ref=float(lp_all[k_ref]), lpips_cm=float(lp_all[k_cm]),
                         lpips_best=float(lp_all[k_gt]), picks=[(p[0], p[1]) for p in d["picks"]]))
        r = rows[-1]
        print(f"{v:8s} img{i} raw_ok={r['raw_ok']!s:5} raw={r['psnr_raw']:5.2f} ref={r['psnr_ref']:5.2f} "
              f"cm={r['psnr_cm']:5.2f} best={r['psnr_best']:5.2f} picks={r['picks']}")
    out[v] = rows

json.dump(out, open("../mode_probe/select_results.json", "w"), indent=1)
print("\nvariant   raw_sign_ok | PSNR raw / ref-face fix / CM fix / GT-sign bound | LPIPS raw / ref / CM / bound")
for v, rows in out.items():
    m = lambda k: np.mean([r[k] for r in rows])
    print(f"{v:8s}  {sum(r['raw_ok'] for r in rows):2d}/10 | {m('psnr_raw'):5.2f} / {m('psnr_ref'):5.2f} / {m('psnr_cm'):5.2f} / {m('psnr_best'):5.2f}"
          f" | {m('lpips_raw'):.3f} / {m('lpips_ref'):.3f} / {m('lpips_cm'):.3f} / {m('lpips_best'):.3f}"
          f" | ref_ok {sum(r['ref_ok'] for r in rows)}/10 cm_ok {sum(r['cm_ok'] for r in rows)}/10")
