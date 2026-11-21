"""Summarize random_steps.py results: table + strip plot.  Run: python random_steps_analyze.py"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_steps")
TASKS = ["deblur_gauss", "deblur_nl", "phase_retrieval"]
CONDS = ["shipped", "matched", "jitter", "rand_sorted", "rand_unsorted", "wide_sorted"]
LABEL = {"shipped": "shipped\n(seeds)", "matched": "shipped,\nδ=0", "jitter": "random in\nauthors' ranges",
         "rand_sorted": "random σ≤1\ndecreasing", "rand_unsorted": "random σ≤1\nany order",
         "wide_sorted": "random σ≤80\ndecreasing"}

res = {}
for t in TASKS:
    cf = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/{t}/c*.json"))]
    for c in cf:
        for k in c["rows"][0]:
            c[k] = float(np.mean([r[k] for r in c["rows"]]))
    res[t] = cf

for t in TASKS:
    cf = res[t]
    if not cf:
        continue
    ship = [c for c in cf if c["cond"] == "shipped"]
    sp, sl = np.mean([c["psnr"] for c in ship]), np.mean([c["lpips"] for c in ship])
    print(f"\n=== {t}  ({len(cf)} schedules x {len(cf[0]['rows'])} images)   shipped: PSNR {sp:.2f}  LPIPS {sl:.3f}")
    print(f"{'condition':14s} {'n':>3s} | PSNR median   min    max | LPIPS median   min    max | "
          f"within 0.5dB  beats shipped (PSNR / LPIPS)")
    for k in CONDS:
        g = [c for c in cf if c["cond"] == k]
        if not g:
            continue
        P, L = np.array([c["psnr"] for c in g]), np.array([c["lpips"] for c in g])
        print(f"{k:14s} {len(g):3d} | {np.median(P):11.2f} {P.min():6.2f} {P.max():6.2f} | "
              f"{np.median(L):12.3f} {L.min():6.3f} {L.max():6.3f} | "
              f"{np.mean(P >= sp - 0.5) * 100:9.0f}%   {np.sum(P > sp):3d} / {np.sum(L < sl):3d}")
    if "psnr_oracle_sign" in cf[0]:
        print("   (oracle-sign PSNR, median per condition: " + ", ".join(
            f"{k} {np.median([c['psnr_oracle_sign'] for c in cf if c['cond'] == k]):.2f}"
            for k in CONDS if any(c['cond'] == k for c in cf)) + ")")
    rnd = [c for c in cf if c["cond"] in ("jitter", "rand_sorted", "rand_unsorted", "wide_sorted")]
    if len(rnd) > 5:
        P = np.array([c["psnr"] for c in rnd])
        for name, v in [("first injected sigma", [c["sigma_t"][0] for c in rnd]),
                        ("last injected sigma", [c["sigma_t"][-1] for c in rnd]),
                        ("last sigma given to CM", [c["sigma_cm"][-1] for c in rnd]),
                        ("max sigma given to CM", [max(c["sigma_cm"]) for c in rnd])]:
            r = np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(P)))[0, 1]
            print(f"   rank corr(PSNR, {name}) = {r:+.2f}")
        best = max(rnd, key=lambda c: c["psnr"])
        print(f"   best random: {best['cond']} PSNR {best['psnr']:.2f} LPIPS {best['lpips']:.3f} "
              f"sigma_t {np.round(best['sigma_t'], 3).tolist()} sigma_cm {np.round(best['sigma_cm'], 3).tolist()}")

# ---- figure: one dot = one schedule (mean over the 10 images)
done = [t for t in TASKS if res[t]]
fig, axes = plt.subplots(2, len(done), figsize=(5.2 * len(done), 7.2), squeeze=False)
INK, MUTED, DOT, REF = "#1f2328", "#6b7280", "#3b6fb6", "#1f2328"
rng = np.random.default_rng(0)
for j, t in enumerate(done):
    for i, (key, name) in enumerate([("psnr", "PSNR (dB) — higher is better"), ("lpips", "LPIPS — lower is better")]):
        ax = axes[i, j]
        ship = np.mean([c[key] for c in res[t] if c["cond"] == "shipped"])
        ax.axhline(ship, color=REF, lw=1, ls=(0, (4, 3)), zorder=1)
        for x, k in enumerate(CONDS):
            v = np.array([c[key] for c in res[t] if c["cond"] == k])
            if len(v) == 0:
                continue
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=30, color=DOT, alpha=0.75,
                       edgecolor="white", linewidth=0.8, zorder=3)
            if len(v) > 2:
                ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([LABEL[k] for k in CONDS] if i == 1 else [], fontsize=8, color=MUTED)
        ax.set_ylabel(name, fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title(f"{t}  ({len(res[t][0]['sigma_t'])} steps)", fontsize=11, color=INK, loc="left")
            ax.annotate("shipped", xy=(len(CONDS) - 0.55, ship), xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="right")
fig.suptitle("PnP-CM with random noise schedules (CelebA-HQ, 10 images). One dot = one schedule; bar = median; "
             "dashed = shipped schedule", fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{root}/random_steps.png", dpi=150, facecolor="white")
print(f"\nfigure: {root}/random_steps.png")
