"""Summarize calibrate.py results.  Run: python calibrate_analyze.py"""
import glob, json, os
import numpy as np

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_steps")
m = lambda rows, k: float(np.mean([r[k] for r in rows]))
rank = lambda v: np.argsort(np.argsort(v))

for t in ["deblur_gauss", "deblur_nl", "phase_retrieval"]:
    test = {c["id"]: c for c in (json.load(open(f)) for f in glob.glob(f"{root}/{t}/c*.json"))}
    pool = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/{t}/cal/c*.json"))]
    pool = [c for c in pool if c["id"] in test]
    ship = [c for c in test.values() if c["cond"] == "shipped"]
    sp, sl = np.mean([m(c["rows"], "psnr") for c in ship]), np.mean([m(c["rows"], "lpips") for c in ship])
    print(f"\n================ {t}   shipped on the 10 test images: PSNR {sp:.2f}  LPIPS {sl:.3f}")

    if pool:
        tp = np.array([m(test[c["id"]]["rows"], "psnr") for c in pool])
        tl = np.array([m(test[c["id"]]["rows"], "lpips") for c in pool])
        cand = np.array([c["cond"] != "shipped" for c in pool])  # selection may not pick the shipped schedule
        s0 = next(c for c in pool if c["cond"] == "shipped")
        print(f"-- pick the best of {cand.sum()} random/δ=0 schedules by calibration score (shipped excluded)")
        print(f"   {'criterion':22s} cal score (shipped) | test PSNR  LPIPS | rank-corr cal vs test | test-oracle pick")
        for n in (4, 8):
            cp = np.array([m(c["cal_rows"][:n], "psnr") for c in pool])
            cl = np.array([m(c["cal_rows"][:n], "lpips") for c in pool])
            rp, rl = m(s0["cal_rows"][:n], "psnr"), m(s0["cal_rows"][:n], "lpips")
            for name, cs, ts, ref in [("PSNR", cp, tp, f"{rp:.2f}"), ("LPIPS", -cl, -tl, f"{rl:.3f}"),
                                      ("both", np.minimum(cp - rp, (rl - cl) / 0.03), np.minimum(tp - sp, (sl - tl) / 0.03), "0")]:
                i = np.where(cand)[0][np.argmax(cs[cand])]
                j = np.where(cand)[0][np.argmax(ts[cand])]
                rc = np.corrcoef(rank(cs[cand]), rank(ts[cand]))[0, 1]
                print(f"   cal{n} {name:17s} {abs(cs[i]):9.3f} ({ref:>6s}) | {tp[i]:9.2f}  {tl[i]:.3f} | {rc:21.2f} | "
                      f"{tp[j]:.2f} / {tl[j]:.3f}")

    es = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/{t}/cal/es_*.json")) if not f.endswith("_hist.json")]
    if es:
        print("-- evolution-strategy search on calibration images, winner scored on the 10 test images")
        print(f"   {'run':28s} evals | test PSNR (Δ)      LPIPS (Δ)       | injected sigma -> sigma given to CM")
        for r in es:
            P = [m(x, "psnr") for x in r["test"]]
            L = [m(x, "lpips") for x in r["test"]]
            print(f"   {r['tag']:28s} {r['n_evals']:5d} | {np.mean(P):6.2f} ({np.mean(P) - sp:+.2f})   "
                  f"{np.mean(L):.3f} ({np.mean(L) - sl:+.3f}) | {np.round(r['sigma_t'], 3).tolist()} -> {np.round(r['sigma_cm'], 3).tolist()}")
