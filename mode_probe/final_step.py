"""What is the best way to do the final refinement after two good steps?  (linear tasks)

Base: the first two shipped steps with shaped noise.  A third, final step is then one of
  none        stop after two steps                                                    (2 network calls)
  shipped3    the shipped third step (sigma 0.01, almost no noise)                     (3)
  dc          data-consistency solve only, no network call; penalty rho searched       (2)
  cm          one CM call; injected sigma, level told to the CM and rho searched       (3)
  avg4        four CM calls from four noise draws, averaged; same search               (6)
  pick4       four CM calls, keep the one whose A(x) is closest to y; same search      (6)
Each family's settings are chosen on the 8 calibration images by PSNR - LPIPS/0.03, then scored on the 10 test
images with 3 sampler seeds.  For reference the best setting by calibration PSNR and by calibration LPIPS is
also reported for the cm family.

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/final_step.py <task>
"""
import os, sys, json, itertools
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

base = [dict(S.SHIP[0]), dict(S.SHIP[1])]
mu3 = S.SHIP[2]["mu"]
fam = {"none": [None], "shipped3": [dict(S.SHIP[2])],
       "dc": [dict(st=0.0, scm=0.002, rho=r, mu=mu3, dc=True) for r in (0.1, 0.3, 1, 3, 10, 30)],
       "cm": [dict(st=a, scm=max(a * f, c), rho=r, mu=mu3) for a in (0.0, 0.01, 0.03, 0.06, 0.1)
              for f, c in ((1.0, 0.005), (1.5, 0.01), (3.0, 0.03)) for r in (1, 3, 10, 30)]}
for name, pick in (("avg4", False), ("pick4", True)):
    fam[name] = [dict(st=a, scm=a * f, rho=r, mu=mu3, K=4, pick=pick) for a in (0.03, 0.06, 0.1, 0.15)
                 for f in (1.0, 1.5) for r in (3, 10)]

q = lambda p, l: p - l / 0.03
rows = {}
print(f"\n######## {TASK}: final-step options after two shaped steps (test: 10 images x 3 seeds)")
print("family     calls | chosen on calibration (by PSNR - LPIPS/0.03)                 | test PSNR / LPIPS")
for name, cands in fam.items():
    ev = []
    for c in cands:
        st = base + ([c] if c else [])
        p, l, tab = S.cal_eval(st, "shaped", True)
        ev.append((p, l, st, tab))
    sel = {"q": max(ev, key=lambda e: q(e[0], e[1]))}
    if name == "cm":
        sel["psnr"], sel["lpips"] = max(ev, key=lambda e: e[0]), min(ev, key=lambda e: e[1])
    for how, (p, l, st, tab) in sel.items():
        tp, tl = S.test_eval(st, "shaped", tab)
        c = st[2] if len(st) > 2 else {}
        d = f"sigma {c.get('st', 0):.3f} told {c.get('scm', 0):.3f} rho {c.get('rho', 0):g}" if c else "-"
        calls = 2 + (0 if (not c or c.get("dc")) else c.get("K", 1))
        label = name if how == "q" else f"cm ({how})"
        rows[label] = dict(cal=(p, l), test=(tp, tl), step=c)
        print(f"{label:10s} {calls:5d} | {d:34s} cal {p:.2f}/{l:.3f}        | {tp:.2f} / {tl:.3f}", flush=True)
json.dump(rows, open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_final_step.json"), "w"))
