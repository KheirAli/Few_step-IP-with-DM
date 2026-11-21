"""Drop the measurement update in the last step(s): let the CM alone refine the estimate.

A "prior-only" step skips the data-fit solve and the dual update; the CM is applied to the previous estimate
plus optional noise:  x <- CM(x + sigma * eps, told).

Linear tasks (base = first two shipped steps, shaped noise):
  none / shipped3      references (2 and 3 network calls)
  prior1               one prior-only step, (sigma, told) searched on the calibration images
  prior2               two prior-only steps with the same setting
Nonlinear deblur (base = shipped 4 steps, shaped noise), where the measurement update is the slow part:
  drop4, drop34        steps 4 / 3-4 keep their shipped noise but lose the measurement update
  extra                4 normal steps plus one prior-only step at small noise

Run from the PnP-CM repo root:  CUDA_VISIBLE_DEVICES=g python ../mode_probe/prior_only.py <task>
"""
import os, sys, json
import numpy as np, torch

TASK = sys.argv[1]
sys.argv = ["steps.py", "lib", TASK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S

q = lambda p, l: p - l / 0.03
P = lambda a, t: dict(st=a, scm=t, rho=1.0, mu=0.0, prior=True)
rows = {}


def report(label, st, calls, seeds=3, cal=None):
    p, l, tab = cal if cal else S.cal_eval(st, "shaped", True)
    tp, tl = S.test_eval(st, "shaped", tab, seeds=seeds)
    c = st[-1]
    d = f"sigma {c['st']:.3f} told {c['scm']:.3f}" if c.get("prior") else "-"
    rows[label] = dict(cal=(p, l), test=(tp, tl), last=c)
    print(f"{label:16s} {calls:5d} | {d:24s} cal {p:.2f}/{l:.3f} | {tp:.2f} / {tl:.3f}", flush=True)


print(f"\n######## {TASK}: last step(s) without a measurement update (test: 10 images)")
print("variant          calls | last step                                  | test PSNR / LPIPS")
if S.LIN:
    base = [dict(S.SHIP[0]), dict(S.SHIP[1])]
    report("none", base, 2)
    report("shipped3", base + [dict(S.SHIP[2])], 3)
    grid = [(a, max(a * f, c)) for a in (0.0, 0.01, 0.03, 0.06, 0.1, 0.15) for f, c in ((1.0, 0.005), (1.5, 0.01), (3.0, 0.03))]
    for name, n in (("prior1", 1), ("prior2", 2)):
        ev = []
        for a, t in grid:
            st = base + [P(a, t)] * n
            ev.append((S.cal_eval(st, "shaped", True), st))
        for how, pick in (("balanced", max(ev, key=lambda e: q(e[0][0], e[0][1]))), ("psnr", max(ev, key=lambda e: e[0][0])),
                          ("lpips", min(ev, key=lambda e: e[0][1]))):
            report(f"{name} ({how})", pick[1], 2 + n, cal=pick[0])
else:
    sh = [dict(s) for s in S.SHIP]
    report("shipped 4", sh, 4, seeds=2)
    report("drop4", sh[:3] + [dict(sh[3], prior=True)], 4, seeds=2)
    report("drop34", sh[:2] + [dict(sh[2], prior=True), dict(sh[3], prior=True)], 4, seeds=2)
    for a, t in ((0.03, 0.045), (0.1, 0.15)):
        report(f"extra s={a}", sh + [P(a, t)], 5, seeds=2)
json.dump(rows, open(os.path.abspath(f"../mode_probe/random_steps/steps/{TASK}_prior_only.json"), "w"))
