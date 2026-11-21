"""Candidate 8-step noise schedules for a task, built from its shipped 4-step schedule.  Prints 'name sigmas positions' lines.
'positions' place every new step on the shipped schedule's axis (0 = first shipped step ... 3 = last), so the tuned told-level
ratio and penalty weight can be interpolated instead of being replaced."""
import sys, numpy as np
SHIP = {"sr_bicubic": [0.465, 0.245, 0.01, 0.01], "deblur_gauss": [0.327, 0.132, 0.01, 0.01], "inpainting": [0.327, 0.132, 0.01, 0.01],
        "jpeg": [0.694, 0.571, 0.351, 0.01], "deblur_nl": [0.694, 0.635, 0.56, 0.459]}
s = SHIP[sys.argv[1]]; f = lambda a: ",".join("%.4f" % v for v in a)
def sig_at(p):                                                 # noise level at a fractional shipped position (geometric between steps)
    i = int(np.floor(min(p, 2.999))); t = p - i; return float(np.exp((1 - t) * np.log(s[i]) + t * np.log(s[i + 1])))
def out(name, pos): print(name, f([sig_at(p) for p in pos]), f(pos))
out("ship4", [0, 1, 2, 3])
out("twice", [0, 0, 1, 1, 2, 2, 3, 3])                         # every shipped step used twice
out("interleave", [0, 0.5, 1, 1.5, 2, 2.5, 3, 3])              # a midpoint inserted between shipped steps
out("even", list(np.linspace(0, 3, 8)))                        # 8 steps spread evenly along the shipped path
out("early", [0, 0.33, 0.67, 1, 1.5, 2, 2.5, 3])               # more steps in the high-noise part
out("late", [0, 0.5, 1, 1.5, 2, 2.33, 2.67, 3])                # more steps in the low-noise part
out("hold", [0, 0, 0, 1, 1, 2, 3, 3])                          # repeat the first (highest-noise) step three times
