"""Candidate 2-step schedules: two positions on the task's shipped 4-step path (0 = first shipped step ... 3 = last).
Noise level, told-level ratio and penalty are interpolated from the shipped values at those positions."""
import sys, numpy as np
SHIP = {"sr_bicubic": [0.465, 0.245, 0.01, 0.01], "deblur_gauss": [0.327, 0.132, 0.01, 0.01], "inpainting": [0.327, 0.132, 0.01, 0.01],
        "jpeg": [0.694, 0.571, 0.351, 0.01], "deblur_nl": [0.694, 0.635, 0.56, 0.459]}
s = SHIP[sys.argv[1]]; f = lambda a: ",".join("%.4f" % v for v in a)
def sig_at(p):
    i = int(np.floor(min(p, 2.999))); t = p - i; return float(np.exp((1 - t) * np.log(s[i]) + t * np.log(s[i + 1])))
for name, pos in [("p0_1", [0, 1]), ("p0_2", [0, 2]), ("p0_3", [0, 3]), ("p1_2", [1, 2]), ("p1_3", [1, 3]), ("p05_2", [0.5, 2]), ("p05_25", [0.5, 2.5]), ("p0_15", [0, 1.5])]:
    print(name, f([sig_at(p) for p in pos]), f(pos))
