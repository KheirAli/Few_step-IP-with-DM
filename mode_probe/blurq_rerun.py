"""Rerun of the two blur-then-quantise variants hit by out-of-memory shards, one at a time on the GPUs with free memory (run from PnP-CM/)."""
import subprocess, os
L = "../mode_probe"; src = open(f"{L}/blurq_exact.py").read(); exec(src[src.index("import json"):src.index("ps = [launch(k, ex")])
N = ["posterior mean, 8 alternations", "posterior mean (T x0.3), 8 alternations"]; R = {}
for k, n in enumerate(N):
    launch(f"r{k}", V[n], f"{L}/data_gen512", "128:152 152:176", 0).wait(); R[n] = sc(f"bqr{k}test_zband8_blurq_deblur_nl"); print(f"BQR generated images: {n}: {R[n][0]:.2f}/{R[n][1]:.3f} (n={R[n][2]}) | current 27.29/0.342", flush=True)
subprocess.run(f"rm -rf {L}/exp_fin_bqr*test_*", shell=True); b = max(R, key=lambda n: R[n][0] if R[n][2] == 48 else -1); launch("rT", V[b], f"{L}/data_paper", "0:54 54:107", 0).wait(); r = sc("bqrTtest_zband8_blurq_deblur_nl")
print(f"BQR TEST {b}: {r[0]:.2f}/{r[1]:.3f} (n={r[2]}) | current 26.62/0.351", flush=True)
