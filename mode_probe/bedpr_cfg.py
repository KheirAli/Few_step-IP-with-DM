"""Bedroom unmasked phase retrieval: the six configurations (exec() from PnP-CM/).  JOBS = [(tag, key, cfg string, steps, description)].
4-call schedules: bedpr_sched_{free,direct}.json (bedpr_sched.py).  8-call schedules: geometric from the first level chosen in bedpr_s8.json (bedpr_s8.py)."""
import json, os, numpy as np
L = "../mode_probe"; gs = lambda a, n: [float(v) for v in np.geomspace(a, 0.03, n)]; sg = lambda r: f"SIGS={','.join(f'{v:.4f}' for v in r['sig'])} SIGPOS=0,1,2,3 SIGTOLD={r['told']}"
BENV = {"DSET": "bedroom", "MEANFACE": "mean_bed_gen.pt"}
_B = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=80"; NET = f"{_B} PRD_INIT=netfit PRD_NET=v2k8 PRD_NETREG=0 PRD_NETR=48"; FREE = f"{_B} PRD_INIT=regfit4 PRD_NSTART=16"
ND = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32"
_f = json.load(open(f"{L}/bedpr_sched_free.json"))["4"] if os.path.exists(f"{L}/bedpr_sched_free.json") else None; _d = json.load(open(f"{L}/bedpr_sched_direct.json"))["4"] if os.path.exists(f"{L}/bedpr_sched_direct.json") else None
_s8 = json.load(open(f"{L}/bedpr_s8.json")) if os.path.exists(f"{L}/bedpr_s8.json") else {}; S8 = lambda k: f"SIGS={','.join(f'{v:.4f}' for v in gs(_s8[k], 8))} TILT=0.5"
JOBS = []
if _f: JOBS += [("bedprF4", "pruF4", f"deblur_nl {FREE} {sg(_f)} TILT=0.5", "4", "training-free iterative start"), ("bedprN4", "pruN4", f"deblur_nl {NET} {sg(_f)} TILT=0.5", "4", "predictor-in-the-fit iterative start")]
if _d: JOBS += [("bedprD4", "prdD4", f"deblur_nl {ND} TILT=0.5 {sg(_d)}", "4", "non-iterative start")]
if "free" in _s8: JOBS += [("bedprF8", "pruF8", f"deblur_nl {FREE} {S8('free')}", "8", "training-free iterative start"), ("bedprN8", "pruN8", f"deblur_nl {NET} {S8('free')}", "8", "predictor-in-the-fit iterative start")]
if "direct" in _s8: JOBS += [("bedprD8", "prdD8", f"deblur_nl {ND} {S8('direct')}", "8", "non-iterative start")]
# start-stage cache (unified.py PRD_W0CACHE): one folder per (image set, start variant); the non-iterative start is not cached
W0 = lambda tag_or_kind, split: {"PRD_W0CACHE": os.path.realpath(f"{L}/bedpr_w0") + f"/{split}_" + ("free" if ("F" in tag_or_kind or tag_or_kind == "free") else "net")} if not ("D" in tag_or_kind or tag_or_kind == "direct") else {}
