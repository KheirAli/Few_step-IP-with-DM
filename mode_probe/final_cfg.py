"""The FINAL 'Ours' configuration of every CelebA-HQ table row: FINAL[row] = (deg, env string, combination-weight file, steps, key).  exec() this file from PnP-CM/."""
import json, os, numpy as np
L = "../mode_probe"; _ch = json.load(open(f"{L}/clean_protocol_choices.json")); _chm = json.load(open(f"{L}/clean_protocol_choicesm.json"))
_src = open(f"{L}/clean_protocol.py").read(); exec(_src[_src.index("LIN = ["):_src.index("PREV = {")])
sg = lambda s, t: f"SIGS={','.join(f'{v:.4f}' for v in s)} SIGPOS=0,1,2,3 SIGTOLD={t}"; gs = lambda a, n: [float(v) for v in np.geomspace(a, 0.03, n)]
FINAL = {}
for t in ("sr_bicubic", "deblur_gauss", "inpainting", "deblur_nl", "hdr", "quant", "onebit", "gamma", "blurclip", "ct30", "ct60"):
    c = _ch[t]; deg, ex = [(d, e) for n, d, e, b in CAND[t] if n == c["variant"]][0]; FINAL[t] = (deg, f"{ex} {sg(c['sig'], c['told'])}", f"cln{t}_zband8_{t}.pt", 4, t)
c = _ch["jpeg"]; FINAL["jpeg"] = ("jpeg", f"JPROJ=1 JCOLD=1 JEV=use JEVS=0.35 JEVFILE={os.path.realpath(L + '/jpeg_ev_table.json')} JFINAL=1 {sg(c['sig'], c['told'])}", "jz_zband8_jpeg.pt", 4, "jpeg")
c = _ch["blurq"]; FINAL["blurq"] = ("deblur_nl", f"NEWOP=blurq SPLIT_EV=0.3 {sg(c['sig'], c['told'])}", "bqrT_zband8_blurq.pt", 4, "blurq")
for R in (4, 8): c = _chm[f"mri{R}"]; FINAL[f"mri{R}"] = ("deblur_nl", f"NEWOP=mri MRI_R={R} RHOX=1 {sg(c['sig'], c['told'])}", f"clnmmri{R}_zband8_mri{R}.pt", 4, f"mri{R}")
for K in (30, 60): FINAL[f"ctnl{K}"] = ("deblur_nl", f"NEWOP=ctnl CT_K={K} RHOX=0.03 CTNL_N=1 CTNL_MU=1 SIGS=0.9000,0.4500,0.5940,0.1650 SIGPOS=0,1,2,3 SIGTOLD=1.5", f"ctnlO{K}_zband8_ctnl{K}.pt", 4, f"ctnl{K}")
_x = json.load(open(f"{L}/sched_cfg_extra.json"))["prmask__p2_0.1_fw"]; _r = json.load(open(f"{L}/sched_cd2_prmask__p2_0.1_fw.json"))["4"]
FINAL["prmask"] = ("phase_retrieval", f"{_x[1]} {sg(_r['sig'], _r['told'])}", "p2fw_zband8_prmask.pt", 4, "prmask")
FINAL["cdp2"] = ("deblur_nl", "NEWOP=prcdp PR_MASKS=2 RHOX=0.3 PR_ISO_WARM=50 SIGS=0.1500,0.1375,0.1800,0.0900 SIGPOS=0,1,2,3 SIGTOLD=1.5", "mfC_zband8_cdp2.pt", 4, "cdp2")
_B = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=80"; _NET = f"{_B} PRD_INIT=netfit PRD_NET=v2k8 PRD_NETREG=0 PRD_NETR=48"; _FREE = f"{_B} PRD_INIT=regfit4 PRD_NSTART=16"
_ND = "NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_NET=v2k8 PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32"
_p = json.load(open(f"{L}/sched_cd2_pru_t05.json"))["4"]; _S4 = f"{sg(_p['sig'], _p['told'])} TILT=0.5"; _S8 = f"SIGS={','.join(f'{v:.4f}' for v in gs(1.5, 8))} TILT=0.5"
FINAL["pr_unmasked_4"] = ("deblur_nl", f"{_NET} {_S4}", "prtN4_zband8_pruN4.pt", 4, "pruN4"); FINAL["pr_unmasked_8"] = ("deblur_nl", f"{_NET} {_S8}", "prtN8_zband8_pruN8.pt", 8, "pruN8")
FINAL["pr_unmasked_free_4"] = ("deblur_nl", f"{_FREE} {_S4}", "prtF4_zband8_pruF4.pt", 4, "pruF4")
_d = json.load(open(f"{L}/sched_cd2_prd_t05.json"))["4"]; FINAL["pr_unmasked_noniter_4"] = ("deblur_nl", f"{_ND} TILT=0.5 {sg(_d['sig'], _d['told'])}", "a2D4_zband8_prdD4.pt", 4, "prdD4")
