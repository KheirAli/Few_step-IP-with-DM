# Structured combinations of the step outputs: full run on server 2 (2026-11-21)

512 model-generated calibration images, 107 test images, 4 calls, 150 fits, no failures. PSNR / LPIPS; brackets = PSNR change against scalar weights.
Base configuration per task as in `launch_llew.sh` (no frequency-weighted data-fit). Masked phase retrieval is scored as produced.
## Job 3 (server 2 run): frequency-band, gated and data-fit-term combinations, 107 test images, 4 calls, calibrated on generated images
| Task | Scalar (9 numbers) | 8 frequency bands | gated by step-to-step change | scalar + data-fit term | 8 bands + data-fit term |
|---|---|---|---|---|---|
| sr_bicubic | 27.96/0.285 | 28.08/0.302 (+0.12) | 27.98/0.287 (+0.02) | 27.92/0.294 (-0.04) | 28.12/0.303 (+0.16) |
| deblur_gauss | 29.50/0.249 | 29.67/0.255 (+0.17) | 29.53/0.248 (+0.03) | 29.51/0.250 (+0.01) | 29.67/0.253 (+0.17) |
| inpainting | 29.52/0.213 | 29.96/0.205 (+0.44) | 29.53/0.214 (+0.00) | 29.55/0.211 (+0.03) | 29.96/0.205 (+0.44) |
| jpeg | 26.35/0.363 | 26.57/0.345 (+0.23) | 26.39/0.360 (+0.05) | 26.45/0.348 (+0.10) | 26.63/0.345 (+0.28) |
| jpegtrue | 26.62/0.359 | 26.66/0.359 (+0.04) | 26.65/0.359 (+0.03) | 26.63/0.357 (+0.01) | 26.68/0.355 (+0.06) |
| deblur_nl | 26.01/0.365 | 26.18/0.345 (+0.17) | 26.03/0.365 (+0.02) | 26.00/0.365 (-0.01) | 26.21/0.342 (+0.20) |
| hdr | 25.17/0.302 | 25.59/0.291 (+0.41) | 25.63/0.304 (+0.46) | 25.66/0.289 (+0.48) | 25.93/0.272 (+0.75) |
| quant | 30.55/0.229 | 30.70/0.228 (+0.15) | 30.59/0.231 (+0.04) | 30.63/0.225 (+0.08) | 30.83/0.216 (+0.29) |
| onebit | 28.45/0.271 | 29.45/0.241 (+1.00) | 28.47/0.269 (+0.02) | 28.54/0.270 (+0.09) | 29.49/0.241 (+1.04) |
| prmask | 23.65/0.376 | 23.79/0.324 (+0.14) | 23.66/0.390 (+0.01) | 23.71/0.341 (+0.06) | 23.89/0.309 (+0.24) |

```
LLEFIT2 j3cal3_scalar_sr_bicubic_sr_bicubic step 4 mode scalar n=512: scalar solution +0.027 -0.053 -0.250 +1.281 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_sr_bicubic_sr_bicubic step 4 mode band8 n=512: overall +0.008 -0.134 +0.420 +0.707 | last term per bin +0.71 +0.71 +0.71 +0.71 +0.72 +0.72 +0.64 +0.63
LLEFIT2 j3cal3_gate_sr_bicubic_sr_bicubic step 4 mode gate n=512: overall +0.026 -0.034 -0.248 +1.260 | last term per bin +1.26 +1.26 +1.27 +1.26 +1.26 +1.26
LLEFIT2 j3cal3_zscalar_sr_bicubic_sr_bicubic step 4 mode zscalar n=512: overall +0.212 -1.315 +2.125 -1.022 +1.019 | last term per bin +1.02 | data-fit term per bin -1.02
LLEFIT2 j3cal3_zband8_sr_bicubic_sr_bicubic step 4 mode zband8 n=512: overall +0.124 -0.893 +1.864 -0.624 +0.540 | last term per bin +0.54 +0.54 +0.54 +0.54 +0.53 +0.55 +0.53 +0.52 | data-fit term per bin -0.62 -0.62 -0.62 -0.62 -0.63 -0.63 -0.64 -0.66
LLEFIT2 j3cal3_scalar_deblur_gauss_deblur_gauss step 4 mode scalar n=512: scalar solution -0.008 +0.044 +0.110 +0.855 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_deblur_gauss_deblur_gauss step 4 mode band8 n=512: overall -0.009 +0.004 +0.544 +0.461 | last term per bin +0.46 +0.46 +0.46 +0.46 +0.46 +0.47 +0.50 +0.45
LLEFIT2 j3cal3_gate_deblur_gauss_deblur_gauss step 4 mode gate n=512: overall -0.006 +0.041 +0.110 +0.855 | last term per bin +0.85 +0.86 +0.88 +0.85 +0.85 +0.85
LLEFIT2 j3cal3_zscalar_deblur_gauss_deblur_gauss step 4 mode zscalar n=512: overall -0.048 +0.298 -0.512 +0.410 +0.853 | last term per bin +0.85 | data-fit term per bin +0.41
LLEFIT2 j3cal3_zband8_deblur_gauss_deblur_gauss step 4 mode zband8 n=512: overall -0.021 +0.058 +0.393 +0.108 +0.463 | last term per bin +0.46 +0.46 +0.46 +0.46 +0.46 +0.47 +0.50 +0.47 | data-fit term per bin +0.11 +0.11 +0.11 +0.11 +0.11 +0.11 +0.09 +0.17
LLEFIT2 j3cal3_scalar_inpainting_inpainting step 4 mode scalar n=512: scalar solution +0.010 -0.146 -0.244 +1.380 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_inpainting_inpainting step 4 mode band8 n=512: overall -0.006 -0.374 +0.248 +1.133 | last term per bin +1.13 +1.13 +1.13 +1.14 +1.15 +1.17 +1.13 +0.94
LLEFIT2 j3cal3_gate_inpainting_inpainting step 4 mode gate n=512: overall +0.009 -0.146 -0.238 +1.375 | last term per bin +1.37 +1.37 +1.37 +1.37 +1.37 +1.37
LLEFIT2 j3cal3_zscalar_inpainting_inpainting step 4 mode zscalar n=512: overall +0.002 -0.124 -0.164 -0.018 +1.304 | last term per bin +1.30 | data-fit term per bin -0.02
LLEFIT2 j3cal3_zband8_inpainting_inpainting step 4 mode zband8 n=512: overall -0.011 -0.247 +0.164 +0.002 +1.093 | last term per bin +1.09 +1.09 +1.09 +1.10 +1.11 +1.12 +1.11 +0.93 | data-fit term per bin +0.00 +0.00 +0.00 +0.01 +0.01 +0.03 -0.02 -0.08
LLEFIT2 j3cal3_scalar_jpeg_jpeg step 4 mode scalar n=512: scalar solution -0.194 -0.246 -0.161 +1.575 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_jpeg_jpeg step 4 mode band8 n=512: overall +0.038 -0.911 +0.191 +1.690 | last term per bin +1.71 +1.69 +1.69 +1.67 +1.68 +1.69 +1.47 +1.33
LLEFIT2 j3cal3_gate_jpeg_jpeg step 4 mode gate n=512: overall -0.190 -0.237 -0.170 +1.571 | last term per bin +1.57 +1.57 +1.57 +1.59 +1.93 +2.18
LLEFIT2 j3cal3_zscalar_jpeg_jpeg step 4 mode zscalar n=512: overall -0.043 -0.884 +1.431 -0.774 +1.275 | last term per bin +1.28 | data-fit term per bin -0.77
LLEFIT2 j3cal3_zband8_jpeg_jpeg step 4 mode zband8 n=512: overall -0.021 -1.015 +1.346 -0.719 +1.417 | last term per bin +1.43 +1.42 +1.41 +1.40 +1.40 +1.42 +1.28 +1.23 | data-fit term per bin -0.70 -0.72 -0.73 -0.74 -0.76 -0.79 -0.63 -0.51
LLEFIT2 j3cal3_scalar_jpegtrue_jpeg step 4 mode scalar n=512: scalar solution -0.126 -0.061 -0.065 +1.246 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_jpegtrue_jpeg step 4 mode band8 n=512: overall -0.041 -0.391 +0.160 +1.270 | last term per bin +1.27 +1.27 +1.27 +1.26 +1.27 +1.31 +1.23 +1.22
LLEFIT2 j3cal3_gate_jpegtrue_jpeg step 4 mode gate n=512: overall -0.122 -0.060 -0.067 +1.244 | last term per bin +1.24 +1.24 +1.24 +1.27 +1.41 +1.24
LLEFIT2 j3cal3_zscalar_jpegtrue_jpeg step 4 mode zscalar n=512: overall -0.128 -0.179 +0.166 -0.082 +1.216 | last term per bin +1.22 | data-fit term per bin -0.08
LLEFIT2 j3cal3_zband8_jpegtrue_jpeg step 4 mode zband8 n=512: overall -0.065 -0.372 +0.302 -0.074 +1.207 | last term per bin +1.21 +1.21 +1.20 +1.19 +1.21 +1.26 +1.21 +1.18 | data-fit term per bin -0.08 -0.07 -0.07 -0.05 -0.08 -0.13 -0.07 -0.03
LLEFIT2 j3cal3_scalar_deblur_nl_deblur_nl step 4 mode scalar n=512: scalar solution -0.061 -0.030 +0.471 +0.616 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_deblur_nl_deblur_nl step 4 mode band8 n=512: overall +0.015 -0.259 +0.858 +0.376 | last term per bin +0.38 +0.38 +0.38 +0.39 +0.49 +0.44 +0.32 +0.21
LLEFIT2 j3cal3_gate_deblur_nl_deblur_nl step 4 mode gate n=512: overall -0.055 -0.016 +0.451 +0.615 | last term per bin +0.61 +0.61 +0.61 +0.63 +0.68 +0.58
LLEFIT2 j3cal3_zscalar_deblur_nl_deblur_nl step 4 mode zscalar n=512: overall -0.049 -0.062 +0.571 -0.097 +0.634 | last term per bin +0.63 | data-fit term per bin -0.10
LLEFIT2 j3cal3_zband8_deblur_nl_deblur_nl step 4 mode zband8 n=512: overall +0.002 -0.291 +0.996 -0.106 +0.392 | last term per bin +0.39 +0.39 +0.39 +0.41 +0.52 +0.49 +0.35 +0.21 | data-fit term per bin -0.11 -0.11 -0.11 -0.11 -0.12 -0.18 -0.16 -0.12
LLEFIT2 j3cal3_scalar_hdr_deblur_nl step 4 mode scalar n=512: scalar solution +0.058 +0.088 +0.530 +0.341 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_hdr_deblur_nl step 4 mode band8 n=512: overall +0.049 -0.101 +0.824 +0.238 | last term per bin +0.27 +0.26 +0.25 +0.25 +0.26 +0.27 +0.23 +0.15
LLEFIT2 j3cal3_gate_hdr_deblur_nl step 4 mode gate n=512: overall +0.059 +0.010 +0.641 +0.301 | last term per bin +0.32 +0.33 +0.33 +0.33 +0.28 +0.22
LLEFIT2 j3cal3_zscalar_hdr_deblur_nl step 4 mode zscalar n=512: overall +0.026 +0.113 +0.401 +0.177 +0.283 | last term per bin +0.28 | data-fit term per bin +0.18
LLEFIT2 j3cal3_zband8_hdr_deblur_nl step 4 mode zband8 n=512: overall +0.002 +0.016 +0.606 +0.189 +0.182 | last term per bin +0.20 +0.19 +0.18 +0.17 +0.18 +0.20 +0.20 +0.13 | data-fit term per bin +0.20 +0.20 +0.20 +0.20 +0.20 +0.18 +0.11 +0.11
LLEFIT2 j3cal3_scalar_quant_deblur_nl step 4 mode scalar n=512: scalar solution +0.002 +0.012 +0.644 +0.335 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_quant_deblur_nl step 4 mode band8 n=512: overall +0.007 -0.135 +0.855 +0.265 | last term per bin +0.26 +0.27 +0.27 +0.27 +0.28 +0.28 +0.28 +0.23
LLEFIT2 j3cal3_gate_quant_deblur_nl step 4 mode gate n=512: overall +0.001 +0.009 +0.646 +0.335 | last term per bin +0.33 +0.33 +0.34 +0.34 +0.33 +0.31
LLEFIT2 j3cal3_zscalar_quant_deblur_nl step 4 mode zscalar n=512: overall +0.026 -0.047 +0.844 -0.166 +0.333 | last term per bin +0.33 | data-fit term per bin -0.17
LLEFIT2 j3cal3_zband8_quant_deblur_nl step 4 mode zband8 n=512: overall +0.023 -0.166 +1.046 -0.172 +0.262 | last term per bin +0.26 +0.26 +0.27 +0.27 +0.29 +0.29 +0.28 +0.22 | data-fit term per bin -0.17 -0.17 -0.17 -0.19 -0.23 -0.26 -0.15 -0.11
LLEFIT2 j3cal3_scalar_onebit_deblur_nl step 4 mode scalar n=512: scalar solution +0.000 +0.048 +0.372 +0.567 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_onebit_deblur_nl step 4 mode band8 n=512: overall -0.010 -0.047 +0.714 +0.333 | last term per bin +0.33 +0.33 +0.34 +0.38 +0.46 +0.51 +0.47 +0.15
LLEFIT2 j3cal3_gate_onebit_deblur_nl step 4 mode gate n=512: overall +0.001 +0.038 +0.380 +0.568 | last term per bin +0.56 +0.57 +0.57 +0.58 +0.56 +0.47
LLEFIT2 j3cal3_zscalar_onebit_deblur_nl step 4 mode zscalar n=512: overall +0.009 +0.111 +0.229 +0.095 +0.545 | last term per bin +0.55 | data-fit term per bin +0.09
LLEFIT2 j3cal3_zband8_onebit_deblur_nl step 4 mode zband8 n=512: overall +0.004 +0.001 +0.593 +0.068 +0.327 | last term per bin +0.32 +0.32 +0.33 +0.37 +0.44 +0.50 +0.45 +0.16 | data-fit term per bin +0.07 +0.07 +0.07 +0.08 +0.08 +0.05 +0.06 +0.03
LLEFIT2 j3cal3_scalar_prmask_phase_retrieval step 4 mode scalar n=512: scalar solution +0.058 -0.002 +0.532 +0.401 | spatial std of the last weight 0.000
LLEFIT2 j3cal3_band8_prmask_phase_retrieval step 4 mode band8 n=512: overall +0.136 -2.559 +4.270 -0.867 | last term per bin -0.89 -0.88 -0.88 -0.88 -0.84 -0.83 -0.33 -0.31
LLEFIT2 j3cal3_gate_prmask_phase_retrieval step 4 mode gate n=512: overall +0.036 +0.056 +0.528 +0.371 | last term per bin +0.37 +0.37 +0.42 +0.59 -0.31 +0.37
LLEFIT2 j3cal3_zscalar_prmask_phase_retrieval step 4 mode zscalar n=512: overall +0.036 +0.017 +1.142 -0.198 -0.002 | last term per bin -0.00 | data-fit term per bin -0.20
LLEFIT2 j3cal3_zband8_prmask_phase_retrieval step 4 mode zband8 n=512: overall +0.015 -0.846 +2.749 -0.126 -0.793 | last term per bin -0.80 -0.80 -0.80 -0.80 -0.79 -0.72 -0.24 -0.24 | data-fit term per bin -0.12 -0.12 -0.12 -0.12 -0.12 -0.15 -0.19 -0.20
```

# Best configuration per task + structured combinations (2026-11-21)

128 generated calibration images, 107 test images, 4 calls, 81 fits, no failures. Base = each task's best configuration (frequency-weighted data-fit where used; HDR first-step sigma 3.0).
The masked phase-retrieval row is NOT valid: the fit did not align the channel signs of the calibration targets (fixed afterwards, rerun pending).
| Task | Scalar (9 numbers) | 8 frequency bands | 8 bands + data-fit term |
|---|---|---|---|
| sr_bicubic | 27.96/0.285 | 28.08/0.301 (+0.12) | 28.12/0.303 (+0.16) |
| deblur_gauss | 29.50/0.249 | 29.67/0.255 (+0.17) | 29.67/0.253 (+0.17) |
| inpainting | 29.88/0.199 | 30.22/0.198 (+0.34) | 30.23/0.198 (+0.35) |
| jpeg | 26.78/0.352 | 26.94/0.341 (+0.16) | 27.02/0.338 (+0.24) |
| deblur_nl | 26.34/0.343 | 26.41/0.333 (+0.07) | 26.43/0.332 (+0.08) |
| hdr | 25.88/0.282 | 26.09/0.276 (+0.21) | 26.32/0.269 (+0.45) |
| quant | 30.83/0.218 | 30.99/0.218 (+0.16) | 31.05/0.209 (+0.22) |
| onebit | 29.44/0.233 | 30.23/0.214 (+0.79) | 30.30/0.213 (+0.86) |
| prmask | 23.99/0.335 | 24.10/0.337 (+0.10) | 24.17/0.332 (+0.18) |

# Noise-schedule tuning (2026-11-21), no extrapolation, PSNR on 24 real selection images

| Task | Current | Full search (112 runs) | Coordinate search (19 runs) | Best error-proportional rule (c) |
|---|---|---|---|---|
| HDR | 22.44 | 24.75 | 23.69 | 23.44 (c=3) |
| One-bit | 26.85 | 27.78 | 28.00 | 27.83 (c=1.5) |
| Quantisation | 28.31 | 28.96 | 28.77 | 28.70 (c=3) |
| Inpainting | 28.50 | 28.79 | 28.57 | 27.81 (c=1) |
| SR | 27.06 | 27.15 | 27.08 | 27.05 (c=1.5) |
| Gaussian deblur | 28.36 | 28.44 | 28.36 | 28.02 (c=1) |

Scripts: sched_search.py, sched_cd.py, sched_rule.py; schedules in sched_search_<task>.json / sched_cd_<task>.json. Masked phase retrieval with sign-aligned fit: scalar 27.48/0.300, band8 27.87/0.276, zband8 27.93/0.271.

# Schedule tuned WITH the combination in the objective, for 4 / 3 / 2 calls (2026-11-22)

sched_cd2.py (19 / 15 / 11 runs per task on generated images), then in-loop zband8 weights (128 generated images) and the 107 test images. PSNR / LPIPS.

| Task | Previous best, 4 calls | Tuned, 4 calls | Tuned, 3 calls | Tuned, 2 calls |
|---|---|---|---|---|
| hdr | 26.32/0.263 | 26.42/0.268 | 25.61/0.289 | 24.55/0.285 |
| quant | 31.46/0.177 | 31.31/0.182 | 31.14/0.185 | 30.94/0.199 |
| onebit | 30.30/0.213 | 30.30/0.213 | 29.76/0.220 | 29.18/0.249 |
| sr_bicubic | 28.12/0.303 | 28.28/0.297 | 28.18/0.305 | 28.11/0.308 |
| inpainting | 30.23/0.198 | 30.29/0.207 | 29.79/0.221 | 29.19/0.260 |
| deblur_gauss | 29.67/0.253 | 29.76/0.263 | 29.44/0.274 | 29.02/0.276 |

# CT (2026-11-23)

Linear CT: data-fit in closed form from a precomputed decomposition of the Radon operator (`ct_svd.py`, `CT_SVD=1`): 29.20/0.255 (30 views), 31.50/0.200 (60 views), within 0.002 dB of the conjugate-gradient version on every test image; 0.21 s per image.

Nonlinear CT, y = 1 - exp(-s) + noise 0.01 (`NEWOP=ctnl`), 107 test images, PSNR / LPIPS. Ours: penalty, schedule and combination weights all chosen on model-generated images (`ctnl_full.py`).

| Method | Calls | 30 views | 60 views |
|---|---|---|---|
| Logarithm, then linear closed form | 4 | 24.20/0.421 | 25.53/0.405 |
| PnP-CM, learning rate tuned | 4 | 23.93/0.355 | 25.24/0.326 |
| Ours, untuned | 4 | 25.99/0.356 | 26.93/0.351 |
| Ours, tuned, scalar extrapolation | 4 | 27.01/0.279 | 28.15/0.249 |
| Ours, tuned, 8 bands + data-fit term | 4 | 27.62/0.278 | 28.87/0.246 |
| DPS, CelebA-HQ prior | 1000 | 24.15/0.278 | 24.80/0.270 |
| DAPS, CelebA-HQ prior | ~1000 | 27.65/0.264 | 28.55/0.254 |

Statistical ideas (`three_ideas.py`, tuned on generated images): expected-value update on blur-then-clip 22.49/0.446 -> 23.62/0.433; on nonlinear CT no effect; eigenface regression start for HDR negative; Poisson CT (1000 photons, 30 views) Lambert-W update 25.33/0.363 vs logarithm of counts 24.77/0.372.
Hessian commutator (relative): linear CT 0.0000, nonlinear CT 0.017 (Gaussian loss), 0.056 (Poisson loss).

Nonlinear CT, number of alternations inside the data-fit (tuned config, zband8 refitted on generated images, 107 test, PSNR/LPIPS; `CTNL_N`, `CTNL_MU`):
| Data-fit | 30 views | 60 views |
|---|---|---|
| 1 pass (no alternation, coupling weight 2) | 26.94/0.322 | 28.45/0.279 |
| 4 passes | 27.58/0.280 | 28.88/0.246 |
| 8 passes | 27.62/0.278 | 28.87/0.246 |

# Round 3 (2026-11-23), `round3.py`; tuning on generated images, 107 test images, PSNR/LPIPS

Nonlinear CT with NO alternation in the data-fit (one per-ray solve + one linear formula per step; penalty 0.03, coupling weight 1, sigma 0.90,0.45,0.59,0.17, told x1.5):
scalar 26.66/0.295 (30 views), 27.93/0.251 (60 views); 8 bands + data-fit term **27.47/0.292**, **28.73/0.254**. (4-pass version 27.58/0.280, 28.88/0.246; DAPS 27.65/0.264, 28.55/0.254.)

Coded-diffraction phase retrieval, 2 random phase masks (`NEWOP=prcdp`), start = mean face + 50 magnitude-fit passes, penalty 0.3, sigma 0.15,0.14,0.18,0.09, told x1.5:
ours scalar 33.72/0.132, 8 bands + data-fit term **34.45/0.126**; PnP-CM gradient solver (learning rate chosen on generated images, its own schedule) 28.65/0.186.

Blur then clip, data-fit variants: 8-pass split 22.49/0.446, expected-value split 23.62/0.433, active set (predicted clipped set + quadratic solve) 21.30/0.472.
Unpadded Fourier magnitudes (square unitary transform, exact data-fit), 24 selection images: 13.33 dB (fails: the phase is not determined).

# LSUN bedroom (2026-11-23), `bedroom.py`; 58 test images, PSNR/LPIPS, 4 calls

Consistency model cd_bedroom256_lpips. Ours: every task's CelebA-HQ configuration UNCHANGED; only the combination weights refitted on 128 images generated by the bedroom model. PnP-CM: in-framework (exact linear step / its gradient solver), CelebA-HQ settings (none are published for bedroom). Masked phase retrieval scored with the sign fixed by the truth for all methods.

| Task | PnP-CM | Ours, scalar extrapolation | Ours, 8 bands + data-fit term |
|---|---|---|---|
| sr_bicubic | 23.89/0.375 | 24.59/0.360 | 24.72/0.367 |
| deblur_gauss | 25.83/0.308 | 26.14/0.298 | 26.44/0.291 |
| inpainting | 25.77/0.264 | 26.20/0.254 | 26.58/0.255 |
| jpeg | 23.80/0.403 | 25.04/0.366 | 25.12/0.358 |
| deblur_nl | 23.53/0.356 | 24.15/0.380 | 24.31/0.373 |
| hdr | 22.93/0.301 | 25.28/0.260 | 25.78/0.245 |
| quant | 25.71/0.283 | 29.90/0.210 | 29.98/0.207 |
| onebit | 16.05/0.653 | 27.78/0.270 | 28.50/0.234 |
| prmask | 21.97/0.453 | 24.50/0.405 | 24.74/0.386 |
| ct30 | n/a | 25.14/0.366 | 25.52/0.346 |
| ct60 | n/a | 28.09/0.242 | 28.42/0.226 |
| ctnl30 | 20.91/0.488 | 23.74/0.399 | 23.96/0.394 |
| ctnl60 | 22.21/0.438 | 25.04/0.347 | 25.37/0.335 |
| gamma | 25.35/0.234 | 30.26/0.205 | 30.67/0.191 |
| blurq | 22.47/0.474 | 23.06/0.466 | 23.53/0.428 |
| blurclip | 21.37/0.467 | 22.61/0.479 | 22.70/0.445 |

# Unmasked phase retrieval, CLEAN PROTOCOL (2026-11-24, `pr_clean.py`)

All choices made on 24 model-generated images (classical loop 80 passes with offset; first noise level 1.0 for 4 calls, 1.5 for 8 calls; predictor radius 48), then 107 test images x 3 measurement-noise draws, PSNR/LPIPS:
training-free 23.62/0.369 (4 calls), 24.38/0.336 (8 calls), 0.9% below 14 dB; learned phase predictor 24.07/0.356 (4), 24.78/0.325 (8), 0.0% below 14 dB.
(Previous numbers, with settings chosen on test images: 23.14/0.387, 24.07/0.344; 23.75, 24.48.)

# LSUN bedroom, complete table (2026-11-24); 58 test images, PSNR/LPIPS

Ours = CelebA-HQ settings unchanged, combination weights refitted on generated bedroom images, 4 calls; nonlinear CT = no alternation. PnP-CM (+ scalar extrapolation fitted on generated bedroom images), 4 calls. DPS / DAPS with google/ddpm-ema-bedroom-256, ~1000 calls. Masked phase retrieval: sign fixed by the truth for all.

| Task | PnP-CM | PnP-CM + extrapolation | DPS | DAPS | Ours |
|---|---|---|---|---|---|
| SR x4 | 23.89/0.375 | 24.35/0.368 | 21.22/0.429 | 22.07/0.542 | 24.72/0.367 |
| Gaussian deblur | 25.83/0.308 | 25.85/0.305 | 21.98/0.372 | 22.84/0.505 | 26.44/0.291 |
| Inpainting | 25.77/0.264 | 25.86/0.264 | 24.54/0.330 | 25.67/0.318 | 26.58/0.255 |
| JPEG | 23.80/0.403 | 24.61/0.384 | 20.73/0.451 | 26.23/0.313 | 25.12/0.358 |
| Nonlinear deblur | 23.53/0.356 | 23.65/0.366 | 21.45/0.411 | 25.18/0.328 | 24.31/0.373 |
| HDR | 22.93/0.301 | 25.26/0.236 | 15.05/0.521 | 25.49/0.260 | 25.78/0.245 |
| Quantisation | 25.71/0.283 | 29.67/0.209 | 25.01/0.338 | 25.13/0.372 | 29.98/0.207 |
| One-bit | 16.05/0.653 | 19.99/0.572 | 26.34/0.261 | 19.92/0.560 | 28.50/0.234 |
| Masked phase retrieval | 21.97/0.453 | 21.63/0.463 | 22.19/0.418 | 24.52/0.383 | 24.74/0.386 |
| Linear CT 60 views | - | - | 23.54/0.352 | 28.32/0.257 | 28.42/0.226 |
| Linear CT 30 views | - | - | 22.42/0.375 | - | 25.52/0.346 |
| Nonlinear CT 60 views | 22.21/0.438 | 22.44/0.438 | 19.53/0.464 | 25.05/0.354 | 25.46/0.319 |
| Nonlinear CT 30 views | 20.91/0.488 | 21.12/0.488 | 19.32/0.470 | 23.85/0.403 | 24.10/0.380 |
| Tone curve | 25.35/0.234 | 27.47/0.210 | 23.81/0.351 | 28.50/0.282 | 30.67/0.191 |
| Blur then quantise | 22.47/0.474 | 22.66/0.475 | 21.45/0.403 | 21.75/0.562 | 23.53/0.428 |
| Blur then clip | 21.37/0.467 | 21.56/0.486 | 12.28/0.614 | 24.50/0.373 | 22.70/0.445 |
| Unmasked phase retrieval | - | - | 9.98/0.740 | 20.06/0.457 | not run |

# Black-box pointwise data-fit (2026-11-24, `GENERIC=1`, `blackbox_pointwise_prox`)

No formula for the operator: it is evaluated once on 257 constant images, giving a per-pixel table; the data-fit is the exact grid minimiser. 107 test images, 4 calls, no combination step, PSNR/LPIPS, hand-derived formula vs black-box:
HDR 23.18/0.357 vs 23.21/0.356; quantisation 28.37/0.268 vs 28.41/0.266; one-bit (nearest-value version) 23.88/0.358 vs 23.84/0.358; tone curve 28.28/0.235 vs 28.28/0.235; wrap-around 9.47/0.707 vs 9.47/0.707 (fails either way).

# Ambiguity check for coordinate-wise measurement functions (2026-11-24, `ambiguity.py`)

Share of coordinates for which the plausible values (objective within 4 sigma_y^2 of the best) form 2+ separate pieces, when the current estimate is off by 0.5 / 0.15 of the half-range:
HDR 0.3% / 0.1%; quantisation 2.8% / 2.1%; one-bit 0.0% / 0.0%; tone curve 0.0% / 0.0%; exponential attenuation (CT ray) 0.0% / 0.0%; wrap-around 86.9% / 0.6%; magnitude with the sign lost 49.5% / 3.8%.
The two operators flagged (wrap-around, magnitude) are the two on which the method fails without a starting-point search.

# CLEAN PROTOCOL, CelebA-HQ (2026-11-24, `clean_protocol.py`)

Every per-task choice (data-fit variant, noise schedule, told-level factor) made on 48 model-generated images with one rule (one pass of coordinate search, combination in the score, best score adopted); in-loop weights (8 bands + data-fit term) fitted on 128 generated images; 107 test images once. PSNR/LPIPS.

| Task | Chosen on generated images | Clean protocol, test | Previous number |
|---|---|---|---|
| sr_bicubic | plain; sigma 0.26 0.44 0.17 0.09; told x1.0 | 28.28/0.297 | 28.28/0.297 |
| deblur_gauss | fw; sigma 0.33 0.24 0.09 0.17; told x1.0 | 29.98/0.253 | 29.76/0.263 |
| inpainting | fw; sigma 0.33 0.24 0.01 0.01; told x1.0 | 30.29/0.207 | 30.29/0.207 |
| jpeg | softfw; sigma 2.29 1.03 0.35 0.01; told x1.2 | 27.19/0.336 | 27.02/0.338 |
| deblur_nl | fw; sigma 2.29 0.35 0.31 0.25; told x1.2 | 26.35/0.333 | 26.43/0.332 |
| hdr | plain; sigma 1.25 2.10 0.31 0.14; told x1.5 | 26.42/0.268 | 26.42/0.268 |
| quant | fw; sigma 0.21 0.19 0.31 0.25; told x1.5 | 31.31/0.182 | 31.46/0.177 |
| onebit | evfw; sigma 0.69 0.64 0.56 0.46; told x1.0 | 30.30/0.213 | 30.30/0.213 |
| prmask | fw; sigma 0.33 0.24 0.03 0.03; told x1.2 | 28.20/0.273 | 27.93/0.271 |
| blurclip | split; sigma 1.25 2.10 0.17 0.14; told x1.0 | 24.67/0.387 | 23.62/0.433 (no combination) |
| blurq | split; sigma 0.21 2.10 0.31 0.25; told x1.0 | 26.62/0.351 | 25.52/0.413 (no combination) |
| gamma | plain; sigma 0.21 0.19 0.31 0.25; told x1.5 | 33.22/0.147 | 29.21/0.217 (no combination) |
| ct30 | r1; sigma 0.90 0.45 0.33 0.17; told x1.5 | 30.03/0.223 | 29.20/0.255 (no combination) |
| ct60 | r1; sigma 0.28 0.25 0.33 0.17; told x1.5 | 32.25/0.171 | 31.50/0.200 (no combination) |

# NON-ITERATIVE unmasked phase retrieval (2026-11-24, `pr_direct.py`, `PRD_INIT=netdirect PRD_TV=0`)

One pass of the phase predictor -> hypothesis with the best low-frequency magnitude match -> coarse image = inverse transform of (measured magnitude x predicted phase) -> 4 or 8 model calls with the per-coefficient magnitude fit. No fit, no multi-start, no classical loop. First noise level chosen on 24 generated images (0.7 / 1.0). 107 test images x 3 noise draws, PSNR/LPIPS:
4 calls 22.87/0.385, 8 calls 23.72/0.345; 0.0% below 14 dB (5th percentile 18.1 / 18.7 dB).
Iterative versions for comparison: training-free 23.62/0.369, 24.38/0.336 (0.9% failures); with the predictor inside the fit 24.07/0.356, 24.78/0.325 (0.0%). DAPS 26.70/0.308.
On generated images the non-iterative version matched the training-free iterative one (24.55 vs 24.37 at 4 calls); on real test images it is 0.7 dB lower.

# ATTRIBUTION ABLATION and generic data-fit in the full pipeline (2026-11-24, `attribution.py`)

Same pipeline in every arm (schedule, noise tilt, 4 calls, 8-band + data-fit combination refitted on 128 generated images for that arm); ONLY the data step differs. Gradient solver = PnP-CM's, learning rate (4 values) and iterations (100/400) chosen on generated images. 107 test images; PSNR/LPIPS, time per image (shared GPUs).

| Task | Ours (closed form) | Gradient solver, tuned | Generic table, minimiser | Generic table, mean |
|---|---|---|---|---|
| hdr | 26.32/0.269, 0.30s | 26.17/0.263, 0.47s (lr 0.005, 100 it) | 26.33/0.269, 0.21s | 25.57/0.289, 0.30s |
| quant | 30.84/0.216, 0.33s | 31.08/0.196, 1.08s (lr 0.02, 400 it) | 30.89/0.214, 1.05s | 30.79/0.203, 0.27s |
| onebit | 29.49/0.241, 0.22s | 28.90/0.255, 3.58s (lr 0.05, 400 it) | 25.89/0.340, 1.79s | 27.90/0.295, 0.72s |
| gamma | 32.04/0.173, 0.32s | 32.00/0.173, 1.27s (lr 0.005, 400 it) | 32.04/0.173, 0.30s | 31.18/0.175, 0.35s |
| blurclip | 24.33/0.400, 0.28s | 23.90/0.406, 4.18s (lr 0.005, 400 it) | - | - |
| ctnl30 | 27.47/0.292, 0.56s | 25.64/0.378, 3.85s (lr 0.02, 400 it) | - | - |

Nonlinear CT row: the gradient arm inherits our penalty (0.03), which suits our data step; tuned separately with its own penalty it reached 23.93 without the combination. Gradient-solver sensitivity on generated images (worst / best setting): HDR 17.1 / 25.6, one-bit 14.0 / 22.3, tone curve 25.3 / 29.3, blur then clip 21.2 / 25.7, quantisation 26.7 / 27.8 dB.

# Mean face rebuilt from generated images (2026-11-24, `meanface_rerun.py`)

`mean_face01.pt` was the mean of 100 REAL images (two of them test images). With `mean_face_gen.pt` (mean of 512 generated images, now the default), 107 test images:
masked phase retrieval (attempt 1, clean config) 28.38/0.270, channel signs right 98.1% (was 28.20/0.273); coded diffraction 34.45/0.126 (unchanged).

# Real MRI on a second dataset: M4Raw (2026-11-24, `m4raw_convert.py`, `m4raw_run.sh`)

Public brain k-space (Zenodo 8056074): 0.3 T, 4 coils, T2, repetitions averaged; 40 slices (5 central slices of 8 validation subjects), padded 256 -> 320, ESPIRiT coil maps.
PnP-CM's own real-MRI pipeline with its fastMRI BRAIN consistency model (trained at 3 T: a real domain shift), knee hyperparameters, 4 calls. PSNR ± std / SSIM:

| Undersampling | PnP-CM | PnP-CM + noise tilt 0.5 |
|---|---|---|
| R = 4 | 24.55 ± 5.74 / 0.730 | 24.69 ± 5.65 / 0.733 |
| R = 8 | 23.46 ± 4.98 / 0.703 | 23.76 ± 4.87 / 0.709 |

Only the noise tilt is tested here (the combination step is not wired into the real-MRI pipeline). Absolute values are low and spread widely across slices, as expected under the field-strength shift.

# Unmasked phase retrieval (iterative start) with the shared components (2026-11-24, `pr_today.py`)

Noise tilt 0.5, schedule tuned on generated images (4 calls: 1.80, 0.31, 0.17, 0.09), in-loop 8-band + data-fit combination fitted on 128 generated images. 107 test images x 3 noise draws, PSNR/LPIPS:

| Version | 4 calls | 8 calls | Below 14 dB | Before these components |
|---|---|---|---|---|
| Training-free start | 24.18/0.371 | 25.18/0.328 | 0.6% / 0.9% | 23.62/0.369, 24.38/0.336 |
| Predictor inside the fit | 24.63/0.362 | 25.69/0.317 | 0.0% | 24.07/0.356, 24.78/0.325 |

DAPS (about 1000 calls): 26.70/0.308.

# Attempt 2 results (2026-11-24, `attempt2.py`); choices on generated images; test set once

Non-iterative unmasked phase retrieval (predictor -> one inverse transform -> model calls) with noise tilt 0.5, tuned schedule (0.70, 0.44, 0.15, 0.02) and the 8-band + data-fit combination, 107 x 3 noise draws:
4 calls 23.63/0.377, 8 calls 25.01/0.336, 0.0% below 14 dB (without these components 22.87/0.385, 23.72/0.345). Iterative start with the same components: 24.18/0.371, 25.18/0.328 (training-free); 24.63/0.362, 25.69/0.317 (predictor inside the fit).

Masked phase retrieval inside the operator class (two-piece data step with the true operator, `PR2=1`, coupling 0.3, no frequency-weighted arm), 107 test images, as produced:
28.11/0.268 (2.8% of images below 14 dB, i.e. a wrong channel sign) | attempt 1 (reduction to inpainting, frequency-weighted, generated mean face) 28.38/0.270.

# Overnight queue, 24 November 2026 (`overnight.sh`); raw output

## 1_masked_pr_attempt2

P2 tuning on generated images: p2_0.1_plain: 28.82 -> 28.98 | p2_0.1_fw: 28.92 -> 29.26 | p2_0.3_plain: 28.83 -> 28.85 | p2_0.3_fw: 28.97 -> 29.06
P2 chosen: prmask__p2_0.1_fw
P2 TEST masked phase retrieval, attempt 2 (p2_0.1_fw), as produced: 28.57/0.270 (n=107) | attempt 1 (reduction to inpainting, generated mean face) 28.38/0.270

## 2_new_modes

| Task | 8 bands + last data-fit image (current) | 8 bands + all data-fit images | 8 bands x measurement region |
|---|---|---|---|
| hdr | 26.42/0.268 | 26.57/0.267 | 26.77/0.255 |
| inpainting | 30.29/0.207 | 30.31/0.206 | 30.32/0.208 |
| blurclip | 24.67/0.387 | 24.84/0.387 | 24.91/0.377 |
| sr_bicubic | 28.28/0.297 | 28.32/0.297 | - |
| quant | 31.31/0.182 | 31.34/0.180 | - |
| onebit | 30.30/0.213 | 30.31/0.211 | - |
| jpeg | 27.19/0.336 | 27.22/0.339 | - |
| deblur_gauss | 29.98/0.253 | 29.98/0.251 | - |
| gamma | 33.22/0.147 | 33.25/0.147 | - |

## 3_mri_celeba

CHOICE mri4: r1: 30.57->31.10 | r3: 30.60->30.77 || chosen r1, sigma 0.900 0.250 0.180 0.165, told x1.2
CHOICE mri8: r1: 27.36->28.12 | r3: 27.38->27.87 || chosen r1, sigma 1.650 0.825 0.330 0.165, told x1.2
| Task | Chosen on generated images | Clean protocol, test | Previous number |
|---|---|---|---|
| mri4 | r1; sigma 0.90 0.25 0.18 0.17; told x1.2 | 31.42/0.189 | - |
| mri8 | r1; sigma 1.65 0.82 0.33 0.17; told x1.2 | 27.89/0.293 | - |

## 4_bedroom_clean

CHOICE sr_bicubic: plain: 27.51->27.63 | fw: 27.62->27.67 || chosen fw, sigma 0.465 0.245 0.165 0.165, told x1.5
CHOICE deblur_gauss: plain: 29.37->29.39 | fw: 29.54->29.59 || chosen fw, sigma 0.327 0.238 0.010 0.165, told x1.5
CHOICE inpainting: plain: 29.48->29.86 | fw: 29.82->30.23 || chosen fw, sigma 0.589 0.238 0.010 0.028, told x1.0
CHOICE jpeg: soft: 26.21->26.46 | softfw: 26.49->26.79 | true: 26.49->26.77 || chosen softfw, sigma 2.290 1.028 0.351 0.015, told x1.5
CHOICE deblur_nl: plain: 26.21->26.40 | fw: 26.57->26.69 || chosen fw, sigma 2.290 0.191 0.560 0.252, told x1.5
CHOICE hdr: plain: 26.87->28.18 | rx2: 26.89->27.64 || chosen plain, sigma 0.208 0.191 0.308 0.252, told x1.0
CHOICE quant: plain: 31.52->32.64 | fw: 31.77->32.53 || chosen plain, sigma 0.208 0.191 0.560 0.138, told x1.0
CHOICE onebit: nearest: 26.50->26.88 | ev: 29.76->30.07 | evfw: 30.44->30.61 || chosen evfw, sigma 0.382 1.143 0.560 0.459, told x1.0
CHOICE blurclip: split: 24.02->24.77 | ev: 25.15->25.45 || chosen ev, sigma 2.290 1.143 1.848 0.252, told x1.0
CHOICE blurq: split: 25.83->26.03 || chosen split, sigma 0.208 2.095 0.560 0.252, told x1.5
CHOICE gamma: plain: 32.73->33.90 || chosen plain, sigma 0.208 0.349 0.308 0.252, told x1.0
CHOICE ct30: r1: 27.58->29.18 | r3: 27.74->28.63 || chosen r1, sigma 0.900 0.825 0.330 0.165, told x1.0
CHOICE ct60: r1: 31.15->32.21 | r3: 31.45->31.58 || chosen r1, sigma 0.500 0.450 0.330 0.165, told x1.2
CHOICE mri4: r1: 30.55->31.23 | r3: 30.60->30.80 || chosen r1, sigma 0.900 0.450 0.330 0.165, told x1.2
CHOICE mri8: r1: 27.02->27.81 | r3: 27.05->27.62 || chosen r1, sigma 1.650 0.825 0.330 0.165, told x1.5
| Task | Chosen on generated images | Clean protocol, test | Previous number |
|---|---|---|---|
| sr_bicubic | fw; sigma 0.47 0.24 0.17 0.17; told x1.5 | 24.81/0.362 | - |
| deblur_gauss | fw; sigma 0.33 0.24 0.01 0.17; told x1.5 | 26.71/0.283 | - |
| inpainting | fw; sigma 0.59 0.24 0.01 0.03; told x1.0 | 27.08/0.244 | - |
| jpeg | softfw; sigma 2.29 1.03 0.35 0.01; told x1.5 | 25.21/0.355 | - |
| deblur_nl | fw; sigma 2.29 0.19 0.56 0.25; told x1.5 | 24.08/0.392 | - |
| hdr | plain; sigma 0.21 0.19 0.31 0.25; told x1.0 | 26.03/0.231 | - |
| quant | plain; sigma 0.21 0.19 0.56 0.14; told x1.0 | 31.17/0.166 | - |
| onebit | evfw; sigma 0.38 1.14 0.56 0.46; told x1.0 | 28.41/0.236 | - |
| blurclip | ev; sigma 2.29 1.14 1.85 0.25; told x1.0 | 22.60/0.442 | - |
| blurq | split; sigma 0.21 2.10 0.56 0.25; told x1.5 | 23.78/0.412 | - |
| gamma | plain; sigma 0.21 0.35 0.31 0.25; told x1.0 | 31.92/0.173 | - |
| ct30 | r1; sigma 0.90 0.82 0.33 0.17; told x1.0 | 26.34/0.298 | - |
| ct60 | r1; sigma 0.50 0.45 0.33 0.17; told x1.2 | 28.85/0.198 | - |
| mri4 | r1; sigma 0.90 0.45 0.33 0.17; told x1.2 | 28.60/0.218 | - |
| mri8 | r1; sigma 1.65 0.82 0.33 0.17; told x1.5 | 25.19/0.331 | - |

## 5_mri_pnp

MRIB PnP-CM, celeba, MRI-type sampling R=4: 28.44/0.252 (n=107)
MRIB PnP-CM, celeba, MRI-type sampling R=8: 26.13/0.302 (n=107)
MRIB PnP-CM, bedroom, MRI-type sampling R=4: 26.52/0.254 (n=58)
MRIB PnP-CM, bedroom, MRI-type sampling R=8: 24.05/0.334 (n=58)

## 6_timing

| Configuration | Seconds per image (median of 10) |
|---|---|
| sr_bicubic | 0.19 |
| deblur_gauss | 0.19 |
| inpainting | 0.19 |
| jpeg | 0.22 |
| deblur_nl | 0.27 |
| hdr | 0.19 |
| quant | 0.19 |
| onebit | 0.19 |
| prmask | 1.26 |
| blurclip | 0.20 |
| blurq | 0.19 |
| gamma | 0.20 |
| ct30 | 0.22 |
| ct60 | 0.26 |
| ctnl30 (no alternation) | 0.24 |
| ctnl60 (no alternation) | 0.30 |
| unmasked PR, non-iterative, 4 calls | 1.05 |
| unmasked PR, iterative training-free, 4 calls | 2.23 |
| PnP-CM gradient solver, HDR | 0.42 |
| PnP-CM gradient solver, JPEG | 3.52 |
| PnP-CM gradient solver, nonlinear deblur | 4.10 |
| PnP-CM, SR (exact linear step) | 0.18 |

# Spectrum of the injected noise: four versions (2026-11-24, `noise_arms.py`)

Each task's clean-protocol configuration (same data step, schedule, told level, 8-band combination refitted per arm on 128 generated images); 107 test images once.
White | tilt 0.5 (current) | matched = spectrum from the measured error per radial band so that error + noise is white (`NOISE=shape`, tables measured on
generated images) | fitted = one power gain per octave band (`NBG`), one round of coordinate search from the tilt's profile on 48 generated images.
Schedules were tuned with the tilt on (favours the tilt over white and matched).

| Task | White | Tilt 0.5 (current) | Matched to measured error | Fitted band gains |
|---|---|---|---|---|
| sr_bicubic | 27.69/0.306 | 28.28/0.297 | 28.30/0.292 | 28.34/0.300 |
| deblur_gauss | 29.47/0.242 | 29.98/0.253 | 29.87/0.244 | 30.11/0.264 |
| inpainting | 30.03/0.205 | 30.29/0.207 | 30.12/0.203 | 30.41/0.212 |
| jpeg | 26.63/0.335 | 27.19/0.336 | 26.94/0.327 | 27.25/0.342 |
| deblur_nl | 25.56/0.355 | 26.35/0.333 | 25.87/0.332 | 26.45/0.341 |
| quant | 30.76/0.188 | 31.31/0.182 | 31.03/0.181 | 31.43/0.189 |
| onebit | 29.28/0.237 | 30.30/0.213 | 29.77/0.212 | 30.51/0.231 |
| gamma | 32.63/0.156 | 33.22/0.147 | 32.67/0.155 | 33.45/0.147 |
| blurclip | 23.89/0.399 | 24.67/0.387 | 24.32/0.381 | 24.82/0.393 |
| blurq | 25.77/0.363 | 26.62/0.351 | 26.31/0.345 | 26.78/0.362 |
| ct30 | 28.34/0.290 | 30.03/0.223 | 29.12/0.240 | 30.37/0.238 |
| ct60 | 31.31/0.199 | 32.25/0.171 | 31.79/0.174 | 32.57/0.182 |
| hdr | 25.51/0.277 | 26.42/0.268 | 26.06/0.274 | 26.55/0.268 |

Fitted profiles are nearly identical across tasks: the tilt's profile with band 4 (or 5) lowered by x0.4. Literature and next test (operator-domain coloured renoising, FIRE arXiv 2501.17468; residual-based variance): see session notes.

# Coloured renoising in the operator's coordinates, after FIRE (2026-11-24, `fire_arms.py`, `NOISE=fire`)

Error of the model input tabulated per coordinate group on generated images (singular-value bins for linear operators, measurement-value bins for
pointwise operators); injected noise variance per group chosen so that error + noise is white. Same data step, schedule and band combination
(refitted per arm); 107 test images once. "Same power": injected power as in the schedule, told level unchanged. "Total = told": FIRE's rule.

| Task | White | Tilt 0.5 (current) | Operator-domain renoising, same injected power | Operator-domain renoising, total = told level |
|---|---|---|---|---|
| sr_bicubic | 27.69/0.306 | 28.28/0.297 | 28.23/0.290 | 28.16/0.283 |
| deblur_gauss | 29.47/0.242 | 29.98/0.253 | 29.72/0.239 | 29.59/0.241 |
| inpainting | 30.03/0.205 | 30.29/0.207 | 30.03/0.205 | 29.51/0.224 |
| ct30 | 28.34/0.290 | 30.03/0.223 | 28.56/0.276 | 28.45/0.283 |
| ct60 | 31.31/0.199 | 32.25/0.171 | 31.62/0.181 | 31.76/0.177 |
| hdr | 25.51/0.277 | 26.42/0.268 | 25.54/0.271 | 25.56/0.274 |
| quant | 30.76/0.188 | 31.31/0.182 | 30.76/0.188 | 31.04/0.183 |
| onebit | 29.28/0.237 | 30.30/0.213 | 29.28/0.237 | 29.62/0.222 |
| gamma | 32.63/0.156 | 33.22/0.147 | 32.84/0.148 | 34.24/0.118 |

Measured: on inpainting, quantisation and one-bit the input error is nearly uniform across the operator's coordinates (e.g. one-bit 0.32-0.35 rms per
group), so this noise is practically white there (results equal the white arm). It varies strongly only on deblur (0.06-0.39), HDR step 1 and CT.
Not supported as a replacement for the tilt: below the tilt in PSNR on all 9 tasks (same power). Exception to follow up: tone curve with total = told
level, 34.24/0.118 (+1.0 dB) - a level effect (less injected noise), not a colouring effect (same-power arm 32.84).
JPEG, nonlinear deblur, blur+clip, blur+quantise not run (composite coordinates).

# JPEG: exact data step (2026-11-24, `jpeg_exact.py`, `JEV`)

Check (16 generated images): re-encoding the noisy measurement recovers every coefficient's quantisation interval exactly - projecting the TRUE image
onto the recovered intervals changes it by 0.00000. So the data term is a hard constraint. New step (`JEV=use`): per-coefficient posterior mean = mean of
N(c, s_k^2) truncated to the interval, s_k tabulated per DCT position and step on generated images (x `JEVS`); one closed-form update per step, no
frequency weighting; final image projected into the intervals (`JFINAL=1`). Same schedule as the clean protocol. Variants on generated images (128:176):

JX belief std of luma coefficients (0..255 scale), step 2: DC 179.5, first AC 70.2, highest AC 4.5 | luma quantisation steps: DC 160, highest 255
JX generated images: soft_fw (current): 27.57/0.341 (n=48)
JX generated images: soft_fw + final projection: 27.58/0.341 (n=48)
JX generated images: hard_fw + final projection: 27.56/0.330 (n=48)
JX generated images: hard + final projection: 27.37/0.332 (n=48)
JX generated images: posterior mean: 27.48/0.332 (n=48)
JX generated images: posterior mean + final projection: 27.49/0.332 (n=48)
JX generated images: posterior mean x0.5 + final projection: 27.69/0.325 (n=48)
JX chosen on generated images: posterior mean x0.5 + final projection
JX TEST posterior mean x0.5 + final projection: 27.40/0.324 (n=107) | current 27.19/0.336 | DAPS 28.15/0.259 | PnP-CM 26.05/0.358
JY width factor on generated images: x0.15: 27.55/0.329 | x0.25: 27.66/0.325 | x0.35: 27.70/0.323 | x0.5: 27.69/0.325 | x0.7: 27.60/0.329
JY chosen width x0.35; schedule tuning on generated images: 27.38 -> 27.52, sigma 1.260 1.028 1.158 0.028 told x1.2
JY generated images, tuned schedule: 27.76/0.330 (old schedule 27.70/0.323)
JY TEST posterior mean x0.35, tuned schedule: 27.45/0.338 (n=107) | first exact version 27.40/0.324 | previous 27.19/0.336 | DAPS 28.15/0.259
JZ TEST posterior mean x0.35, unchanged schedule: 27.38/0.324 (n=107)

Test (107 images): previous soft+fw 27.19/0.336 | posterior mean x0.5 27.40/0.324 | x0.35 27.38/0.324 | x0.35 + retuned schedule 27.45/0.338.
Reading: the exact step gives about +0.2 dB and -0.012 LPIPS with ONE closed-form update per step (was 7); width factor is flat between 0.25 and 0.5;
schedule retuning (PSNR criterion) buys +0.05 dB and loses the LPIPS gain - not adopted. DAPS 28.15/0.259 still ahead.

# Blur then 3-bit quantisation: posterior mean and no alternation (2026-11-26, `blurq_exact.py`, `SPLIT_EV`, `SPLIT_N`)

Half a quantisation step (0.125) is close to the noise level (0.1): the level is NOT recoverable exactly (unlike JPEG). Tested: per-coordinate posterior
mean over the 8 levels in place of the nearest level, and one per-coordinate update + one linear solve in place of 8 alternations. Same schedule.
Two shards ran out of GPU memory (GPUs shared): the "T x0.3" variant has 24 of 48 images, and one step's weights of "posterior mean, 8 alternations" were fitted on 64 images.

BQ generated images: nearest level, 8 alternations (current): 27.29/0.342 (n=48)
BQ generated images: posterior mean, 8 alternations: 27.35/0.356 (n=48)
BQ generated images: posterior mean (T x0.3), 8 alternations: 27.52/0.340 (n=24)
BQ generated images: posterior mean, no alternation, coupling 2: 26.84/0.371 (n=48)
BQ generated images: posterior mean, no alternation, coupling 8: 26.85/0.360 (n=48)
BQ generated images: nearest level, no alternation, coupling 2: 26.77/0.361 (n=48)
BQ generated images: nearest level, no alternation, coupling 8: 26.80/0.359 (n=48)
BQ chosen on generated images: posterior mean, 8 alternations | best without alternation: posterior mean, no alternation, coupling 8
BQ TEST posterior mean, 8 alternations: 26.82/0.364 (n=107) | current 26.62/0.351 | DAPS 24.76/0.388 | DPS 24.63/0.276
BQ TEST posterior mean, no alternation, coupling 8: 26.25/0.369 (n=107) | current 26.62/0.351 | DAPS 24.76/0.388 | DPS 24.63/0.276

Rerun without memory failures (`blurq_rerun.py`):
BQR generated images: posterior mean, 8 alternations: 27.36/0.356 (n=48) | current 27.29/0.342
BQR generated images: posterior mean (T x0.3), 8 alternations: 27.48/0.340 (n=48) | current 27.29/0.342
BQR TEST posterior mean (T x0.3), 8 alternations: 26.90/0.349 (n=107) | current 26.62/0.351
Adopted candidate: posterior mean with belief factor 0.3 (SPLIT_EV=0.3), 8 alternations: 26.90/0.349 vs 26.62/0.351. Without alternation: 26.25/0.369 (not adopted).

# Real MRI (M4Raw T2, 40 slices) with our pipeline (2026-11-26, `mri_ours.py`, `mri_ours_run.sh`, `mri_exact2.sh`, `mri_exact3.sh`)

Inside PnP-CM's own MRI code (brain consistency model, shipped schedule, 4 calls). Ours = noise tilt 0.5 + in-loop 8-band combination with the data-fit
image as a term, weights fitted step by step on 80 GENERATED slices (2-step model samples, k-space simulated with the coil maps of 4 subjects that are
not in the test set). Exact data step (`MRIEXACT=1`): the mask depends on one axis only, so after an inverse FFT along the fully sampled axis the
problem splits into one W x W system per image row, solved directly (checked: residual 2e-7; 100 CG iterations converge to it within 5e-6; PnP-CM's
10 iterations are 15% away from the minimiser at the first step's penalty). Penalty factor chosen on generated slices.

| Undersampling | PnP-CM | PnP-CM + tilt | Ours, 10-iteration solver | Ours, exact row-wise step |
|---|---|---|---|---|
| R = 4 | 24.55/0.730 | 24.69/0.733 | 25.05/0.741 | 26.94/0.783 (penalty x3; generated: x1 34.84, x3 35.29, x10 34.01, x30 32.99) |
| R = 8 | 23.46/0.703 | 23.76/0.709 | 24.80/0.720 | 24.23/0.707 (penalty x1; generated: x1 31.43, x3 31.40, x10 30.60) |

Generated-slice PSNR of the 10-iteration version: 34.74 (R=4), 31.24 (R=8). Time per slice about 1.8 s for both data steps (GPUs shared).

# Filling the empty table cells (2026-11-26, `fill_cells.py`; memory-safe rerun, no failed batches)

PnP-CM = same loop, its own gradient solver (exact linear step for linear CT and MRI-type sampling), no tilt, shipped schedule. Scalar extrapolation weights fitted on generated images of the same dataset. Unmasked phase retrieval scored with the better of the two 180-degree flips.

| | Dataset | Task | Method | PSNR/LPIPS |
|---|---|---|---|---|
FILL | celeba | gamma | PnP-CM + scalar extrapolation | 28.83/0.187 |
FILL | celeba | blurq | PnP-CM + scalar extrapolation | 25.34/0.362 |
FILL | celeba | blurclip | PnP-CM + scalar extrapolation | 23.18/0.423 |
FILL | celeba | ct60 | PnP-CM + scalar extrapolation | 30.57/0.198 |
FILL | celeba | ct30 | PnP-CM + scalar extrapolation | 28.32/0.261 |
FILL | celeba | ctnl60 | PnP-CM + scalar extrapolation | 25.68/0.310 |
FILL | celeba | ctnl30 | PnP-CM + scalar extrapolation | 24.06/0.349 |
FILL | celeba | mri4 | PnP-CM + scalar extrapolation | 29.40/0.224 |
FILL | celeba | mri8 | PnP-CM + scalar extrapolation | 26.46/0.292 |
FILL | celeba | cdp2 | PnP-CM + scalar extrapolation | 31.36/0.176 |
FILL | celeba | pru | PnP-CM + scalar extrapolation | 13.77/0.758 |
FILL | celeba | ct60 | PnP-CM | 28.45/0.245 |
FILL | celeba | ct30 | PnP-CM | 27.45/0.278 |
FILL | celeba | pru | PnP-CM | 13.49/0.795 |
FILL | bedroom | ct60 | PnP-CM + scalar extrapolation | 27.29/0.257 |
FILL | bedroom | ct30 | PnP-CM + scalar extrapolation | 24.63/0.373 |
FILL | bedroom | mri4 | PnP-CM + scalar extrapolation | 27.01/0.249 |
FILL | bedroom | mri8 | PnP-CM + scalar extrapolation | 24.28/0.340 |
FILL | bedroom | cdp2 | PnP-CM + scalar extrapolation | 30.68/0.176 |
FILL | bedroom | pru | PnP-CM + scalar extrapolation | 14.02/0.729 |
FILL | bedroom | ct60 | PnP-CM | 26.57/0.261 |
FILL | bedroom | ct30 | PnP-CM | 24.45/0.362 |
FILL | bedroom | cdp2 | PnP-CM | 28.18/0.186 |
FILL | bedroom | pru | PnP-CM | 13.65/0.765 |
FILL | bedroom | jpeg | Ours | 25.35/0.359 |
FILL | bedroom | blurq | Ours | 23.90/0.410 |
FILL | bedroom | cdp2 | Ours | 34.92/0.102 |

# Real MRI: k-space interpolation map and regular sampling (2026-11-26/27, `mri_ours.py` MRIDATA=grappa|sense, `mri_maps.sh`)

grappa: PnP-CM's random-line pattern; missing lines predicted from the nearest acquired line on each side (all coils, 5 taps), weights fitted on the scan's own centre lines; then one division per pixel. sense: every R-th line, no centre block; exact R x R system per alias group (checked: operator share outside the groups <= 1e-7). Penalty factor chosen on generated slices among x1, x3, x10 (grappa best at the upper edge, sense at the lower edge).

generated slices, gaussian1d R=4, grappa step, penalty x1: PSNR 26.44
generated slices, gaussian1d R=4, grappa step, penalty x3: PSNR 31.38
generated slices, gaussian1d R=4, grappa step, penalty x10: PSNR 33.71
M4RAW T2, gaussian1d R=4, ours with the grappa data step, penalty x10: PSNR (mean ± std): 25.25 ± 1.50;SSIM (mean ± std): 0.748 ± 0.066; data steps 323.4 ms |  lines left empty 0.0 of 240 missing on average; interpolation alone PSNR 19.57 
generated slices, gaussian1d R=8, grappa step, penalty x1: PSNR 18.84
generated slices, gaussian1d R=8, grappa step, penalty x3: PSNR 22.66
generated slices, gaussian1d R=8, grappa step, penalty x10: PSNR 27.98
M4RAW T2, gaussian1d R=8, ours with the grappa data step, penalty x10: PSNR (mean ± std): 24.24 ± 1.41;SSIM (mean ± std): 0.683 ± 0.082; data steps 152.0 ms |  lines left empty 58.0 of 280 missing on average; interpolation alone PSNR 16.48 
M4RAW T2, regular1d R=4, PnP-CM (10 solver iterations, no tilt, no combination): PSNR (mean ± std): 21.98 ± 5.04;SSIM (mean ± std): 0.701 ± 0.119; data steps 38.2 ms
generated slices, regular1d R=4, sense step, penalty x1: PSNR 32.34
generated slices, regular1d R=4, sense step, penalty x3: PSNR 29.63
generated slices, regular1d R=4, sense step, penalty x10: PSNR 26.81
M4RAW T2, regular1d R=4, ours with the sense data step, penalty x1: PSNR (mean ± std): 22.87 ± 4.17;SSIM (mean ± std): 0.710 ± 0.104; data steps 37.3 ms | MRISENSE R=4: systems of size 4, 25600 per slice; share of the operator outside the alias groups 9.5e-08  
M4RAW T2, regular1d R=8, PnP-CM (10 solver iterations, no tilt, no combination): PSNR (mean ± std): 19.61 ± 3.90;SSIM (mean ± std): 0.648 ± 0.107; data steps 28.6 ms
generated slices, regular1d R=8, sense step, penalty x1: PSNR 24.33
generated slices, regular1d R=8, sense step, penalty x3: PSNR 23.33
generated slices, regular1d R=8, sense step, penalty x10: PSNR 22.23
M4RAW T2, regular1d R=8, ours with the sense data step, penalty x1: PSNR (mean ± std): 21.62 ± 2.55;SSIM (mean ± std): 0.668 ± 0.088; data steps 38.6 ms | MRISENSE R=8: systems of size 8, 12800 per slice; share of the operator outside the alias groups 0.0e+00  

[exited with code 0]

# Real MRI, selection and weights on NOISE-MATCHED generated slices (2026-11-27, `mri_noisy.sh`, MRINOISE=1)

Wider penalty sweeps on noise-free generated slices picked settings that failed on real scans at R=8 (interpolation x100: 23.37/0.674; regular x0.3: 18.29/0.612). Fix: complex Gaussian noise added to the simulated k-space at the level read from the outer k-space corners of the 4 calibration subjects (std 0.043 per sample, image max 1). All weights and penalty factors refitted on these 80 slices; 40 test slices once per data step. These replace the earlier real-MRI numbers of our method. PnP-CM: 24.55/0.730 (R=4), 23.46/0.703 (R=8) random lines; 21.98/0.701, 19.61/0.648 regular lines.

noisy generated slices: 80; noise std per k-space sample, mean over calibration slices: 0.0430
generated slices, gaussian1d R=4, cg step, penalty x1: PSNR 28.78
M4RAW T2, gaussian1d R=4, ours with the cg data step, penalty x1: PSNR (mean ± std): 29.26 ± 2.69;SSIM (mean ± std): 0.827 ± 0.074;
generated slices, gaussian1d R=4, exact step, penalty x1: PSNR 28.88
generated slices, gaussian1d R=4, exact step, penalty x3: PSNR 31.70
generated slices, gaussian1d R=4, exact step, penalty x10: PSNR 31.08
generated slices, gaussian1d R=4, exact step, penalty x30: PSNR 30.58
M4RAW T2, gaussian1d R=4, ours with the exact data step, penalty x3: PSNR (mean ± std): 29.55 ± 2.67;SSIM (mean ± std): 0.833 ± 0.068;
generated slices, gaussian1d R=4, grappa step, penalty x3: PSNR 18.81
generated slices, gaussian1d R=4, grappa step, penalty x10: PSNR 26.71
generated slices, gaussian1d R=4, grappa step, penalty x30: PSNR 28.97
generated slices, gaussian1d R=4, grappa step, penalty x100: PSNR 29.01
M4RAW T2, gaussian1d R=4, ours with the grappa data step, penalty x100: PSNR (mean ± std): 26.64 ± 2.33;SSIM (mean ± std): 0.715 ± 0.091;
generated slices, gaussian1d R=8, cg step, penalty x1: PSNR 26.79
M4RAW T2, gaussian1d R=8, ours with the cg data step, penalty x1: PSNR (mean ± std): 27.38 ± 1.93;SSIM (mean ± std): 0.775 ± 0.081;
generated slices, gaussian1d R=8, exact step, penalty x1: PSNR 26.91
generated slices, gaussian1d R=8, exact step, penalty x3: PSNR 29.43
generated slices, gaussian1d R=8, exact step, penalty x10: PSNR 28.90
generated slices, gaussian1d R=8, exact step, penalty x30: PSNR 28.40
M4RAW T2, gaussian1d R=8, ours with the exact data step, penalty x3: PSNR (mean ± std): 27.50 ± 1.87;SSIM (mean ± std): 0.778 ± 0.077;
generated slices, gaussian1d R=8, grappa step, penalty x3: PSNR 13.67
generated slices, gaussian1d R=8, grappa step, penalty x10: PSNR 21.49
generated slices, gaussian1d R=8, grappa step, penalty x30: PSNR 26.16
generated slices, gaussian1d R=8, grappa step, penalty x100: PSNR 26.46
M4RAW T2, gaussian1d R=8, ours with the grappa data step, penalty x100: PSNR (mean ± std): 24.46 ± 1.46;SSIM (mean ± std): 0.658 ± 0.087;
generated slices, regular1d R=4, sense step, penalty x0.3: PSNR 21.36
generated slices, regular1d R=4, sense step, penalty x1: PSNR 27.08
generated slices, regular1d R=4, sense step, penalty x3: PSNR 28.36
generated slices, regular1d R=4, sense step, penalty x10: PSNR 26.62
M4RAW T2, regular1d R=4, ours with the sense data step, penalty x3: PSNR (mean ± std): 26.96 ± 2.52;SSIM (mean ± std): 0.802 ± 0.075;
generated slices, regular1d R=8, sense step, penalty x0.3: PSNR 19.81
generated slices, regular1d R=8, sense step, penalty x1: PSNR 22.85
generated slices, regular1d R=8, sense step, penalty x3: PSNR 23.44
generated slices, regular1d R=8, sense step, penalty x10: PSNR 22.45
M4RAW T2, regular1d R=8, ours with the sense data step, penalty x3: PSNR (mean ± std): 23.88 ± 1.39;SSIM (mean ± std): 0.723 ± 0.069;

[exited with code 0]

# LSUN bedroom, unmasked phase retrieval (2026-11-27, agent run, scripts `bedpr_*.py`, files `*_bed.pt`)

Start stage rebuilt from generated bedroom images only (eigen-images, regression, offset, phase predictor trained on 65k model samples for 8k steps - a
shorter run than for faces). Schedules tuned on generated bedroom images; 58 test images x 3 noise draws, scored as produced (no flip allowance).

| Start | 4 calls | 8 calls | Below 14 dB |
|---|---|---|---|
| Non-iterative | 16.79/0.626 | 16.83/0.602 | 17.8% / 16.1% |
| Training-free iterative | 16.58/0.612 | 16.61/0.588 | 21.3% / 20.7% |
| Predictor inside the fit | 16.85/0.594 | 16.96/0.567 | 20.7% / 20.7% |

DAPS 20.06/0.457 (32.8% below 14 dB), DPS 9.98/0.740, PnP-CM 13.65/0.765, PnP-CM + scalar extrapolation 14.02/0.729.
The start image is the bottleneck: about 16.1 dB on test for all three starts; the bedroom predictor reaches 18.1 dB on held-out model samples (faces 22.9),
512 eigen-images reach 22.3 dB. `unified.py`: `_bed` files read when DSET=bedroom; optional start-stage cache `PRD_W0CACHE`.
