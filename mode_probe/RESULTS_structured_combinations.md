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
