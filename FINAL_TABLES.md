# Final tables (27 November 2026)

PSNR / LPIPS. Ours and PnP-CM: 4 model calls unless noted. DPS and DAPS: about 1000 calls, prior trained on the same dataset.
Every "Ours" row uses the same noise tilt (0.5) and the same combination (8 frequency bands, data-fit image as a term); only the data step differs.
All choices and weights come from model-generated images; test sets are used once. One run per method.

## CelebA-HQ (107 test images)

| Task | PnP-CM | PnP-CM + scalar extrapolation | DPS | DAPS | Ours |
|---|---|---|---|---|---|
| SR x4 | 27.24/0.287 | 27.45/0.284 | 24.34/0.287 | 25.47/0.414 | 28.28/0.297 |
| Gaussian deblur | 28.92/0.249 | 29.02/0.246 | 25.12/0.260 | 26.60/0.340 | 29.98/0.253 |
| Inpainting 70% | 29.06/0.204 | 29.31/0.204 | 27.19/0.249 | 27.56/0.299 | 30.29/0.207 |
| JPEG QF5 | 26.05/0.358 | 26.46/0.355 | 24.05/0.302 | 28.15/0.259 | 27.38/0.324 |
| Nonlinear deblur | 24.88/0.302 | 25.13/0.307 | 23.19/0.305 | 26.26/0.312 | 26.35/0.333 |
| HDR | 23.17/0.359 | 25.07/0.279 | 25.62/0.272 | 25.88/0.271 | 26.42/0.268 |
| 3-bit quantisation | 25.51/0.284 | 30.19/0.200 | 27.75/0.260 | 25.25/0.378 | 31.31/0.182 |
| One-bit | 19.03/0.516 | 23.16/0.447 | 28.82/0.217 | 22.38/0.512 | 30.30/0.213 |
| Tone curve | 27.25/0.217 | 28.83/0.187 | 27.50/0.239 | 29.62/0.245 | 33.22/0.147 |
| Blur then quantise | 24.51/0.365 | 25.34/0.362 | 24.63/0.276 | 24.76/0.388 | 26.90/0.349 |
| Blur then clip | 22.91/0.419 | 23.18/0.423 | 22.57/0.333 | 27.00/0.276 | 24.67/0.387 |
| Linear CT, 60 views | 28.45/0.245 | 30.57/0.198 | 28.75/0.218 | 30.59/0.227 | 32.25/0.171 |
| Linear CT, 30 views | 27.45/0.278 | 28.32/0.261 | 27.70/0.231 | 29.67/0.230 | 30.03/0.223 |
| Nonlinear CT, 60 views | 25.24/0.326 | 25.68/0.310 | 24.80/0.270 | 28.55/0.254 | 28.73/0.254 |
| Nonlinear CT, 30 views | 23.93/0.355 | 24.06/0.349 | 24.15/0.278 | 27.65/0.264 | 27.47/0.292 |
| MRI-type sampling, 4x | 28.44/0.252 | 29.40/0.224 | 27.00/0.249 | 30.04/0.251 | 31.42/0.189 |
| MRI-type sampling, 8x | 26.13/0.302 | 26.46/0.292 | 25.70/0.257 | 28.76/0.267 | 27.89/0.293 |
| Masked phase retrieval | 22.93/0.463 (8 calls) | 24.02/0.416 | 24.79/0.308 | 27.03/0.318 | 28.57/0.270 |
| Coded diffraction, 2 masks | 28.65/0.186 | 31.36/0.176 | 31.28/0.193 | 26.73/0.318 | 34.45/0.126 |
| Unmasked phase retrieval | 13.49/0.795 | 13.77/0.758 | 13.43/0.643 | 26.70/0.308 | 24.63/0.362 (4 calls), 25.69/0.317 (8 calls) |

## LSUN bedroom (58 test images)

| Task | PnP-CM | PnP-CM + scalar extrapolation | DPS | DAPS | Ours |
|---|---|---|---|---|---|
| SR x4 | 23.89/0.375 | 24.35/0.368 | 21.22/0.429 | 22.07/0.542 | 24.81/0.362 |
| Gaussian deblur | 25.83/0.308 | 25.85/0.305 | 21.98/0.372 | 22.84/0.505 | 26.71/0.283 |
| Inpainting 70% | 25.77/0.264 | 25.86/0.264 | 24.54/0.330 | 25.67/0.318 | 27.08/0.244 |
| JPEG QF5 | 23.80/0.403 | 24.61/0.384 | 20.73/0.451 | 26.23/0.313 | 25.35/0.359 |
| Nonlinear deblur | 23.53/0.356 | 23.65/0.366 | 21.45/0.411 | 25.18/0.328 | 24.08/0.392 |
| HDR | 22.93/0.301 | 25.26/0.236 | 15.05/0.521 | 25.49/0.260 | 26.03/0.231 |
| 3-bit quantisation | 25.71/0.283 | 29.67/0.209 | 25.01/0.338 | 25.13/0.372 | 31.17/0.166 |
| One-bit | 16.05/0.653 | 19.99/0.572 | 26.34/0.261 | 19.92/0.560 | 28.41/0.236 |
| Tone curve | 25.35/0.234 | 27.47/0.210 | 23.81/0.351 | 28.50/0.282 | 31.92/0.173 |
| Blur then quantise | 22.47/0.474 | 22.66/0.475 | 21.45/0.403 | 21.75/0.562 | 23.90/0.410 |
| Blur then clip | 21.37/0.467 | 21.56/0.486 | 12.28/0.614 | 24.50/0.373 | 22.60/0.442 |
| Linear CT, 60 views | 26.57/0.261 | 27.29/0.257 | 23.54/0.352 | 28.32/0.257 | 28.85/0.198 |
| Linear CT, 30 views | 24.45/0.362 | 24.63/0.373 | 22.42/0.375 | 26.97/0.276 | 26.34/0.298 |
| Nonlinear CT, 60 views | 22.21/0.438 | 22.44/0.438 | 19.53/0.464 | 25.05/0.354 | 25.46/0.319 |
| Nonlinear CT, 30 views | 20.91/0.488 | 21.12/0.488 | 19.32/0.470 | 23.85/0.403 | 24.10/0.380 |
| MRI-type sampling, 4x | 26.52/0.254 | 27.01/0.249 | 24.54/0.330 | 28.47/0.256 | 28.60/0.218 |
| MRI-type sampling, 8x | 24.05/0.334 | 24.28/0.340 | 23.18/0.343 | 26.44/0.291 | 25.19/0.331 |
| Masked phase retrieval | 21.97/0.453 | 21.63/0.463 | 22.19/0.418 | 24.52/0.383 | 24.74/0.386 |
| Coded diffraction, 2 masks | 28.18/0.186 | 30.68/0.176 | 30.02/0.201 | 26.86/0.302 | 34.92/0.102 |
| Unmasked phase retrieval | 13.65/0.765 | 14.02/0.729 | 9.98/0.740 | 20.06/0.457 | 16.85/0.594 (4 calls), 16.96/0.567 (8 calls) |

## Real MRI, M4Raw (40 brain slices, PSNR / SSIM, 4 calls)

| Sampling | PnP-CM | Ours, PnP-CM's 10-iteration data step | Ours, exact data step | Ours, k-space interpolation |
|---|---|---|---|---|
| Random lines, 4x | 24.55/0.730 | 29.26/0.827 | 29.55/0.833 (one system per image row) | 26.64/0.715 |
| Random lines, 8x | 23.46/0.703 | 27.38/0.775 | 27.50/0.778 (one system per image row) | 24.46/0.658 |
| Every 4th line | 21.98/0.701 | - | 26.96/0.802 (one 4x4 system per alias group) | - |
| Every 8th line | 19.61/0.648 | - | 23.88/0.723 (one 8x8 system per alias group) | - |

## Notes

- DAPS leads on PSNR: CelebA-HQ - JPEG, blur then clip, nonlinear CT 30 views, MRI-type 8x, unmasked phase retrieval; bedroom - JPEG, nonlinear deblur, blur then clip,
  linear CT 30 views, MRI-type 8x, unmasked phase retrieval. On LPIPS another method is better on roughly half the rows.
- PnP-CM = the same loop with its own gradient solver (exact linear step for linear operators), no tilt, shipped schedule; on bedroom it uses its CelebA-HQ hyperparameters.
  PnP-CM on real MRI uses its knee settings (not re-tuned for these 0.3 T brain scans).
- DPS / DAPS step sizes are borrowed from the closest shipped task (face settings on bedroom). DAPS on linear CT 30 views uses a 3x smaller step (the default diverged to about 5 dB).
- Masked phase retrieval: ours is scored as produced; the baselines with the sign fixed by the truth. Unmasked phase retrieval: PnP-CM rows take the better of the two 180-degree flips.
- Bedroom rows for nonlinear CT and masked phase retrieval use the face settings with weights refitted on generated bedroom images. Bedroom unmasked phase retrieval uses a start stage
  rebuilt from generated bedroom images (weaker than the face one; predictor trained for a shorter run).
- Real MRI: weights and penalty chosen on generated slices with measurement noise matched to the scans; the reference images are themselves noisy.

## Wall clock per image (final configurations; one image at a time; median of 10)

No GPU was idle for 20 hours, so these were measured on a GPU shared with another user's job (62-68% utilised by it). PnP-CM was measured back to back
in the same session, so ratios are reliable; absolute values are inflated about 2.4x (PnP-CM SR: 0.44 s here, 0.18 s on an idle GPU on 24 November).
"Loop" = model calls + data steps + combination; "total" also includes the start stage.

| Row | Total s / image | Loop only s |
|---|---|---|
| sr_bicubic | 0.47 | 0.46 |
| deblur_gauss | 0.51 | 0.50 |
| inpainting | 0.52 | 0.52 |
| deblur_nl | 0.70 | 0.69 |
| hdr | 0.47 | 0.47 |
| quant | 0.50 | 0.49 |
| onebit | 0.46 | 0.46 |
| gamma | 0.49 | 0.49 |
| blurclip | 0.48 | 0.48 |
| ct30 | 0.55 | 0.52 |
| ct60 | 0.63 | 0.57 |
| jpeg | 0.51 | 0.49 |
| blurq | 0.50 | 0.50 |
| mri4 | 0.46 | 0.45 |
| mri8 | 0.47 | 0.46 |
| ctnl30 | 0.58 | 0.55 |
| ctnl60 | 0.73 | 0.68 |
| prmask | 1.64 | 0.49 |
| cdp2 | 0.49 | 0.47 |
| pr_unmasked_4 | 4.14 | 0.47 |
| pr_unmasked_8 | 4.66 | 0.92 |
| pr_unmasked_free_4 | 9.07 | 0.47 |
| pr_unmasked_noniter_4 | 1.39 | 0.46 |
| PnP-CM sr_bicubic (exact linear step) | 0.44 | 0.44 |
| PnP-CM hdr | 0.90 | 0.90 |
| PnP-CM jpeg | 9.30 | 9.30 |
| PnP-CM deblur_nl | 7.59 | 7.59 |
| PnP-CM blurq | 1.02 | 1.02 |
| PnP-CM ctnl30 | 1.99 | 1.96 |
| PnP-CM cdp2 | 1.06 | 1.06 |
| PnP-CM prmask | 17.03 | 17.03 |
| PnP-CM mri4 (exact linear step) | 0.44 | 0.44 |

## RAM baseline (Reconstruct Anything Model; public pretrained checkpoint, one forward pass)

`mode_probe/ram_baseline.py`. RAM takes LINEAR operators only, so it has entries for the linear tasks; the nonlinear tasks (JPEG, nonlinear deblur, HDR,
quantisation, one-bit, tone curve, blur + pointwise, nonlinear CT, phase retrieval) have none. Same operators and noise level as ours; each operator is
rescaled to unit norm (an equivalent restatement of the same measurement; without it RAM gives 20 dB on SR and fails on CT). Not trained on either dataset.

| Task | RAM, CelebA-HQ | Ours, CelebA-HQ | RAM, bedroom | Ours, bedroom |
|---|---|---|---|---|
| SR x4 | 28.18/0.338 | 28.28/0.297 | 24.96/0.381 | 24.81/0.362 |
| Gaussian deblur | 30.19/0.285 | 29.98/0.253 | 27.42/0.290 | 26.71/0.283 |
| Inpainting 70% | 29.37/0.293 | 30.29/0.207 | 26.90/0.309 | 27.08/0.244 |
| Linear CT, 60 views | 30.85/0.263 | 32.25/0.171 | 27.60/0.298 | 28.85/0.198 |
| Linear CT, 30 views | 28.86/0.320 | 30.03/0.223 | 25.52/0.372 | 26.34/0.298 |
| MRI-type sampling, 4x | 29.29/0.300 | 31.42/0.189 | 27.11/0.295 | 28.60/0.218 |
| MRI-type sampling, 8x | 25.81/0.399 | 27.89/0.293 | 24.06/0.405 | 25.19/0.331 |

RAM time per image on a shared GPU: 0.06-0.09 s (CT 0.5-1.0 s, where the operator and its adjoint dominate); ours 0.46-0.63 s in the same conditions.
RAM leads on PSNR for Gaussian deblur on both datasets and SR on bedroom; ours has the better LPIPS on every row.
