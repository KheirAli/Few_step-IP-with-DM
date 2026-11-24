# Morning report, 24 November 2026

Everything queued overnight finished (04:10 to 07:33), with no failed batches. All choices were made on model-generated images; test sets were used once.
PSNR / LPIPS. Ours and PnP-CM: 4 model calls. DPS and DAPS: about 1000 calls, with a prior trained on the same dataset.

## 0. One extrapolation for every task

Every "Ours" number in sections 1 and 2 uses the SAME combination step: 8 radial frequency bands, terms = earlier passed-on images + this step's
data-fit image + the model output, weights fitted by least squares on generated images (`zband8`; audited from the weight files on 2026-11-24:
all are kind=band, 8 bands, data-fit term on). Only the data step differs between tasks. The region-wise variant in section 4 is a side
experiment, is NOT used in any table and is not part of the method.

## 1. CelebA-HQ, 107 test images

| Task | PnP-CM | PnP-CM + scalar extrapolation | DPS | DAPS | Ours |
|---|---|---|---|---|---|
| SR x4 | 27.24/0.287 | 27.45/0.284 | 24.34/0.287 | 25.47/0.414 | 28.28/0.297 |
| Gaussian deblur | 28.92/0.249 | 29.02/0.246 | 25.12/0.260 | 26.60/0.340 | 29.98/0.253 |
| Inpainting 70% | 29.06/0.204 | 29.31/0.204 | 27.19/0.249 | 27.56/0.299 | 30.29/0.207 |
| JPEG QF5 | 26.05/0.358 | 26.46/0.355 | 24.05/0.302 | 28.15/0.259 | 27.19/0.336 |
| Nonlinear deblur | 24.88/0.302 | 25.13/0.307 | 23.19/0.305 | 26.26/0.312 | 26.35/0.333 |
| HDR | 23.17/0.359 | 25.07/0.279 | 25.62/0.272 | 25.88/0.271 | 26.42/0.268 |
| 3-bit quantisation | 25.51/0.284 | 30.19/0.200 | 27.75/0.260 | 25.25/0.378 | 31.31/0.182 |
| One-bit | 19.03/0.516 | 23.16/0.447 | 28.82/0.217 | 22.38/0.512 | 30.30/0.213 |
| Tone curve | 27.25/0.217 | - | 27.50/0.239 | 29.62/0.245 | 33.22/0.147 |
| Blur then quantise | 24.51/0.365 | - | 24.63/0.276 | 24.76/0.388 | 26.62/0.351 |
| Blur then clip | 22.91/0.419 | - | 22.57/0.333 | 27.00/0.276 | 24.67/0.387 |
| Linear CT, 60 views | 28.40 | - | 28.75/0.218 | 30.59/0.227 | 32.25/0.171 |
| Linear CT, 30 views | 26.89 | - | 27.70/0.231 | failed | 30.03/0.223 |
| Nonlinear CT, 60 views | 25.24/0.326 | - | 24.80/0.270 | 28.55/0.254 | 28.73/0.254 |
| Nonlinear CT, 30 views | 23.93/0.355 | - | 24.15/0.278 | 27.65/0.264 | 27.47/0.292 |
| MRI-type sampling, R=4 | 28.44/0.252 | - | - | - | 31.42/0.189 |
| MRI-type sampling, R=8 | 26.13/0.302 | - | - | - | 27.89/0.293 |
| Masked phase retrieval | 22.93/0.463 (8 calls) | 24.02/0.416 | 24.79/0.308 | 27.03/0.318 | 28.57/0.270 |
| Coded diffraction, 2 masks | 28.65/0.186 | - | - | - | 34.45/0.126 |
| Unmasked phase retrieval, 4 calls | ~13 | - | 13.43/0.643 | 26.70/0.308 | 24.18/0.371 (training-free), 24.63/0.362 (predictor) |
| Unmasked phase retrieval, 8 calls | | | | | 25.18/0.328 (training-free), 25.69/0.317 (predictor) |

Notes: nonlinear CT = data step without alternation. Masked phase retrieval = attempt 2, solved inside the operator class, scored as produced
(baselines are scored with the sign fixed by the truth). Unmasked phase retrieval, non-iterative start: 23.63/0.377 (4 calls), 25.01/0.336 (8 calls), no failures.

## 2. LSUN bedroom, 58 test images

"Transfer" = CelebA-HQ settings unchanged, only the combination weights refitted. "Tuned" = every choice re-made on generated bedroom images (new tonight).

| Task | PnP-CM | PnP-CM + scalar extrapolation | DPS | DAPS | Ours, transfer | Ours, tuned on bedroom |
|---|---|---|---|---|---|---|
| SR x4 | 23.89/0.375 | 24.35/0.368 | 21.22/0.429 | 22.07/0.542 | 24.72/0.367 | 24.81/0.362 |
| Gaussian deblur | 25.83/0.308 | 25.85/0.305 | 21.98/0.372 | 22.84/0.505 | 26.44/0.291 | 26.71/0.283 |
| Inpainting | 25.77/0.264 | 25.86/0.264 | 24.54/0.330 | 25.67/0.318 | 26.58/0.255 | 27.08/0.244 |
| JPEG | 23.80/0.403 | 24.61/0.384 | 20.73/0.451 | 26.23/0.313 | 25.12/0.358 | 25.21/0.355 |
| Nonlinear deblur | 23.53/0.356 | 23.65/0.366 | 21.45/0.411 | 25.18/0.328 | 24.31/0.373 | 24.08/0.392 |
| HDR | 22.93/0.301 | 25.26/0.236 | 15.05/0.521 | 25.49/0.260 | 25.78/0.245 | 26.03/0.231 |
| Quantisation | 25.71/0.283 | 29.67/0.209 | 25.01/0.338 | 25.13/0.372 | 29.98/0.207 | 31.17/0.166 |
| One-bit | 16.05/0.653 | 19.99/0.572 | 26.34/0.261 | 19.92/0.560 | 28.50/0.234 | 28.41/0.236 |
| Tone curve | 25.35/0.234 | 27.47/0.210 | 23.81/0.351 | 28.50/0.282 | 30.67/0.191 | 31.92/0.173 |
| Blur then quantise | 22.47/0.474 | 22.66/0.475 | 21.45/0.403 | 21.75/0.562 | 23.53/0.428 | 23.78/0.412 |
| Blur then clip | 21.37/0.467 | 21.56/0.486 | 12.28/0.614 | 24.50/0.373 | 22.70/0.445 | 22.60/0.442 |
| Linear CT, 60 views | - | - | 23.54/0.352 | 28.32/0.257 | 28.42/0.226 | 28.85/0.198 |
| Linear CT, 30 views | - | - | 22.42/0.375 | - | 25.52/0.346 | 26.34/0.298 |
| Nonlinear CT, 60 views | 22.21/0.438 | 22.44/0.438 | 19.53/0.464 | 25.05/0.354 | 25.46/0.319 | - |
| Nonlinear CT, 30 views | 20.91/0.488 | 21.12/0.488 | 19.32/0.470 | 23.85/0.403 | 24.10/0.380 | - |
| MRI-type sampling, R=4 | 26.52/0.254 | - | - | - | - | 28.60/0.218 |
| MRI-type sampling, R=8 | 24.05/0.334 | - | - | - | - | 25.19/0.331 |
| Masked phase retrieval | 21.97/0.453 | 21.63/0.463 | 22.19/0.418 | 24.52/0.383 | 24.74/0.386 | - |

## 3. Real MRI, public M4Raw dataset (0.3 T brain, T2, 40 slices, 4 calls), PSNR / SSIM

| Undersampling | PnP-CM | PnP-CM + our noise tilt |
|---|---|---|
| R = 4 | 24.55 / 0.730 | 24.69 / 0.733 |
| R = 8 | 23.46 / 0.703 | 23.76 / 0.709 |

Brain model trained at 3 T (domain shift); only the tilt is tested here.

## 4. New this night

**Masked phase retrieval inside the operator class, like for like.** Attempt 2 with the frequency-weighted fit: 28.57/0.270, against 28.38/0.270 for attempt 1
(reduction to inpainting). On generated images 29.26 against 28.64. So the task can be presented as a member of the class with no loss.

**Unmasked phase retrieval with the shared components** (noise tilt, tuned schedule, band combination): +0.6 to +0.9 dB for the iterative start,
+0.8 to +1.3 dB for the non-iterative start. Gap to DAPS at 8 calls: 1.0 dB (best iterative), 1.7 dB (non-iterative).

**Two new combination methods** (107 CelebA-HQ test images, each task's clean configuration):

| Task | 8 bands + last data-fit image (current) | 8 bands + all data-fit images | 8 bands x measurement region |
|---|---|---|---|
| HDR | 26.42/0.268 | 26.57/0.267 | 26.77/0.255 |
| Blur then clip | 24.67/0.387 | 24.84/0.387 | 24.91/0.377 |
| Inpainting | 30.29/0.207 | 30.31/0.206 | 30.32/0.208 |
| SR | 28.28/0.297 | 28.32/0.297 | - |
| Quantisation | 31.31/0.182 | 31.34/0.180 | - |
| One-bit | 30.30/0.213 | 30.31/0.211 | - |
| JPEG | 27.19/0.336 | 27.22/0.339 | - |
| Gaussian deblur | 29.98/0.253 | 29.98/0.251 | - |
| Tone curve | 33.22/0.147 | 33.25/0.147 | - |

Region-wise weights help where the measurement splits the image into well-measured and unmeasured pixels by clipping (+0.35 dB on HDR, +0.24 dB on blur
then clip, with better LPIPS), not on inpainting. Adding all data-fit images gives +0.0 to +0.17 dB. These are not folded into the main tables.

## 5. Time per image, one image at a time on an idle GPU

| Task | Ours | PnP-CM |
|---|---|---|
| SR, deblur, inpainting, HDR, quantisation, one-bit, blur combinations, tone curve | 0.19-0.20 s | 0.18 s (linear), 0.42 s (HDR, gradient solver) |
| JPEG | 0.22 s | 3.52 s |
| Nonlinear deblur | 0.27 s | 4.10 s |
| Linear CT, 30 / 60 views | 0.22 / 0.26 s | - |
| Nonlinear CT, 30 / 60 views | 0.24 / 0.30 s | 0.84 / 1.57 s (measured earlier on loaded GPUs) |
| Masked phase retrieval | 1.26 s | 10.8 s (8 calls, measured earlier) |
| Unmasked phase retrieval, non-iterative / iterative | 1.05 / 2.23 s | - |

DPS and DAPS: 33-90 s (DAPS up to 520 s on JPEG), measured earlier.

## 6. What to keep in mind

- One run per method; 107 and 58 test images; differences under about 0.2 dB are not wins.
- The data step itself buys speed and no solver tuning, not accuracy: with the pipeline fixed, a tuned gradient solver matches it on HDR, tone curve and
  quantisation. The accuracy comes from the shared components (combination, noise tilt, calibrated schedule), which PnP-CM was not given beyond scalar extrapolation.
- "Closed form" is repeated a fixed number of times on some tasks (frequency-weighted fit: 7 solves per step; blur combinations: 8 alternations).
- DAPS still leads on JPEG, blur then clip and unmasked phase retrieval on CelebA-HQ, and on JPEG, nonlinear deblur and blur then clip on bedroom.
  On LPIPS another method is better on about half the tasks.
- Candidate lists and the choice of the band combination came from earlier work in which test numbers were visible; 104 of 107 CelebA-HQ test images exist in CelebA-HQ.
- Tuning on bedroom lowered two tasks slightly (nonlinear deblur -0.23 dB, blur then clip -0.10 dB): the generated-image score is not a perfect guide.
- Baseline step sizes are borrowed from face settings; PnP-CM on bedroom uses CelebA-HQ hyperparameters.
- Not run: our method on unmasked phase retrieval for bedroom; DPS / DAPS on MRI-type sampling and coded diffraction; the combination step inside the real-MRI pipeline.

Details and every earlier experiment: `mode_probe/RESULTS_structured_combinations.md`; independent check: `mode_probe/AGENT_CHECK.md`; attempt 1 / attempt 2: `mode_probe/ATTEMPTS.md`.
