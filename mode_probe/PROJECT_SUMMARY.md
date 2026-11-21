# Few-step consistency-model solvers without an inner optimiser: project summary (2026-11-20)

Setting: PnP-CM (CVPR'26) loop = data-fit, add noise, one consistency-model call; 4 calls (2 also tested).
Data: CelebA-HQ 256, 107 of the paper's 300 validation images, noise 0.1 on [-1,1]; selection on 24 separate images.
All code in `PNP/mode_probe/` (`unified.py` is the sampler); baselines in `PNP/baselines_src/`.

## 1. Contributions

1. **Closed-form data-fits for separable nonlinear operators** (HDR/clipping, quantisation, one-bit, tone curve, JPEG, Fourier magnitude):
   no inner optimiser, no learning rate, no operator gradient. 2-40x faster than PnP-CM on nonlinear tasks, higher PSNR on all.
2. **Expected-value data-fit for one-bit measurements** (mean of a truncated Gaussian instead of the nearest feasible point):
   23.9 -> 27.8 dB at 4 calls, 16.3 -> 27.1 dB at 2 calls.
3. **Reductions for non-separable / multi-solution operators**: masked phase retrieval = HIO on the known support + inpainting;
   nonlinear deblur = fitted linear filter + true-operator correction; "linear then pointwise" = alternation of two closed forms.
4. **Frequency-weighted data-fit** solved by re-using the closed forms (proximal-gradient passes); rule: every step except the first;
   off when a large part of the image is unobserved in the pixel domain (clipping) and for CT.
5. **Tilted noise** (power ~ frequency-rank^0.5): +0.1..+0.7 dB, also on real MRI with PnP-CM's own pipeline (3 slices only).
6. **Linear extrapolation over the step outputs** (method of Zhang et al., NeurIPS'25; adopted, not ours): helps every task,
   +1.2..+2.3 dB on HDR / quantisation / one-bit. Our additions still to be tested: closed-form weights, split for nonlinear operators.
7. **Analysis**: the few-step sampler is effectively deterministic (seed-to-seed 0.03-0.14 dB); errors are bias; candidate selection,
   diversity terms and re-distribution of calls do not help; per-step and per-frequency diagnosis of each failing task.

## 2. Results at 4 calls (PSNR / LPIPS, 107 test images)

| Task | PnP-CM | Ours | DPS (1000) | DAPS (~1000) | time ours / PnP-CM |
|---|---|---|---|---|---|
| SR x4 | 27.24/0.287 | 27.96/0.285 | 23.31/0.321 | 24.10/0.554 | 0.2 / 0.18 s |
| Gaussian deblur | 28.92/0.249 | 29.50/0.246 | 24.33/0.286 | 26.00/0.445 | 0.2 / 0.18 s |
| Inpainting 70% | 29.06/0.204 | 29.87/0.199 | 26.44/0.263 | 27.80/0.285 | 0.2 / 0.18 s |
| JPEG QF5 | 26.05/0.358 | 26.77/0.354 | 22.81/0.341 | 28.11/0.270 | 0.3 / 3.2 s |
| Nonlinear deblur | 24.88/0.302 | 26.31/0.335 | 22.69/0.332 | 26.17/0.339 | 0.3 / 4.2 s |
| HDR | 23.17/0.359 | 25.22/0.307 | 23.06/0.298 | 25.43/0.282 | 0.2 / 0.42 s |
| 3-bit quantisation | 25.51/0.284 | 30.87/0.216 | 26.45/0.281 | 24.34/0.395 | 0.2 / 0.42 s |
| One-bit | 19.03/0.516 | 29.39/0.239 | 28.64/0.210 | 23.22/0.431 | 0.2 / 0.42 s |
| Phase retrieval, masked | 22.93/0.463 (8) | 26.78/0.392 | 24.12/0.329 | 27.38/0.298 | 1.5 / 10.8 s |
| Phase retrieval, unmasked | n/a | 23.14/0.387 (24.07 at 8) | 13.52/0.638 | 25.84/0.308 | 2.3 s |
| CT 60 / 30 views (faces) | 31.11 / 28.48 | 31.50 / 29.20 | pending | pending | ~2 s |
| Tone curve | 27.25/0.217 | 29.21/0.217 | pending | pending | 0.3 / 0.55 s |
| Blur + 3-bit quantisation | 24.51/0.365 | 25.52/0.413 | pending | pending | 0.3 / 0.63 s |
| Blur + clip | 22.91/0.419 | 22.49/0.446 | pending | pending | 0.25 / 0.64 s |
| Wrap-around (modulo) | 5.51 | 9.47 (fails) | pending | pending | |

DPS 34-41 s/image; DAPS 33-37 s (cheap operators), ~300-520 s (nonlinear deblur, JPEG).
Two calls: within 0.2-0.7 dB of four on most tasks (also true of PnP-CM's own solver).

## 3. Tested and not supported

Posterior-mean + calibrated-covariance "single principle"; fully calibrated penalty; expected-value fit on quantisation;
dropping the dual variable; explicit warm starts where the data-fit is exact; higher told level in the last step; true operator in the
last step; best-of-K selection; diversity (repulsion) term; two short runs averaged / forked / two-sample first step;
BM3D in the phase-retrieval coarse loop; learned low-frequency phase predictor (ties the training-free version on small nets).

## 4. Open

PnP-CM + extrapolation (fairness); closed-form extrapolation weights; second dataset (LSUN bedroom CM is public);
full fastMRI validation set; 300 images and 3 seeds; DPS/DAPS tuned at our noise; LPIPS on nonlinear deblur; JPEG and phase retrieval vs DAPS.

## 5. Theoretical claim (see the session notes for the full statement)

Class S: A(x) = f(Tx), T orthogonal (or unitary), f acting coordinate-wise.
(P1) The step's data-fit  min_z 1/2 |f(Tz) - y|^2 + rho/2 |z - v|^2  splits into n scalar problems and its GLOBAL minimiser is
     z* = T' c*, c*_i = argmin_c 1/2 (f_i(c) - y_i)^2 + rho/2 (c - (Tv)_i)^2, even when f is non-convex or discontinuous.
     Cost O(K n) for piecewise-linear f with K pieces (clip 3, quantiser L, sign 2); closed form for |.| on complex coefficients.
(P2) If every coordinate's solution set f_i^{-1}(y_i) is an interval (clip, quantiser, sign, monotone curves, JPEG), the feasible set
     {z : f(Tz) = y} is convex (a box in the transform domain) and the noiseless data-fit is the projection onto it.
     If it is not (circle for |.|, lattice for modulo), the feasible set is non-convex and a selection stage is required.
     This convex / non-convex preimage criterion matches all 15 operators tested.
(P3) With a frequency weight 0 < W <= I, the passes z <- datafit(z - W(z - v)) are forward-backward iterations and converge to the
     weighted minimiser when the data term is convex.
