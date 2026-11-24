# Independent check (2026-11-24)

Read-only review of `unified.py`, the protocol scripts, `README.md`, `ATTEMPTS.md`, `RESULTS_structured_combinations.md`.
Numerical checks were CPU-only (scripts in `/tmp/agchk/`: `pr2.py`, `nl.py`, `ov.py`). No existing file was edited, no GPU job started.
`attempt2.py` was still running during the check (tuning stage only, no test numbers yet).

Verdict summary

| # | Item | Verdict |
|---|---|---|
| 1a | Operator is `|F P D x|`, D in {-1,0,+1}, F unitary | correct |
| 1b | `(FPD)^H (FPD) = D^2` | correct |
| 1c | `_prox_pr2` does what its comment says; no bugs found | correct as code; "exact data step" would be incorrect: it is one splitting step |
| 2 | Wiener test (`NLDIAG`) | numbers reproduce, "a per-pixel curve gives nothing" correct; "the 7.6% is nonlinearity / 92% linear" overstated |
| 3 | Non-iterative unmasked PR start | correct (no loop, no ground truth, no leak); the `PRD_KAPPA` block is not used by any reported number |
| 4 | "Every choice on generated images, test used once" | overstated; three concrete violations below |
| 5 | README vs results | several statements unsupported; data-step accuracy advantage is not claimed outright but is not disclaimed either |

## 1. Masked phase retrieval, attempt 2 (`PR2=1`, `_prox_pr2`)

(a) Correct. `PhaseRetrievalOperator.forward` = `fft2_m(pad(x * modulation)).abs()`. `modulation = -(g==0) + (g==7)` with `g` uniform on 0..7, so values are {-1,0,+1} and 2/8 of pixels are kept (measured 24.6%). `fft2_m` (centred FFT, `norm="ortho"`) is unitary: `|Fp|/|p| = 1.000005`, `ifft(fft(p)) - p` relative error 1.7e-7. Pad = 128 per side (256 -> 512) for oversample 4.

(b) Correct. On a random real image `|M^H M x - D^2 x| / |x| = 4e-8` (single precision; `fft2_m` casts to complex64), adjoint test `<Mx,u> = <x,M^H u>` agrees to 7 digits, with `M^H u = D * crop(Re F^-1 u)`.

(c) The code matches its comment, and I found no bug:
- Real part: correct. The noisy `y` is not Hermitian-symmetric, so `F^-1 u` is complex; `Re` is the right adjoint for a real image.
- Zero-mask pixels: left equal to `v` (verified, max change 0). This is the exact answer there, since the objective does not see them.
- Sign: `|M(-x)| = |M x|` per channel, so working in the sign-corrected frame is harmless. The step keeps the phase of `v`, so it can never repair a wrong channel sign; the sign comes entirely from the start (HIO + `PRSIGN=prior`). In the attempt-1 clean run 4 of 103 logged test images have a wrong-sign channel (`sign_ok` mean 0.975).
- Scaling: objective is `1/2 | |Mz| - y |^2 + rho/2 |z - v|^2` with no `1/sigma_y^2`, the same convention as `svd_prox`; the masked operator acts on [-1,1] images directly, so no factor 4 (unlike `priso_prox`). `PR2_RS` is a free multiplier tuned on generated images.
- Negative `y + r|c|` (noisy `y` below zero at weak coefficients) flips the phase by pi; rare for r >= 0.3, harmless.

It is NOT the exact minimiser. The range of `M` has ~16k real dimensions per channel inside a 524k-dimensional coefficient space, so projecting the per-coefficient solution back is one splitting step. Algebraically the step is exactly
`z_kept = (ER(v) + r v) / (1 + r)`, with `ER(v) = M^H[ y * Mv/|Mv| ]` one error-reduction (magnitude-projection) step (verified: relative difference 4e-8).
Measured on a generated image (true objective at `v` / after `_prox_pr2` / after L-BFGS from that point):

| anchor noise, rho | J(v) | J(prox) | J(L-BFGS) | kept-pixel PSNR prox / L-BFGS |
|---|---|---|---|---|
| 0.1, rho 0.1, no meas. noise | 119.5 | 40.0 | 20.0 | 30.8 / 40.7 dB |
| 0.1, rho 1 | 119.5 | 81.3 | 78.9 | 28.4 / 29.4 |
| 0.1, rho 10 | 119.5 | 113.58 | 113.57 | 26.40 / 26.42 |
| 0.5, rho 0.1, sigma_y 0.05 | 3913 | 1870 | 1423 | 16.8 / 24.8 |

The gap is large for rho <= 1 and negligible for rho >= 10.

What can be claimed: the masked operator is in the class `f(Lx)` with `L^H L` diagonal; each of the two pieces is closed-form and exact for its own sub-problem; the composition always lowers the objective in the cases tested but is one splitting step. Attempt 2 differs from attempt 1 only in the target for the kept pixels: fixed HIO output (attempt 1) versus one magnitude projection of the current image (attempt 2). Both still need the HIO search (4 x 1050 FFT iterations) and the prior-based sign. Also: attempt 1's reported number uses the `fw` variant; `attempt2.py` has no `fw` candidate for PR2, so the comparison against 28.20 is not like for like.

## 2. Nonlinear deblur, attempt 2 (`NLDIAG`)

Reproduced on CPU exactly: identity 0.1976, linear filter 0.0759, filter + curve 0.0761. No saved log of the original run was found under `logs_x/`.

Logic checks, none of which hides a gain:
- Fit (files 0-7) and test (files 8-15) of `datacal` are disjoint; `datacal` has 20 images; none is a near-duplicate of a test image (nearest 32x32 RMS distance 0.117).
- 257 knots instead of 65: 0.0761. No monotone projection: 0.0761. (`cummax` is not an isotonic fit, but it does not matter here.)
- Clamp range +-1.3 covers all of `Kx` (range -1.06..1.22).
- In-sample (fit images) the curve gives 0.0596 -> 0.0592, so the null result is not a generalisation artefact.
- Cross-channel pointwise cubic map of the three filtered channels (covers "per-channel curve too restrictive"): interior 0.0613 -> 0.0614.

So "a static per-pixel function after this filter explains nothing more" is correct and robust.

What is NOT justified is reading the 7.6% as the size of the nonlinearity:
- 64% of the held-out residual energy sits in the 24-pixel border (34% of the area): circular FFT filter versus the network's padding. Interior-only residual is 6.1%.
- In-sample residual is 6.0% (5.3% interior) versus 7.6% held out: part of it is estimation error of `K` (a 3x3 complex matrix per frequency from 16 samples).
- Direct nonlinearity tests that involve no fitted filter: midpoint test `|A((a+b)/2) - (A(a)+A(b))/2| / |A(mid)|` = 2.7% (interior, 4 held-out pairs; 2.2% relative to `|A(a) - A(b)|`); shift test `|A(shift x) - shift A(x)| / |A x|` = 0.6-1.7%.

So the operator is about 97% affine on this measure, and most of the 7.6% is error of the particular linear surrogate (borders, not exactly shift-invariant, K estimation), not nonlinearity. "92% linear" understates linearity, and "the network's nonlinearity is not coordinate-wise" is only supported in the weaker form "a curve after a shift-invariant filter does not capture what the filter misses".

Stronger test to run (GPU, 1 minute): decompose the residual directly. For held-out `x`, with `x0` = the filter-deconvolved start: (i) `|A(x) - A(x0) - J(x0)(x - x0)| / |A(x)|` with `torch.func.jvp` (true nonlinearity; `DIAGLIN=1` already computes this, report it); (ii) the same residual for `K`; (iii) the curve test applied to the residual of the Jacobian model on the interior only. If (i) is ~2-3%, the paper statement is "nearly linear, not shift-invariant", and a better linear model (non-circular boundary) is the lever, not a curve.

## 3. Unmasked phase retrieval, attempt 2 (`netdirect_init`, `PRD_KAPPA`)

Start: correct as described. One forward pass of `Net2` on features of the measured magnitudes, one FFT of the K=8 hypotheses, `argmin` of the magnitude mismatch inside the disc, one inverse FFT. With the `pr_direct.py` / `attempt2.py` settings (`PRD_K=0 PRD_TV=0 PRD_FINAL=0`, `PRD_TOPK` unset) no HIO, no TV loop, no L-BFGS, no multi-start runs. No ground truth is used. The hypothesis selection uses only `amp` and the network output: no leak. Scoring uses `rec["psnr"]` (upright image), not the better of the two flips.

Qualifiers on "non-iterative":
- `PRD_KAPPA` is not set by any script (`grep` finds it only in `unified.py`). The "partly linear" data step is therefore unused in every reported number (22.87 / 23.72 and the `attempt2.py` runs); those use the plain per-coefficient magnitude fit. The KAPPA algebra itself is correct (completing the square gives the stated anchor and weight), but it counts the measured magnitude twice inside the disc (target `c` has modulus `amp`), so it is not noise-calibrated.
- The data step is one splitting pass, not the exact data-fit (the padded transform is redundant; the code's own `PRD_ADMM` comment says so).
- "No search over starts" is true only up to the pick among 8 hypotheses.
- It is no longer training-free: a second network trained on 256k model samples. Tuning on generated images favours it (in-distribution): 24.55 vs 24.37 on generated images became 22.87 vs 23.62 on test.
- `PRD_NET=v2k8` and `PRD_NETR=32` are fixed in the scripts, not chosen by a logged generated-image search. `PRD_OFFSET=1` loads a 3.4 GB file that this path never uses.

## 4. Protocol

Violations of "every choice on generated images, no real data":
1. `mean_face01.pt` is the mean of the 100 REAL images in `data100`, not of model samples (32x32 RMS difference to the `data100` mean 0.0001; to the `data_gen512` mean 0.066). It sets the channel signs for masked PR (`PRSIGN=prior`, clean-protocol 28.20) and is the start for coded diffraction (34.45). Two `data100` images are test images (`00021.png` = `img_127.png`, `00066.png` = `img_176.png`; RMS 0.0035); the README says one. `ATTEMPTS.md` and code comments say "mean face of model samples".
2. `reg_best.pt` (the regression start of the "training-free" unmasked PR, `regfit4`) was selected in `reg_pr.py` by `report(...)`, which returns the mean PSNR on the first 12 images of `data_paper` (test). The clean number 23.62 / 24.38 and the "learned predictor" number 24.07 / 24.78 (`netfit` uses `creg`) depend on it. The non-iterative path does not.
3. The combination mode "8 bands + data-fit term" was picked from the Job-3 table, which compares modes on the 107 test images.
4. Nonlinear deblur's filter `K` is fitted on 8 real CelebA-HQ images (`datacal`; disjoint from test). Not a leak, but it is real data.

Weaker points:
- `clean_protocol.py` searches only a short candidate list "compared earlier on real selection images", from base schedules (`LIN`, `SR`, `JP`, `NL`) and constants (`TILT=0.5`, `FWP`, `FWMIN`, `OBMEAN`, `RHOX`, `HIO_K/N`, `SPLIT_EV`) fixed in earlier work on real selection / test images. One-bit keeps its base schedule unchanged (29.873 -> 29.873). Accurate wording: "within a fixed candidate list, variant and schedule were chosen on generated images".
- "Test set used once" holds per script, not for the project: each task was evaluated on the same 107 images many times before (the "Previous number" column).
- Same generated images for choosing and reporting: `sched_cd2.py` "tuned" values are the maximum over 19 runs on the 48 tuning images; `pr_clean.py` / `pr_direct.py` generated-image numbers (24.55 vs 24.37) are best-of-sweep on the 24 tuning images. Fine for selection, not for quoting.
- Schedule tuning uses generated images 0:48, weights 0:128 (overlap); harmless because reporting is on test.
- `attribution.py`: arms use older configurations (HDR first sigma 3.0, no `fw`), so its "Ours" column differs from the clean numbers (HDR 26.32 vs 26.42, quant 30.84 vs 31.31, one-bit 29.49 vs 30.30, tone curve 32.04 vs 33.22). The gradient arm's learning rate was chosen without the combination. Test images are not used for selection.
- `sched_cd2.py` aligns channel signs with the truth when scoring `prmask` on generated images (tuning only; test is scored as produced).

`JNONE=1`: `unified.py` never reads `JNONE`. The variant therefore runs the default path: no `Pb.prox`, cold start, the built-in iterative solver on the true JPEG operator (gradients of the operator). So it does select the true-operator JPEG fit, but that fit is the iterative one, not a closed form; it also relies on `JLIN` / `JPROJ` not being set in the parent environment. It lost the selection (26.95 vs 27.36 for `softfw`), so no reported number depends on it.

## 5. README / ATTEMPTS versus the results

1. "Solved exactly, so no inner optimiser" (Idea) and "one closed-form update" (tier 1). `FW=1` runs `FWK=6` proximal-gradient iterations, each a closed-form solve: 7 solves per sampler step. The clean protocol chose an `fw` variant for 7 of 14 tasks (Gaussian deblur, inpainting, JPEG, nonlinear deblur, quantisation, one-bit, masked PR). For nonlinear deblur with `MEC=step` that is also 7 true-operator calls per step, not 1.
2. Tier 2 "two closed-form updates, applied once per sampler step". Blur-then-clip and blur-then-quantise (24.67, 26.62) use `split_prox` with `n=8` alternations per step. Only nonlinear CT (`CTNL_N=1`) and the zero-padded PR fit are single-pass.
3. Attribution. The README does not claim an accuracy advantage of the data step in so many words, but "Part of the gap ... not from the data step alone" understates the result: with the same pipeline the tuned gradient solver is equal or better on HDR (26.17/0.263 vs 26.32/0.269), quantisation (31.08/0.196 vs 30.84/0.216) and tone curve (32.00 vs 32.04). The closed form is ahead on one-bit (+0.59 dB), blur-then-clip (+0.43) and nonlinear CT (+1.83, confounded: the gradient arm inherits our penalty). The README should state this and rest the data-step claim on speed (0.2-0.3 s vs 0.5-4 s), no learning rate, and robustness (gradient-solver worst/best spread). The headline table's gap to PnP-CM on pointwise tasks is due to the shared components.
4. Table header "every choice made on model-generated images, test set used once" contradicts "Known limits" bullet 2 ("some settings were chosen on real selection images or on the test set; being redone") and section 4 above. One of the two is stale.
5. LPIPS limit bullet is incomplete. Ours is also behind on inpainting (0.207 vs PnP-CM 0.204), unmasked PR, nonlinear CT 30 views (0.292 vs DAPS 0.264, DPS 0.278), blur-then-quantise (0.351 vs DPS 0.276), blur-then-clip (0.387 vs DAPS 0.276, DPS 0.333).
6. PSNR leads inside single-run noise: nonlinear deblur 26.35 vs DAPS 26.26 (LPIPS worst of all five methods), nonlinear CT 60 views 28.73 vs 28.55, bedroom masked PR 24.74 vs 24.52 (oracle sign), bedroom linear CT 60 28.42 vs 28.32. Nonlinear CT 30 views is behind DAPS (27.47 vs 27.65). "Best PSNR on 13 of 16" (bedroom) counts correctly but includes these.
7. Timing "phase retrieval 1.5-2.3 s". Logged medians (shared GPUs): masked 1.42 s, non-iterative unmasked 1.27 s, training-free unmasked (the 23.62 headline) 7.5 s.
8. Tier table: coded diffraction's start is "mean face + 50 magnitude-fit passes" (iterative, real-image mean face). JPEG's chosen variant is soft interval projection + frequency weighting, not the tier-1 one-update form.
9. `ATTEMPTS.md` section 3: "92% linear" and "nonlinearity is not in the class" (see section 2). `ATTEMPTS.md` header "all tuning on model-generated images; test set used once" (see section 4). Section 1: "mean face of model samples" is wrong.
10. Masked PR row: ours is scored as produced; confirm the PnP-CM / DPS / DAPS entries use the same sign convention (bedroom uses the oracle sign for all).

## What can be said in the paper

Masked phase retrieval. "The masked operator is `|F P D x|` with `(FPD)^H(FPD) = D^2` diagonal, so it belongs to the class; the data step is one closed-form magnitude update per Fourier coefficient followed by a diagonal back-projection, equal to a damped error-reduction step. It is a single splitting step, not the exact minimiser, and the method still needs a classical multi-start HIO stage and a prior-based channel sign."

Unmasked phase retrieval. "A learned phase predictor (trained on model samples) gives a start with one network pass and one inverse FFT, with no classical loop and no ground truth; each sampler step then applies one per-coefficient magnitude fit (a splitting step for the padded transform). This reaches 22.87 / 23.72 dB at 4 / 8 calls with no failures, below the iterative start and below DAPS (26.70)."

Nonlinear deblur. "The learned blur is close to affine (midpoint test about 3% of the output norm). A shift-invariant 3x3-per-frequency filter fitted on 16 images leaves 7.6% unexplained on held-out images, mostly at the borders, and a per-pixel curve after the filter does not reduce it; we therefore use the linear closed form with a per-step correction from the true operator."

Do not quote yet:
- Masked PR attempt 2: no test number exists (run in progress), and the comparison to 28.20 needs an `fw`-matched arm.
- 28.20 (masked PR) and 34.45 (coded diffraction) as "no real data": the mean face is from real images, two of them test images. Rebuild it from model samples and rerun.
- 23.62 / 24.38 and 24.07 / 24.78 (unmasked PR) as "clean protocol": `reg_best.pt` was selected on 12 test images. Reselect on generated images and rerun.
- "92% linear" / 7.6% as a nonlinearity measure: it is a surrogate-error measure.
- Any non-iterative unmasked PR number described with the "partly linear" (`PRD_KAPPA`) data step: never run.
- 24.55 vs 24.37 (generated images) and the `CD2 ... tuned` values: best-of-sweep on the tuning images.
- Attribution "Ours" numbers as headline numbers (older configs); nonlinear-CT attribution gap of 1.83 dB (confounded penalty).
- "Phase retrieval 1.5-2.3 s" for the training-free unmasked variant (7.5 s logged).
- Sub-0.2 dB PSNR leads over DAPS (item 6): one run each, 107 / 58 images.
