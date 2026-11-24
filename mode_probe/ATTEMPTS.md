# Tasks outside the operator class: attempt 1 and attempt 2

The paper's class: `A(x) = f(Lx)`, `L` linear, `f` acting on each coordinate. Tier 1: `L` distance-preserving (one closed-form
data step). Tier 2: `L` linear with a cheap decomposition (two closed-form pieces: per coordinate, then linear).
Three tasks were handled outside this class. This file keeps the original algorithms ("attempt 1") and records the mappings
into the class ("attempt 2"). All tuning on model-generated images; test set used once.

## 1. Masked phase retrieval (PnP-CM's task)

Operator (`PnP-CM/functions/measurements.py`, `PhaseRetrievalOperator`): `y = |F P D x| + noise`, `D` = known mask with entries
in {-1, 0, +1} (1 pixel in 4 kept), `P` = zero padding 256 -> 512, `F` = unitary FFT.

**Attempt 1 (reduction, outside the class).** `INIT=hio HIO_K=4 HIO_N=1000 PRLIN=1 PRSIGN=prior LINEXACT=1 [FW=1 FWP=0.5]`
1. Classical hybrid input-output on the known support, 4 starts x 1000 iterations, FFTs only -> the kept pixels (up to a sign per colour channel).
2. Sign per channel from positive correlation with the mean face of model samples (`PRSIGN=prior`).
3. The kept pixels are treated as a LINEAR measurement (inpainting, 25% of the pixels); 4 model calls with the closed-form linear data step.
Result (clean protocol, as produced): 28.20 / 0.273.

**Attempt 2 (inside the class, tier 2).** Add `PR2=1 [PR2_RS=r]`. Same start (steps 1-2). In the sampler the data step uses the TRUE
operator: `L = F P D` is linear and `L'L = D^2` is diagonal in pixels, `f = |.|` is coordinate-wise.
- piece 1, per Fourier coefficient: keep the phase of `c = (L v)_k`, magnitude `(y_k + r |c|) / (1 + r)`;
- piece 2, back to the image, diagonal in pixels: kept pixels take `D * crop(F^-1 u)`, the others keep `v`.
Applied once per step (`_prox_pr2` in `unified.py`). The start is still a search stage, as the ambiguity check predicts for magnitudes.
Result: see `RESULTS_structured_combinations.md` (run `attempt2.py`).

## 2. Unmasked phase retrieval (DPS / DAPS task)

Operator: `y = |F P x| + noise`, padding 256 -> 384. In the class (tier 2); what was outside the "exact, non-iterative" goal is the start.

**Attempt 1 (iterative start).** `PRD_INIT=regfit4 PRD_NSTART=16 PRD_OFFSET=1 PRD_TV=80` (training-free) or `PRD_INIT=netfit PRD_NET=v2k8 PRD_NETR=48`
(predictor inside the fit): regression -> 16 eigenface fits (L-BFGS) -> 80 passes of magnitude fit + total variation -> 4 / 8 model calls with the
per-coefficient magnitude fit. Results: 23.62/0.369, 24.38/0.336 (training-free); 24.07/0.356, 24.78/0.325 (with predictor).

**Attempt 2 (non-iterative).** `PRD_INIT=netdirect PRD_TV=0 PRD_NETR=32`: one pass of the phase predictor -> hypothesis with the best
low-frequency magnitude match -> coarse image = inverse transform of (measured magnitude x predicted phase) -> 4 / 8 model calls.
Results without the shared components: 22.87/0.385, 23.72/0.345, no failures. With noise tilt, tuned schedule and the band combination: `attempt2.py`.

## 3. Nonlinear deblur (learned blur network)

**Attempt 1 (surrogate + correction).** `WARMX=lin NLKER=1 MEC=step [FW=1 FWP=0.5 FWMIN=0.4]`: a linear filter `K` (3x3 colour mixing per
frequency) is fitted to the network on 8 calibration images; each step solves the linear problem in closed form against the measurement
corrected by the true operator's mismatch at the current image (`y - (A(v) - K v)`, one operator call per step). Result: 26.35 / 0.333.

**Attempt 2 (mapping into the class: "linear filter, then a per-pixel curve", `y ~ g(K x)`).** Tested with `NLDIAG=1`: on 8 held-out
calibration images the linear filter leaves 7.6% of the operator output unexplained, and adding a fitted monotone per-pixel curve leaves 7.6%
(no reduction). The network's departure from linearity is not coordinate-wise, so this mapping gives nothing; attempt 1 stands.
Consequence for the paper: nonlinear deblur is "92% linear" and is handled as a tier-1 linear problem with a model-error correction; it is the
one task whose nonlinearity is not in the class.
