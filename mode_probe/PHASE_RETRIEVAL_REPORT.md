# Few-step phase retrieval with a consistency model: status report (2026-11-19)

All numbers: CelebA-HQ 256, the first 107 of the PnP-CM paper's 300 validation images, PSNR / LPIPS (VGG), one scoring code for
every method. Measurement noise 0.05 on the [0,1] magnitudes (unmasked task) or 0.1 on the [-1,1] scale (masked task).
Unmasked results are averaged over 3 measurement-noise draws (321 runs); differences are paired, uncertainty is over images.

## 1. Two tasks

| | Masked (PnP-CM paper) | Unmasked (DPS / DAPS) |
|---|---|---|
| Measurement | magnitudes of the FFT of the image times a known +-1 mask that keeps 1 pixel in 4 | magnitudes of the FFT of the zero-padded image (384 frame), no mask |
| What makes it hard | sign per channel | everything: the magnitudes do not single out an image |

## 2. Masked phase retrieval: solved by reduction to inpainting

Classical HIO on the known support recovers the kept pixels (4 starts x 1000 iterations, FFTs only); the rest is linear
inpainting, solved by the same closed-form 4-step sampler as every other task.

| Method | Model calls | PSNR / LPIPS |
|---|---|---|
| PnP-CM (rerun) | 8 | 22.93 / 0.463 |
| DPS | 1000 | 24.12 / 0.329 |
| **Ours** | **4** | **25.47 / 0.431** |
| DAPS | ~1000 | 27.38 / 0.298 |

Time per image 1.26 s against 14.8 s for PnP-CM. Eight steps make ours worse (24.48); keep 4.

## 3. Unmasked phase retrieval

### 3.1 Pipeline

1. **Regression start.** Ridge regression from low-frequency log-magnitudes to 512 eigenface coefficients. Eigenfaces and
   regression are built only from samples of the consistency model.
2. **Fit.** L-BFGS on the coefficients against the measured magnitudes (64 -> 512 components), with a constant offset that
   absorbs the average mismatch of the subspace. 16 starts around the regression prediction, fitted in one batch; the best
   magnitude fit is kept.
3. **Classical plug-and-play loop.** 40 iterations of: noise-aware magnitude fit (exact per-coefficient minimiser, phase kept),
   then a total-variation denoiser.
4. **Sampler.** 4 or 8 consistency-model calls; the data step is the same noise-aware magnitude fit.

Steps 1-3 use no model calls and no real images.

### 3.2 Results

| Method | Model calls | PSNR / LPIPS | below 14 dB | time / image |
|---|---|---|---|---|
| DPS | 1000 | 13.52 / 0.638 | | |
| Ours, 4-start fit only (state on 11-18 morning) | 4 / 8 | 20.46 / 21.61 | 2.2% | 2.0 s |
| + constant offset | 4 / 8 | 20.95 / 22.05 | 1.9% | |
| + denoiser loop | 4 / 8 | 22.37 / 23.23 | 2.2% | 2.3 s |
| + 16 starts, batched (**training-free method**) | 4 / 8 | **23.14 / 0.387** / **24.07 / 0.344** | 0.9% | 2.3 / 2.4 s |
| + learned phase predictor in place of the starts | 4 / 8 | **23.75 / 0.370** / **24.48 / 0.333** | 0.0% | 2.3 / 2.5 s |
| DAPS, one run | ~1000 | 25.84 / 0.308 | | |
| DAPS, best of 4 chosen with ground truth | ~4000 | 29.44 | | |

Learned predictor against the training-free method: +0.60 +- 0.24 dB at 4 steps, +0.41 +- 0.27 dB at 8 steps.

### 3.3 The learned predictor

A network (81M parameters) maps the autocorrelation and low-frequency log-magnitudes to 8 face hypotheses at 64x64. It is
trained only on 256,000 samples of the consistency model (1.8 h, one GPU). For each hypothesis, its phases inside a
low-frequency disc (radius 32 of 192) are used as soft complex targets in the eigenface fit, followed by a free refit, so
predicted phases are never locked in. Best magnitude fit wins. The network takes 1 ms.

Scaling: 8k samples / small network 18.1 dB (low-resolution face, held-out); 64k / small network 21.4 dB (saturated);
256k / larger network 24.1 dB.

Do NOT add regression starts to the learned hypotheses: the magnitude loss then prefers the regression candidates and the gain
disappears (23.18 / 24.09).

### 3.4 Diagnostics that explain the remaining gap

- **Oracle low-frequency phases** (48 non-test images, coarse PSNR, from 21.4 dB): true phases inside radius 4 / 8 / 16 / 32
  give 22.2 / 23.1 / 24.5 / 26.4 dB. Radius 4 (about 50 phases per channel) is enough to rescue the failed images.
- **Failures are wrong low-frequency basins**: phase accuracy of the coarse face in the 2-4 band is 0.96 on typical images and
  0.30 on failed ones. More starts or the predictor fix them.
- **The refinement stage works**: given the best face the subspace can represent (23.1 dB) the sampler reaches 26.6 dB in 8 steps.
- **Selection is not the bottleneck**: the magnitude loss picks within 0.3 dB of the best candidate among regression starts.

### 3.5 What was tried and did not help

| Idea | Outcome |
|---|---|
| Classical HIO / mean-face phases as the start | 13-14 dB |
| Coarse-to-fine, frequency weighting, anchoring, mismatch prediction, local PCA, full-resolution linearised correction | no gain |
| BM3D in place of total variation | worse alone; +0.3 dB at best in combination; 100x slower |
| Image-prior score for candidate selection | no gain over magnitude loss |
| Two 4-step runs + selection by fit (8 calls) | 23.25, i.e. -0.82 dB against one 8-step run |
| One model call inside the coarse stage | level at 4 calls, -0.22 dB at 8 |
| Small phase network (64k samples) | ties 16 plain starts |

## 4. Caveats

- 107 of the paper's 300 images; one sampler seed.
- DPS and DAPS use an FFHQ prior and their own settings, not tuned for this noise level.
- The 8-step schedule and the loop settings were chosen on images outside the test set.
- With the learned predictor the unmasked method is no longer training-free (it still uses no real images).

## 5. Reproduce (from `PNP/PnP-CM`)

- Training-free, 4 and 8 steps: `ONLY="b16" ../mode_probe/launch_pall.sh`  (runs `fin_pa_b16_{4,8}_m{0,1,2}`)
- Learned predictor: `../mode_probe/launch_pnet2.sh`  (runs `fin_pb_n2r32_{4,8}_m{0,1,2}`)
- Predictor training: `VAL=<non-test images> python ../mode_probe/phase_net2.py`; samples from `gen_bank.py` in `mode_probe/bank64k/` (48 GB)
- Diagnostics: `oracle_phase.py`, `cand_rank.py`, `bm3d_sel.py`, `phase_net.py`
- Sampler options: `PRD_INIT=regfit4|netfit`, `PRD_NSTART`, `PRD_BATCH`, `PRD_OFFSET`, `PRD_TV`, `PRD_NET`, `PRD_NETR`, `PRD_TOPK`, `PRD_CMC`
