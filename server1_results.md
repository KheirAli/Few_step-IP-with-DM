# Results from server 1

(Server 1 appends here. One section per job from `server_1.md`: what ran, the table, run time, failures.)

## Job 0: set up and verify (2026-11-20)

- Machine: `marigold`, 8 x NVIDIA RTX A6000 (48 GB), driver 580.178.04. GPU 6 is in use by another user and was not used.
- Environment: Python 3.11 (conda), torch 2.11.0+cu128 (CUDA 12.8), torchvision 0.26.0, numpy 2.4.6; the other packages pinned to `requirements_frozen.txt`.
- Files not in git: checkpoint from the Dropbox link in `PnP-CM/README.md`, later confirmed byte-identical to the copy sent from server 2. `GOPRO_wVAE.pth` from the Google Drive link in the bkse README (461,123,490 bytes; not compared with server 2's copy). Masks and kernels came with the PnP-CM clone and are byte-identical to server 2's.
- Verify run (SR, images 0:8, GPU 0): finished without error, 8 `raw_<i>.pt` files written. 0.34 s per image after the first (first image 22.7 s, model load and warm-up). Mean over the 8 images: 27.81 dB / 0.283 LPIPS.

## Job 1: three sampler seeds, 107 test images, 4 calls (PSNR / LPIPS)

| Task | Reference (seed 1, original server) | Seed 1 | Seed 2 | Seed 3 | Per-image sd across seeds (dB) | Images |
|---|---|---|---|---|---|---|
| sr_bicubic | 27.96/0.285 | 27.96/0.285 | 27.97/0.285 | 27.96/0.284 | 0.03 | 107/107/107 |
| deblur_gauss | 29.50/0.246 | 29.50/0.247 | 29.50/0.247 | 29.49/0.246 | 0.03 | 107/107/107 |
| inpainting | 29.87/0.199 | 29.88/0.200 | 29.87/0.200 | 29.87/0.200 | 0.04 | 107/107/107 |
| jpeg | 26.77/0.354 | 26.78/0.354 | 26.78/0.354 | 26.77/0.355 | 0.05 | 107/107/107 |
| deblur_nl | 26.31/0.335 | missing | missing | missing | n/a | 0/0/0 |
| hdr | 25.22/0.307 | 25.13/0.324 | 25.14/0.324 | 25.14/0.325 | 0.05 | 107/107/107 |
| quant | 30.87/0.216 | 30.87/0.216 | 30.88/0.216 | 30.88/0.217 | 0.04 | 107/107/107 |
| onebit | 29.39/0.239 | 29.39/0.239 | 29.41/0.239 | 29.40/0.239 | 0.06 | 107/107/107 |
| prmask | 26.78/0.392 | 26.72/0.395 | 26.71/0.397 | 26.79/0.394 | 0.62 | 107/107/107 |

- Run: `NGPU=6 ./launch_seeds.sh` (GPUs 0-5, 12 jobs at a time), 23 min wall time for all 54 jobs. `unified.py` and the launch settings unchanged.
- Median seconds per image with two jobs per GPU: sr_bicubic 0.49, deblur_gauss 0.34, inpainting 0.35, jpeg 0.57, hdr 0.46, quant 0.56, onebit 0.50, prmask 0.43 (sampler time only; the hybrid input-output initialisation of prmask is not in this number).
- **Failed: deblur_nl, all three seeds, both shards.** `NLKER=1` reads `mode_probe/datacal/celeba_hq/celeba_hq/*.png`, which is not in git, not in `sync_large.sh` and not in the archive sent. Error: `AttributeError: 'int' object has no attribute 'dtype'` at `unified.py:1628` (no calibration images found, so `XX` stays 0).
- **Seed 1 does not reproduce the reference on hdr:** 25.13 / 0.324 here against 25.22 / 0.307. Seeds 2 and 3 give the same (25.14 / 0.324, 25.14 / 0.325), so it is not sampler noise.
- prmask seed 1 is 26.72 against 26.78; the per-image sd across seeds is 0.62 dB, and seed 3 gives 26.79.
- The other six tasks reproduce the reference to 0.01 dB and 0.001 LPIPS.
