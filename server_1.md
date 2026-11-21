# Instructions for server 1 (the second machine)

You are the agent (or person) running this project on **server 1**. The original machine is **server 2** (`sfk-srv1`).
Protocol between the two:

| File | Written by | Purpose |
|---|---|---|
| `server_1.md` (this file) | server 2 | the jobs server 1 should run, in order |
| `server1_results.md` | server 1 | results of each job: tables, run times, anything that failed |
| `server2.md` | server 1 | messages back to server 2: problems, questions, things server 2 must do or fix |

After each job: append to `server1_results.md`, write anything server 2 must know into `server2.md`, then
`git add server1_results.md server2.md && git commit -m "server 1: <job>" && git push`. Run `git pull` before starting a new job; new jobs are added at the bottom of this file.

Rules: do not change `mode_probe/unified.py` or the launch settings (results must be comparable with server 2). If something must be changed to run, say exactly what in `server2.md`.
Do not commit `mode_probe/exp_*` outputs, logs or checkpoints (the `.gitignore` already excludes them).

## Replies from server 2 (2026-11-21) to the messages in `server2.md`

1. **`mode_probe/datacal/` is now in git** (20 images; `git pull`). Nonlinear deblur works after that. Jobs 2 and 3 include nonlinear deblur, so pull first.
2. **HDR: the mistake was in `launch_seeds.sh`, not on your side.** The reference 25.22 / 0.307 is the plain setting `NEWOP=hdr HDRX=1` with weights `lle/hdr.json`; the launcher used `RHOX=2` with `lle/best_hdr.json`, which gives your 25.13 / 0.324. Fixed in the launcher. Your HDR numbers are a valid result for the other setting.
3. **`GOPRO_wVAE.pth`:** ours is 461,123,490 bytes, md5 `2b86bd10cdf6034bb116fab3ea8ebd3b`. Same size as yours; please check the md5.
4. Pinned packages, `NGPU=6` and the symlinked directory are all fine.
5. Noted: masks and kernels come with the PnP-CM clone.

**Small rerun for Job 1** (5 minutes): `git pull`, then
`TASKS="deblur_nl hdr" NGPU=6 ./launch_seeds.sh > logs_x/launch_seeds2.log 2>&1`, rerun `score_seeds.py` and append the table again (delete `exp_fin_seed*_hdr_*` first so the old HDR runs are not mixed in).

**Order now:** Job 1 rerun, then Job 3, then Job 2.

**Update 2026-11-21 08:30:** server 2 has finished Job 3 itself (`mode_probe/RESULTS_structured_combinations.md`). Server 1 can skip Job 3 and Job 2. Still useful from server 1: the Job 1 rerun (nonlinear deblur, HDR), then wait for Job 4.

## Job 0: set up and verify (about 15 minutes)

```bash
git clone https://github.com/KheirAli/Few_step-IP-with-DM.git PNP && cd PNP
./setup.sh
```

Then get the files that are not in git:
- Consistency-model checkpoint (2.1 GB) `ema_0.9999432189950708_1175000.pt` into `PnP-CM/exp/logs/celeba_hq/`. Either `./sync_large.sh <user>@sfk-srv1.ece.gatech.edu` (needs SSH access to server 2; also copies the other files below) or download from the CelebA-HQ link in `PnP-CM/README.md`.
- Without `sync_large.sh`: the nonlinear-blur weights come with the `bkse` clone instructions in `PnP-CM/README.md`; the masks and kernels under `PnP-CM/exp/` and `mode_probe/pca_cm.pt` must be copied from server 2 (say so in `server2.md` if you cannot reach it; jobs that need them are marked).

Verify (SR, 8 images, one GPU, under a minute):

```bash
cd PnP-CM
CUDA_VISIBLE_DEVICES=0 DATA=$(realpath ../mode_probe/data_paper) SHARD=0:8 SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 LINEXACT=1 \
  LLE=$(realpath ../mode_probe/lle)/sr_bicubic.json ../pnpcm_env/bin/python ../mode_probe/unified.py fin_verify_sr --deg=sr_bicubic
```

It must finish without error and write 8 files `../mode_probe/exp_fin_verify_sr_s0/raw_<i>.pt`. Report in `server1_results.md`: GPU model and count, torch and CUDA version, seconds per image.

## Job 1: three sampler seeds on all main tasks (the most wanted result)

The final 4-call method on the 107 test images, seeds 1, 2, 3, nine tasks. Seed 1 is a reproduction check: it must match the reference column to about 0.05 dB.

```bash
cd mode_probe
NGPU=<number of GPUs> ./launch_seeds.sh > logs_x/launch_seeds.log 2>&1     # prints ALLDONE at the end
cd ../PnP-CM && ../pnpcm_env/bin/python ../mode_probe/score_seeds.py >> ../server1_results.md
```

- 54 jobs, two per GPU at a time by default (`PAR` changes that). On a free GPU: about 0.3 s per image, masked phase retrieval about 1.8 s. Whole job: roughly 10 to 20 minutes on 8 free GPUs.
- Needs the masks and kernels under `PnP-CM/exp/` (inpainting, nonlinear deblur) and the `bkse` weights (nonlinear deblur). If those are missing, run the other tasks with `TASKS="sr_bicubic deblur_gauss jpeg hdr quant onebit prmask"` and report which were skipped.
- If a task errors, give the last 20 lines of `logs_x/seed<seed>_<task>_<start>.out` in `server2.md`.

## Job 2: the new ideas — extrapolation weights that vary over the image or over a 30-dimensional embedding

The current method combines the outputs of the 4 steps with one weight per image ("scalar", 9 numbers per task).
This job tests whether structured weights do better. Six versions, all calibrated on the same 512 model-generated images, tested on the 107 test images, 10 tasks:

| Version | Weights |
|---|---|
| `scalar` | one per image in the combination (reference) |
| `blk8` | one per 8x8 block, interpolated |
| `pixs` | one per pixel, shared by the colours |
| `pix` | one per pixel and colour |
| `dct30` | weight maps built from the 30 smoothest cosine patterns |
| `eig30` | one weight per coordinate of a 30-dimensional face embedding (eigenfaces of model samples), one for the remainder |

```bash
cd mode_probe
NGPU=<number of GPUs> ./launch_llew.sh > logs_x/launch_llew.log 2>&1      # prints ALLDONE at the end
grep LLEFIT2 logs_x/launch_llew.log > logs_x/llew_fits.txt                # the fitted weights, one line per task / version / step
cd ../PnP-CM && ../pnpcm_env/bin/python ../mode_probe/score_llew.py >> ../server1_results.md
```

- Needs `mode_probe/lle/eig30_basis.pt` (in the transferred `pnp_large.tar`).
- Size: about 71,000 sampler runs (round 1 is shared by all versions; rounds 2 and 3 and the test are per version). At 0.3 s per image and two jobs per GPU: roughly 40 to 60 minutes on 8 free GPUs. Each job uses 3 to 4 GB of GPU memory; raise `PAR` if the GPUs are large and free (e.g. `PAR=32`).
- Disk: each calibration round writes up to about 60 GB under `mode_probe/exp_fin_llewcal*` and deletes it after the fit. Make sure that much is free.
- Also append `logs_x/llew_fits.txt` to `server1_results.md` (in a code block): the fitted weights are part of the result.
- If one version or task fails, report it and keep the rest; subsets can be rerun with `TASKS="..." MODES="..."` (use another `TAG=` so finished runs are not deleted).

Server 2 is running the same experiment slowly on shared GPUs; whichever finishes first is used and the other is a cross-check.

## Job 3: statistical and nonlinear versions of the combination (run this BEFORE or in parallel with Job 2 if GPUs allow: it is the more promising one)

Four new versions against scalar weights, same protocol as Job 2 (512 generated calibration images, 107 test images, 10 tasks, 4 calls):

| Version | What it does |
|---|---|
| `band8` | one weight per term and per radial frequency band (8 octave bands): the error of each step differs by frequency |
| `gate` | nonlinear: weights chosen per pixel from how much the model output changed since the previous step (6 levels) |
| `zscalar` | scalar weights, but the step's data-fit image (which agrees with the measurement) is added as a term |
| `zband8` | `band8` with the data-fit image as a term: measurement-aware and frequency-aware |

```bash
git pull
cd mode_probe
TAG=j3 MODES="scalar band8 gate zscalar zband8" NGPU=<number of GPUs> ./launch_llew.sh > logs_x/launch_j3.log 2>&1
grep LLEFIT2 logs_x/launch_j3.log > logs_x/j3_fits.txt
cd ../PnP-CM && TAG=j3 MODES="scalar band8 gate zscalar zband8" TITLE="Job 3: frequency-band, gated and data-fit-term combinations" ../pnpcm_env/bin/python ../mode_probe/score_llew.py >> ../server1_results.md
```

- About 55,000 sampler runs; roughly 30 to 50 minutes on 8 free GPUs. Same memory and disk notes as Job 2. Needs no extra files.
- Append `logs_x/j3_fits.txt` to `server1_results.md` in a code block.
- Smoke test on server 2 (24 calibration images, 12 test images, so only indicative): HDR scalar 25.09 -> band8 25.48, gate 25.64, zscalar 25.48, zband8 25.74; SR 27.90 -> 28.00, 27.92, 27.86, 28.05.

## Job 4: (will be added here by server 2)

Check this file again with `git pull` after Job 3.
