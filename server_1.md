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

## Job 2: (will be added here by server 2)

Check this file again after Job 1 with `git pull`.
