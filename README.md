# Few-step consistency-model solver for inverse problems (built on PnP-CM)

Working repository. Status, results and method description: `mode_probe/PROJECT_SUMMARY.md`,
`mode_probe/PHASE_RETRIEVAL_REPORT.md`.

## Two-server workflow

`server_1.md` = jobs for the second machine; it answers in `server1_results.md` (results) and `server2.md` (messages back).

## What is tracked

| Path | Content |
|---|---|
| `mode_probe/unified.py` | the sampler (all tasks, all options are environment variables) |
| `mode_probe/launch_*.sh`, `score_*.py`, `lle_fit*.py` | experiment launchers, scoring, extrapolation-weight fitting |
| `mode_probe/lle/*.json`, `mode_probe/cal/` | fitted extrapolation weights, calibration tables |
| `mode_probe/data_paper/` | the 107 test images (from the PnP-CM paper's validation set) |
| `mode_probe/data100/` | 100 CelebA-HQ images: 0:24 selection, 24:48 calibration |
| `mode_probe/data_gen/`, `data_gen512/` | 24 / 512 images sampled from the consistency model (calibration without real data) |
| `baselines_src/` | our additions to DPS and DAPS: extra operators, CelebA-HQ prior wrapper, job queue, patches |

Not tracked: upstream code (cloned by `setup.sh`), checkpoints, outputs (`mode_probe/exp_*`), logs.

## Set up on a new server

```bash
git clone https://github.com/KheirAli/Few_step-IP-with-DM.git PNP && cd PNP
./setup.sh                                   # clones PnP-CM (pinned), builds the environment, rewrites absolute paths
./sync_large.sh <user>@sfk-srv1.ece.gatech.edu   # checkpoint and large data files (2.6 GB)
```

`requirements_frozen.txt` lists the exact package versions of the original environment (Python 3.11, torch 2.11 / CUDA 12.8).

## Run

Everything runs from `PnP-CM/`:

```bash
cd PnP-CM
# one task, 4 calls, test images 0..53 on GPU 0 (SR as an example)
CUDA_VISIBLE_DEVICES=0 DATA=$(realpath ../mode_probe/data_paper) SHARD=0:54 SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 \
  MODE=unified TILT=0.5 LINEXACT=1 LLE=$(realpath ../mode_probe/lle)/sr_bicubic.json \
  ../pnpcm_env/bin/python ../mode_probe/unified.py fin_demo_sr --deg=sr_bicubic
# output: ../mode_probe/exp_fin_demo_sr_s0/{raw_<i>.pt, stats.json}
```

Per-task options are the `CFG` table at the top of each launcher (e.g. `mode_probe/launch_llepx.sh`).

Current experiments (weights of the extrapolation that vary over the image):

```bash
cd mode_probe
./launch_llepx.sh > logs_x/launch_llepx.log 2>&1     # scalar / 8x8 blocks / per pixel / per pixel and colour, 512 generated calibration images
./launch_llepy_now.sh > logs_x/launch_llepy_now.log 2>&1   # 30 smooth maps / 30-dimensional face embedding (its GPU list GL=(...) puts most jobs on GPU 4; edit)
```

The launchers assume 8 GPUs (`CUDA_VISIBLE_DEVICES=$((g%8))`, 16 jobs at a time); edit those two numbers for another machine.

## Baselines

```bash
git clone https://github.com/zhangbingliang2019/DAPS.git baselines_src/DAPS && git -C baselines_src/DAPS checkout 0b18272
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git baselines_src/diffusion-posterior-sampling && git -C baselines_src/diffusion-posterior-sampling checkout effbde7
git -C baselines_src/DAPS apply ../patches/DAPS/changes.patch && cp -r baselines_src/patches/DAPS/configs baselines_src/DAPS/
git -C baselines_src/diffusion-posterior-sampling apply ../patches/DPS/changes.patch && cp -r baselines_src/patches/DPS/{cfg_paper,configs} baselines_src/diffusion-posterior-sampling/
```

CelebA-HQ prior: `google/ddpm-celebahq-256` from Hugging Face into `baselines_src/ckpt_celeba/ddpm-celebahq-256`, enabled with `CELEBA_HF=1`
(`baselines_src/hf_wrap.py`). Jobs: `queue_jobs_c.txt` + `queue_worker_c.sh <gpu>`. Both baselines need their own environments (see their READMEs).
