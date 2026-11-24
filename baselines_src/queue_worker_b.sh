#!/bin/bash
# DPS and DAPS on LSUN BEDROOM with a bedroom diffusion prior (google/ddpm-ema-bedroom-256).  One worker per GPU.   jobs: "DPSC <cfg>" | "DAPSC <task> [n01]"
gpu=$1; B=/usr/scratch/akheirandish3/PNP/baselines_src; PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python; export CELEBA_HF=1 HF_DM_PATH=$B/ckpt_bedroom/ddpm-ema-bedroom-256
while true; do
  job=$(flock $B/queue_b.lock bash -c "head -1 $B/queue_jobs_b.txt; sed -i 1d $B/queue_jobs_b.txt"); [ -z "$job" ] && break
  set -- $job
  if [ "$1" = DPSC ]; then cd $B/diffusion-posterior-sampling; CUDA_VISIBLE_DEVICES=$gpu $PY sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config_fixed.yaml --task_config=cfg_bed/$2.yaml --gpu 0 --save_dir results_bedroom/$2 > logb_$2.out 2>&1
  else cd $B/DAPS; extra=""; name=$2; [ "$3" = n01 ] && extra="task.pixel.operator.sigma=0.1" && name=${2}_n01
    CUDA_VISIBLE_DEVICES=$gpu $PY posterior_sample.py +data=test-ffhq +model=celebahq_hf +task=$2 +sampler=edm_daps task_group=pixel save_dir=results_bedroom num_runs=1 sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=12 data.root=dataset/paper-bedroom data.start_id=0 data.end_id=58 name=$name gpu=0 save_traj=False $extra > logb_$name.out 2>&1
  fi
done
