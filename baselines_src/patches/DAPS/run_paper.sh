#!/bin/bash
# DAPS (upstream code, FFHQ pixel model, its own operator definitions and default DAPS-1K settings) on the PnP-CM paper's validation images.
cd /usr/scratch/akheirandish3/PNP/baselines_src/DAPS
PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python; N=${N:-107}
run(){ gpu=$1; task=$2; runs=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY posterior_sample.py +data=test-ffhq +model=ffhq256ddpm +task=$task +sampler=edm_daps task_group=pixel save_dir=results_paper num_runs=$runs \
    sampler.diffusion_scheduler_config.num_steps=5 sampler.annealing_scheduler_config.num_steps=200 batch_size=12 data.root=dataset/paper-celeba data.start_id=0 data.end_id=$N \
    name=$task gpu=0 save_traj=False > log_$task.out 2>&1; }
run 0 down_sampling 1 & run 1 gaussian_blur 1 & run 2 inpainting_rand 1 & run 3 nonlinear_blur 1 & run 4 hdr 1 & run 5 phase_retrieval 4 &
wait; echo ALLDONE
