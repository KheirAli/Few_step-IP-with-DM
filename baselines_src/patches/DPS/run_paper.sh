#!/bin/bash
# DPS (upstream code, FFHQ model, 1000 steps, its own operators and default step sizes) on the PnP-CM paper's validation images: one task per GPU.
cd /usr/scratch/akheirandish3/PNP/baselines_src/diffusion-posterior-sampling
PY=/usr/scratch/akheirandish3/DAPS_project/DAPS/.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=cfg_paper/$2.yaml --gpu 0 --save_dir results_paper > log_$2.out 2>&1; }
run 0 super_resolution & run 2 gaussian_deblur & run 4 inpainting & run 6 phase_retrieval & run 7 nonlinear_deblur &
wait; echo ALLDONE
