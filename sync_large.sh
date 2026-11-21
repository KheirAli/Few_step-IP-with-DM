#!/bin/bash
# Copy the files that are too large for git from the original server.   usage: ./sync_large.sh user@host [all]
# default: what the 4-call sampler needs for every task (about 2.6 GB). "all" adds the phase-predictor networks and MRI models (about 5 GB more).
set -e; SRC=$1; OLD=/usr/scratch/akheirandish3/PNP; ROOT=$(cd "$(dirname "$0")" && pwd)
[ -z "$SRC" ] && { echo "usage: $0 user@host [all]"; exit 1; }
R="rsync -avP"
$R $SRC:$OLD/PnP-CM/exp/logs/celeba_hq/ $ROOT/PnP-CM/exp/logs/celeba_hq/                 # consistency model, 2.1 GB
$R $SRC:$OLD/PnP-CM/bkse/experiments/pretrained/ $ROOT/PnP-CM/bkse/experiments/pretrained/   # nonlinear-blur network
$R --exclude image_samples --exclude logs --exclude datasets $SRC:$OLD/PnP-CM/exp/ $ROOT/PnP-CM/exp/   # masks, kernels
$R $SRC:$OLD/mode_probe/pca_cm.pt $ROOT/mode_probe/                                      # eigenfaces (unmasked phase retrieval, 30-dim weights), 385 MB
$R $SRC:$OLD/mode_probe/lle/eig30_basis.pt $ROOT/mode_probe/lle/
if [ "$2" = "all" ]; then
  $R $SRC:$OLD/mode_probe/phase_net_k8biglong.pt $SRC:$OLD/mode_probe/phase_net_v2k8.pt $ROOT/mode_probe/
  $R $SRC:$OLD/PnP-CM/exp/logs/fast_mri/ $ROOT/PnP-CM/exp/logs/fast_mri/
  $R $SRC:$OLD/baselines_src/ckpt_celeba/ $ROOT/baselines_src/ckpt_celeba/
fi
