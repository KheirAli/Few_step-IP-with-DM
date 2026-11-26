#!/bin/bash
# bedpr_retry.sh <min free MiB> <command...> : run the command (from PnP-CM/) on the GPU with the most free memory; rerun (up to 30 times) if it fails, e.g. out of memory
cd /usr/scratch/akheirandish3/PNP/PnP-CM; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; need=$1; shift
for try in $(seq 1 30); do g=$(bash ../mode_probe/bedpr_pick.sh $need); echo "RETRY try $try gpu $g: $*"; CUDA_VISIBLE_DEVICES=$g "$@" && exit 0; sleep $((20 + RANDOM % 20)); done; echo "RETRY GAVE UP: $*"; exit 1
