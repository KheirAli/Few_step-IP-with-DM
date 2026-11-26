#!/bin/bash
# bedpr_prefill.sh <free|net> <shard a:b> : compute and store the start stage of generated bedroom images (data_bedgen); sampler output is discarded
cd /usr/scratch/akheirandish3/PNP/PnP-CM; kind=$1; sh=$2; L=../mode_probe
B="NEWOP=prdaps PRD_K=0 PRD_FINAL=0 PRD_SOFT=1 PRD_BATCH=1 PRD_OFFSET=1 PRD_TV=80"
[ $kind = free ] && C="$B PRD_INIT=regfit4 PRD_NSTART=16" || C="$B PRD_INIT=netfit PRD_NET=v2k8 PRD_NETREG=0 PRD_NETR=48"
tag=fin_bedprpre_${kind}_${sh%%:*}
bash $L/bedpr_retry.sh 8000 env DATA=$(realpath $L/data_bedgen) SHARD=$sh SAMPLE_SEED=1 RESTART=never NOISE=white NOLOG=1 MODE=unified TILT=0.5 DSET=bedroom MEANFACE=mean_bed_gen.pt LLE=/nonexistent_weights.json $C PRD_W0CACHE=$(realpath $L)/bedpr_w0/gen_$kind SIGS=1.0000,0.3107,0.0965,0.0300 ../pnpcm_env/bin/python $L/unified.py $tag --deg=deblur_nl
rm -rf $L/exp_${tag}_s${sh%%:*}; echo PREFILL DONE $kind $sh
