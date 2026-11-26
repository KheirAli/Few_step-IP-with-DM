#!/bin/bash
# bedpr_fit.sh <tag> : fit the zband8 combination weights of one bedroom configuration on generated images 0:128 with launch_llewd.sh (SKIPTEST),
# one sampler process at a time on the GPU with the most free memory; repeated (up to 4 times) if a round lost images (e.g. out of memory).
cd /usr/scratch/akheirandish3/PNP/PnP-CM; L=../mode_probe; tag=$1
read KEY NSTEP CFGSTR < <(../pnpcm_env/bin/python -c "
exec(open('../mode_probe/bedpr_cfg.py').read())
t, k, c, n, d = [j for j in JOBS if j[0] == '$tag'][0]; print(k, n, c + ''.join(f' {a}={b}' for a, b in W0(t[5:], 'gen').items()))")
[ -z "$KEY" ] && { echo "FIT $tag: no configuration"; exit 1; }
for try in 1 2 3 4; do g=$(bash $L/bedpr_pick.sh 8000)
  DSET=bedroom MEANFACE=mean_bed_gen.pt TAG=$tag KEY=$KEY CFGSTR="$CFGSTR" NSTEP=$NSTEP MODES=zband8 CALSH="0:64 64:128" CALDATA=$L/data_bedgen SKIPTEST=1 NGPU=8 PAR=1 GOFF=$g GPULIST="$g" bash $L/launch_llewd.sh > $L/logs_x/launch_${tag}.log 2>&1
  ok=$(grep -c "LLEFIT2.* n=128:" $L/logs_x/launch_${tag}.log); echo "FIT $tag try $try gpu $g: $ok of $((NSTEP-1)) rounds complete"
  [ "$ok" = "$((NSTEP-1))" ] && [ -f $L/lle/${tag}_zband8_${KEY}.pt ] && { echo "FIT DONE $tag"; exit 0; }; done; echo "FIT FAILED $tag"; exit 1
