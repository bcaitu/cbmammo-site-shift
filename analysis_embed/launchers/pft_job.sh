#!/bin/bash
# pft_job.sh <gpu> <backbone> <set> <seed> : few-shot with 30 partial fine-tune draws (fresh draws, --seed 1)
gpu=$1; bb=$2; st=$3; s=$4
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu
mkdir -p results/embed/pft logs/analysis
out=results/embed/pft/fewshot_pft_${bb}_${st}_s${s}.csv
[ -s "$out" ] && { echo "skip $out"; exit 0; }
sleep $((RANDOM % 25))
for attempt in 1 2; do
  echo "== pft $bb $st s$s gpu$gpu attempt$attempt $(date)"
  python -u -m cbmammo.fewshot --cb runs/cb_embed_${bb}_s${s} --opaque runs/opq_embed_${bb}_s${s} --set $st \
    --ks 25,100 --draws 30 --seed 1 --partial_ft --partial_ft_ks 25,100 --partial_ft_draws 30 --device cuda \
    --out $out > logs/analysis/pft_${bb}_${st}_s${s}.log 2>&1 && break
  sleep 20
done
[ -s "$out" ] && echo "== OK $bb $st s$s $(date)" || echo "== FAILED $bb $st s$s $(date)"
