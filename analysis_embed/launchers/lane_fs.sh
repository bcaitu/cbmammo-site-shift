#!/bin/bash
# few-shot site repair for one backbone (dinov2|ft2), seeds 0-2, on GPU $2
bb=$1; gpu=$2
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu
mkdir -p results/embed
for s in 0 1 2; do
  for set in test_cdd_cesm test_cmmd; do
    echo "== fewshot $bb s$s $set (+partial_ft) $(date)"
    python -u -m cbmammo.fewshot --cb runs/cb_embed_${bb}_s$s --opaque runs/opq_embed_${bb}_s$s --set $set \
      --ks 10,25,50,100,200 --draws 30 --partial_ft --partial_ft_ks 25,100 --partial_ft_draws 10 --device cuda \
      --out results/embed/fewshot_${bb}_${set}_s$s.csv 2>&1 | grep -vi warn | tail -n 12 || true
  done
  for set in test_embed_ge test_embed; do
    echo "== fewshot $bb s$s $set $(date)"
    python -u -m cbmammo.fewshot --cb runs/cb_embed_${bb}_s$s --opaque runs/opq_embed_${bb}_s$s --set $set \
      --ks 25,50,100,200,400 --draws 30 --device cuda \
      --out results/embed/fewshot_${bb}_${set}_s$s.csv 2>&1 | grep -vi warn | tail -n 12 || true
  done
done
echo "FS_LANE_FINISHED $bb $(date)"
