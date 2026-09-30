#!/bin/bash
# few-shot on EMBED sets with the corrected k grid (only ~245 patients carry a malignant label in each set)
bb=$1; gpu=$2
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu
for s in 0 1 2; do
  for set in test_embed_ge test_embed; do
    echo "== fewshot $bb s$s $set (+partial_ft) $(date)"
    python -u -m cbmammo.fewshot --cb runs/cb_embed_${bb}_s$s --opaque runs/opq_embed_${bb}_s$s --set $set \
      --ks 10,25,50,100 --draws 30 --partial_ft --partial_ft_ks 25,100 --partial_ft_draws 10 --device cuda \
      --out results/embed/fewshot_${bb}_${set}_s$s.csv 2>&1 | grep -vi warn | tail -n 12 || true
  done
done
echo "FS_EMBED_LANE_FINISHED $bb $(date)"
