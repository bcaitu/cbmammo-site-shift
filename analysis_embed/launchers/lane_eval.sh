#!/bin/bash
# evaluation tables + CB-vs-opaque comparison + oracle decomposition for the 12 EMBED-inclusive runs (CPU only)
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
mkdir -p results/embed
RUNS=""; CMP=""
for bb in dinov2 ft2; do for s in 0 1 2; do
  RUNS="$RUNS runs/cb_embed_${bb}_s$s runs/opq_embed_${bb}_s$s"
  CMP="$CMP runs/cb_embed_${bb}_s$s:runs/opq_embed_${bb}_s$s"
done; done
echo "== evaluate $(date)"
python -u -m cbmammo.evaluate --runs $RUNS --out results/embed --bootstrap 300 --compare $CMP 2>&1 | grep -vi warn || true
for bb in dinov2 ft2; do for s in 0 1 2; do
  echo "== oracle $bb s$s $(date)"
  python -u -m cbmammo.oracle_gap --run runs/cb_embed_${bb}_s$s --out results/embed/oracle_${bb}_s$s.csv --bootstrap 200 2>&1 | grep -vi warn || true
done; done
echo "EVAL_LANE_FINISHED $(date)"
