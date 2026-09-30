#!/bin/bash
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
mkdir -p results/embed
for bb in dinov2 ft2; do for s in 0 1 2; do echo "$bb $s"; done; done | xargs -P 3 -L 1 bash -c \
  'echo "== oracle $0 s$1 $(date)"; python -u -m cbmammo.oracle_gap --run runs/cb_embed_$0_s$1 --out results/embed/oracle_$0_s$1.csv --bootstrap 200 2>&1 | grep -vi warn | tail -n 3'
echo "ORACLE_LANE_FINISHED $(date)"
