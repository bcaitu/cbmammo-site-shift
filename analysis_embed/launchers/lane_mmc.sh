#!/bin/bash
# MMC Astana: inference of the 12 EMBED-inclusive runs -> pack -> eval. Patient-level data never leaves the server.
gpu=$1
source ~/envs/cbmammo/bin/activate; cd ~/cbmammo
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
L=~/local_kz; E=~/local_kz_embed
mkdir -p $E/dumps; chmod 700 $E $E/dumps
for x in manifests mmc_astana cache_mmc; do [ -e $E/$x ] || ln -s $L/$x $E/$x; done
for arm in cb opq; do for bb in dinov2 ft2; do for s in 0 1 2; do
  run=${arm}_embed_${bb}_s$s; al=${arm}_${bb}_s$s     # alias keeps the arm_backbone_seed pattern that mmc_eval.py parses
  t0=$(date +%s)
  python -u -m cbmammo.dump_new --run runs/$run --manifest $L/manifests/mmc_astana.jsonl --name test_mmc_astana \
     --cache_dir $L/cache_mmc --out_dir $E/dumps/$al --workers 8 > $E/dumps/log_$al.txt 2>&1
  echo "$run -> $al rc=$? $(( $(date +%s)-t0 ))s"
done; done; done
python ~/cbmammo/analysis_embed/mmc_pack.py $E $E/mmc_bundle_embed.npz
chmod 600 $E/mmc_bundle_embed.npz
python ~/cbmammo/analysis_embed/mmc_eval.py --bundle $E/mmc_bundle_embed.npz --out $E/results_mmc --draws 30 --boot 2000 --jobs 16 2>&1 | tail -n 60
echo "MMC_LANE_FINISHED $(date)"
