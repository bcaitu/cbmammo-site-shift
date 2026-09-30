# cbmammo-site-shift

Code and aggregate results for

> Orazayev Y., Abdikenov B. **Diagnosing and Repairing Site Shift in a BI-RADS Concept Bottleneck Model for Mammography: A Multi-Cohort Evaluation.** Manuscript in preparation (MDPI). Astana IT University, Astana, Kazakhstan.

The study trains a descriptor-level concept bottleneck model (CBM, 30 BI-RADS concepts) and a matched opaque model on two-view mammograms (CBIS-DDSM, VinDr-Mammo, EMBED), then asks (i) how much accuracy the bottleneck costs zero-shot, (ii) how much of the external error arises before the bottleneck (oracle-concept decomposition) and (iii) whether refitting only the 30-input head on a few dozen local patients repairs the model, compared with matched opaque baselines (head refit, partial fine-tuning). External targets: CDD-CESM, CMMD, EMBED patients held out by imaging vendor, and a Kazakhstani clinic (MMC Astana).

## What is in this repository

| Path | Content |
|---|---|
| `cbmammo/` | model, training, evaluation, oracle decomposition, few-shot repair, dataset builders (`cbmammo/prepare`), report extractor used by the CDD-CESM builder |
| `configs/` | `cb_embed_dinov2.yaml`, `opq_embed_dinov2.yaml` (EMBED-inclusive models, the paper's main results), `cb_dinov2.yaml`, `opq_dinov2.yaml` (trained without EMBED, ablation), `smoke.yaml` |
| `analysis_embed/` | post-training analysis: `evaluate_rest.py`, `vendor_split.py`, MMC Astana packing/evaluation (`mmc_pack.py`, `mmc_eval.py`) and the shell launchers that were used (`launchers/`, site-specific) |
| `results/tables/` | aggregate result tables behind every number in the manuscript (no patient-level data) |
| `manuscript_tools/` | scripts that turn `results/tables/` into the manuscript tables and figures |
| `scripts/make_synthetic.py`, `tests/` | synthetic-data smoke test |

## What is not included

- **Image data and manifests.** Datasets are distributed by their owners under their own terms (table below). Manifests are rebuilt from the raw data with `cbmammo.prepare`.
- **MMC Astana data.** The clinic's de-identified mammograms and registry labels cannot be shared. The code that produced the reported aggregate results (`mmc_pack.py`, `mmc_eval.py`, `cbmammo/dump_new.py`) is included; the aggregate outputs are in `results/tables/mmc_*.csv`.
- **Model weights and training runs.** Not released here.
- **Report-generation stage and temporal models** of the same code base, which this paper does not use.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| CBIS-DDSM | TCIA | cite Lee et al. 2017 and the TCIA paper (Clark et al. 2013) |
| VinDr-Mammo | PhysioNet | credentialed access |
| EMBED | Emory / AWS Open Data | open-data release used; data-use terms apply |
| CMMD | TCIA | cite Cui et al. 2021 and the TCIA paper |
| CDD-CESM | Khaled et al. 2022 | public |
| INbreast | Moreira et al. 2012 | obtained from the original authors |

**EMBED split.** Patients are assigned to train / validation / test by a hash of the patient id (10% / 15% held out). Every patient with at least one non-Hologic 2D examination is moved, with *all* of their records, to a vendor-holdout test set (`test_embed_ge`, `test_embed_fuji`). The holdout is therefore defined by patient, not by examination: most malignancy-labelled breasts in `test_embed_ge` were nevertheless imaged on Hologic systems (131 GE, 186 Hologic, 7 Fujifilm of 324; `analysis_embed/vendor_split.py`). Malignancy labels exist only for biopsied breasts.

## Environment

Python 3.10.12, PyTorch 2.6.0 (CUDA 12.4), torchvision 0.21.0, timm 1.0.30, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, joblib 1.6.0 (see `requirements.txt`). Training used NVIDIA H200 GPUs; head refits, oracle decomposition and most tables run on CPU.

```
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Reproducing the analysis

1. **Manifests** (one per dataset; check the printed drop counts):
   ```
   python -m cbmammo.prepare cbis_ddsm --root /data/CBIS-DDSM   --out manifests/cbis_ddsm.jsonl
   python -m cbmammo.prepare vindr     --root /data/vindr-mammo --out manifests/vindr.jsonl
   python -m cbmammo.prepare embed     --root /data/EMBED       --out manifests/embed.jsonl --kw legend_csv=/data/EMBED/tables/clinical_legend.csv path_prefix=/data/EMBED
   python -m cbmammo.prepare cmmd      --root /data/CMMD        --out manifests/cmmd.jsonl
   python -m cbmammo.prepare cdd_cesm  --root /data/CDD-CESM    --out manifests/cdd_cesm.jsonl
   python -m cbmammo.prepare inbreast  --root /data/INbreast    --out manifests/inbreast.jsonl
   python -m cbmammo.prepare count manifests/*.jsonl     # images per descriptor
   ```
   Adjust the paths; `cache_dir` in the configs (default `cache/prep`) holds preprocessed images. Items marked `VERIFY` in `cbmammo/prepare/builders.py` depend on the exact layout of each dataset release.
2. **Training** (3 seeds x {CBM, opaque} x {frozen, tuned encoder}; each run writes `runs/<name>_s<seed>/`):
   ```
   # frozen DINOv2-B/14
   python -m cbmammo.train_stage1 --config configs/cb_embed_dinov2.yaml  --seed 0
   python -m cbmammo.train_stage1 --config configs/opq_embed_dinov2.yaml --seed 0
   # last two blocks tuned (learning-rate multiplier 0.1)
   python -m cbmammo.train_stage1 --config configs/cb_embed_dinov2.yaml  --seed 0 --override name=cb_embed_ft2  encoder.trainable_blocks=2 train.enc_lr_mult=0.1
   python -m cbmammo.train_stage1 --config configs/opq_embed_dinov2.yaml --seed 0 --override name=opq_embed_ft2 encoder.trainable_blocks=2 train.enc_lr_mult=0.1
   ```
   Models without EMBED (ablation): `configs/cb_dinov2.yaml`, `configs/opq_dinov2.yaml` with the same overrides (`name=cb_ft2` ...). Key training settings are in the configs: AdamW, lr 3e-4, weight decay 0.05, batch 16, at most 20 epochs, patience 4, `embed_neg_ratio: 1.0` (fresh 1:1 resampling of EMBED BI-RADS-1 breasts each epoch), `dataset_balanced_val: true`.
3. **Evaluation tables, CB-vs-opaque comparison, oracle decomposition:** `analysis_embed/launchers/lane_eval.sh` (uses `cbmammo.evaluate`, `cbmammo.oracle_gap`; `analysis_embed/evaluate_rest.py` is a faster equivalent of the linear-probe/subgroup/compare stages).
4. **Vendor stratification:** `python analysis_embed/vendor_split.py`.
5. **Few-shot repair** (30 draws per seed; head refits use stored dumps, `--partial_ft` needs images and a GPU): `analysis_embed/launchers/lane_fs.sh`, `lane_fs_embed.sh`, `pft_job.sh`. Example:
   ```
   python -m cbmammo.fewshot --cb runs/cb_embed_dinov2_s0 --opaque runs/opq_embed_dinov2_s0 --set test_cdd_cesm \
       --ks 10,25,50,100,200 --draws 30 --partial_ft --partial_ft_ks 25,100 --partial_ft_draws 10 --device cuda --out results/embed/fewshot.csv
   ```
   Tables 5-6 of the manuscript use the run launched by `pft_job.sh` (`--seed 1`, 30 partial fine-tuning draws); the curves in Figure 2 use the earlier run with different draw seeds. Overlapping means differ by 0.004 on average and by at most 0.025, which indicates the draw-to-draw noise of the protocol.
6. **Local cohort:** `analysis_embed/launchers/lane_mmc.sh` (requires the clinic data; not distributed). `mmc_eval.py` excludes the second study of each of three pixel-identical pairs; pass their pseudonymous ids with `MMC_DROP_STUDIES=id1,id2,id3` (they are not stored in the public code).
7. **Manuscript tables and figures** from the aggregate tables:
   ```
   python -c "exec(open('manuscript_tools/paper_tables.py').read()); print(list(tabs))"
   python manuscript_tools/fig_fewshot_public.py
   (cd results/tables && python ../../manuscript_tools/make_mmc_latex.py && python ../../manuscript_tools/mmc_figure.py)
   ```
   The launchers are written for one server (`~/envs/cbmammo`, `~/cbmammo`, GPU ids); adapt them before use.

## Smoke test (CPU, synthetic images, no downloads)

```
python scripts/make_synthetic.py
python -m cbmammo.train_stage1 --config configs/smoke.yaml
python -m cbmammo.evaluate --runs runs/smoke_cb_s0 --out results_smoke
python -m pytest -q tests
```

## Results tables

`results/tables/` holds: zero-shot AUC (`A_malignant_auc.csv`, `A2_cb_minus_opaque_auc.csv`), oracle decomposition (`B_oracle_*.csv`), few-shot repair (`C_fewshot_*.csv`, `E_pft_*.csv`), ablation without EMBED (`D_ablation_*.csv`), concept-level metrics (`concepts_all.csv`, `compare_all.csv`), vendor stratification (`vendor_split_zero_shot.csv`) and MMC Astana aggregates (`mmc_*.csv`, `mmc_cohort.json`). All rows are aggregates over patients or draws; none is patient-level.

## Citation and licence

See `CITATION.cff`. Code licence: Apache-2.0 (see `LICENSE`). The datasets keep their own licences and data-use terms (see the table above).
