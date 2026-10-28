# Revision analyses (October 2026)

Scripts and aggregate result tables of the analyses added in the revision of the manuscript
"Diagnosing Site Shift and Testing Head-Only Repair in a BI-RADS Concept Bottleneck Model for Mammography: A Multi-Cohort Evaluation".
They use the stored outputs of the models trained for the first version (no model was retrained, except for the partial fine-tuning baseline in E).

| Script | Analysis | Output tables (`results/`) |
|---|---|---|
| `rev1_A.py` | Zero-shot CBM minus opaque model, seed-ensembled models, paired patient bootstrap | `A_*.csv` |
| `rev1_B.py` | Controlled concept substitution (predicted, hard, oracle) on complete-label subsets; BI-RADS strata; label coverage | `B_*.csv`, `B2_*.csv`, `B3_*.csv` |
| `rev1_C.py` | Form of the head: source-fitted linear head, deployed MLP, MLP with shrinkage, on repeated target draws | `C_*.csv` |
| `rev1_D.py` | Nested resampling of target patients and training seeds for the few-shot contrasts | `D_*.csv` |
| `rev1_E.py` | Partial fine-tuning baseline (last block and fusion layers), sensitivity to learning rate, epochs and blocks, with cost | `E_pft_*.csv` |
| `rev1_T.py` | CPU timing of the head refits | `E_cpu_timing_head_refits.csv` |
| `build_tables.py` | Builds the manuscript tables from the result tables | `E_pft_paired_with_headrefit.csv` |

`rev1_common.py` holds the shared loaders. The scripts read the per-patient score and feature dumps of the trained models, which are not released
(patient privacy for the Kazakhstani data, size for the public sets); the tables in `results/` contain aggregate values and per-draw AUCs only, no patient identifiers.
Paths inside the scripts point to the authors' server and have to be changed to rerun them.
The code of `cbmammo/` in this repository is the one the scripts import.
