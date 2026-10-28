
"""Shared helpers for the Paper 1 (MAKE) revision analyses. Reads stage-1 dumps in ~/cbmammo/runs; no retraining."""
import os, sys, json, glob, time, warnings
START = os.getcwd()
OUT = os.path.join(START, "out"); os.makedirs(OUT, exist_ok=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import rankdata
ROOT = os.path.expanduser("~/cbmammo")
sys.path.insert(0, ROOT); os.chdir(ROOT)
from cbmammo import data as D
from cbmammo.concepts import CONCEPT_HEADS, HEADS, IGNORE
from cbmammo.oracle_gap import Head, HIDX, GROUPS

RUNS = os.path.join(ROOT, "runs")
ENC = {"frozen": ("cb_embed_dinov2", "opq_embed_dinov2"), "tuned": ("cb_embed_ft2", "opq_embed_ft2")}
SEEDS = (0, 1, 2)
SETS = ("cbis_ddsm", "cdd_cesm", "cmmd", "embed", "embed_ge")
GROUP_HEADS = {**{k: list(v) for k, v in GROUPS.items()}, "all": [h.name for h in CONCEPT_HEADS]}

_R = None
def recs():
    global _R
    if _R is None:
        ms = ["manifests/cbis_ddsm.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl"]
        _R = {r["sample_id"]: r for r in D.load_manifest(ms)}
    return _R

_C = {}
def dump(run, seed, name):
    k = (run, seed, name)
    if k not in _C:
        _C[k] = torch.load(f"{RUNS}/{run}_s{seed}/dump_{name}.pt", map_location="cpu")
    return _C[k]

def patients(d):
    R = recs()
    return np.array([R[i]["patient_id"] if i in R else i for i in d["sample_id"]])

def head_for(run, seed):
    return Head(torch.load(f"{RUNS}/{run}_s{seed}/best.pt", map_location="cpu")).eval()

def fast_auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def cluster_index(g):
    ug, inv = np.unique(g, return_inverse=True)
    return [np.flatnonzero(inv == j) for j in range(len(ug))]

def resample(idx_by, rng):
    return np.concatenate([idx_by[j] for j in rng.integers(0, len(idx_by), len(idx_by))])

def ci(vals, q=(2.5, 97.5)):
    v = np.array([x for x in vals if x == x])
    return (float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))) if len(v) else (np.nan, np.nan)

def target_arrays(d, tgt="malignant"):
    y = d["labels"][:, HIDX[tgt]].numpy()
    return y, y != IGNORE

def save(df, name):
    p = os.path.join(OUT, name); df.to_csv(p, index=False); print("saved", p, df.shape, flush=True)
