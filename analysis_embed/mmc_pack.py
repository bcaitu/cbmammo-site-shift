"""Pack the 12 MMC inference dumps + per-breast metadata into one compact npz (no pixels, pseudonymous ids only).
Usage: python mmc_pack.py <local_kz_dir> <out_npz>"""
import csv
import json
import os
import sys

import numpy as np
import torch

L, OUT = sys.argv[1], sys.argv[2]
R = [json.loads(l) for l in open(os.path.join(L, "manifests", "mmc_astana.jsonl"))]
by = {r["sample_id"]: r for r in R}
resp = {r["study"]: r for r in csv.DictReader(open(os.path.join(L, "mmc_astana", "response_labels.csv"), encoding="utf-8"))}
out = {}
ids = None
for run in sorted(os.listdir(os.path.join(L, "dumps"))):
    f = os.path.join(L, "dumps", run, "dump_test_mmc_astana.pt")
    if not os.path.isfile(f):
        continue
    d = torch.load(f, map_location="cpu")
    sid = list(d["sample_id"])
    if ids is None:
        ids = sid
    assert sid == ids, run
    for k in ("p_birads", "p_malignant", "p_mass", "p_calc", "cond", "feats"):
        out[f"{run}|{k}"] = d[k].numpy().astype(np.float32)
meta = {k: [] for k in ("sample_id", "study", "side", "group", "patient_group", "device", "study_M", "breast_M",
                        "cancer_breast", "mlo_fallback", "has_cc", "has_mlo", "response", "response_quality",
                        "regression_pct", "month_match")}
for s in ids:
    r = by[s]; m = r["mmc"]; rr = resp.get(r["study_id"], {})
    reg = rr.get("rep_regression_pct") or rr.get("reg_regression_pct") or ""
    for k, v in (("sample_id", s), ("study", r["study_id"]), ("side", r["side"]), ("group", m["group"]),
                 ("patient_group", r["patient_id"]), ("device", m["device"]), ("study_M", m["study_M"] or ""),
                 ("breast_M", "" if m["breast_M"] is None else str(m["breast_M"])), ("cancer_breast", int(m["cancer_breast"])),
                 ("mlo_fallback", int(m["mlo_fallback"])), ("has_cc", int(bool(r["views"]["CC"]))),
                 ("has_mlo", int(bool(r["views"]["MLO"]))), ("response", m["response"]), ("response_quality", m["response_quality"]),
                 ("regression_pct", reg), ("month_match", rr.get("month_match", ""))):
        meta[k].append(v)
for k, v in meta.items():
    out[f"meta|{k}"] = np.array(v)
np.savez_compressed(OUT, **out)
print(json.dumps({"runs": sorted({k.split("|")[0] for k in out if not k.startswith("meta|")}), "n": len(ids),
                  "bytes": os.path.getsize(OUT)}))
