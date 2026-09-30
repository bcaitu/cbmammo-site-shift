
import json, glob, os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser("~/cbmammo"))
from cbmammo.concepts import HEADS, IGNORE
from cbmammo.metrics import cluster_bootstrap, safe_auc
HIDX = {h.name: j for j, h in enumerate(HEADS)}
meta = {}
for l in open("manifests/embed.jsonl"):
    r = json.loads(l); meta[r["sample_id"]] = (r["patient_id"], r["embed_meta"].get("vendor"))
rows = []
for run in sorted(glob.glob("runs/*_embed_*_s[0-2]")):
    name = os.path.basename(run)
    for ds in ("test_embed", "test_embed_ge"):
        d = torch.load(f"{run}/dump_{ds}.pt", map_location="cpu")
        ids = list(d["sample_id"]); y = d["labels"][:, HIDX["malignant"]].numpy()
        s = d["p_malignant"].float().numpy()
        s = s[:, 1] if s.ndim == 2 else s
        pid = np.array([meta[i][0] for i in ids]); ven = np.array([meta[i][1] for i in ids])
        m0 = y != IGNORE
        for sub in ["all"] + sorted(set(ven[m0])):
            m = m0 & ((ven == sub) if sub != "all" else True)
            if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
            a, lo, hi = cluster_bootstrap(safe_auc, y[m], s[m], pid[m], 500)
            rows.append(dict(run=name, set=ds, vendor=sub, n=int(m.sum()), n_pos=int(y[m].sum()), n_pat=len(set(pid[m])), auc=a, lo=lo, hi=hi))
df = pd.DataFrame(rows); os.makedirs("results/embed/summary", exist_ok=True); df.to_csv("results/embed/summary/vendor_split_zero_shot.csv", index=False)
df["arm"] = df.run.str.split("_").str[0]; df["bb"] = df.run.str.split("_").str[2]
print(df.groupby(["set","vendor","bb","arm"]).agg(n=("n","first"), pos=("n_pos","first"), pat=("n_pat","first"), auc=("auc","mean"), sd=("auc","std")).round(3).to_string())
