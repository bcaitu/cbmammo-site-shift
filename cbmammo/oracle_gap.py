"""Concept-level decomposition of external-set error (cb arm only).

For each test set, the target heads (BI-RADS, malignancy) are evaluated with
  pred   : predicted concepts (the deployed model)
  oracle : ground-truth concepts wherever labelled, predicted elsewhere
  oracle_<group> : ground truth only for one concept group (density / mass / calc / other)
The pred->oracle gain is the part of the error caused by concept PERCEPTION (x->c);
what remains at oracle is REASONING (c->y) error or unlabelled-concept error.

python -m cbmammo.oracle_gap --run runs/cb_dinov2_s0 --out results/oracle_gap.csv
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from .concepts import CONCEPT_HEADS, HEADS, IGNORE
from .metrics import cluster_bootstrap, macro_f1, safe_auc
from .model import ConceptModel
from . import data as D

GROUPS = {"density": ["density"], "mass": ["mass", "mass_shape", "mass_margin"],
          "calc": ["calc", "calc_morphology", "calc_distribution"], "other": ["asymmetry", "distortion"]}
HIDX = {h.name: j for j, h in enumerate(HEADS)}


class Head(torch.nn.Module):
    """Only the concept->target part of the cb model (no encoder needed)."""
    def __init__(self, sd):
        super().__init__()
        n_c = sum(h.n for h in CONCEPT_HEADS)
        self.target_mlp = torch.nn.Sequential(torch.nn.Linear(n_c, 128), torch.nn.GELU(), torch.nn.Dropout(0.0))
        from .concepts import TARGET_HEADS
        self.target_clf = torch.nn.ModuleDict({h.name: torch.nn.Linear(128, h.n) for h in TARGET_HEADS})
        self.load_state_dict({k: v for k, v in sd.items() if k.startswith(("target_mlp", "target_clf"))})

    def forward(self, probs):
        return ConceptModel.targets_from_concepts(self, probs)


def gt_coverage(d, use_gt: set, mask=None) -> float:
    """Fraction of rows (within `mask`, default all) where EVERY head in `use_gt` is actually
    labelled -- i.e. how much of 'oracle_<group>' is real ground truth vs falling back to the
    model's own prediction for an unlabelled row. Always 1.0 for the empty set ('pred')."""
    if not use_gt:
        return 1.0
    m = mask if mask is not None else np.ones(len(d["sample_id"]) if "sample_id" in d else d["labels"].shape[0], bool)
    ok = np.ones(int(m.sum()) if hasattr(m, "sum") else len(m), bool)
    for name in use_gt:
        y = d["labels"][:, HIDX[name]].numpy()[m]
        ok &= y != IGNORE
    return float(ok.mean())


def build_probs(d, use_gt: set):
    probs = {}
    for h in CONCEPT_HEADS:
        p = d[f"p_{h.name}"].clone().float()
        if h.name in use_gt:
            y = d["labels"][:, HIDX[h.name]]
            m = y != IGNORE
            p[m] = F.one_hot(y[m], h.n).float()
        probs[h.name] = p
    # re-derive raw (ungated) descriptor probs, then gate with (possibly oracle) parent
    for h in CONCEPT_HEADS:
        if h.parent:
            raw = probs[h.name] / probs[h.name].sum(1, keepdim=True).clamp_min(1e-6)
            probs[h.name] = raw * probs[h.parent][:, 1:2]
    return probs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=500); a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "run.json")))["config"]
    assert cfg["arm"] == "cb", "oracle decomposition needs the concept-bottleneck arm"
    head = Head(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu")).eval()
    recs = {r["sample_id"]: r for r in D.load_manifest(cfg["data"]["train_manifests"] + cfg["data"].get("test_manifests", []))}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.run, "dump_test_*.pt"))) + [os.path.join(a.run, "dump_val.pt")]:
        d = torch.load(p, map_location="cpu"); name = os.path.basename(p)[5:-3]
        g = np.array([recs[i]["patient_id"] for i in d["sample_id"]])
        dens = np.array([recs[i]["values"].get("density") or "NA" for i in d["sample_id"]])
        conds = {"pred": set(), "oracle": {h.name for h in CONCEPT_HEADS}, **{f"oracle_{k}": set(v) for k, v in GROUPS.items()}}
        for tgt in ("malignant", "birads"):
            y = d["labels"][:, HIDX[tgt]].numpy(); m = y != IGNORE
            if m.sum() < 20 or len(np.unique(y[m])) < 2:
                continue
            for cname, gt in conds.items():
                with torch.no_grad():
                    pr = head(build_probs(d, gt))[tgt].softmax(-1).numpy()
                for sub, mm in [("all", m)] + [(f"density_{k}", m & (dens == k)) for k in "ABCD"]:
                    if mm.sum() < 20 or len(np.unique(y[mm])) < 2:
                        continue
                    f1 = cluster_bootstrap(macro_f1, y[mm], pr[mm].argmax(1), g[mm], a.bootstrap)
                    auc = cluster_bootstrap(safe_auc, y[mm], pr[mm], g[mm], a.bootstrap)
                    rows.append({"set": name, "target": tgt, "concepts": cname, "subset": sub, "n": int(mm.sum()),
                                 "coverage": round(gt_coverage(d, gt, mm), 3),
                                 "f1": f1[0], "f1_lo": f1[1], "f1_hi": f1[2], "auc": auc[0], "auc_lo": auc[1], "auc_hi": auc[2]})
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); df.to_csv(a.out, index=False)
    piv = df[df.subset == "all"].pivot_table(index=["set", "target"], columns="concepts", values="auc").round(3)
    pd.set_option("display.width", 250); print(piv.to_string())


if __name__ == "__main__":
    main()
