"""Remaining stages of cbmammo.evaluate (linear probe, subgroups, CB-vs-opaque compare) run as parallel tasks.
Identical computations to evaluate.main; the only change is that the linear-probe classifier is fitted once per
(run, head) instead of once per (run, test set, head) -- the fit does not depend on the test set."""
import glob, json, os, sys
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cbmammo import evaluate as E
from cbmammo import data as D
from cbmammo.concepts import HEADS, IGNORE
from cbmammo.metrics import macro_f1, safe_auc

OUT = "results/embed"; PARTS = os.path.join(OUT, "parts"); os.makedirs(PARTS, exist_ok=True)
NB = 300
RUNS = [f"runs/{a}_embed_{bb}_s{s}" for bb in ("dinov2", "ft2") for s in (0, 1, 2) for a in ("cb", "opq")]
PAIRS = [(f"runs/cb_embed_{bb}_s{s}", f"runs/opq_embed_{bb}_s{s}") for bb in ("dinov2", "ft2") for s in (0, 1, 2)]


def recs_all():
    man = sorted({m for r in RUNS for m in (lambda c: c["data"]["train_manifests"] + c["data"].get("test_manifests", []))(
        json.load(open(os.path.join(r, "run.json")))["config"])})
    return {r["sample_id"]: r for r in D.load_manifest(man)}


def lp_cached(run):
    tr = E._load(run, "train")
    X = tr["feats"].numpy(); sc = StandardScaler().fit(X); Xs = sc.transform(X)
    clfs = {}
    for j, h in enumerate(HEADS):
        y = tr["labels"][:, j].numpy(); m = y != IGNORE
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        clfs[j] = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(Xs[m], y[m])
    rows = []
    for p in sorted(glob.glob(os.path.join(run, "dump_test_*.pt"))):
        d = torch.load(p, map_location="cpu"); name = os.path.basename(p)[5:-3]
        Xt = sc.transform(d["feats"].numpy())
        for j, h in enumerate(HEADS):
            if j not in clfs:
                continue
            yt = d["labels"][:, j].numpy(); mt = yt != IGNORE
            if mt.sum() < 10 or len(np.unique(yt[mt])) < 2:
                continue
            clf = clfs[j]
            prob = np.zeros((mt.sum(), h.n)); prob[:, clf.classes_] = clf.predict_proba(Xt[mt])
            rows.append({"run": E.run_label(run), "set": name, "head": h.name, "n": int(mt.sum()),
                         "f1": macro_f1(yt[mt], prob.argmax(1)), "auc": safe_auc(yt[mt], prob)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(PARTS, f"lp_{E.run_label(run)}.csv"), index=False); return len(df)


def sub_one(run):
    df = E.subgroup_table([run], recs_all(), NB); df.to_csv(os.path.join(PARTS, f"sub_{E.run_label(run)}.csv"), index=False); return len(df)


def cmp_one(pair):
    df = E.compare(pair[0], pair[1], recs_all(), nb=NB); df.to_csv(os.path.join(PARTS, f"cmp_{E.run_label(pair[0])}.csv"), index=False); return len(df)


if __name__ == "__main__":
    tasks = [delayed(lp_cached)(r) for r in RUNS] + [delayed(sub_one)(r) for r in RUNS] + [delayed(cmp_one)(p) for p in PAIRS]
    out = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 10, verbose=5)(tasks)
    for stage, name in (("lp", "linear_probe"), ("sub", "subgroups"), ("cmp", "compare")):
        fs = sorted(glob.glob(os.path.join(PARTS, f"{stage}_*.csv")))
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(name, len(fs), "parts merged")
    print("EVALUATE_REST_DONE")
