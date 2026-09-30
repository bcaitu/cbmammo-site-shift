"""MMC Astana local-site evaluation of the trained stage-1 runs (packed inference dumps, see mmc_pack.py).

Endpoints
  retro       study-level M3-M4 vs M1-M2 (the clinic's BI-RADS-analog categories) on retrospective
              treatment-naive studies; score = max over the study's breasts of P(BI-RADS>=3) [primary]
              or P(malignant) [secondary]. Not a malignancy endpoint.
  outpatient  breast-level M6 (confirmed cancer) vs M1-M2, registry conclusions with an explicit side.
  post_nact   exploratory: imaged cancer breast after neoadjuvant chemotherapy, high-quality response
              labels; residual disease (partial/stable/progression) vs complete radiological response.
Few-shot site repair (retro endpoint): k labelled patient groups, all other patient groups evaluated,
30 draws per k and seed. Arms: CB refit of the concept->target head on the max-pooled 30-d concept
vector; opaque refit on the max-pooled 512-d bottleneck z; linear probe on pooled encoder features.
Regularisation either fixed as in the paper (C=0.3 CB, 0.01 opaque/probe) or chosen by inner
cross-validation within the k labelled cases (same grid for every arm). Sensitivity variant 'offset':
the refit is a correction to the deployed head (offset = logit of the deployed score), C by inner CV.
CIs: cluster bootstrap over patient groups.

python mmc_eval.py --bundle mmc_bundle.npz --out results_mmc [--draws 30 --boot 2000 --jobs 8]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
# Pseudonymous ids of the second study of each pixel-identical pair (three pairs, listed in the clinic export's known_duplicates.csv),
# passed as a comma-separated list so that no study identifiers are stored in the public code:  MMC_DROP_STUDIES=id1,id2,id3
DROP = {x for x in os.environ.get("MMC_DROP_STUDIES", "").split(",") if x}
GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FIXED_C = {"cb": 0.3, "opq": 0.01, "probe": 0.01}
KS = (10, 25, 50, 100)


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = pd.DataFrame({k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("meta|")})
    runs = sorted({k.split("|")[0] for k in z.files if not k.startswith("meta|")})
    P = {r: {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith(r + "|")} for r in runs}
    return meta, P


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot(y, scores, g, B, seed=0):
    """Cluster bootstrap over g. scores: list of score vectors; returns CIs of each AUC and of s0 - s1."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    out = []
    for _ in range(B):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        if len(np.unique(y[ix])) < 2:
            continue
        a = [roc_auc_score(y[ix], s[ix]) for s in scores]
        out.append(a + ([a[0] - a[1]] if len(a) == 2 else []))
    out = np.array(out)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def lr(C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def fit_predict(Xf, yf, Xe, C, std):
    if std:
        sc = StandardScaler().fit(Xf); Xf, Xe = sc.transform(Xf), sc.transform(Xe)
    return lr(C).fit(Xf, yf).predict_proba(Xe)[:, 1]


def offset_lr(Xf, yf, of, Xe, oe, C, std):
    """Logistic regression with a fixed per-case offset (logit of the deployed model's score):
    the refit learns a correction to the deployed head. Objective = C * sum(balanced log-loss) + 0.5 ||w||^2
    (same scaling as sklearn), intercept unpenalised. C -> 0 reproduces the deployed ranking."""
    from scipy.optimize import minimize
    if std:
        sc = StandardScaler().fit(Xf); Xf, Xe = sc.transform(Xf), sc.transform(Xe)
    n, d = Xf.shape
    cw = np.where(yf == 1, n / (2 * max(yf.sum(), 1)), n / (2 * max(n - yf.sum(), 1)))

    def f(t):
        w, b = t[:d], t[d]
        z = of + b + Xf @ w
        pz = 1 / (1 + np.exp(-z))
        loss = C * np.sum(cw * (np.logaddexp(0, z) - yf * z)) + 0.5 * w @ w
        g = C * (cw * (pz - yf))
        return loss, np.concatenate([Xf.T @ g + w, [g.sum()]])
    t = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x
    return oe + t[d] + Xe @ t[:d]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def choose_C_offset(X, y, o, std, seed):
    nf = int(min(5, y.sum(), len(y) - y.sum()))
    if nf < 2:
        return None
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
    best, bestC = -1.0, None
    for C in GRID:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            oof[te] = offset_lr(X[tr], y[tr], o[tr], X[te], o[te], C, std)
        a = auc(y, oof)
        if a > best + 1e-9:
            best, bestC = a, C
    return bestC


def choose_C(X, y, std, seed):
    """Inner CV on the k labelled cases: pooled out-of-fold AUC; ties -> stronger regularisation.
    Returns None when either class has fewer than 2 cases (caller falls back to the fixed C)."""
    nf = int(min(5, y.sum(), len(y) - y.sum()))
    if nf < 2:
        return None
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
    best, bestC = -1.0, None
    for C in GRID:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            oof[te] = fit_predict(X[tr], y[tr], X[te], C, std)
        a = auc(y, oof)
        if a > best + 1e-9:
            best, bestC = a, C
    return bestC


# ------------------------------------------------------------------ study pooling
def study_pool(meta, rows):
    """rows: boolean mask of breasts to use -> (studies, patient_group per study, list of breast-index arrays)."""
    m = meta[rows]
    studies = list(dict.fromkeys(m.study))
    idx = [np.flatnonzero((meta.study == s).values & rows) for s in studies]
    pg = [meta.patient_group.values[i[0]] for i in idx]
    return np.array(studies), np.array(pg), idx


def pooled(X, idx):
    return np.stack([X[i].max(0) for i in idx])


# ------------------------------------------------------------------ few-shot task
def one_draw(k, draw, seed, y, pg, feats, zero):
    rng = np.random.default_rng(1000 * seed + 17 * k + draw)
    groups = np.unique(pg)
    for _ in range(200):
        fitg = set(rng.choice(groups, k, replace=False))
        fit = np.array([i for i in range(len(y)) if pg[i] in fitg]); ev = np.array([i for i in range(len(y)) if pg[i] not in fitg])
        if len(np.unique(y[fit])) == 2 and len(np.unique(y[ev])) == 2:
            break
    rows = []
    base = dict(k=k, draw=draw, n_fit=len(fit), n_fit_pos=int(y[fit].sum()), n_eval=len(ev), n_eval_pos=int(y[ev].sum()))
    for name, s in zero.items():
        rows.append({**base, "method": f"{name}_zero", "C": np.nan, "auc": auc(y[ev], s[ev])})
    for arm, (X, std) in feats.items():
        Cf = FIXED_C[arm]
        rows.append({**base, "method": f"{arm}_refit_fixedC", "C": Cf, "auc": auc(y[ev], fit_predict(X[fit], y[fit], X[ev], Cf, std))})
        Cc = choose_C(X[fit], y[fit], std, draw)
        rows.append({**base, "method": f"{arm}_refit_cvC", "C": Cc if Cc is not None else Cf, "cv_fallback": Cc is None,
                     "auc": auc(y[ev], fit_predict(X[fit], y[fit], X[ev], Cc if Cc is not None else Cf, std))})
    for arm in ("cb", "opq"):
        X, std = feats[arm]
        o = logit(zero[arm])
        Cc = choose_C_offset(X[fit], y[fit], o[fit], std, draw)
        C_use = Cc if Cc is not None else GRID[0]
        rows.append({**base, "method": f"{arm}_offset_cvC", "C": C_use, "cv_fallback": Cc is None,
                     "auc": auc(y[ev], offset_lr(X[fit], y[fit], o[fit], X[ev], o[ev], C_use, std))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=30); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, P = load(a.bundle)
    for c in ("cancer_breast", "mlo_fallback", "has_cc", "has_mlo"):
        meta[c] = meta[c].astype(int)
    meta["dev"] = np.where(meta.device.str.upper().str.startswith("SIEMENS"), "Siemens",
                           np.where(meta.device.str.upper().str.startswith("AGFA"), "Agfa", "other"))
    runs = sorted(P)
    pairs = sorted({r.split("_", 1)[1] for r in runs})        # dinov2_s0 ... ft2_s2
    sc = {r: {"birads3": P[r]["p_birads"][:, 2:].sum(1), "malig": P[r]["p_malignant"][:, 1]} for r in runs}
    for arm in ("cb", "opq"):                                 # seed ensembles (mean probability over seeds)
        for bb in ("dinov2", "ft2"):
            rs = [r for r in runs if r.startswith(f"{arm}_{bb}_s")]
            if rs:
                sc[f"{arm}_{bb}_ens"] = {k: np.mean([sc[r][k] for r in rs], 0) for k in ("birads3", "malig")}
    res, info = [], {}

    def add(endpoint, subset, name, score, y, s, g, extra=None):
        lo, hi = boot(y, [s], g, a.boot)
        arm, bb, seed = name.split("_")
        res.append({"endpoint": endpoint, "subset": subset, "arm": arm, "backbone": bb, "seed": seed, "score": score,
                    "n": len(y), "n_pos": int(y.sum()), "n_groups": len(np.unique(g)), "auc": auc(y, s), "lo": lo[0], "hi": hi[0],
                    **(extra or {})})

    # ---------------- retro zero-shot (study level)
    retro = meta.group.str.startswith("retro_M").values & ~meta.study.isin(DROP).values
    st, pg, idx = study_pool(meta, retro)
    yM = np.array([int(meta.study_M.values[i[0]]) for i in idx]); y = (yM >= 3).astype(int)
    dev = np.array([meta.dev.values[i[0]] for i in idx])
    info["retro"] = {"studies": len(st), "patient_groups": len(np.unique(pg)), "pos_M3_M4": int(y.sum()),
                     "M_counts": {int(k): int(v) for k, v in zip(*np.unique(yM, return_counts=True))},
                     "devices": {k: int(v) for k, v in zip(*np.unique(dev, return_counts=True))},
                     "breasts": int(retro.sum()), "mlo_fallback_breasts": int(meta.mlo_fallback.values[retro].sum())}
    for name in sc:
        for score in ("birads3", "malig"):
            s = pooled(sc[name][score][:, None], idx)[:, 0]
            add("retro_M34_vs_M12", "all", name, score, y, s, pg)
            for d in ("Siemens", "Agfa"):
                m = dev == d
                if m.sum() >= 20 and len(np.unique(y[m])) == 2:
                    add("retro_M34_vs_M12", f"device_{d}", name, score, y[m], s[m], pg[m])
    # paired CB - opaque on the retro endpoint (P(BI-RADS>=3))
    for suffix in [p for p in pairs] + ["dinov2_ens", "ft2_ens"]:
        cbn, opn = f"cb_{suffix}", f"opq_{suffix}"
        if cbn in sc and opn in sc:
            s0 = pooled(sc[cbn]["birads3"][:, None], idx)[:, 0]; s1 = pooled(sc[opn]["birads3"][:, None], idx)[:, 0]
            lo, hi = boot(y, [s0, s1], pg, a.boot)
            bb, seed = suffix.split("_")
            res.append({"endpoint": "retro_M34_vs_M12", "subset": "all", "arm": "cb-opq", "backbone": bb, "seed": seed,
                        "score": "birads3", "n": len(y), "n_pos": int(y.sum()), "n_groups": len(np.unique(pg)),
                        "auc": auc(y, s0) - auc(y, s1), "lo": lo[2], "hi": hi[2]})

    # ---------------- outpatient breast-level M6 vs M1-M2
    bm = pd.to_numeric(meta.breast_M, errors="coerce").values
    nodup = ~meta.study.isin(DROP).values
    op = (meta.group.values == "outpatient") & np.isin(bm, [1, 2, 6]) & nodup
    yo = (bm[op] == 6).astype(int); go = meta.patient_group.values[op]
    info["outpatient"] = {"breasts": int(op.sum()), "pos_M6": int(yo.sum()), "patient_groups": int(len(np.unique(go))),
                          "duplicate_breasts_excluded": int(((meta.group.values == "outpatient") & np.isin(bm, [1, 2, 6]) & ~nodup).sum())}
    if op.sum() >= 10 and 0 < yo.sum() < op.sum():
        for name in sc:
            for score in ("malig", "birads3"):
                add("outpatient_M6_vs_M12", "breast", name, score, yo, sc[name][score][op], go)
    # any group: breasts with an explicit-side registry M (sensitivity, includes retro studies with sided registry rows)
    anyb = np.isin(bm, [1, 2, 6]) & (meta.group.values != "post_nact") & ~meta.study.isin(DROP).values
    ya = (bm[anyb] == 6).astype(int); ga = meta.patient_group.values[anyb]
    info["sided_registry_any_group"] = {"breasts": int(anyb.sum()), "pos_M6": int(ya.sum()),
                                        "by_group": meta.group[anyb].value_counts().to_dict()}
    if anyb.sum() >= 10 and 0 < ya.sum() < anyb.sum() and anyb.sum() > op.sum():
        for name in sc:
            add("sided_M6_vs_M12", "breast_all_groups", name, "malig", ya, sc[name]["malig"][anyb], ga)

    # ---------------- post-NACT exploratory
    pn = nodup & (meta.group.values == "post_nact") & (meta.cancer_breast.values == 1) & (meta.response_quality.values == "high") \
        & np.isin(meta.response.values, ["complete", "partial", "stable", "progression"])
    yr = (meta.response.values[pn] != "complete").astype(int); gr = meta.patient_group.values[pn]
    regp = pd.to_numeric(meta.regression_pct.values[pn], errors="coerce")
    info["post_nact_duplicates_excluded"] = int(((meta.group.values == "post_nact") & (meta.cancer_breast.values == 1) &
                                                 (meta.response_quality.values == "high") & ~nodup).sum())
    info["post_nact"] = {"cancer_breasts_high_quality": int(pn.sum()), "residual": int(yr.sum()), "complete": int((1 - yr).sum()),
                         "with_regression_pct": int(np.isfinite(regp).sum())}
    corr = []
    if pn.sum() >= 10 and 0 < yr.sum() < pn.sum():
        for name in sc:
            add("post_nact_residual_vs_complete", "cancer_breast_hq", name, "malig", yr, sc[name]["malig"][pn], gr)
    for name in sc:
        f = np.isfinite(regp)
        if f.sum() >= 10:
            r, p = spearmanr(regp[f], sc[name]["malig"][pn][f])
            corr.append({"run": name, "n": int(f.sum()), "spearman_regression_pct_vs_Pmalig": r, "p": p})
    pd.DataFrame(corr).to_csv(os.path.join(a.out, "mmc_post_nact_corr.csv"), index=False)

    pd.DataFrame(res).to_csv(os.path.join(a.out, "mmc_results.csv"), index=False)
    json.dump(info, open(os.path.join(a.out, "mmc_cohort.json"), "w"), indent=1, default=str)
    print(json.dumps(info, default=str))

    # ---------------- few-shot site repair on the retro endpoint
    tasks = []
    for pair in pairs:
        cbn, opn = f"cb_{pair}", f"opq_{pair}"
        if cbn not in P or opn not in P:
            continue
        seed = int(pair.split("_s")[1])
        feats = {"cb": (pooled(P[cbn]["cond"], idx), False), "opq": (pooled(P[opn]["cond"], idx), True),
                 "probe": (pooled(P[cbn]["feats"], idx), True)}
        zero = {"cb": pooled(sc[cbn]["birads3"][:, None], idx)[:, 0], "opq": pooled(sc[opn]["birads3"][:, None], idx)[:, 0]}
        for k in KS:
            for d in range(a.draws):
                tasks.append((pair, k, d, seed, feats, zero))
    out = Parallel(n_jobs=a.jobs, verbose=0)(delayed(one_draw)(k, d, seed, y, pg, feats, zero) for (pair, k, d, seed, feats, zero) in tasks)
    fs = pd.DataFrame([{**r, "backbone": t[0].split("_")[0], "seed": t[3]} for t, rows in zip(tasks, out) for r in rows])
    fs.to_csv(os.path.join(a.out, "mmc_fewshot_draws.csv"), index=False)
    g = fs.groupby(["backbone", "k", "method"])["auc"]
    summ = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "p2.5": g.quantile(0.025), "p97.5": g.quantile(0.975), "n_draws": g.size()}).reset_index()
    # paired per-draw differences (same draw, same eval set): CB refit - opaque refit, CB refit - CB zero
    w = fs.pivot_table(index=["backbone", "seed", "k", "draw"], columns="method", values="auc").reset_index()
    for a_, b_ in (("cb_refit_cvC", "opq_refit_cvC"), ("cb_refit_fixedC", "opq_refit_fixedC"), ("cb_refit_cvC", "cb_zero"),
                   ("opq_refit_cvC", "opq_zero"), ("cb_refit_cvC", "probe_refit_cvC"), ("cb_offset_cvC", "opq_offset_cvC"),
                   ("cb_offset_cvC", "cb_zero"), ("opq_offset_cvC", "opq_zero")):
        w[f"{a_}-{b_}"] = w[a_] - w[b_]
    dcols = [c for c in w.columns if "-" in str(c)]
    dd = w.groupby(["backbone", "k"])[dcols].agg(["mean", lambda x: (x > 0).mean()])
    dd.columns = [f"{c}|{'mean' if s == 'mean' else 'frac_gt0'}" for c, s in dd.columns]
    summ.to_csv(os.path.join(a.out, "mmc_fewshot_summary.csv"), index=False)
    dd.reset_index().to_csv(os.path.join(a.out, "mmc_fewshot_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["backbone", "k"], columns="method", values="mean").round(3).to_string())
    print(dd.round(3).to_string())


if __name__ == "__main__":
    main()
