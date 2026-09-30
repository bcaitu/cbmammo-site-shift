"""Few-shot adaptation to a new site from stage-1 dumps (no image recomputation, except the
optional --partial_ft opaque baseline, which needs the raw images).

Adaptation set: k PATIENTS drawn from the target test set (both classes required);
evaluation: all remaining patients of that set. Repeated over n_draws random draws.

cb arm
  cb_zero              deployed model, no adaptation
  cb_recal_concept     per-concept recalibration (multinomial LR on log-probs, fit on the k
                       cases' concept labels), then the ORIGINAL concept->target head  [perception]
  cb_refit_head_fixedC new concept->target logistic head on predicted concepts, C fixed as in the
                       paper (0.3)                                                    [reasoning]
  cb_refit_head_cvC    same, C chosen by inner cross-validation within the k fit cases
  cb_both              recal_concept then refit_head (fixed C)
  cb_recal_score_*     1-D Platt recalibration of the deployed TARGET score (slope+intercept
                       fit on the k cases). Cannot change AUC (monotone transform of one score);
                       reported for CALIBRATION ONLY (ece_before/ece_after, platt_slope/intercept
                       columns on the *_zero rows, not a separate 'method').
  cb_correction_cvC    correction to the DEPLOYED logit (offset = logit(cb_zero), L2-penalised
                       delta, C by inner CV) -- revision-with-shrinkage-toward-the-existing-model
                       in the clinical-updating sense; C->0 reproduces cb_zero exactly.
opaque arm (matched control, same protocol)
  opq_zero, opq_refit_head_fixedC, opq_refit_head_cvC, opq_correction_cvC
  opq_partial_ft       (only if --partial_ft; only at the requested k's) unfreeze the opaque
                       model's last encoder block + fusion + head and fine-tune on the k
                       patients' raw images -- the black-box counterpart to a "cheap repair",
                       so the paper's "the box cannot be cheaply repaired" claim is checked
                       against fine-tuning, not just a linear refit.
both arms
  probe_feats          logistic regression on frozen pooled encoder features (black-box few-shot)

python -m cbmammo.fewshot --cb runs/cb_dinov2_s0 --opaque runs/opq_dinov2_s0 --set test_cdd_cesm \
    --out results/fewshot.csv [--partial_ft --partial_ft_ks 25,100 --partial_ft_draws 10]
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from .concepts import CONCEPT_HEADS, HEADS, IGNORE
from .metrics import safe_auc
from .oracle_gap import Head, HIDX
from . import data as D

warnings.filterwarnings("ignore")

GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FIXED_C = {"cb": 0.3, "opq": 0.01, "probe": 0.01}


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gate(probs):
    out = {}
    for h in CONCEPT_HEADS:
        out[h.name] = probs[h.name]
    for h in CONCEPT_HEADS:
        if h.parent:
            raw = out[h.name] / out[h.name].sum(1, keepdims=True).clip(1e-6)
            out[h.name] = raw * out[h.parent][:, 1:2]
    return out


def recalibrate(d, idx_fit, idx_all):
    """Return concept probs for idx_all after per-head recalibration fitted on idx_fit."""
    probs = {}
    for h in CONCEPT_HEADS:
        p = d[f"p_{h.name}"].numpy().astype(np.float64)
        if h.parent:                                             # ungate for fitting
            p = p / p.sum(1, keepdims=True).clip(1e-6)
        q = p[idx_all].copy()
        y = d["labels"][:, HIDX[h.name]].numpy()
        f = [i for i in idx_fit if y[i] != IGNORE]
        if len(f) >= 6 and len(np.unique(y[f])) >= 2:
            X = np.log(p.clip(1e-6))
            m = lr(1.0).fit(X[f], y[f])
            q = np.full((len(idx_all), h.n), 1e-6)
            q[:, m.classes_] = m.predict_proba(X[idx_all])
            q = q / q.sum(1, keepdims=True)
        probs[h.name] = q
    return gate(probs)


def cond_of(probs):
    return np.concatenate([probs[h.name] for h in CONCEPT_HEADS], 1)


def fit_predict(Xf, yf, Xe, C, std=False):
    if std:
        sc = StandardScaler().fit(Xf); Xf, Xe = sc.transform(Xf), sc.transform(Xe)
    return lr(C).fit(Xf, yf).predict_proba(Xe)[:, 1]


def offset_lr(Xf, yf, of, Xe, oe, C, std=False):
    """Logistic regression with a fixed per-case offset (logit of the deployed model's score):
    the fit learns an ADDITIVE CORRECTION to the deployed head, penalised by C * balanced
    log-loss + 0.5||w||^2 (sklearn's scaling convention; intercept unpenalised). C -> 0
    reproduces the deployed ranking exactly (pure shrinkage toward the existing model, as
    recommended for model revision with a small update sample)."""
    if std:
        sc = StandardScaler().fit(Xf); Xf, Xe = sc.transform(Xf), sc.transform(Xe)
    n, d_ = Xf.shape
    cw = np.where(yf == 1, n / (2 * max(yf.sum(), 1)), n / (2 * max(n - yf.sum(), 1)))

    def f(t):
        w, b = t[:d_], t[d_]
        z = of + b + Xf @ w
        pz = 1 / (1 + np.exp(-z))
        loss = C * np.sum(cw * (np.logaddexp(0, z) - yf * z)) + 0.5 * w @ w
        g = C * (cw * (pz - yf))
        return loss, np.concatenate([Xf.T @ g + w, [g.sum()]])
    t = minimize(f, np.zeros(d_ + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x
    return oe + t[d_] + Xe @ t[:d_]


def choose_C(X, y, std, seed, offset=None):
    """Inner-CV choice of C on the k fit cases only (pooled out-of-fold AUC); returns None when
    either class has fewer than 2 cases, so the caller falls back to the fixed C."""
    nf = int(min(5, y.sum(), len(y) - y.sum()))
    if nf < 2:
        return None
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed)
    best, bestC = -1.0, None
    for C in GRID:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            oof[te] = (fit_predict(X[tr], y[tr], X[te], C, std) if offset is None else
                       offset_lr(X[tr], y[tr], offset[tr], X[te], offset[te], C, std))
        a = safe_auc(y, oof)
        if a == a and a > best + 1e-9:            # a == a filters NaN (a fold with one class)
            best, bestC = a, C
    return bestC


def platt(y_fit, p_fit, p_all):
    """1-D logistic (Platt) recalibration of a single score: fit slope+intercept on
    logit(p_fit) -> y_fit, apply to p_all. A monotone (slope>0) transform of one score cannot
    change AUC or ranking -- use this for calibration reporting only."""
    x = logit(p_fit).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(x, y_fit)
    return m.predict_proba(logit(p_all).reshape(-1, 1))[:, 1], float(m.coef_[0, 0]), float(m.intercept_[0])


def ece(y, p, n_bins=10):
    """Expected calibration error, equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi >= 1 else p < hi)
        if m.sum():
            e += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(e)


# ------------------------------------------------------------------ partial fine-tune (opaque)
def partial_finetune_opaque(run_dir, records_by_id, fit_ids, ev_ids, target, dev,
                             epochs=6, lr_=1e-4, batch_size=8):
    """Unfreeze the opaque model's last encoder block + fusion + pool/to_z/target head and
    fine-tune on the k fit patients' raw breast images; return predicted P(target=1) on ev_ids.
    This is the black-box counterpart to the concept-head refit: both get k patients and a
    cheap budget (a handful of epochs on a handful of images), so the paper's claim that the
    box resists cheap repair is checked against gradient fine-tuning, not only a linear probe.
    """
    from .model import build, head_weights, masked_ce
    from .train_stage1 import class_weights as _cw
    cfg = json.load(open(os.path.join(run_dir, "run.json")))["config"]
    assert cfg["arm"] == "opaque", "partial_finetune_opaque needs an opaque-arm run"
    model = build(cfg).to(dev)
    sd = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev)
    model.load_state_dict(sd, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.enc.m.blocks[-1:]:
        for p in blk.parameters():
            p.requires_grad = True
    for mod in (model.fusion, model.pool, model.to_z, model.clf):
        for p in mod.parameters():
            p.requires_grad = True
    model.enc.trainable = True                      # so BatchNorm/eval-mode logic (none here) and best.pt saving would include it if reused
    train_recs = [records_by_id[i] for i in fit_ids]
    eval_recs = [records_by_id[i] for i in ev_ids]
    size = tuple(cfg["data"].get("size", [1024, 640]))
    mk = lambda rs, train: torch.utils.data.DataLoader(
        D.BreastDataset(rs, size, train, cfg["data"].get("clahe", True), cfg["data"].get("cache_dir")),
        batch_size=min(batch_size, max(1, len(rs))), shuffle=train, num_workers=2, collate_fn=D.collate, drop_last=False)
    tl, el = mk(train_recs, True), mk(eval_recs, False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.01)
    hw = head_weights("opaque", cfg["model"].get("aux_lambda", 0.1), cfg["model"].get("target_weight", 1.0))
    cw = _cw(train_recs, dev)
    model.train()
    for _ in range(epochs):
        for b in tl:
            o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
            loss, _ = masked_ce(o["logits"], b["labels"].to(dev), hw, cw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    model.eval(); preds = []
    with torch.no_grad():
        for b in el:
            o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
            preds.append(o["logits"][target].float().softmax(-1)[:, 1].cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cb", required=True); ap.add_argument("--opaque")
    ap.add_argument("--set", default="test_cdd_cesm"); ap.add_argument("--target", default="malignant")
    ap.add_argument("--ks", default="5,10,25,50,100"); ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partial_ft", action="store_true", help="also run the opaque partial-fine-tune baseline (slow: needs real images)")
    ap.add_argument("--partial_ft_ks", default="25,100")
    ap.add_argument("--partial_ft_draws", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.cb, "run.json")))["config"]
    recs = {r["sample_id"]: r for r in D.load_manifest(cfg["data"]["train_manifests"] + cfg["data"].get("test_manifests", []))}
    dcb = torch.load(os.path.join(a.cb, f"dump_{a.set}.pt"), map_location="cpu")
    head = Head(torch.load(os.path.join(a.cb, "best.pt"), map_location="cpu")).eval()
    dop = torch.load(os.path.join(a.opaque, f"dump_{a.set}.pt"), map_location="cpu") if a.opaque else None
    if dop is not None:
        assert dop["sample_id"] == dcb["sample_id"]
    y = dcb["labels"][:, HIDX[a.target]].numpy()
    ok = np.flatnonzero(y != IGNORE)
    pid = np.array([recs[s]["patient_id"] for s in dcb["sample_id"]])
    patients = np.unique(pid[ok])
    feats = dcb["feats"].numpy()
    zero_cb_full = dcb[f"p_{a.target}"].numpy()[:, 1]
    zero_opq_full = dop[f"p_{a.target}"].numpy()[:, 1] if dop is not None else None
    ft_ks = {int(x) for x in a.partial_ft_ks.split(",")} if a.partial_ft else set()
    rng = np.random.default_rng(a.seed)
    rows = []
    for k in [int(x) for x in a.ks.split(",")]:
        for draw in range(a.draws):
            for _ in range(100):
                fitp = set(rng.choice(patients, k, replace=False))
                fit = np.array([i for i in ok if pid[i] in fitp]); ev = np.array([i for i in ok if pid[i] not in fitp])
                if len(np.unique(y[fit])) == 2 and len(np.unique(y[ev])) == 2:
                    break
            res, extra = {}, {}
            res["cb_zero"] = zero_cb_full[ev]
            pr = recalibrate(dcb, list(fit), list(ev))
            with torch.no_grad():
                res["cb_recal_concept"] = head({k2: torch.tensor(v, dtype=torch.float32) for k2, v in pr.items()})[a.target].softmax(-1).numpy()[:, 1]
            cond = dcb["cond"].numpy()
            res["cb_refit_head_fixedC"] = fit_predict(cond[fit], y[fit], cond[ev], FIXED_C["cb"], False)
            Ccv = choose_C(cond[fit], y[fit], False, draw)
            res["cb_refit_head_cvC"] = fit_predict(cond[fit], y[fit], cond[ev], Ccv if Ccv is not None else FIXED_C["cb"], False)
            pr_fit = recalibrate(dcb, list(fit), list(fit))
            res["cb_both"] = fit_predict(cond_of(pr_fit), y[fit], cond_of(pr), FIXED_C["cb"], False)
            o_cb = logit(zero_cb_full)
            Cc_cb = choose_C(cond[fit], y[fit], False, draw, offset=o_cb[fit])
            C_cb = Cc_cb if Cc_cb is not None else GRID[0]
            res["cb_correction_cvC"] = offset_lr(cond[fit], y[fit], o_cb[fit], cond[ev], o_cb[ev], C_cb, False)
            res["probe_feats"] = fit_predict(feats[fit], y[fit], feats[ev], FIXED_C["probe"], True)
            # calibration of the zero-shot score (Platt slope/intercept + ECE before/after; AUC unaffected by construction)
            p_recal, slope, icpt = platt(y[fit], zero_cb_full[fit], zero_cb_full[ev])
            extra["cb_zero"] = {"platt_slope": slope, "platt_intercept": icpt,
                                "ece_before": ece(y[ev], zero_cb_full[ev]), "ece_after": ece(y[ev], p_recal)}
            if dop is not None:
                res["opq_zero"] = zero_opq_full[ev]
                z = dop["cond"].numpy()
                res["opq_refit_head_fixedC"] = fit_predict(z[fit], y[fit], z[ev], FIXED_C["opq"], True)
                Ccv_o = choose_C(z[fit], y[fit], True, draw)
                res["opq_refit_head_cvC"] = fit_predict(z[fit], y[fit], z[ev], Ccv_o if Ccv_o is not None else FIXED_C["opq"], True)
                o_opq = logit(zero_opq_full)
                Cc_opq = choose_C(z[fit], y[fit], True, draw, offset=o_opq[fit])
                C_opq = Cc_opq if Cc_opq is not None else GRID[0]
                res["opq_correction_cvC"] = offset_lr(z[fit], y[fit], o_opq[fit], z[ev], o_opq[ev], C_opq, True)
                p_recal_o, slope_o, icpt_o = platt(y[fit], zero_opq_full[fit], zero_opq_full[ev])
                extra["opq_zero"] = {"platt_slope": slope_o, "platt_intercept": icpt_o,
                                     "ece_before": ece(y[ev], zero_opq_full[ev]), "ece_after": ece(y[ev], p_recal_o)}
                if k in ft_ks and draw < a.partial_ft_draws:
                    ft_pred = partial_finetune_opaque(a.opaque, recs, list(dcb["sample_id"][i] for i in fit),
                                                       list(dcb["sample_id"][i] for i in ev), a.target, torch.device(a.device))
                    res["opq_partial_ft"] = ft_pred
            for mth, s in res.items():
                rows.append({"set": a.set, "target": a.target, "k_patients": k, "n_fit": len(fit), "n_eval": len(ev),
                            "draw": draw, "method": mth, "auc": safe_auc(y[ev], s), **extra.get(mth, {})})
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); df.to_csv(a.out, index=False)
    s = df.groupby(["k_patients", "method"])["auc"].agg(["mean", "std", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)])
    s.columns = ["mean", "sd", "p2.5", "p97.5"]
    pd.set_option("display.width", 250); print(s.round(3).unstack("method")["mean"].to_string()); print(s.round(3).to_string())


if __name__ == "__main__":
    main()
