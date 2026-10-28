
"""Analysis C: is the CDD-CESM gain of the head refit due to local adaptation or to replacing the MLP by a linear head?
Variants on identical draws (k patients adapt, the remaining patients evaluate):
  cb_zero         deployed MLP head
  lin_zero        linear (L2 logistic) head fitted on the SOURCE training dump, no local data
  cb_refit        linear head fitted on the k local patients (the paper's 'head refit', C by inner CV)
  cb_correction   deployed MLP logit as offset + L2-penalised linear correction (paper's 'correction')
  lin_correction  same, offset = source-fitted linear head
  mlp_shrink      the deployed MLP itself fine-tuned on the k patients, L2 shrinkage towards its deployed weights (mu by inner CV)
The concept predictor does not depend on the target head (targets read detached concepts), so heads can be swapped on the same checkpoint."""
from rev1_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from cbmammo.fewshot import choose_C, fit_predict, offset_lr, logit, FIXED_C, GRID
NJ = int(os.environ.get("NJOBS", 40)); NDRAWS = int(os.environ.get("NDRAWS", 30))
sets = os.environ.get("SETS", "cdd_cesm,cmmd,embed,embed_ge").split(",")
KS = [int(x) for x in os.environ.get("KS", "10,25,50,100").split(",")]
MU = (100.0, 10.0, 1.0, 0.1, 0.01, 0.001)

def mlp_fit(P0, Xf, yf, mu, steps=100, lr=0.01):
    torch.set_num_threads(1)
    P = [p.clone().requires_grad_(True) for p in P0]
    X = torch.tensor(Xf, dtype=torch.float32); y = torch.tensor(yf, dtype=torch.long)
    n1 = int(y.sum()); n0 = len(y) - n1
    w = torch.tensor([len(y) / (2 * max(n0, 1)), len(y) / (2 * max(n1, 1))], dtype=torch.float32)
    opt = torch.optim.Adam(P, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = F.gelu(X @ P[0].T + P[1]) @ P[2].T + P[3]
        loss = F.cross_entropy(z, y, weight=w) + mu * sum(((p - p0) ** 2).sum() for p, p0 in zip(P, P0))
        loss.backward(); opt.step()
    return [p.detach() for p in P]
def mlp_logit(P, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32); z = F.gelu(Xt @ P[0].T + P[1]) @ P[2].T + P[3]
    return (z[:, 1] - z[:, 0]).numpy()
def choose_mu(P0, X, y, seed):
    nf = int(min(5, y.sum(), len(y) - y.sum()))
    if nf < 2: return None
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed); best, bm = -1.0, None
    for mu in MU:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            oof[te] = mlp_logit(mlp_fit(P0, X[tr], y[tr], mu), X[te])
        a = fast_auc(y, oof)
        if a == a and a > best + 1e-9: best, bm = a, mu
    return bm

def make_draws(y, pid, k, n, seed):
    rng = np.random.default_rng(seed); ups = np.unique(pid); out = []
    for _ in range(n):
        for _ in range(100):
            fp = rng.choice(ups, k, replace=False); inf = np.isin(pid, fp)
            fit, ev = np.flatnonzero(inf), np.flatnonzero(~inf)
            if len(np.unique(y[fit])) == 2 and len(np.unique(y[ev])) == 2: break
        out.append((fit, ev))
    return out

def task(enc, seed, s, k, T):
    X, y, pid, zero, lin, P0 = T["X"], T["y"], T["pid"], T["cb_zero"], T["lin"], T["P0"]
    o_cb, o_lin = logit(zero), lin
    rows = []
    for dr, (fit, ev) in enumerate(make_draws(y, pid, k, NDRAWS, 7919 * seed + 31 * k + len(s))):
        res = {"cb_zero": zero[ev], "lin_zero": lin[ev]}
        Cc = choose_C(X[fit], y[fit], False, dr)
        res["cb_refit"] = fit_predict(X[fit], y[fit], X[ev], Cc if Cc is not None else FIXED_C["cb"], False)
        C1 = choose_C(X[fit], y[fit], False, dr, offset=o_cb[fit]); C1 = C1 if C1 is not None else GRID[0]
        res["cb_correction"] = offset_lr(X[fit], y[fit], o_cb[fit], X[ev], o_cb[ev], C1, False)
        C2 = choose_C(X[fit], y[fit], False, dr, offset=o_lin[fit]); C2 = C2 if C2 is not None else GRID[0]
        res["lin_correction"] = offset_lr(X[fit], y[fit], o_lin[fit], X[ev], o_lin[ev], C2, False)
        mu = choose_mu(P0, X[fit], y[fit], dr); mu = mu if mu is not None else MU[0]
        res["mlp_shrink"] = mlp_logit(mlp_fit(P0, X[fit], y[fit], mu), X[ev])
        for m_, sc in res.items():
            rows.append(dict(encoder=enc, seed=seed, set=s, k=k, draw=dr, variant=m_, auc=fast_auc(y[ev], sc), n_fit=len(fit), n_eval=len(ev)))
    return rows

tasks, zs = [], []
for enc, (cb, opq) in ENC.items():
    for seed in SEEDS:
        tr = torch.load(f"{RUNS}/{cb}_s{seed}/dump_train.pt", map_location="cpu"); va = dump(cb, seed, "val")
        ytr, mtr = target_arrays(tr); Xtr = tr["cond"].numpy()[mtr].astype(np.float64); ytr = (ytr[mtr] == 1).astype(int); del tr
        yva, mva = target_arrays(va); Xva = va["cond"].numpy()[mva].astype(np.float64); yva = (yva[mva] == 1).astype(int)
        best = (-1, None, None)
        for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
            a = fast_auc(yva, m.decision_function(Xva))
            if a > best[0]: best = (a, C, m)
        lin = best[2]; hd = head_for(cb, seed)
        sd = hd.state_dict(); P0 = [sd["target_mlp.0.weight"], sd["target_mlp.0.bias"], sd["target_clf.malignant.weight"], sd["target_clf.malignant.bias"]]
        print(enc, seed, "source linear head: C =", best[1], "val AUC", round(best[0], 4), "deployed val AUC", round(fast_auc(yva, va["p_malignant"].numpy()[mva, 1]), 4),
              "n_train_labelled", len(ytr), flush=True)
        for s in sets:
            name = "test_" + s; d = dump(cb, seed, name); y, ok = target_arrays(d)
            X = d["cond"].numpy()[ok].astype(np.float64); yy = (y[ok] == 1).astype(int)
            T = dict(X=X, y=yy, pid=patients(d)[ok], cb_zero=d["p_malignant"].numpy()[ok, 1].astype(np.float64), lin=lin.decision_function(X), P0=P0)
            zs.append(dict(encoder=enc, seed=seed, set=s, n=len(yy), auc_deployed=fast_auc(yy, T["cb_zero"]), auc_source_linear=fast_auc(yy, T["lin"]),
                           src_C=best[1], val_auc_linear=best[0]))
            for k in KS:
                if s.startswith("embed") and k > 100: continue
                tasks.append((enc, seed, s, k, T))
print("tasks", len(tasks), flush=True)
t0 = time.time()
out = Parallel(n_jobs=NJ, verbose=5)(delayed(task)(*t) for t in tasks)
print("done in", round(time.time() - t0), "s", flush=True)
df = pd.DataFrame([r for rs in out for r in rs]); save(df, "C_functional_form_draws.csv"); save(pd.DataFrame(zs), "C_zero_shot_source_linear.csv")
# paired summary
W = df.pivot_table(index=["encoder", "set", "k", "seed", "draw"], columns="variant", values="auc").reset_index()
CONTR = [("lin_zero", "cb_zero"), ("cb_refit", "cb_zero"), ("cb_refit", "lin_zero"), ("cb_correction", "cb_zero"), ("lin_correction", "lin_zero"), ("mlp_shrink", "cb_zero"), ("cb_refit", "mlp_shrink")]
sm = []
for (enc, s, k), g in W.groupby(["encoder", "set", "k"]):
    row = dict(encoder=enc, set=s, k=k, n_draws=len(g), **{f"mean_{v}": g[v].mean() for v in ("cb_zero", "lin_zero", "cb_refit", "cb_correction", "lin_correction", "mlp_shrink")})
    for a, b in CONTR:
        dlt = (g[a] - g[b]).values; row[f"{a}-{b}"] = dlt.mean(); row[f"{a}-{b}_lo"], row[f"{a}-{b}_hi"] = ci(dlt); row[f"{a}-{b}_fracgt0"] = float((dlt > 0).mean())
    sm.append(row)
save(pd.DataFrame(sm), "C_functional_form_summary.csv")
