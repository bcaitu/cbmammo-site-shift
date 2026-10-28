
"""Analysis D: nested resampling for the paired few-shot contrasts.
Outer replicate b: (1) resample the target patients with replacement (finite-population uncertainty), (2) within it run M adaptation draws
(k distinct patients adapt, the remaining resampled patients, with multiplicity, evaluate), (3) record the mean paired AUC difference per
training seed. Interval = percentile of the outer replicates; training seeds are either held fixed (mean of the 3 seeds) or resampled with replacement."""
from rev1_common import *
from joblib import Parallel, delayed
from cbmammo.fewshot import choose_C, fit_predict, offset_lr, logit, FIXED_C, GRID
NJ = int(os.environ.get("NJOBS", 40)); B = int(os.environ.get("B", 200)); M = int(os.environ.get("M", 15))
sets = os.environ.get("SETS", "cdd_cesm").split(","); KS = [int(x) for x in os.environ.get("KS", "25,100").split(",")]
CONTR = [("cb_refit", "cb_zero"), ("cb_refit", "opq_refit"), ("cb_correction", "cb_zero"), ("cb_zero", "opq_zero"), ("cb_refit", "opq_zero")]

def one_seed(T, fit, ev, dr):
    X, Z, y = T["X"], T["Z"], T["y"]; zc, zo = T["cb_zero"], T["opq_zero"]; r = {}
    r["cb_zero"], r["opq_zero"] = zc[ev], zo[ev]
    Cc = choose_C(X[fit], y[fit], False, dr); r["cb_refit"] = fit_predict(X[fit], y[fit], X[ev], Cc if Cc is not None else FIXED_C["cb"], False)
    oc = logit(zc); C1 = choose_C(X[fit], y[fit], False, dr, offset=oc[fit]); C1 = C1 if C1 is not None else GRID[0]
    r["cb_correction"] = offset_lr(X[fit], y[fit], oc[fit], X[ev], oc[ev], C1, False)
    Co = choose_C(Z[fit], y[fit], True, dr); r["opq_refit"] = fit_predict(Z[fit], y[fit], Z[ev], Co if Co is not None else FIXED_C["opq"], True)
    return r

def replicate(enc, s, k, b, Ts, rows_of, ups):
    rng = np.random.default_rng(100003 * b + 17 * k + len(s) + (7 if enc == "tuned" else 0))
    boot = rng.choice(len(ups), len(ups), replace=True); U, cnt = np.unique(boot, return_counts=True)
    y0 = Ts[0]["y"]; acc = np.full((3, len(CONTR)), np.nan); sums = np.zeros((3, len(CONTR))); nn = 0
    for dr in range(M):
        for _ in range(50):
            pick = rng.choice(len(U), k, replace=False); mask = np.ones(len(U), bool); mask[pick] = False
            fit = np.concatenate([rows_of[U[j]] for j in pick])
            ev = np.concatenate([np.tile(rows_of[U[j]], cnt[j]) for j in np.flatnonzero(mask)])
            if len(np.unique(y0[fit])) == 2 and len(np.unique(y0[ev])) == 2: break
        else: continue
        for si, T in enumerate(Ts):
            res = one_seed(T, fit, ev, dr); au = {m: fast_auc(T["y"][ev], v) for m, v in res.items()}
            for ci_, (a, bb) in enumerate(CONTR): sums[si, ci_] += au[a] - au[bb]
        nn += 1
    if nn == 0: return b, acc
    return b, sums / nn

rows, summ = [], []
for enc, (cb, opq) in ENC.items():
    for s in sets:
        name = "test_" + s; Ts = []
        for seed in SEEDS:
            d, do = dump(cb, seed, name), dump(opq, seed, name); y, ok = target_arrays(d)
            Ts.append(dict(X=d["cond"].numpy()[ok].astype(np.float64), Z=do["cond"].numpy()[ok].astype(np.float64), y=(y[ok] == 1).astype(int),
                           cb_zero=d["p_malignant"].numpy()[ok, 1].astype(np.float64), opq_zero=do["p_malignant"].numpy()[ok, 1].astype(np.float64)))
            pid = patients(d)[ok]
        ups, inv = np.unique(pid, return_inverse=True); rows_of = [np.flatnonzero(inv == j) for j in range(len(ups))]
        for k in KS:
            if s.startswith("embed") and k > 100: continue
            t0 = time.time()
            out = Parallel(n_jobs=NJ)(delayed(replicate)(enc, s, k, b, Ts, rows_of, ups) for b in range(B))
            arr = np.stack([o[1] for o in sorted(out, key=lambda z: z[0])])          # [B, 3, n_contrasts]
            rng = np.random.default_rng(5); sel = rng.integers(0, 3, (arr.shape[0], 3))
            for ci_, (a, bb) in enumerate(CONTR):
                fixed = np.nanmean(arr[:, :, ci_], 1); res = np.array([np.nanmean(arr[i, sel[i], ci_]) for i in range(arr.shape[0])])
                summ.append(dict(encoder=enc, set=s, k=k, contrast=f"{a} - {bb}", mean=float(np.nanmean(fixed)), lo_seeds_fixed=ci(fixed)[0], hi_seeds_fixed=ci(fixed)[1],
                                 lo_seeds_resampled=ci(res)[0], hi_seeds_resampled=ci(res)[1], frac_gt0=float(np.nanmean(fixed > 0)), n_outer=int(np.isfinite(fixed).sum()),
                                 sd_between_seeds=float(np.nanstd(np.nanmean(arr[:, :, ci_], 0), ddof=1))))
            for i in range(arr.shape[0]):
                for si in range(3):
                    for ci_, (a, bb) in enumerate(CONTR): rows.append(dict(encoder=enc, set=s, k=k, b=i, seed=si, contrast=f"{a} - {bb}", value=arr[i, si, ci_]))
            print(enc, s, k, "done in", round(time.time() - t0), "s", flush=True)
            tag = "_".join(sets)
            save(pd.DataFrame(summ), f"D_nested_resampling_summary_{tag}.csv")
save(pd.DataFrame(rows), "D_nested_resampling_replicates_" + "_".join(sets) + ".csv")
