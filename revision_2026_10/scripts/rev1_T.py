
"""Wall-clock CPU time of one k=25 adaptation for the head refits (single thread), CDD-CESM, frozen encoder, seed 0, 30 draws (same draws as analysis C)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
from rev1_common import *
from cbmammo.fewshot import choose_C, fit_predict, offset_lr, logit, FIXED_C, GRID
K = 25; rows = []
for enc, (cb, opq) in ENC.items():
    d, do = dump(cb, 0, "test_cdd_cesm"), dump(opq, 0, "test_cdd_cesm"); y, ok = target_arrays(d)
    X = d["cond"].numpy()[ok].astype(np.float64); Z = do["cond"].numpy()[ok].astype(np.float64); yy = (y[ok] == 1).astype(int); pid = patients(d)[ok]
    zc = d["p_malignant"].numpy()[ok, 1].astype(np.float64); oc = logit(zc)
    rng = np.random.default_rng(7919 * 0 + 31 * K + len("cdd_cesm")); ups = np.unique(pid)
    for dr in range(30):
        for _ in range(100):
            fp = rng.choice(ups, K, replace=False); inf = np.isin(pid, fp); fit, ev = np.flatnonzero(inf), np.flatnonzero(~inf)
            if len(np.unique(yy[fit])) == 2 and len(np.unique(yy[ev])) == 2: break
        t0 = time.perf_counter(); Cc = choose_C(X[fit], yy[fit], False, dr); fit_predict(X[fit], yy[fit], X[ev], Cc if Cc is not None else FIXED_C["cb"], False); t_cb = time.perf_counter() - t0
        t0 = time.perf_counter(); C1 = choose_C(X[fit], yy[fit], False, dr, offset=oc[fit]); C1 = C1 if C1 is not None else GRID[0]; offset_lr(X[fit], yy[fit], oc[fit], X[ev], oc[ev], C1, False); t_corr = time.perf_counter() - t0
        t0 = time.perf_counter(); Co = choose_C(Z[fit], yy[fit], True, dr); fit_predict(Z[fit], yy[fit], Z[ev], Co if Co is not None else FIXED_C["opq"], True); t_op = time.perf_counter() - t0
        rows.append(dict(encoder=enc, draw=dr, t_cb_refit_s=t_cb, t_cb_correction_s=t_corr, t_opq_refit_s=t_op))
df = pd.DataFrame(rows); save(df, "E_cpu_timing_head_refits.csv")
print(df.groupby("encoder").median(numeric_only=True).round(3))
