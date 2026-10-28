
"""Analysis A: seed-ensemble paired AUC difference CBM - opaque with a patient-clustered bootstrap (replaces the 'mean of per-seed limits' in Table 2)."""
import sys; from rev1_common import *
B = int(os.environ.get("B", 2000)); sets = os.environ.get("SETS", ",".join(SETS)).split(",")
rows, diag = [], []
for enc, (cb, opq) in ENC.items():
    for s in sets:
        name = "test_" + s
        dcb = [dump(cb, k, name) for k in SEEDS]; dop = [dump(opq, k, name) for k in SEEDS]
        assert all(d["sample_id"] == dcb[0]["sample_id"] for d in dcb + dop)
        y, ok = target_arrays(dcb[0]); y = (y[ok] == 1).astype(int); g = patients(dcb[0])[ok]
        if y.sum() < 5 or (1 - y).sum() < 5: continue
        pc = np.stack([d["p_malignant"].numpy()[ok, 1] for d in dcb]); po = np.stack([d["p_malignant"].numpy()[ok, 1] for d in dop])
        ec, eo = pc.mean(0), po.mean(0)
        idx_by = cluster_index(g); rng = np.random.default_rng(0)
        bc, bo, bd = [], [], []; per_seed = {k: [] for k in range(3)}
        for _ in range(B):
            ii = resample(idx_by, rng); yy = y[ii]
            a, b = fast_auc(yy, ec[ii]), fast_auc(yy, eo[ii])
            if a == a and b == b: bc.append(a); bo.append(b); bd.append(a - b)
            for k in range(3):
                a_, b_ = fast_auc(yy, pc[k][ii]), fast_auc(yy, po[k][ii])
                if a_ == a_ and b_ == b_: per_seed[k].append(a_ - b_)
        seed_auc_cb = [fast_auc(y, pc[k]) for k in range(3)]; seed_auc_op = [fast_auc(y, po[k]) for k in range(3)]
        seed_diff = [a - b for a, b in zip(seed_auc_cb, seed_auc_op)]
        # old Table-2 style: mean of per-seed bootstrap limits
        lims = [ci(per_seed[k]) for k in range(3)]
        rows.append(dict(encoder=enc, set=s, n=len(y), n_pos=int(y.sum()), n_patients=len(idx_by),
            auc_cbm_ens=fast_auc(y, ec), auc_cbm_ens_lo=ci(bc)[0], auc_cbm_ens_hi=ci(bc)[1],
            auc_opq_ens=fast_auc(y, eo), auc_opq_ens_lo=ci(bo)[0], auc_opq_ens_hi=ci(bo)[1],
            diff_ens=fast_auc(y, ec) - fast_auc(y, eo), diff_lo=ci(bd)[0], diff_hi=ci(bd)[1],
            frac_boot_diff_gt0=float(np.mean(np.array(bd) > 0)),
            mean_seed_diff=float(np.mean(seed_diff)), mean_seed_lo=float(np.mean([l[0] for l in lims])), mean_seed_hi=float(np.mean([l[1] for l in lims])),
            n_seeds_excl0=int(sum((l[0] > 0) or (l[1] < 0) for l in lims)),
            cbm_seed_aucs=";".join(f"{v:.4f}" for v in seed_auc_cb), opq_seed_aucs=";".join(f"{v:.4f}" for v in seed_auc_op)))
        # do the three seeds give genuinely different score vectors?
        from scipy.stats import spearmanr
        for arm, P in (("cbm", pc), ("opq", po)):
            for i in range(3):
                for j in range(i + 1, 3):
                    diag.append(dict(encoder=enc, set=s, arm=arm, seed_a=i, seed_b=j, spearman=float(spearmanr(P[i], P[j])[0]),
                                     n_identical_scores=int((P[i] == P[j]).sum()), n=len(y), n_unique_a=int(len(np.unique(P[i])))))
        print(enc, s, rows[-1]["diff_ens"], rows[-1]["diff_lo"], rows[-1]["diff_hi"], flush=True)
save(pd.DataFrame(rows), "A_seed_ensemble_table2.csv"); save(pd.DataFrame(diag), "A_seed_vector_diagnostics.csv")
