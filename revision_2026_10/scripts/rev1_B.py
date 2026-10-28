
"""Analysis B: controlled oracle substitution.
 pred   : stored soft concept probabilities
 hard   : argmax one-hot of the predicted concepts of the group (same format as the reference input, no new information)
 oracle : reference concepts wherever labelled (one-hot), prediction elsewhere
Gain from information = oracle - hard; effect of input format alone = hard - pred.
Subsets: 'full' = breasts (with a malignancy label) in which every head of the group is labelled (oracle is complete for the group);
         'all_labelled' = all breasts with a malignancy label (the partial oracle of Table 4).
B2: labelled vs unlabelled characteristics.  B3: zero-shot AUC by reference BI-RADS stratum (endpoint comparability)."""
from rev1_common import *
B = int(os.environ.get("B", 1000)); sets = os.environ.get("SETS", ",".join(SETS)).split(",")
def probs_mode(d, group, mode):
    raw = {}
    for h in CONCEPT_HEADS:
        p = d[f"p_{h.name}"].float().clone()
        if h.parent: p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
        raw[h.name] = p
    for h in CONCEPT_HEADS:
        if h.name in group:
            if mode == "hard":
                raw[h.name] = F.one_hot(raw[h.name].argmax(1), h.n).float()
            elif mode == "oracle":
                y = d["labels"][:, HIDX[h.name]]; m = y != IGNORE
                r = raw[h.name].clone(); r[m] = F.one_hot(y[m], h.n).float(); raw[h.name] = r
    return {h.name: (raw[h.name] * raw[h.parent][:, 1:2] if h.parent else raw[h.name]) for h in CONCEPT_HEADS}
def score(head, d, group, mode):
    with torch.no_grad():
        return head(probs_mode(d, group, mode))["malignant"].softmax(-1)[:, 1].numpy()
rows, chars, strata = [], [], []
heads_cache = {}
for enc, (cb, opq) in ENC.items():
    for s in sets:
        name = "test_" + s
        dd = [dump(cb, k, name) for k in SEEDS]
        d0 = dd[0]; y, ok = target_arrays(d0); g = patients(d0)
        hd = [head_for(cb, k) for k in SEEDS]
        # sanity: re-scored predictions vs stored target probabilities
        chk = max(float(np.abs(score(hd[k], dd[k], [], "pred") - dd[k]["p_malignant"].numpy()[:, 1]).max()) for k in range(3))
        print(enc, s, "max |rescored - stored| =", round(chk, 5), flush=True)
        lab = {h.name: (d0["labels"][:, HIDX[h.name]].numpy() != IGNORE) for h in CONCEPT_HEADS}
        for grp, hn in GROUP_HEADS.items():
            full = np.all([lab[h] for h in hn], 0)
            for subset, mask in (("full", ok & full), ("all_labelled", ok)):
                if subset == "all_labelled" and grp == "all": continue
                m = mask; yy = (y[m] == 1).astype(int)
                if yy.sum() < 10 or (1 - yy).sum() < 10:
                    rows.append(dict(encoder=enc, set=s, group=grp, subset=subset, n=int(m.sum()), n_pos=int(yy.sum()), note="too few cases")); continue
                S = {md: np.stack([score(hd[k], dd[k], hn, md)[m] for k in range(3)]) for md in ("pred", "hard", "oracle")}
                idx_by = cluster_index(g[m]); rng = np.random.default_rng(1)
                bs = {k: [] for k in ("pred", "hard", "oracle", "hard-pred", "oracle-hard", "oracle-pred")}
                for _ in range(B):
                    ii = resample(idx_by, rng); y2 = yy[ii]
                    a = {md: np.nanmean([fast_auc(y2, S[md][k][ii]) for k in range(3)]) for md in S}
                    if any(v != v for v in a.values()): continue
                    for md in a: bs[md].append(a[md])
                    bs["hard-pred"].append(a["hard"] - a["pred"]); bs["oracle-hard"].append(a["oracle"] - a["hard"]); bs["oracle-pred"].append(a["oracle"] - a["pred"])
                pt = {md: float(np.mean([fast_auc(yy, S[md][k]) for k in range(3)])) for md in S}
                row = dict(encoder=enc, set=s, group=grp, subset=subset, n=int(m.sum()), n_pos=int(yy.sum()), n_patients=len(idx_by),
                           auc_pred=pt["pred"], auc_hard=pt["hard"], auc_oracle=pt["oracle"])
                for nm, val in (("hard-pred", pt["hard"] - pt["pred"]), ("oracle-hard", pt["oracle"] - pt["hard"]), ("oracle-pred", pt["oracle"] - pt["pred"])):
                    lo, hi = ci(bs[nm]); row[nm.replace("-", "_minus_")] = val; row[nm.replace("-", "_minus_") + "_lo"] = lo; row[nm.replace("-", "_minus_") + "_hi"] = hi
                rows.append(row)
        # B2: characteristics, labelled-full vs not (among malignancy-labelled breasts)
        bi = d0["labels"][:, HIDX["birads"]].numpy(); dens = d0["labels"][:, HIDX["density"]].numpy()
        for grp, hn in GROUP_HEADS.items():
            if enc != "frozen": continue
            full = np.all([lab[h] for h in hn], 0)
            for nm, m in (("group fully labelled", ok & full), ("not fully labelled", ok & ~full)):
                if m.sum() == 0: continue
                bk = bi[m]; bk = bk[bk != IGNORE]; dk = dens[m]; dk = dk[dk != IGNORE]
                chars.append(dict(set=s, group=grp, stratum=nm, n_breasts=int(m.sum()), n_patients=int(len(np.unique(g[m]))), malignant_frac=float((y[m] == 1).mean()),
                                  birads_known=int(len(bk)), birads_ge4_frac=float((bk >= 3).mean()) if len(bk) else np.nan, birads_ge3_frac=float((bk >= 2).mean()) if len(bk) else np.nan,
                                  density_known=int(len(dk)), density_CD_frac=float((dk >= 2).mean()) if len(dk) else np.nan))
        # B3: zero-shot ensemble AUC by reference BI-RADS stratum
        bi_ok = bi != IGNORE
        if bi_ok[ok].sum() > 50:
            do = [dump(opq, k, name) for k in SEEDS]
            ec = np.mean([d["p_malignant"].numpy()[:, 1] for d in dd], 0); eo = np.mean([d["p_malignant"].numpy()[:, 1] for d in do], 0)
            for sname, m in (("all malignancy-labelled", ok), ("reference BI-RADS >= 3", ok & bi_ok & (bi >= 2)), ("reference BI-RADS >= 4", ok & bi_ok & (bi >= 3))):
                yy = (y[m] == 1).astype(int)
                if yy.sum() < 10 or (1 - yy).sum() < 10: continue
                idx_by = cluster_index(g[m]); rng = np.random.default_rng(2); dl = []
                for _ in range(B):
                    ii = resample(idx_by, rng); a, b = fast_auc(yy[ii], ec[m][ii]), fast_auc(yy[ii], eo[m][ii])
                    if a == a and b == b: dl.append(a - b)
                lo, hi = ci(dl)
                strata.append(dict(encoder=enc, set=s, stratum=sname, n=int(m.sum()), n_pos=int(yy.sum()), prevalence=float(yy.mean()), auc_cbm_ens=fast_auc(yy, ec[m]), auc_opq_ens=fast_auc(yy, eo[m]),
                                   diff=fast_auc(yy, ec[m]) - fast_auc(yy, eo[m]), diff_lo=lo, diff_hi=hi))
save(pd.DataFrame(rows), "B_controlled_oracle.csv"); save(pd.DataFrame(chars), "B2_label_characteristics.csv"); save(pd.DataFrame(strata), "B3_birads_strata.csv")
