"""LaTeX fragments for the MMC Astana (Kazakhstan) cohort, generated from the aggregate result tables so that the
manuscript numbers match mmc_results.csv / mmc_fewshot_summary.csv / mmc_cohort.json exactly. Writes mmc_fragments.json."""
import json

import pandas as pd

res = pd.read_csv("mmc_results.csv"); summ = pd.read_csv("mmc_fewshot_summary.csv"); pairs = pd.read_csv("mmc_fewshot_paired.csv")
coh = json.load(open("mmc_cohort.json")); corr = pd.read_csv("mmc_post_nact_corr.csv")
R, O, P = coh["retro"], coh["outpatient"], coh["post_nact"]


def z(ep, sub, sc, arm, bb, seed="ens"):
    r = res[(res.endpoint == ep) & (res.subset == sub) & (res.score == sc) & (res.arm == arm) & (res.backbone == bb) & (res.seed == seed)]
    assert len(r) == 1, (ep, sub, sc, arm, bb)
    return r.iloc[0]


def ci(r):
    return f"{r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]"


def f(bb, m, k):
    r = summ[(summ.backbone == bb) & (summ.method == m) & (summ.k == k)]
    assert len(r) == 1
    return r.iloc[0]["mean"]


def pdm(bb, k, col):
    return pairs[(pairs.backbone == bb) & (pairs.k == k)].iloc[0][col]


rows = [("Retrospective, all", "retro_M34_vs_M12", "all", "birads3"),
        ("\\quad Siemens (DR)", "retro_M34_vs_M12", "device_Siemens", "birads3"),
        ("\\quad Agfa (CR)", "retro_M34_vs_M12", "device_Agfa", "birads3"),
        ("Outpatient, M6 vs M1--M2", "outpatient_M6_vs_M12", "breast", "malig"),
        ("Post-NACT, residual vs complete", "post_nact_residual_vs_complete", "cancer_breast_hq", "malig")]
tab = ["\\begin{table}[t]\\centering\\small",
       "\\caption{Zero-shot AUC at the Kazakhstani clinic (seed ensembles; 95\\% CIs from a bootstrap over patients). "
       "Retrospective studies are scored by the maximum over breasts of $P(\\text{BI-RADS}\\geq3)$; outpatient and post-NACT breasts by "
       "$P(\\text{malignant})$. $n$ = studies (retrospective) or breasts; positives in parentheses.}\\label{tab:mmc}",
       "\\begin{tabular}{lrcccc}\\toprule",
       "& & \\multicolumn{2}{c}{Frozen DINOv2-B} & \\multicolumn{2}{c}{Fine-tuned (last 2 blocks)}\\\\\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
       "Endpoint & $n$ (pos.) & CBM & Opaque & CBM & Opaque\\\\\\midrule"]
for lab, ep, sub, sc in rows:
    r0 = z(ep, sub, sc, "cb", "dinov2")
    cells = [ci(z(ep, sub, sc, a, b)) for a, b in (("cb", "dinov2"), ("opq", "dinov2"), ("cb", "ft2"), ("opq", "ft2"))]
    tab.append(f"{lab} & {int(r0.n)} ({int(r0.n_pos)}) & " + " & ".join(cells) + "\\\\")
tab += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
d0 = z("retro_M34_vs_M12", "all", "birads3", "cb-opq", "dinov2"); d1 = z("retro_M34_vs_M12", "all", "birads3", "cb-opq", "ft2")
cb_c = corr[corr.run == "cb_dinov2_ens"].iloc[0]
F = {
    "table": "\n".join(tab),
    "n_retro": R["studies"], "n_groups": R["patient_groups"], "n_pos": R["pos_M3_M4"], "M": R["M_counts"], "dev": R["devices"],
    "n_out": O["breasts"], "pos_out": O["pos_M6"], "n_pn": P["cancer_breasts_high_quality"], "pn_res": P["residual"], "pn_cr": P["complete"],
    "zs_cb": [round(z("retro_M34_vs_M12", "all", "birads3", "cb", b).auc, 3) for b in ("dinov2", "ft2")],
    "zs_opq": [round(z("retro_M34_vs_M12", "all", "birads3", "opq", b).auc, 3) for b in ("dinov2", "ft2")],
    "d_frozen": ci(d0), "d_ft": ci(d1),
    "fs": {bb: {m: {k: round(f(bb, m, k), 3) for k in (10, 25, 50, 100)} for m in summ.method.unique()} for bb in ("dinov2", "ft2")},
    "corr_off_vs_opq_off_k25": round(pdm("dinov2", 25, "cb_offset_cvC-opq_offset_cvC|mean"), 3),
    "corr_frac_k25": round(pdm("dinov2", 25, "cb_offset_cvC-opq_offset_cvC|frac_gt0"), 2),
    "out_cb": ci(z("outpatient_M6_vs_M12", "breast", "malig", "cb", "dinov2")), "out_opq": ci(z("outpatient_M6_vs_M12", "breast", "malig", "opq", "dinov2")),
    "pn_cb": ci(z("post_nact_residual_vs_complete", "cancer_breast_hq", "malig", "cb", "dinov2")),
    "rho": round(float(cb_c.spearman_regression_pct_vs_Pmalig), 2), "rho_p": round(float(cb_c.p), 2), "rho_n": int(cb_c.n)}
gain = max(f(bb, m, k) - f(bb, f"{m.split('_')[0]}_zero", k) for bb in ("dinov2", "ft2")
           for m in ("cb_refit_cvC", "cb_offset_cvC", "opq_refit_cvC", "opq_offset_cvC") for k in (10, 25, 50, 100))
F["max_gain"] = round(gain, 3)
json.dump(F, open("mmc_fragments.json", "w"), indent=1)
print(F["table"]); print({k: v for k, v in F.items() if k not in ("table", "fs")}); print(F["fs"]["dinov2"]["cb_refit_fixedC"], F["fs"]["dinov2"]["opq_refit_fixedC"])
