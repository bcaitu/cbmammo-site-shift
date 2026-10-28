
"""Builds the LaTeX tables of the MAKE revision from the aggregate result CSVs (analysis outputs in out/, repo tables in repo/results/tables).
Usage: python build_tables.py <out_dir with rev1 CSVs> <repo_tables_dir> <tex_dir>"""
import os, sys, numpy as np, pandas as pd
OUTD, REPO, TEX = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/"
os.makedirs(TEX, exist_ok=True)
SETLAB = {"cbis_ddsm":"CBIS-DDSM (internal)","cdd_cesm":"CDD-CESM","cmmd":"CMMD","embed":"EMBED, Hologic","embed_ge":"EMBED, GE-holdout patients"}
def sg(x, nd=3, interval=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    x = float(x); v = round(x, nd)
    if v == 0 and not interval: return f"{0:.{nd}f}"
    neg = x < 0
    return ("$-$" if neg else "$+$") + f"{abs(v):.{nd}f}"
def iv(lo, hi, nd=3): return f"[{sg(lo,nd,True)}, {sg(hi,nd,True)}]"
def f3(x, nd=3): return f"{x:.{nd}f}"
def pm(m, s): return f"{m:.3f}{{\\scriptsize$\\pm${s:.3f}}}"
def w(name, s): open(TEX + name, "w").write(s); print("wrote", name)

# ------------------------------------------------------------- Table 2 (analysis A)
A = pd.read_csv(OUTD + "A_seed_ensemble_table2.csv"); AM = pd.read_csv(REPO + "A_malignant_auc.csv")
CFG = {("cb","frozen"):"cb_embed_dinov2",("cb","tuned"):"cb_embed_ft2",("opq","frozen"):"opq_embed_dinov2",("opq","tuned"):"opq_embed_ft2"}
def msd(arm, enc, s):
    r = AM[(AM.config==CFG[(arm,enc)])&(AM.set=="test_"+s)].iloc[0]; return r.auc_mean, r.auc_sd
rows = []
for s in ("cbis_ddsm","cdd_cesm","cmmd","embed","embed_ge"):
    cells = [SETLAB[s], f"{int(A[A.set==s].n.iloc[0]):,}"]
    for enc in ("frozen","tuned"):
        r = A[(A.set==s)&(A.encoder==enc)].iloc[0]
        cells += [pm(*msd("cb",enc,s)), pm(*msd("opq",enc,s)), f"{sg(r.diff_ens)} {iv(r.diff_lo, r.diff_hi)}"]
    rows.append(" & ".join(cells) + "\\\\")
w("table2_zeroshot.tex", r"""\begin{table}[H]
\small
\caption{Zero-shot malignancy AUC of the concept bottleneck model (CBM) and the matched opaque model. Columns CBM and Opaque: mean$\pm$SD over three training seeds. The difference is the AUC of the seed-ensembled CBM (mean predicted probability over the three seeds) minus that of the seed-ensembled opaque model, with a 95\% interval from a paired patient-clustered bootstrap (2,000 resamples of patients; both models scored on the same resample). $n$ = breasts with a malignancy label (EMBED: biopsied breasts only).}\label{tab:zeroshot}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}\toprule
 & & \multicolumn{3}{c}{Frozen DINOv2-B} & \multicolumn{3}{c}{Last 2 blocks tuned}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Test set & $n$ & CBM & Opaque & CBM $-$ opaque (seed ensemble) & CBM & Opaque & CBM $-$ opaque (seed ensemble)\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------- Table 3 (reference BI-RADS strata)
B3 = pd.read_csv(OUTD + "B3_birads_strata.csv")
SN = {"all malignancy-labelled":"All labelled breasts","reference BI-RADS >= 3":"Reference BI-RADS $\\geq$3","reference BI-RADS >= 4":"Reference BI-RADS $\\geq$4"}
rows = []
for s in ("cbis_ddsm","cdd_cesm","embed","embed_ge"):
    first = True
    for st in SN:
        x = B3[(B3.set==s)&(B3.stratum==st)]; r0 = x.iloc[0]
        cells = [SETLAB[s] if first else "", SN[st], f"{int(r0.n)}", f"{r0.prevalence*100:.0f}\\%"]; first = False
        for enc in ("frozen","tuned"):
            r = x[x.encoder==enc].iloc[0]
            if st == "all malignancy-labelled":      # identical resamples and values as Table 2
                a = A[(A.set==s)&(A.encoder==enc)].iloc[0]; cells += [f3(a.auc_cbm_ens), f3(a.auc_opq_ens), f"{sg(a.diff_ens)} {iv(a.diff_lo, a.diff_hi)}"]
            else:
                cells += [f3(r.auc_cbm_ens), f3(r.auc_opq_ens), f"{sg(r['diff'])} {iv(r.diff_lo, r.diff_hi)}"]
        rows.append(" & ".join(cells) + "\\\\")
    rows[-1] += "\\addlinespace"
w("table3_strata.tex", r"""\begin{table}[H]
\small
\caption{Zero-shot malignancy AUC of the seed-ensembled models within strata defined by the reference BI-RADS category of the breast (breasts with a reference category only). Malignant = share of malignant breasts among the labelled breasts of the stratum. The difference is CBM minus opaque with a 95\% paired patient-clustered bootstrap interval (2,000 resamples for all labelled breasts, as in Table~\ref{tab:zeroshot}; 1,000 for the strata). The strata use the reference reading and are descriptive, not an independent endpoint.}\label{tab:strata}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrcccccc}\toprule
 & & & & \multicolumn{3}{c}{Frozen DINOv2-B} & \multicolumn{3}{c}{Last 2 blocks tuned}\\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Test set & Stratum & $n$ & Malignant & CBM & Opaque & CBM $-$ opaque & CBM & Opaque & CBM $-$ opaque\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------- Table 4 (controlled oracle) and appendix tables
Bo = pd.read_csv(OUTD + "B_controlled_oracle.csv")
GL = {"density":"Density","mass":"Mass (shape, margin)","calc":"Calcification","other":"Asymmetry, distortion"}
rows = []
for s in ("cbis_ddsm","cdd_cesm","embed","embed_ge"):
    first = True
    for g in ("density","mass","calc","other"):
        x = Bo[(Bo.set==s)&(Bo.group==g)&(Bo.subset=="full")&(Bo.note.isna())]
        if len(x) == 0: continue
        r0 = x.iloc[0]
        if r0.n_pos < 10 or r0.n - r0.n_pos < 10: continue
        dag = "$^{\\dagger}$" if r0.n < 50 else ""
        cells = [SETLAB[s] if first else "", GL[g], f"{int(r0.n)} ({int(r0.n_pos)})" + dag]; first = False
        for enc in ("frozen","tuned"):
            r = x[x.encoder==enc].iloc[0]
            cells += [f3(r.auc_pred), f3(r.auc_hard), f3(r.auc_oracle), f"{sg(r.oracle_minus_hard)} {iv(r.oracle_minus_hard_lo, r.oracle_minus_hard_hi)}"]
        rows.append(" & ".join(cells) + "\\\\")
    rows[-1] += "\\addlinespace"
w("table4_oracle.tex", r"""\begin{table}[H]
\small
\caption{Controlled concept substitution for malignancy (CBM, AUC averaged over three seeds), restricted to breasts in which \emph{every} head of the group is labelled, so that the substitution is complete for that group. Pred: predicted soft probabilities; Hard: the predicted concepts of the group replaced by their one-hot arg-max (the format of the reference input, without new information); Oracle: reference values. The last column is the information effect (oracle minus hard) with a 95\% patient-clustered bootstrap interval (1,000 resamples; the mean over seeds is resampled jointly). $n$ = breasts (malignant). The subsets are selected by the presence of labels (for the mass group, breasts with a mass), so their AUCs are not comparable with whole-set AUCs. $^{\dagger}$ fewer than 50 breasts. CMMD has no descriptor labels and is not shown; groups with fewer than 10 cases of either class are omitted.}\label{tab:oracle}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrcccccccc}\toprule
 & & & \multicolumn{4}{c}{Frozen DINOv2-B} & \multicolumn{4}{c}{Last 2 blocks tuned}\\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}
Test set & Group & $n$ (pos.) & Pred & Hard & Oracle & Oracle $-$ hard & Pred & Hard & Oracle & Oracle $-$ hard\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

rows = []
for s in ("cbis_ddsm","cdd_cesm","cmmd","embed","embed_ge"):
    first = True
    for g in ("density","mass","calc","other"):
        x = Bo[(Bo.set==s)&(Bo.group==g)&(Bo.subset=="all_labelled")&(Bo.note.isna())]
        cells = [SETLAB[s] if first else "", GL[g]]; first = False
        for enc in ("frozen","tuned"):
            r = x[x.encoder==enc].iloc[0]; cells += [sg(r.oracle_minus_pred), sg(r.hard_minus_pred), sg(r.oracle_minus_hard)]
        rows.append(" & ".join(cells) + "\\\\")
    rows[-1] += "\\addlinespace"
w("tableA_oracle_partial.tex", r"""\begin{table}[H]
\small
\caption{Partial concept substitution for malignancy on all labelled breasts (CBM, mean over three seeds; the whole-set design, extended by the hard-prediction control). Columns give AUC changes: oracle minus pred (reference values substituted wherever labelled), hard minus pred (the group replaced by its arg-max one-hot prediction) and oracle minus hard (information effect). Coverage of each group is given in Table~\ref{tab:labelchars} and in the text; CMMD supplies presence labels for masses and calcifications only.}\label{tab:oracle_partial}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}\toprule
 & & \multicolumn{3}{c}{Frozen DINOv2-B} & \multicolumn{3}{c}{Last 2 blocks tuned}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Test set & Group & Oracle $-$ pred & Hard $-$ pred & Oracle $-$ hard & Oracle $-$ pred & Hard $-$ pred & Oracle $-$ hard\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

ch = pd.read_csv(OUTD + "B2_label_characteristics.csv"); rows = []
for s in ("cbis_ddsm","cdd_cesm","embed","embed_ge"):
    first = True
    for st, lab in (("group fully labelled","All mass heads labelled"),("not fully labelled","Not fully labelled")):
        r = ch[(ch.set==s)&(ch.group=="mass")&(ch.stratum==st)].iloc[0]
        rows.append(" & ".join([SETLAB[s] if first else "", lab, f"{int(r.n_breasts)}", f"{int(r.n_patients)}", f"{r.malignant_frac*100:.0f}\\%",
                                f"{r.birads_ge4_frac*100:.0f}\\% ({int(r.birads_known)})", f"{r.density_CD_frac*100:.0f}\\%"]) + "\\\\"); first = False
    rows[-1] += "\\addlinespace"
w("tableA_labelchars.tex", r"""\begin{table}[H]
\small
\caption{Breasts with and without a complete mass annotation (all three mass heads labelled), among the breasts with a malignancy label. Malignant: share of malignant breasts. BI-RADS 4--5: share of breasts with a reference category of 4 or 5 (number of breasts with a reference category in parentheses). Dense: share of breasts of density C or D.}\label{tab:labelchars}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrr}\toprule
Test set & Stratum & Breasts & Patients & Malignant & BI-RADS 4--5 & Dense\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------- Table (analysis C: head form control)
S = pd.read_csv(OUTD + "C_functional_form_summary.csv")
rows = []
for enc, lab in (("frozen","Frozen DINOv2-B"),("tuned","Last 2 blocks tuned")):
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for s in ("cdd_cesm","cmmd","embed","embed_ge"):
        for k in (25, 100):
            r = S[(S.encoder==enc)&(S.set==s)&(S.k==k)].iloc[0]
            rows.append(" & ".join([SETLAB[s] if k == 25 else "", str(k), f3(r.mean_cb_zero), f3(r.mean_lin_zero), f3(r.mean_mlp_shrink), f3(r.mean_cb_refit), f3(r.mean_cb_correction),
                                    f"{sg(r['cb_refit-lin_zero'])} {iv(r['cb_refit-lin_zero_lo'], r['cb_refit-lin_zero_hi'])}"]) + "\\\\")
w("table_headform.tex", r"""\begin{table}[H]
\small
\caption{Control for the form of the concept-to-diagnosis head (CBM; mean AUC over 3 seeds $\times$ 30 new draws of $k$ adaptation patients). Deployed: the original 128-unit MLP head. Source linear: an L2 logistic head fitted on the source training data (no local patients). MLP tuned: the deployed MLP fine-tuned on the $k$ patients with an L2 penalty towards its deployed weights (penalty chosen by inner cross-validation). Linear refit: the head refit of the main analysis. Correction: deployed logit plus a penalised linear correction. The last column is linear refit minus source linear head (mean [2.5--97.5\% range over 90 draws]).}\label{tab:headform}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrccccc>{\centering\arraybackslash}p{4.2cm}}\toprule
Target set & $k$ & Deployed & Source linear & MLP tuned & Linear refit & Correction & Linear refit $-$ source linear\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------- Table (analysis D: nested resampling)
DDf = pd.read_csv(OUTD + "D_nested_resampling_summary_all.csv")
def cell(r):
    ex = (r.lo_seeds_resampled > 0) or (r.hi_seeds_resampled < 0)
    s = f"{sg(r['mean'])} {iv(r.lo_seeds_resampled, r.hi_seeds_resampled)}"
    return "\\textbf{" + s + "}" if ex else s
rows = []
for enc, lab in (("frozen","Frozen DINOv2-B"),("tuned","Last 2 blocks tuned")):
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{lab}}}}}\\\\")
    for s in ("cdd_cesm","cmmd","embed","embed_ge"):
        for k in (25, 100):
            x = DDf[(DDf.encoder==enc)&(DDf.set==s)&(DDf.k==k)]
            g = lambda c: x[x.contrast==c].iloc[0]
            rows.append(" & ".join([SETLAB[s] if k == 25 else "", str(k), cell(g("cb_refit - cb_zero")), cell(g("cb_correction - cb_zero")), cell(g("cb_refit - opq_refit"))]) + "\\\\")
w("table_nested.tex", r"""\begin{table}[H]
\small
\caption{Paired AUC differences of the head-refit variants with the sampling of the target patients and of the training seeds included (nested resampling; mean [2.5--97.5\% range over outer replicates], 200 replicates of 15 draws for CDD-CESM and 150 replicates of 10 draws otherwise; seeds resampled with replacement). Bold: range excludes zero. Intervals with the seeds held fixed are in the supplementary tables and are narrower. The partial fine-tuning contrast is not included.}\label{tab:nested}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrccc}\toprule
Target set & $k$ & CBM refit $-$ CBM deployed & CBM correction $-$ CBM deployed & CBM refit $-$ opaque head refit\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------- Tables from analysis E (partial fine-tuning sensitivity, cost)
import glob
CFGLAB = {"default":"Default (lr $10^{-4}$, 6 epochs, last block)","lr3e-5":"lr $3\\times10^{-5}$","lr3e-4":"lr $3\\times10^{-4}$","ep3":"3 epochs","ep12":"12 epochs","blocks2":"Last 2 blocks unfrozen"}
Ef = [pd.read_csv(f) for f in glob.glob(OUTD + "E_pft_*_cdd_cesm_s*.csv")]
if Ef:
    E = pd.concat(Ef); ncfg = E.config.nunique()
    full = E.groupby(["encoder","seed","draw"]).config.nunique().reset_index(); full = full[full.config == 6][["encoder","seed","draw"]]
    E = E.merge(full, on=["encoder","seed","draw"])                      # only draws with all six configurations
    Cd = pd.read_csv(OUTD + "C_functional_form_draws.csv"); Cd = Cd[(Cd.set=="cdd_cesm")&(Cd.k==25)&(Cd.variant=="cb_refit")][["encoder","seed","draw","auc"]].rename(columns={"auc":"auc_cb_refit"})
    M = E.merge(Cd, on=["encoder","seed","draw"]); M["diff"] = M.auc_cb_refit - M.auc
    rows = []; nd = M.groupby("encoder").apply(lambda g: g[["seed","draw"]].drop_duplicates().shape[0])
    for cfg in CFGLAB:
        cells = [CFGLAB[cfg]]
        for enc in ("frozen","tuned"):
            g = M[(M.encoder==enc)&(M.config==cfg)]
            lo, hi = np.percentile(g["diff"], [2.5, 97.5])
            cells += [f3(g.auc.mean()), f3(g.auc_cb_refit.mean()), f"{sg(g['diff'].mean())} {iv(lo, hi)}"]
        rows.append(" & ".join(cells) + "\\\\")
    nfro, ntun = int(nd.get("frozen", 0)), int(nd.get("tuned", 0))
    w("table_pft.tex", r"""\begin{table}[H]
\small
\caption{Sensitivity of the partial fine-tuning baseline on CDD-CESM at $k=25$ (one factor changed at a time from the default; """ + f"{nfro} (frozen) and {ntun} (tuned)" + r""" adaptation draws, the first draws of the head-form runs, so that every row uses the same patients). AUC on the remaining patients, mean over draws; CBM refit: the linear head refit of the main analysis on the same draws; the difference is CBM refit minus partial fine-tuning (mean [2.5--97.5\% range over draws]).}\label{tab:pft}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}\toprule
 & \multicolumn{3}{c}{Frozen DINOv2-B} & \multicolumn{3}{c}{Last 2 blocks tuned}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Partial fine-tuning configuration & Partial FT & CBM refit & CBM refit $-$ partial FT & Partial FT & CBM refit & CBM refit $-$ partial FT\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n\\end{table}\n")
    M.to_csv(OUTD + "E_pft_paired_with_headrefit.csv", index=False)
    # cost table
    d = E[E.config=="default"]; T = pd.read_csv(OUTD + "E_cpu_timing_head_refits.csv")
    tm = T.groupby("encoder").median(numeric_only=True)
    cpu = lambda col: f"{min(tm[col]):.2f}" if abs(tm[col].max()-tm[col].min()) < 0.01 else f"{tm[col].min():.2f}--{tm[col].max():.2f}"
    w("table_cost.tex", r"""\begin{table}[H]
\small
\caption{Cost of one adaptation at $k=25$ (CDD-CESM). CPU times: median of 30 draws on one thread, including the choice of the penalty by inner cross-validation. Partial fine-tuning: median wall-clock time of the training loop (6 epochs) and of the evaluation of the remaining patients, and peak GPU memory, over """ + f"{len(d)}" + r""" runs (both encoders) on a shared server under load from other jobs, so the times are upper bounds.}\label{tab:cost}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrll}\toprule
Strategy & Parameters updated & Inputs needed & Time and memory\\\midrule
CBM head refit (L2 logistic regression) & 31 & Stored 30-dimensional concept vectors & """ + f"{cpu('t_cb_refit_s')} s CPU; negligible memory" + r"""\\
CBM correction of the deployed head & 31 & As above & """ + f"{cpu('t_cb_correction_s')} s CPU" + r"""\\
Opaque head refit & 513 & Stored 512-dimensional vectors & """ + f"{cpu('t_opq_refit_s')} s CPU" + r"""\\
Opaque partial fine-tuning (default) & 16,566,565 (17.2\% of the model) & Images of the adaptation patients and a GPU & """ + f"training {d.t_fit_s.median():.0f} s; evaluation of {int(d.n_eval_breasts.median())} breasts {d.t_eval_s.median():.0f} s; peak GPU memory {d.peak_mem_fit_gib.median():.1f} GiB in training, {d.peak_mem_total_gib.median():.1f} GiB with evaluation" + r"""\\
\bottomrule\end{tabular}}
\end{table}
""")
