"""LaTeX tables of the manuscript, generated from results/tables/*.csv. Run from the repository root: python -c \"exec(open('manuscript_tools/paper_tables.py').read())\" (defines the dict `tabs`)."""

import pandas as pd, numpy as np, json, os
T = "results/tables/"; M = "results/tables/"
SETN = {"test_cbis_ddsm":"CBIS-DDSM (internal)", "test_cdd_cesm":"CDD-CESM", "test_cmmd":"CMMD",
        "test_embed":"EMBED, Hologic", "test_embed_ge":"EMBED, GE-holdout patients"}
BBN = {"dinov2":"Frozen DINOv2-B", "ft2":"Last 2 blocks tuned"}
def f3(x): return f"{x:.3f}"
def sg(x):
    if round(abs(x), 3) == 0: return "0.000"
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"
def pmsd(m, s): return f"{m:.3f}{{\\scriptsize$\\pm${s:.3f}}}"
def ci(m, lo, hi): return f"{sg(m)} [{sg(lo)}, {sg(hi)}]".replace("[$+$", "[$+$")
def wrap(caption, label, body, resize=True, note=None):
    s = "\\begin{table}[H]\n\\small\n\\caption{" + caption + "}\\label{" + label + "}\n\\centering\n"
    s += ("\\resizebox{\\textwidth}{!}{%\n" if resize else "") + body + ("}\n" if resize else "\n")
    if note: s += "\\par\\smallskip\\noindent{\\footnotesize " + note + "}\n"
    return s + "\\end{table}\n"

tabs = {}

# ---- Table: zero-shot
A = pd.read_csv(T+"A_malignant_auc.csv"); A2 = pd.read_csv(T+"A2_cb_minus_opaque_auc.csv")
def cfg(arm, bb): return f"{arm}_embed_{bb}"
order = ["test_cbis_ddsm","test_cdd_cesm","test_cmmd","test_embed","test_embed_ge"]
rows = []
for s in order:
    n = int(A[A.set==s].n.iloc[0]); cells = []
    for bb in ["dinov2","ft2"]:
        g = lambda arm: A[(A.config==cfg(arm,bb))&(A.set==s)].iloc[0]
        cb, oq = g("cb"), g("opq"); d = A2[(A2.bb==bb)&(A2.set==s)].iloc[0]
        cells += [pmsd(cb.auc_mean, cb.auc_sd), pmsd(oq.auc_mean, oq.auc_sd), ci(d.auc_diff, d.lo, d.hi) + f" ({int(d.seeds_CI_excl_0)}/3)"]
    rows.append(f"{SETN[s]} & {n:,} & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{llcccccc}\\toprule\n & & \\multicolumn{3}{c}{Frozen DINOv2-B} & \\multicolumn{3}{c}{Last 2 blocks tuned}\\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nTest set & $n$ & CBM & Opaque & CBM $-$ opaque & CBM & Opaque & CBM $-$ opaque\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["zeroshot"] = wrap("Zero-shot malignancy AUC of the concept bottleneck model (CBM) and the matched opaque model (mean$\\pm$SD over three training seeds). The difference is the mean of seed-matched paired differences with the mean of the per-seed 95\\% patient-clustered bootstrap limits; the fraction in parentheses is the number of seeds whose interval excludes zero. $n$ = breasts with a malignancy label (EMBED: biopsied breasts only).", "tab:zeroshot", body)

# ---- Table: oracle decomposition
Bo = pd.read_csv(T+"B_oracle_all.csv"); Bm = Bo[(Bo.target=="malignant")&(Bo.subset=="all")]
ag = Bm.groupby(["bb","set","concepts"]).agg(auc=("auc","mean"), cov=("coverage","mean")).reset_index()
pv = ag.pivot_table(index=["bb","set"], columns="concepts", values="auc"); cv = ag.pivot_table(index=["bb","set"], columns="concepts", values="cov")
oorder = ["test_cbis_ddsm","test_cdd_cesm","test_embed","test_embed_ge","test_cmmd"]
rows = []
for bb in ["dinov2","ft2"]:
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{BBN[bb]}}}}}\\\\")
    for s in oorder:
        p_, o_ = pv.loc[(bb,s),"pred"], pv.loc[(bb,s),"oracle"]
        cells = [f"{sg(pv.loc[(bb,s),'oracle_'+g]-p_)} ({cv.loc[(bb,s),'oracle_'+g]:.2f})" for g in ["density","mass","calc","other"]]
        rows.append(f"{SETN[s]} & {f3(p_)} & {f3(o_)} & {sg(o_-p_)} & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{lccccccc}\\toprule\nTest set & Predicted & Oracle (all) & Gain & $\\Delta$ density & $\\Delta$ mass & $\\Delta$ calcification & $\\Delta$ other\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["oracle"] = wrap("Oracle-concept decomposition for malignancy (CBM, mean AUC over three seeds). ``Oracle (all)'' replaces every predicted concept by its reference value wherever that concept is labelled; $\\Delta$ columns give the AUC change when only one concept group is replaced, with the fraction of breasts in which \\emph{every} head of that group is labelled in parentheses. CMMD supplies presence labels for masses and calcifications only. Fuji-imaged holdout breasts (13 labelled) are not analysed.", "tab:oracle", body)

# ---- Table: few-shot means
Ep = pd.read_csv(M+"E_pft_means.csv"); Ch = pd.read_csv(T+"C_fewshot_test_embed.csv")
rows = []
for bb in ["dinov2","ft2"]:
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{BBN[bb]}}}}}\\\\")
    for s in ["test_cdd_cesm","test_cmmd","test_embed","test_embed_ge"]:
        for k in [25,100]:
            if s=="test_embed":
                r = Ch[(Ch.bb==bb)&(Ch.k_patients==k)].iloc[0]; dag = "$^{\\dagger}$"
            else:
                r = Ep[(Ep.bb==bb)&(Ep.set==s)&(Ep.k_patients==k)].iloc[0]; dag = ""
            vals = {"cbr": r.cb_refit_head_cvC, "cbc": r.cb_correction_cvC, "oqr": r.opq_refit_head_cvC, "pft": r.opq_partial_ft}
            best = max(vals, key=vals.get)
            fm = lambda key: (f"\\textbf{{{f3(vals[key])}}}" if key==best else f3(vals[key]))
            rows.append(f"{SETN[s]} & {k} & {f3(r.cb_zero)} & {fm('cbr')} & {fm('cbc')} & {f3(r.opq_zero)} & {fm('oqr')} & {fm('pft')}{dag}\\\\")
body = ("\\begin{tabular}{lrcccccc}\\toprule\n & & \\multicolumn{3}{c}{CBM} & \\multicolumn{3}{c}{Opaque}\\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Target set & $k$ & No adaptation & Head refit & Correction & No adaptation & Head refit & Partial fine-tuning\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["fewshot"] = wrap("Few-shot repair on the public target sets: mean malignancy AUC on the remaining patients (3 seeds $\\times$ 30 draws of $k$ adaptation patients). Bold: highest of the four adapted variants in each row. Penalties of the head refits were chosen by inner cross-validation. Rows other than EMBED-Hologic come from the 90-draw run that includes partial fine-tuning; the paired contrasts in Table~\\ref{tab:paired} use the same draws.", "tab:fewshot", body,
     note="$^{\\dagger}$ EMBED-Hologic rows come from an independent earlier 90-draw run, in which partial fine-tuning used 10 draws per seed (30 in total).")

# ---- Table: paired contrasts
Pp = pd.read_csv(M+"E_pft_paired.csv")
cons = [("CBrefit - CBzero","CBM refit $-$ CBM deployed"),("CBrefit - OPQrefit","CBM refit $-$ opaque head refit"),("CBrefit - OPQpartialFT","CBM refit $-$ opaque partial fine-tuning")]
rows = []
for bb in ["dinov2","ft2"]:
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{BBN[bb]}}}}}\\\\")
    for s in ["test_cdd_cesm","test_cmmd","test_embed_ge"]:
        for k in [25,100]:
            cells = []
            for cn,_ in cons:
                r = Pp[(Pp.bb==bb)&(Pp.set==s)&(Pp.k==k)&(Pp.contrast==cn)].iloc[0]
                txt = ci(r["mean"], r.lo, r.hi)
                if r.lo > 0 or r.hi < 0: txt = "\\textbf{" + txt + "}"
                cells.append(txt)
            rows.append(f"{SETN[s]} & {k} & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{lrccc}\\toprule\nTarget set & $k$ & " + " & ".join(c for _,c in cons) + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["paired"] = wrap("Paired AUC differences between adaptation variants, computed draw by draw (mean and 2.5--97.5\\% range over 90 draws = 3 seeds $\\times$ 30 draws). Bold: range excludes zero. Draws share the evaluation patients and therefore describe the variability of the adaptation sample, not sampling of the target population.", "tab:paired", body)

# ---- Table: ablation EMBED
D1 = pd.read_csv(T+"D_ablation_malignant_auc.csv")
rows = []
for arm, an in [("cb","CBM"),("opq","Opaque")]:
    for s in ["test_cbis_ddsm","test_cdd_cesm","test_cmmd"]:
        cells = []
        for bb in ["dinov2","ft2"]:
            r = D1[(D1.arm==arm)&(D1.bb==bb)&(D1.set==s)].iloc[0]
            cells += [pmsd(r.old, r.old_sd), pmsd(r.new, r.new_sd), sg(r.delta)]
        rows.append(f"{an} & {SETN[s]} & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{llcccccc}\\toprule\n & & \\multicolumn{3}{c}{Frozen DINOv2-B} & \\multicolumn{3}{c}{Last 2 blocks tuned}\\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Model & Test set & Without EMBED & With EMBED & $\\Delta$ & Without EMBED & With EMBED & $\\Delta$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["ablation"] = wrap("Effect of adding EMBED to the training data: zero-shot malignancy AUC (mean$\\pm$SD over three seeds) of models trained without EMBED (8,388 breasts) and with EMBED (73,470 breasts, resampled as described in Section~\\ref{sec:training}).", "tab:ablation", body)

# ---- Table: vendor stratification
V = pd.read_csv("results/tables/vendor_split_zero_shot.csv"); V["arm"] = V.run.str.split("_").str[0]; V["bb"] = V.run.str.split("_").str[2]
vg = V.groupby(["set","vendor","bb","arm"]).agg(n=("n","first"), pos=("n_pos","first"), auc=("auc","mean"), sd=("auc","std")).reset_index()
rows = []
for s, v, lab in [("test_embed","hologic","Hologic test set (all Hologic)"),("test_embed_ge","hologic","GE-holdout patients, imaged on Hologic"),("test_embed_ge","ge","GE-holdout patients, imaged on GE")]:
    cells = []
    for bb in ["dinov2","ft2"]:
        cb = vg[(vg.set==s)&(vg.vendor==v)&(vg.bb==bb)&(vg.arm=="cb")].iloc[0]; oq = vg[(vg.set==s)&(vg.vendor==v)&(vg.bb==bb)&(vg.arm=="opq")].iloc[0]
        cells += [pmsd(cb.auc, cb.sd), pmsd(oq.auc, oq.sd), sg(cb.auc-oq.auc)]
    rows.append(f"{lab} & {int(cb.n)} ({int(cb.pos)}) & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{llcccccc}\\toprule\n & & \\multicolumn{3}{c}{Frozen DINOv2-B} & \\multicolumn{3}{c}{Last 2 blocks tuned}\\\\\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Subset of EMBED test breasts & $n$ (malignant) & CBM & Opaque & CBM $-$ opaque & CBM & Opaque & CBM $-$ opaque\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["vendor"] = wrap("Zero-shot malignancy AUC (mean$\\pm$SD over three seeds) of the EMBED-trained models, stratified by the vendor of the imaging system. The 7 Fujifilm-imaged labelled breasts of the holdout patients are not shown.", "tab:vendor", body)

# ---- Table: MMC (from make_mmc_latex output)
F = json.load(open(M+"mmc_fragments.json"))
mm = F["table"].replace("\\begin{table}[t]\\centering\\small", "\\begin{table}[H]\n\\small").replace("\\begin{tabular}{lrcccc}", "\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrcccc}").replace("\\end{tabular}\n\\end{table}", "\\end{tabular}}\n\\end{table}")
mm = mm.replace("Fine-tuned (last 2 blocks)", "Last 2 blocks tuned")
tabs["mmc"] = mm + "\n"

# ---- Appendix tables: concept heads, BI-RADS oracle
co = pd.read_csv(T+"concepts_all.csv"); co["arm"] = co.run.str.split("_").str[0]; co["bb"] = co.run.str.split("_").str[2]
heads = [("density","density"),("mass","mass presence"),("calc","calcification presence"),("mass_margin","mass margin"),("calc_morphology","calcification morphology")]
rows = []
for s in ["test_cdd_cesm","test_embed","test_embed_ge"]:
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{SETN[s]}}}}}\\\\")
    for h, hl in heads:
        cells = []
        for bb in ["dinov2","ft2"]:
            g = co[(co.set==s)&(co["head"]==h)&(co.bb==bb)].groupby("arm").auc.mean()
            cells += [f3(g["cb"]), f3(g["opq"])]
        rows.append(f"\\quad {hl} & " + " & ".join(cells) + "\\\\")
body = ("\\begin{tabular}{lcccc}\\toprule\n & \\multicolumn{2}{c}{Frozen DINOv2-B} & \\multicolumn{2}{c}{Last 2 blocks tuned}\\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nConcept head & CBM & Opaque (auxiliary) & CBM & Opaque (auxiliary)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
tabs["concepts"] = wrap("Concept-level prediction quality (AUC, mean over three seeds; one-vs-rest macro average for multi-class heads) of the CBM and of the auxiliary concept heads of the matched opaque model.", "tab:concepts", body, resize=False)

Bb = Bo[(Bo.target=="birads")&(Bo.subset=="all")]
ab = Bb.groupby(["bb","set","concepts"]).auc.mean().reset_index().pivot_table(index=["bb","set"], columns="concepts", values="auc")
rows = []
for bb in ["dinov2","ft2"]:
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{BBN[bb]}}}}}\\\\")
    for s, sl in [("test_vindr","VinDr-Mammo"),("test_cdd_cesm","CDD-CESM"),("test_inbreast","INbreast"),("test_embed","EMBED, Hologic"),("test_embed_ge","EMBED, GE-holdout patients")]:
        p_ = ab.loc[(bb,s),"pred"]; o_ = ab.loc[(bb,s),"oracle"]
        rows.append(f"{sl} & {f3(p_)} & {f3(o_)} & " + " & ".join(sg(ab.loc[(bb,s),'oracle_'+g]-p_) for g in ["density","mass","calc","other"]) + "\\\\")
body = ("\\begin{tabular}{lccccccc}\\toprule\nTest set & Predicted & Oracle (all) & $\\Delta$ density & $\\Delta$ mass & $\\Delta$ calcification & $\\Delta$ other\\\\\\midrule\n".replace("lccccccc","lcccccc") + "\n".join(rows) + "\n\\bottomrule\\end{tabular}").replace("\\begin{tabular}{lccccccc}","\\begin{tabular}{lcccccc}")
tabs["birads_oracle"] = wrap("Oracle-concept decomposition for the BI-RADS group target (macro one-vs-rest AUC, mean over three seeds). The BI-RADS oracle is partly circular because the same reader assigns descriptors and category.", "tab:biradsoracle", body, resize=False)
