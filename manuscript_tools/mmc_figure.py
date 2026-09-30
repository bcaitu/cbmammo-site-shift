import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import apply_figure_style, panel_letter
"""Figure: MMC Astana zero-shot transfer (a) and few-shot head adaptation on the retrospective endpoint (b, c).
Inputs: mmc_results.csv, mmc_fewshot_draws.csv (aggregate outputs of mmc_eval.py). Output: mmc_fewshot_curves.png/.pdf"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

D = "./"
res = pd.read_csv(D + "mmc_results.csv")
fs = pd.read_csv(D + "mmc_fewshot_draws.csv")
apply_figure_style(sizes=(8, 7, 6))                       
COL = {"cb": "#1f5fa8", "opq": "#c8702a"}
MK = {"dinov2": "o", "ft2": "s"}

fig = plt.figure(figsize=(7.2, 3.45))
outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 2.0], wspace=0.30, left=0.20, right=0.985, bottom=0.25, top=0.78)
inner = outer[1].subgridspec(1, 2, wspace=0.08)
axa = fig.add_subplot(outer[0]); axb = fig.add_subplot(inner[0]); axc = fig.add_subplot(inner[1], sharey=axb)

# ---------------- a: zero-shot, seed ensembles, 95% cluster-bootstrap CI over patients
spec = [("retro_M34_vs_M12", "all", "birads3", "Retrospective, all", "studies"),
        ("retro_M34_vs_M12", "device_Siemens", "birads3", "Retrospective, Siemens", "studies"),
        ("retro_M34_vs_M12", "device_Agfa", "birads3", "Retrospective, Agfa CR", "studies"),
        ("outpatient_M6_vs_M12", "breast", "malig", "Outpatient cancer vs benign", "breasts"),
        ("post_nact_residual_vs_complete", "cancer_breast_hq", "malig", "Post-NACT residual vs complete", "breasts")]
rows = []
for ep, sub, sc, lab, unit in spec:                      # n taken from the results table
    n = int(res[(res.endpoint == ep) & (res.subset == sub) & (res.score == sc) & (res.seed == "ens") & (res.arm == "cb")].n.iloc[0])
    rows.append((ep, sub, sc, f"{lab}\n({n} {unit})"))
off = {("cb", "dinov2"): -0.27, ("opq", "dinov2"): -0.09, ("cb", "ft2"): 0.09, ("opq", "ft2"): 0.27}
for i, (ep, sub, sc, lab) in enumerate(rows):
    for (arm, bb), dy in off.items():
        r = res[(res.endpoint == ep) & (res.subset == sub) & (res.score == sc) & (res.arm == arm) & (res.backbone == bb) & (res.seed == "ens")]
        assert len(r) == 1, (ep, sub, arm, bb)
        r = r.iloc[0]
        axa.errorbar(r.auc, i + dy, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt=MK[bb], color=COL[arm], ms=3.6, lw=0.9,
                     mfc=COL[arm] if bb == "dinov2" else "white", mew=0.9, capsize=0)
axa.axvline(0.5, color="0.6", lw=0.7, ls=(0, (2, 2)), zorder=0)
axa.axhline(2.5, color="0.85", lw=0.6)
axa.set_yticks(range(len(rows))); axa.set_yticklabels([r[3] for r in rows])
axa.set_ylim(4.5, -0.85)
axa.set_xlim(0.2, 1.02); axa.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); axa.set_xlabel("AUC (95% CI)")
for yy, txt in ((-0.62, "score: P(BI-RADS ≥ 3)"), (2.66, "score: P(malignant)")):
    axa.text(0.215, yy, txt, fontsize=6, color="0.35", va="center", zorder=5,
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
fig.text(axa.get_position().x0, 0.975, "Zero-shot transfer is modest on the\nclinic's BI-RADS-analog endpoint", ha="left", va="top", fontsize=8)
panel_letter(axa, "a")                                    

# ---------------- b, c: few-shot on the retrospective endpoint (mean over 3 seeds x 30 draws; bars = IQR)
KS = [10, 25, 50, 100]
series = [("cb", "refit", "cb_refit_cvC", "-"), ("cb", "correction", "cb_offset_cvC", "--"),
          ("opq", "refit", "opq_refit_cvC", "-"), ("opq", "correction", "opq_offset_cvC", "--")]
jit = {"cb_refit_cvC": 0.955, "cb_offset_cvC": 0.985, "opq_refit_cvC": 1.015, "opq_offset_cvC": 1.045}
for ax, bb, sub in ((axb, "dinov2", "Frozen encoder"), (axc, "ft2", "Last 2 blocks fine-tuned")):
    f = fs[fs.backbone == bb]
    for arm in ("cb", "opq"):
        ax.axhline(f[f.method == f"{arm}_zero"]["auc"].mean(), color=COL[arm], lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    for arm, kind, m, ls in series:
        g = f[f.method == m].groupby("k")["auc"]
        mu, q1, q3 = g.mean().reindex(KS), g.quantile(0.25).reindex(KS), g.quantile(0.75).reindex(KS)
        ax.errorbar(np.array(KS) * jit[m], mu, yerr=[np.maximum(mu - q1, 0), np.maximum(q3 - mu, 0)], color=COL[arm], ls=ls,
                    lw=1.1, marker=MK[bb], ms=3.4, mfc=COL[arm] if kind == "refit" else "white", mew=0.9, capsize=0,
                    elinewidth=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
    ax.set_xlim(8, 125); ax.set_xlabel("Labelled local patients (k)")
    ax.text(0.03, 0.97, sub, transform=ax.transAxes, fontsize=7, va="top", ha="left")
axb.set_ylim(0.42, 0.67); axb.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65]); axb.set_ylabel("AUC on held-out studies")
plt.setp(axc.get_yticklabels(), visible=False)

_g = []
for _bb in ("dinov2", "ft2"):
    _f = fs[fs.backbone == _bb]
    for _m in ("cb_refit_cvC", "cb_offset_cvC", "opq_refit_cvC", "opq_offset_cvC"):
        _z = _f[_f.method == _m.split("_")[0] + "_zero"].groupby("k")["auc"].mean()
        _r = _f[_f.method == _m].groupby("k")["auc"].mean()
        _g.append((_r - _z).reindex([10, 25, 50, 100]).max())
GAIN_MAX = max(_g)
print("max mean gain over zero-shot (plotted protocols):", round(GAIN_MAX, 4))
fig.text(axb.get_position().x0, 0.975, f"Head refits on ≤100 local patients gain at most\n{GAIN_MAX:.3f} AUC over the deployed model", ha="left", va="top", fontsize=8)

panel_letter(axb, "b"); panel_letter(axc, "c")             

h = [Line2D([], [], color=COL["cb"], lw=1.4, label="CB (concept bottleneck)"),
     Line2D([], [], color=COL["opq"], lw=1.4, label="Opaque (no concepts)"),
     Line2D([], [], color="0.3", marker="o", ls="", ms=3.4, label="frozen encoder"),
     Line2D([], [], color="0.3", marker="s", ls="", ms=3.4, mfc="white", label="last 2 blocks fine-tuned"),
     Line2D([], [], color="0.3", lw=1.1, ls="-", label="head refit from scratch"),
     Line2D([], [], color="0.3", lw=1.1, ls="--", label="correction to deployed head"),
     Line2D([], [], color="0.3", lw=0.9, ls=(0, (1, 1.5)), label="deployed model (zero-shot)")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4, frameon=False, fontsize=6,
           handlelength=2.2, columnspacing=1.3)
fig.savefig("mmc_fewshot_curves.png", dpi=300); fig.savefig("mmc_fewshot_curves.pdf")
