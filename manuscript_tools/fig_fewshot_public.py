import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from _style import apply_figure_style
"""Figure 2: few-shot repair on the public target sets. Inputs: tabs/C_fewshot_test_*.csv (from embed_analysis_tables.tar.gz) and E_pft_means.csv.
Run from the repository root."""
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
apply_figure_style(sizes=(8, 7, 6))
T, M = "results/tables/", "results/tables/"
COL = {"cb": "#1f5fa8", "opq": "#c8702a"}
sets = [("cdd_cesm", "CDD-CESM"), ("cmmd", "CMMD"), ("embed", "EMBED, Hologic"), ("embed_ge", "EMBED, GE holdout")]
C = {s: pd.read_csv(T + f"C_fewshot_test_{s}.csv") for s, _ in sets}
Epft = pd.read_csv(M + "E_pft_means.csv")
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharey="row")
for r_, (bb, bbl) in enumerate([("dinov2", "Frozen encoder"), ("ft2", "Last 2 blocks fine-tuned")]):
    for c_, (s, sl) in enumerate(sets):
        ax = axes[r_, c_]; d = C[s][C[s].bb == bb].sort_values("k_patients"); k = d.k_patients.values
        ax.plot(k, d.cb_refit_head_cvC, "-o", color=COL["cb"], lw=1.2, ms=3)
        ax.plot(k, d.opq_refit_head_cvC, "-o", color=COL["opq"], lw=1.2, ms=3, mfc="white")
        ax.axhline(d.cb_zero.mean(), color=COL["cb"], lw=0.9, ls=(0, (1, 1.5))); ax.axhline(d.opq_zero.mean(), color=COL["opq"], lw=0.9, ls=(0, (1, 1.5)))
        if s in ("cdd_cesm", "cmmd", "embed_ge"):
            e = Epft[(Epft.bb == bb) & (Epft.set == f"test_{s}")].sort_values("k_patients")
        else:
            e = d.dropna(subset=["opq_partial_ft"])
        ax.plot(e.k_patients.values, e.opq_partial_ft.values, "D", color=COL["opq"], ms=4.2, mew=0.6, mec="white", zorder=5)
        ticks = [10, 25, 50, 100, 200] if s in ("cdd_cesm", "cmmd") else [10, 25, 50, 100]
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in ticks]); ax.minorticks_off(); ax.margins(x=0.08)
        if r_ == 0: ax.text(0.0, 1.04, sl, transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
        if r_ == 1: ax.set_xlabel("Labelled local patients (k)")
        if c_ == 0: ax.set_ylabel("AUC on remaining patients")
for r_ in range(2):
    lo = min(a.get_ylim()[0] for a in axes[r_]); hi = max(a.get_ylim()[1] for a in axes[r_]); axes[r_, 0].set_ylim(lo, hi)
for r_, bbl in enumerate(["Frozen encoder", "Last 2 blocks fine-tuned"]):
    axes[r_, 0].text(0.03, 0.04, bbl, transform=axes[r_, 0].transAxes, fontsize=6.5, ha="left", va="bottom", color="0.25", style="italic")
h = [Line2D([], [], color=COL["cb"], lw=1.2, marker="o", ms=3, label="CBM: head refit"), Line2D([], [], color=COL["opq"], lw=1.2, marker="o", ms=3, mfc="white", label="Opaque: head refit"),
     Line2D([], [], color=COL["opq"], ls="", marker="D", ms=4.2, label="Opaque: partial fine-tuning"), Line2D([], [], color="0.3", lw=0.9, ls=(0, (1, 1.5)), label="Deployed, no adaptation (blue CBM, orange opaque)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.0)
fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.235, wspace=0.12, hspace=0.45)
fig.savefig("fig_fewshot_public.png", dpi=300); fig.savefig("fig_fewshot_public.pdf")
