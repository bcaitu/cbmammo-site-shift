"""Fallbacks for the two figure-style helpers used when the paper figures were drawn (sizes ladder 8/7/6 pt, bold panel letters)."""
import matplotlib as mpl

def apply_figure_style(sizes=(8, 7, 6), **kw):
    base, mid, small = sizes
    mpl.rcParams.update({"font.size": base, "axes.labelsize": base, "axes.titlesize": base, "legend.fontsize": mid,
                         "xtick.labelsize": small, "ytick.labelsize": small, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 300})

def panel_letter(ax, letter, case="lower"):
    ax.text(-0.02, 1.08, letter.lower() if case == "lower" else letter.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
