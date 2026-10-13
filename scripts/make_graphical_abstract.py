"""Graphical abstract for Paper 3 (Energy and AI) — 3-panel story.
(1) The leaderboard is inflated: reported ROC/MAP ~0.9 vs honest full AP ~0.57.
(2) An empirical ceiling: no detector clears full AP ~0.57; no baseline > 0.50.
(3) A guarantee that cuts audits: class-conditional conformal inspects 35% fewer
    consumers at the same 80% per-group detection-rate guarantee.
Landscape, vector PDF + PNG; Tol palette, serif (matches make_figures.py).
"""
from __future__ import annotations
from pathlib import Path
import sys
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
from fig_style import C, apply_style

FIG = (RESULTS_DIR / "figures"); FIG.mkdir(exist_ok=True)
apply_style()
TOL = {"red": C["ours"], "grey": C["base"], "dark": C["ink"], "green": C["good"],
       "blue": C["ours"], "cyan": C["base_lt"], "yellow": C["warn"], "purple": C["warn"]}
mpl.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.4))

# Panel 1: inflation
a1.bar([0, 1], [0.88, 0.567], color=[TOL["grey"], TOL["red"]], width=0.6, edgecolor=TOL["dark"], lw=0.6)
a1.set_xticks([0, 1]); a1.set_xticklabels(["reported\n(ROC, MAP@100)", "honest\n(full AP)"])
a1.set_ylim(0, 1.0); a1.set_ylabel("score")
a1.set_title("(1) The leaderboard is inflated")
a1.annotate("2025 SOTA:\n0.95 $\\rightarrow$ 0.30", xy=(0.5, 0.93), ha="center", fontsize=6.2, color=TOL["dark"])
for x, v in [(0, 0.88), (1, 0.567)]:
    a1.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

# Panel 2: an empirical ceiling -- no detector clears it
det = ["GRU-D", "S-attn", "DKGT", "WDCNN", "MSDM", "HORLN", "MAMRD"]
ap = [0.201, 0.207, 0.302, 0.335, 0.465, 0.500, 0.567]
cols = [TOL["grey"]] * 6 + [TOL["red"]]
a2.bar(range(7), ap, color=cols, width=0.74, edgecolor=TOL["dark"], lw=0.6)
a2.axhline(0.567, color=TOL["dark"], ls="--", lw=0.8)
a2.text(6.4, 0.585, "ceiling 0.57", ha="right", fontsize=6.2, color=TOL["dark"])
a2.set_xticks(range(7)); a2.set_xticklabels(det, rotation=40, ha="right", fontsize=6)
a2.set_ylim(0, 0.8); a2.set_ylabel("full-prevalence AP")
a2.set_title("(2) An empirical ceiling")
a2.text(2.4, 0.70, "no baseline $>0.50$", fontsize=6, color=TOL["dark"], ha="center")

# Panel 3: a guarantee that cuts audits (Move B, MAMRD @ guaranteed 80% per-group detection rate)
a3.bar([0, 1], [0.507, 0.322], color=[TOL["grey"], TOL["green"]], width=0.55,
       edgecolor=TOL["dark"], lw=0.6)
a3.set_xticks([0, 1]); a3.set_xticklabels(["single\nthreshold", "consumption-\nconditional"], fontsize=6.5)
a3.set_ylim(0, 0.72); a3.set_ylabel("consumers inspected")
a3.set_title("(3) A guarantee that cuts audits")
for x, v in [(0, 0.507), (1, 0.322)]:
    a3.text(x, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=7)
a3.annotate("35% fewer\nat the same\n80% guarantee", xy=(1, 0.34), xytext=(1.0, 0.60),
            ha="center", fontsize=6, color=TOL["dark"],
            arrowprops=dict(arrowstyle="->", lw=0.6, color=TOL["dark"]))

fig.suptitle("Trustworthy electricity theft detection: a conformal framework and detector-centric benchmark (MAMRD)",
             fontsize=8.2, y=1.04)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"graphical_abstract.{ext}")
print(f"[done] wrote {FIG}/graphical_abstract.pdf (+png)")
