"""Fig 6 for Paper 3 (Energy and AI) — the consumption-conditional audit guarantee.

(a) MAMRD per-stratum audit-efficiency lift vs guaranteed detection rate: high/low
    consumption lift rises toward the top of the ranking while Flat is floored and
    drops out at the tightest guarantee (too sparse to certify).
(b) High-Flat lift gap at a guaranteed 80% detection rate across all seven detectors
    (BCa 95% CI). Every interval excludes zero: the gap is universal.

Reads results/phase_n/trust/trust_lift_sgcc.csv (written by run_trust_layer.py).
Style matches make_figures.py: Tol palette, Type-42 fonts, Elsevier sizing.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
from fig_style import C, apply_style

N = RESULTS_DIR
FIG = N / "figures"; FIG.mkdir(exist_ok=True)
df = pd.read_csv(N / "trust" / "trust_lift_sgcc.csv")

apply_style()
TOL = {"red": C["ours"], "blue": C["base"], "grey": C["base_lt"], "green": C["good"],
       "dark": C["ink"], "cyan": C["base_lt"], "yellow": C["warn"], "purple": C["warn"]}
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.3,
})
DOUBLE = 7.16

fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))

# ---- panel (a): MAMRD per-stratum lift vs guarantee -------------------------
na = df[df.arm == "a34_NaME"].sort_values("guarantee")
x = na["guarantee"].values
for col, name, c, mk in [("lift_heavy", "High consumption", TOL["red"], "o"),
                         ("lift_light", "Low consumption", TOL["blue"], "s"),
                         ("lift_flat", "Flat", TOL["grey"], "^")]:
    y = na[col].values
    m = ~np.isnan(y)
    axA.plot(x[m], y[m], marker=mk, color=c, label=name)
axA.axhline(1.0, color=TOL["dark"], lw=0.8, ls=":")
axA.text(0.88, 1.04, "random audit", fontsize=6, color=TOL["dark"], ha="center")
axA.set_xticks([0.80, 0.90, 0.95])
axA.invert_xaxis()  # moving right->left = up the ranking (looser guarantee on left)
axA.set_xlabel("Guaranteed theft detection rate $1-\\alpha$")
axA.set_ylabel("Audit-efficiency lift over random")
axA.set_title("(a) MAMRD: lift by consumption level")
axA.legend(loc="upper right")

# ---- panel (b): Heavy-Flat gap @80% across detectors (forest) ---------------
ORDER = [("GRU-D", "GRU-D"), ("HybridAttn", "Self-attn."), ("DKGT", "DKGT-Net"),
         ("WDCNN", "WDCNN"), ("MSDM", "MSDM"), ("HORLN", "HORLN"),
         ("a34_NaME", "MAMRD")]
g = df[df.guarantee == 0.80].set_index("arm")
ys = np.arange(len(ORDER))
for i, (arm, lab) in enumerate(ORDER):
    r = g.loc[arm]
    c = TOL["red"] if arm == "a34_NaME" else TOL["dark"]
    axB.errorbar(r.gap_heavy_flat, i,
                 xerr=[[r.gap_heavy_flat - r.gap_lo], [r.gap_hi - r.gap_heavy_flat]],
                 fmt="o", color=c, ecolor=c, capsize=2.5, markersize=4)
axB.axvline(0.0, color=TOL["dark"], lw=0.8, ls=":")
axB.set_yticks(ys); axB.set_yticklabels([lab for _, lab in ORDER])
axB.set_xlabel("Heavy$-$Flat lift gap (95% CI)")
axB.set_title("(b) The gap is universal")
axB.set_xlim(left=-0.2)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig6_trust_guarantee.{ext}")
print(f"[done] wrote {FIG}/fig6_trust_guarantee.pdf (+png)")
