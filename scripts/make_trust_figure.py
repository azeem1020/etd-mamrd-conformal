"""Fig S6 (fig6_trust_guarantee.pdf) — Group-conditional conformal audit efficiency & hit-rate gap.

(a) MAMRD per-stratum audit-efficiency lift across target marginal recall levels (1 - alpha).
(b) Heavy-minus-Flat audit precision (hit-rate) gap and lift at target marginal recall 1 - alpha = 0.80
    across all seven detectors (95% BCa CI) under the exact order statistic k = ceil((n+1)(1-alpha)).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from repro_config import RESULTS_DIR
from fig_style import C, apply_style
from make_manuscript_figures import bca_ci

N = RESULTS_DIR
FIG = N / "figures"
FIG.mkdir(exist_ok=True)
df = pd.read_csv(N / "trust" / "trust_lift_sgcc.csv")
pc = pd.read_csv(N / "trust" / "trust_sgcc_percell.csv")

apply_style()
TOL = {
    "red": C["ours"], "blue": C["base"], "grey": C["base_lt"], "green": C["good"],
    "dark": C["ink"], "cyan": C["base_lt"], "yellow": C["warn"], "purple": C["warn"],
}
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.4,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.3,
})
DOUBLE = 7.16

fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE, 2.75))

# ---- panel (a): MAMRD per-stratum lift vs target marginal recall 1 - alpha ----
na = df[df.arm == "a34_NaME"].sort_values("guarantee")
x = na["guarantee"].values
for col, name, c, mk in [
    ("lift_heavy", "Heavy consumption (12.2% base)", TOL["red"], "o"),
    ("lift_light", "Light consumption (6.9% base)", TOL["blue"], "s"),
    ("lift_flat", "Flat consumption (3.1% base)", TOL["grey"], "^"),
]:
    y = na[col].values
    m = ~np.isnan(y)
    axA.plot(x[m], y[m], marker=mk, color=c, label=name)
axA.axhline(1.0, color=TOL["dark"], lw=0.8, ls=":")
axA.text(0.88, 1.08, "random audit (1.0x)", fontsize=6, color=TOL["dark"], ha="center")
axA.set_xticks([0.80, 0.90, 0.95])
axA.invert_xaxis()
axA.set_xlabel("Target marginal theft recall $1-\\alpha$")
axA.set_ylabel("Audit-efficiency lift over random")
axA.set_title("(a) MAMRD: lift by consumption group")
axA.legend(loc="upper left")

# ---- panel (b): Heavy-Flat audit hit-rate (precision) gap @ 80% target recall ----
ORDER = [
    ("GRU-D", "GRU-D"), ("HybridAttn", "Self-Attn"), ("DKGT", "DKGT-Net"),
    ("WDCNN", "WDCNN"), ("MSDM", "MSDM"), ("HORLN", "HORLN"),
    ("a34_NaME", "MAMRD"),
]
pc80 = pc[np.isclose(pc["alpha"], 0.20)]
ys = np.arange(len(ORDER))
for i, (arm, lab) in enumerate(ORDER):
    sub = pc80[pc80["arm"] == arm]
    diff = (sub["prec_HEAVY"] - sub["prec_Flat"]).dropna().values * 100.0
    m = float(np.mean(diff))
    lo, hi = bca_ci(diff)
    c = TOL["red"] if arm == "a34_NaME" else TOL["dark"]
    axB.errorbar(m, i, xerr=[[m - lo], [hi - m]],
                 fmt="o", color=c, ecolor=c, capsize=2.5, markersize=4.5)
    axB.text(hi + 0.6, i, f"+{m:.1f}%", va="center", ha="left", fontsize=6.2, color=c)
axB.axvline(0.0, color=TOL["dark"], lw=0.8, ls=":")
axB.set_yticks(ys)
axB.set_yticklabels([lab for _, lab in ORDER])
axB.set_xlabel("Heavy$-$Flat audit hit-rate gap (percentage points, 95% CI)")
axB.set_title("(b) Heavy vs. Flat precision gap at $\\alpha=0.20$")
axB.set_xlim(-2.0, 34.0)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig6_trust_guarantee.{ext}")
plt.close(fig)
print(f"[done] wrote {FIG}/fig6_trust_guarantee.pdf (+png)")
