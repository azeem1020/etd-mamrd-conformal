"""Fig (data characterization) for Paper 3 (Energy and AI).

Single panel, computed directly from the raw SGCC release:
Distribution of per-consumer missing-rate by label (violin): the gap is
structural, not a tail artefact, and is the empirical motivation for treating
missingness as signal (the context-conditional missing-value embedding and the
mask-channel branch). The over-time / non-stationarity (regime) view is out of
scope here and is treated in the companion regime paper; it was previously a
second panel and has been removed to keep this figure to its single point.

Vector PDF + PNG; Tol palette, serif, Type-42 fonts (matches make_figures.py).
Run under conda env tgpu.
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

SRC = SGCC_PATH
FIG = (RESULTS_DIR / "figures"); FIG.mkdir(parents=True, exist_ok=True)
OUT = (RESULTS_DIR / "trust"); OUT.mkdir(parents=True, exist_ok=True)

apply_style()
TOL = {"blue": C["base"], "red": C["honest"], "green": C["good"], "grey": C["base"],
       "dark": C["ink"], "cyan": C["base_lt"], "yellow": C["warn"], "purple": C["warn"]}
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.3,
})
DOUBLE = 7.16

if not Path(SRC).exists() and (OUT / "missingness_summary.csv").exists() and (FIG / "fig_data_missingness.pdf").exists():
    print(f"[sgcc] {SRC} not found; precomputed fig_data_missingness.pdf and summary already present in {FIG}...", flush=True)
    summary = pd.read_csv(OUT / "missingness_summary.csv", index_col=0).squeeze().to_dict()
    print("[done] verified fig_data_missingness.pdf (+png) and missingness_summary.csv")
    for k, v in summary.items():
        print(f"   {k}: {float(v):.3f}")
    sys.exit(0)

print("[load] reading SGCC.csv ...")
df = pd.read_csv(SRC)
df.columns = [c.strip() for c in df.columns]
flag = df["FLAG"].astype(int).values
date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
dates = pd.to_datetime(pd.Series(date_cols), format="%Y/%m/%d", errors="coerce")
order = np.argsort(dates.values)
date_cols = [date_cols[i] for i in order]
dates_sorted = dates.iloc[order].reset_index(drop=True)
print(f"[load] {df.shape[0]} consumers x {len(date_cols)} days; "
      f"theft={flag.sum()} ({100*flag.mean():.2f}%); "
      f"span {dates_sorted.iloc[0].date()} -> {dates_sorted.iloc[-1].date()}")

M = df[date_cols].isna().values  # True == missing
miss_theft = M[flag == 1].mean(axis=0)
miss_honest = M[flag == 0].mean(axis=0)

# per-consumer missing-rate by label
pc = M.mean(axis=1)
pc_theft, pc_honest = pc[flag == 1], pc[flag == 0]

fig, axB = plt.subplots(1, 1, figsize=(3.3, 2.8))

parts = axB.violinplot([100 * pc_honest, 100 * pc_theft], showmedians=True,
                       showextrema=False, widths=0.8)
for b, c, a in zip(parts["bodies"], [TOL["blue"], TOL["red"]], [0.55, 0.85]):
    b.set_facecolor(c); b.set_edgecolor(TOL["dark"]); b.set_alpha(a); b.set_linewidth(0.7)
parts["cmedians"].set_color(TOL["dark"]); parts["cmedians"].set_linewidth(1.0)
axB.set_xticks([1, 2]); axB.set_xticklabels(["Honest", "Theft"])
axB.set_ylabel("Per-consumer missing-rate (%)")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig_data_missingness.{ext}")

# numbers for the manuscript text
summary = {
    "overall_missing_pct": float(100 * M.mean()),
    "theft_missing_pct": float(100 * miss_theft.mean()),
    "honest_missing_pct": float(100 * miss_honest.mean()),
    "theft_pc_median_pct": float(100 * np.median(pc_theft)),
    "honest_pc_median_pct": float(100 * np.median(pc_honest)),
    "pre2016_missing_pct": float(100 * M[:, dates_sorted.values < np.datetime64("2016-01-01")].mean()),
    "post2016_missing_pct": float(100 * M[:, dates_sorted.values >= np.datetime64("2016-01-01")].mean()),
}
pd.Series(summary).to_csv(OUT / "missingness_summary.csv")
print("[done] wrote fig_data_missingness.pdf (+png) and missingness_summary.csv")
for k, v in summary.items():
    print(f"   {k}: {v:.3f}")
