"""Friedman omnibus test + Nemenyi post-hoc critical-difference (CD) diagram for
the seven SGCC detectors, from stored per-cell PR-AUC (NO GPU).

Blocks = the 9 common CV cells (seeds 0-2 x folds 0-2); MAMRD's extra seeds 3-4
are dropped so every detector is matched on identical cells. This mirrors the
Friedman+Nemenyi ranking that the Derzsi benchmark template reports (its Figs 13-15).

Demsar (2006) CD = q_alpha * sqrt(k(k+1)/(6N)); q_0.05 for k=7 is 2.949.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

N = RESULTS_DIR
B = N / "baselines"
FIG = N / "figures"

# (key, glob-dir, display label) — 7 SGCC detectors
DET = [
    ("grud",       B, "GRU-D"),
    ("hybridattn", B, "Self-attn."),
    ("dkgt",       B, "DKGT-Net"),
    ("wdcnn",      B, "WDCNN"),
    ("msdm_net",   B, "MSDM"),
    ("horln",      B, "HORLN"),
    ("a34_cell",   N, "MAMRD"),
]
CELLS = [(s, f) for s in (0, 1, 2) for f in (0, 1, 2)]   # 9 common blocks

# q_alpha (studentized range / sqrt2) for Nemenyi, alpha=0.05 (Demsar Table 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def ap(key, d, s, f):
    row = next(csv.DictReader(open(d / f"{key}_{s}_{f}.csv")))
    return float(row["pr_auc"])


def main():
    # matrix: rows = cells (blocks), cols = detectors
    M = np.array([[ap(k, d, s, f) for (k, d, _) in DET] for (s, f) in CELLS])
    labels = [lab for _, _, lab in DET]
    k = len(DET); Nb = len(CELLS)

    chi2, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    # per-block ranks: rank 1 = best (highest AP) -> negate for rankdata ascending
    ranks = np.array([rankdata(-M[i]) for i in range(Nb)])
    mean_ranks = ranks.mean(axis=0)
    CD = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * Nb))

    order = np.argsort(mean_ranks)  # best (lowest rank) first
    print(f"Friedman chi2={chi2:.3f}, p={p:.3e}  (k={k} detectors, N={Nb} cells)")
    print(f"Nemenyi CD (alpha=0.05) = {CD:.3f} mean-rank units")
    print(f"{'detector':<12}{'meanAP':>8}{'meanRank':>10}")
    for j in order:
        print(f"{labels[j]:<12}{M[:,j].mean():>8.3f}{mean_ranks[j]:>10.3f}")
    best = order[0]
    print("\nDetectors NOT separated from MAMRD by Nemenyi (|rank diff| < CD):")
    for j in order:
        if labels[j] == "MAMRD":
            continue
        diff = mean_ranks[j] - mean_ranks[list(labels).index("MAMRD")]
        sep = "separated" if abs(diff) >= CD else "NOT separated"
        print(f"  MAMRD vs {labels[j]:<11} rank diff={diff:+.3f}  -> {sep}")

    # ---- CD diagram (Demsar horizontal style) ----------------------------
    from fig_style import C, apply_style; apply_style()
    TOL = {"red": C["ours"], "dark": C["ink"], "grey": C["base"]}
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0, 1)
    ax.axis("off")
    # axis line + ticks
    ax.plot([lo, hi], [0.92, 0.92], color=TOL["dark"], lw=1.0)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0.92, 0.95], color=TOL["dark"], lw=0.8)
        ax.text(r, 0.99, str(r), ha="center", va="top", fontsize=7)
    # place detectors: left half on left, right half on right
    srt = list(order)
    half = (k + 1) // 2
    ylsel = 0.78
    for rankpos, j in enumerate(srt):
        mr = mean_ranks[j]
        left = rankpos < half
        yj = ylsel - (rankpos if left else (k - 1 - rankpos)) * 0.135
        xtext = lo - 0.45 if left else hi + 0.45
        ha = "right" if left else "left"
        col = TOL["red"] if labels[j] == "MAMRD" else TOL["dark"]
        fw = "bold" if labels[j] == "MAMRD" else "normal"
        ax.plot([mr, mr], [0.92, yj], color=col, lw=1.0)
        ax.plot([mr, xtext], [yj, yj], color=col, lw=1.0)
        ax.text(xtext + (-0.05 if left else 0.05), yj, f"{labels[j]} ({mr:.2f})",
                ha=ha, va="center", fontsize=7.5, color=col, fontweight=fw)
    # CD bar
    ax.plot([lo, lo + CD], [0.20, 0.20], color=TOL["dark"], lw=2.0)
    ax.plot([lo, lo], [0.18, 0.22], color=TOL["dark"], lw=1.0)
    ax.plot([lo + CD, lo + CD], [0.18, 0.22], color=TOL["dark"], lw=1.0)
    ax.text(lo + CD / 2, 0.10, f"CD = {CD:.2f}", ha="center", va="center", fontsize=7)
    # cliques: connect detectors whose mean ranks differ by < CD
    mr_sorted = mean_ranks[srt]
    y0 = 0.86
    drawn = []
    for a in range(k):
        b = a
        while b + 1 < k and (mr_sorted[b + 1] - mr_sorted[a]) < CD:
            b += 1
        if b > a and not any(a >= x and b <= y for x, y in drawn):
            ax.plot([mr_sorted[a] - 0.03, mr_sorted[b] + 0.03], [y0, y0],
                    color=TOL["grey"], lw=2.5, solid_capstyle="round")
            y0 -= 0.04
            drawn.append((a, b))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig7_cd_diagram.{ext}")
    print(f"\n[done] wrote {FIG}/fig7_cd_diagram.pdf (+png)")


if __name__ == "__main__":
    main()
