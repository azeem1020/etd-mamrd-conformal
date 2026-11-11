"""Generate empirical data-driven figures (Fig. 2, Fig. 3, Fig. 4, Fig. 5) for MAMRD manuscript.

Synchronizes all empirical data-driven vector PDFs directly with the released codebase and
evaluation records under the standardized leak-free protocol:
  - fig2_performance_curves.pdf
  - fig3_ablation_bars.pdf
  - fig4_calibration_reliability.pdf
  - fig5_conformal_tradeoffs.pdf

Note: Figure 1 (MAMRD Architectural Framework) is the conceptual schematic diagram
authored and compiled directly from dedicated vector LaTeX/TikZ source
(submission_package/figures/fig0_architecture_src.tex).
"""
from __future__ import annotations
import csv
import glob
import json
import sys
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Patch

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import RESULTS_DIR
from zkmix_lab.calibration import _stable_logit

N = RESULTS_DIR
FIG = N / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PAL = {
    "blue": "#1f4e78",
    "lblue": "#d9e8f5",
    "red": "#c0392b",
    "lred": "#fadbd8",
    "green": "#1e8449",
    "dgreen": "#145a32",
    "lgreen": "#d4efdf",
    "amber": "#b7950b",
    "lamber": "#fcf3cf",
    "purple": "#6c3483",
    "lpurple": "#ebdef0",
    "grey": "#7f8c8d",
    "lgrey": "#f2f4f4",
    "dark": "#1c2833",
}


def bca_ci(x, B=2000, seed=12345):
    """Same estimator, B and seed as compute_r2_revisions.bca_ci, so Fig. 3 intervals equal Table 5."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        m = float(np.mean(x)) if n else np.nan
        return m, m
    th = x.mean()
    rng = np.random.default_rng(seed)
    bt = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((bt < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)])
    jm = jk.mean()
    dn = 6.0 * (((jm - jk) ** 2).sum() ** 1.5)
    a = ((jm - jk) ** 3).sum() / dn if dn else 0.0
    q = lambda al: float(np.percentile(
        bt, 100.0 * norm.cdf(z0 + (z0 + norm.ppf(al)) / (1.0 - a * (z0 + norm.ppf(al))))
    ))
    return q(0.025), q(0.975)


# =========================================================================== #
# Fig. 2: Main Benchmark Precision-Recall & ROC Curves (SGCC)
# =========================================================================== #
def make_fig2_performance_curves():
    """Generates Figure 2: (a) Precision-Recall curves and (b) ROC curves across
    all re-evaluated baseline detectors and MAMRD directly from raw cell predictions.
    """
    models_config = [
        ("GRU-D",      N / "baselines/grud_{s}_{f}.csv",       {"color": "#2D3436", "lw": 1.3, "ls": (0, (3, 2, 1.5, 2)), "zorder": 1}),
        ("WDCNN",      N / "baselines/wdcnn_{s}_{f}.csv",      {"color": "#C97A10", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2, 1.5, 2)), "zorder": 2}),
        ("Self-Attn",  N / "baselines/hybridattn_{s}_{f}.csv", {"color": "#207D53", "lw": 1.5, "ls": (0, (1.5, 2)), "zorder": 3}),
        ("MSDM",       N / "baselines/msdm_net_{s}_{f}.csv",   {"color": "#6E388C", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2)), "zorder": 4}),
        ("HORLN",      N / "baselines/horln_{s}_{f}.csv",      {"color": "#D33F23", "lw": 1.4, "ls": (0, (5, 2)), "zorder": 5}),
        ("DKGT-Net",   N / "baselines/dkgt_{s}_{f}.csv",       {"color": "#8C6D31", "lw": 1.3, "ls": (0, (1, 1)), "zorder": 5}),
        ("MAMRD",      N / "a34_cell_{s}_{f}.csv",             {"color": "#1B5E94", "lw": 2.2, "ls": "-", "zorder": 6}),
    ]

    fig, (axP, axR) = plt.subplots(1, 2, figsize=(7.16, 2.85), layout="constrained")
    rgrid = np.linspace(0, 1, 200)
    fgrid = np.linspace(0, 1, 200)

    for name, pat, style in models_config:
        prc_list, rocc_list, ap_list, auc_list = [], [], [], []
        seeds = range(5) if name == "MAMRD" else range(3)
        folds = range(3)
        for s in seeds:
            for f in folds:
                fp = Path(str(pat).format(s=s, f=f))
                if not fp.exists():
                    continue
                r = next(csv.DictReader(open(fp)))
                if "val_labels_json" not in r or "val_scores_json" not in r:
                    continue
                y = np.array(json.loads(r["val_labels_json"]), int)
                sc = np.array(json.loads(r["val_scores_json"]), float)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                p_, r_, _ = precision_recall_curve(y, sc)
                o = np.argsort(r_)
                prc_list.append(np.interp(rgrid, r_[o], p_[o]))
                ap_list.append(average_precision_score(y, sc))

                fpr, tpr, _ = roc_curve(y, sc)
                rocc_list.append(np.interp(fgrid, fpr, tpr))
                auc_list.append(roc_auc_score(y, sc))

        if not prc_list:
            continue

        mean_ap = np.mean(ap_list)
        mean_auc = np.mean(auc_list)
        mean_prc = np.mean(prc_list, axis=0)
        mean_rocc = np.mean(rocc_list, axis=0)

        axP.plot(rgrid, mean_prc, color=style["color"], lw=style["lw"], ls=style["ls"],
                 label=f"{name} ({mean_ap:.2f})", zorder=style["zorder"])
        axR.plot(fgrid, mean_rocc, color=style["color"], lw=style["lw"], ls=style["ls"],
                 label=f"{name} ({mean_auc:.2f})", zorder=style["zorder"])

    # Panel (a) styling
    axP.axhline(0.0853, ls=":", lw=0.9, color="#636E72", alpha=0.8, label="Base rate (0.085)", zorder=0)
    axP.set_xlim(-0.02, 1.02)
    axP.set_ylim(-0.02, 1.02)
    axP.set_xlabel("Recall")
    axP.set_ylabel("Precision")
    axP.set_title("(a) Precision–Recall (AP)")
    axP.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DCE1E7", framealpha=0.95, fontsize=7.2)

    # Panel (b) styling
    axR.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#636E72", alpha=0.7, label="Chance (0.50)", zorder=0)
    axR.set_xlim(-0.02, 1.02)
    axR.set_ylim(-0.02, 1.02)
    axR.set_xlabel("False Positive Rate")
    axR.set_ylabel("True Positive Rate")
    axR.set_title("(b) ROC (AUC)")
    axR.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DCE1E7", framealpha=0.95, fontsize=7.2)

    out_pdf = FIG / "fig2_performance_curves.pdf"
    out_png = FIG / "fig2_performance_curves.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("  wrote fig2_performance_curves.pdf")


# =========================================================================== #
# Fig. 3: Ablation Study (Sequential Build-Up & Paired Contrasts, n=9 cells)
# =========================================================================== #
def make_fig3_ablation():
    def load_arm_pr(pat, seeds=(0, 1, 2), folds=(0, 1, 2)):
        out = {}
        for s in seeds:
            for f in folds:
                matches = sorted(glob.glob(str(pat).format(s=s, f=f)))
                if matches:
                    r = next(csv.DictReader(open(matches[0])))
                    out[(s, f)] = (float(r["pr_auc"]), float(r["roc_auc"]))
        return out

    s0 = load_arm_pr(N / "bu/a34_bu0_core_{s}_{f}.csv")
    s1 = load_arm_pr(N / "bu/a34_bu1_ccname_{s}_{f}.csv")
    s2 = load_arm_pr(N / "bu/a34_bu2_mask_{s}_{f}.csv")
    s3 = load_arm_pr(N / "bu/a33_generic_{s}_{f}.csv")
    s4 = load_arm_pr(N / "a34_cell_{s}_{f}.csv")

    # Also load a32 (no CC-NaME, no mask branch) from a32_sgcc.csv
    a32 = {}
    a32_fp = N / "a32_sgcc.csv"
    if a32_fp.exists():
        for r in csv.DictReader(open(a32_fp)):
            a32[(int(r["seed"]), int(r["fold"]))] = (float(r["pr_auc"]), float(r["roc_auc"]))

    common = sorted(set(s0) & set(s1) & set(s2) & set(s3) & set(s4))
    stages = [
        ("S0\nCore", s0),
        ("S1\n+Fill", s1),
        ("S2\n+Mask", s2),
        ("S3\n+Diff", s3),
        ("S4\nMAMRD", s4),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # Panel (a): Sequential build-up bars with 95% BCa CI
    means = [np.mean([st[k][0] for k in common]) for _, st in stages]
    cis = [bca_ci([st[k][0] for k in common]) for _, st in stages]
    yerr = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means, cis)]).T

    x = np.arange(len(stages))
    cols = [PAL["grey"], PAL["lblue"], PAL["lblue"], PAL["lblue"], PAL["blue"]]
    axA.bar(x, means, width=0.54, yerr=yerr, capsize=3,
            color=cols, edgecolor=PAL["dark"], linewidth=0.6)
    for xi, (m, (lo, hi)) in enumerate(zip(means, cis)):
        axA.text(xi, hi + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4, weight="bold")
    axA.set_xticks(x)
    axA.set_xticklabels([lab for lab, _ in stages], fontsize=6.5)
    # Dynamic y-limits with padding so newly retrained values are never clipped
    min_val = min(lo for lo, hi in cis)
    max_val = max(hi for lo, hi in cis)
    axA.set_ylim(min(0.45, min_val - 0.02), max(0.62, max_val + 0.025))
    axA.set_ylabel("PR-AUC (Average Precision)")
    axA.set_title("(a) Sequential build-up ($n=9$ cells)")

    # Panel (b): Paired ablation contrasts (Delta AP + 95% BCa CI)
    contrasts = [
        ("Full Stack vs. S0 Core ($S_4 - S_0$)", [s4[k][0] - s0[k][0] for k in common]),
        ("Drop Missingness + Periodicity ($\\mathcal{A}_{32} - S_4$)", [a32[k][0] - s4[k][0] for k in common if k in a32]),
        ("Drop Diff. & Periodicity ($S_2 - S_4$)", [s2[k][0] - s4[k][0] for k in common]),
        ("Step +Fixed Diff. ($S_3 - S_2$)", [s3[k][0] - s2[k][0] for k in common]),
        ("Step +CC-MVI ($S_1 - S_0$)", [s1[k][0] - s0[k][0] for k in common]),
        ("Step +2D Mask CNN ($S_2 - S_1$)", [s2[k][0] - s1[k][0] for k in common]),
        ("Step +Periodicity ($S_4 - S_3$)", [s4[k][0] - s3[k][0] for k in common]),
    ]
    # Check if controlled S4 no-missingness arm is present
    s4_nomiss = load_arm_pr(N / "bu/a34_no_missingness_{s}_{f}.csv")
    if not s4_nomiss:
        s4_nomiss = load_arm_pr(N / "a34_no_missingness_cell_{s}_{f}.csv")
    if s4_nomiss:
        common_nm = sorted(set(s4) & set(s4_nomiss))
        if common_nm:
            contrasts.insert(2, ("Controlled Drop Missingness ($\\mathcal{A}_{34,\\mathrm{no\\text{-}miss}} - S_4$)",
                                 [s4_nomiss[k][0] - s4[k][0] for k in common_nm]))

    yy = np.arange(len(contrasts))
    all_diffs = []
    for i, (lab, diffs) in enumerate(contrasts):
        m = float(np.mean(diffs))
        lo, hi = bca_ci(diffs)
        all_diffs.extend(diffs)
        sig = (lo > 0) or (hi < 0)
        c = PAL["red"] if (sig and m < 0) else (PAL["blue"] if sig else PAL["grey"])
        axB.plot([lo, hi], [i, i], "-", color=c, lw=1.8)
        axB.plot(m, i, "o", color=c, ms=4.5)
        axB.text(hi + 0.002 if m >= 0 else lo - 0.002, i,
                 f"{m:+.3f}", va="center",
                 ha="left" if m >= 0 else "right", fontsize=6.1, color=PAL["dark"])
    axB.axvline(0.0, ls="--", lw=0.8, color=PAL["dark"], alpha=0.8)
    axB.set_yticks(yy)
    axB.set_yticklabels([c[0] for c in contrasts], fontsize=6.2)
    min_d, max_d = min(all_diffs), max(all_diffs)
    pad = 0.02 * (max_d - min_d)
    axB.set_xlim(min(-0.065, min_d - pad), max(0.070, max_d + pad))
    axB.set_xlabel("Paired $\\Delta$ PR-AUC (95% BCa CI)")
    axB.set_title("(b) Paired build-up & drop-branch contrasts")

    fig.tight_layout()
    fig.savefig(FIG / "fig3_ablation_bars.pdf")
    fig.savefig(FIG / "fig3_ablation_bars.png")
    plt.close(fig)
    print("  wrote fig3_ablation_bars.pdf")


# =========================================================================== #
# Fig. 4: Calibration Reliability & ECE Before/After Affine Recalibration
# =========================================================================== #
def make_fig4_calibration():
    # Compute pooled reliability curves for MAMRD and per-detector ECE before/after affine recalibration
    arms = [
        ("GRU-D", N / "baselines/grud_{s}_{f}.csv", range(3), range(3)),
        ("Self-Attn", N / "baselines/hybridattn_{s}_{f}.csv", range(3), range(3)),
        ("DKGT-Net", N / "baselines/dkgt_{s}_{f}.csv", range(3), range(3)),
        ("WDCNN", N / "baselines/wdcnn_{s}_{f}.csv", range(3), range(3)),
        ("MSDM", N / "baselines/msdm_net_{s}_{f}.csv", range(3), range(3)),
        ("HORLN", N / "baselines/horln_{s}_{f}.csv", range(3), range(3)),
        ("MAMRD", N / "a34_cell_{s}_{f}.csv", range(5), range(3)),
    ]

    def ece_binned(p, y, n_bins=10):
        p = np.clip(p, 0.0, 1.0)
        y = y.astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(p)
        for i in range(n_bins):
            msk = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
            if msk.sum() > 0:
                ece += (msk.sum() / n) * abs(p[msk].mean() - y[msk].mean())
        return float(ece)

    det_names, ece_raw_means, ece_aff_means = [], [], []
    extra_rows = []  # Brier scores and fitted affine parameters (same cells, same split)
    mamrd_raw_p, mamrd_aff_p, mamrd_y = [], [], []

    for name, pat, seeds, folds in arms:
        er_list, ea_list, br_list, ba_list, w_list, b_list = [], [], [], [], [], []
        for s in seeds:
            for f in folds:
                fp = Path(str(pat).format(s=s, f=f))
                if not fp.exists():
                    continue
                r = next(csv.DictReader(open(fp)))
                y = np.array(json.loads(r["val_labels_json"]), int)
                sc = np.clip(np.array(json.loads(r["val_scores_json"]), float), 1e-6, 1.0 - 1e-6)
                ci, ti = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
                lr = LogisticRegression(C=1e4).fit(_stable_logit(sc[ci]).reshape(-1, 1), y[ci])
                p_aff = lr.predict_proba(_stable_logit(sc[ti]).reshape(-1, 1))[:, 1]
                er_list.append(ece_binned(sc[ti], y[ti], n_bins=10))
                ea_list.append(ece_binned(p_aff, y[ti], n_bins=10))
                br_list.append(float(np.mean((sc[ti] - y[ti]) ** 2)))
                ba_list.append(float(np.mean((p_aff - y[ti]) ** 2)))
                w_list.append(float(lr.coef_[0, 0])); b_list.append(float(lr.intercept_[0]))
                if name == "MAMRD":
                    mamrd_raw_p.append(sc[ti])
                    mamrd_aff_p.append(p_aff)
                    mamrd_y.append(y[ti])
        det_names.append(name)
        ece_raw_means.append(float(np.mean(er_list)) if er_list else 0.0)
        ece_aff_means.append(float(np.mean(ea_list)) if ea_list else 0.0)
        extra_rows.append({"brier_raw": float(np.mean(br_list)), "brier_affine": float(np.mean(ba_list)),
                           "w_mean": float(np.mean(w_list)), "w_sd": float(np.std(w_list, ddof=1)),
                           "b_mean": float(np.mean(b_list)), "b_sd": float(np.std(b_list, ddof=1)),
                           "n_cells": len(er_list)})

    raw_all = np.concatenate(mamrd_raw_p)
    aff_all = np.concatenate(mamrd_aff_p)
    y_all = np.concatenate(mamrd_y)

    def bin_curve(p, y, n_bins=10):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        px, py = [], []
        for i in range(n_bins):
            msk = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
            if msk.sum() >= 20:
                px.append(float(p[msk].mean()))
                py.append(float(y[msk].mean()))
        return np.array(px), np.array(py)

    # Values quoted in the text and Fig. 4 (mean over cells; MAMRD 15 cells, baselines 9)
    pd.concat([pd.DataFrame({"detector": det_names, "ece_raw": ece_raw_means, "ece_affine": ece_aff_means}),
               pd.DataFrame(extra_rows)], axis=1).to_csv(
        N / "trust" / "fig4_ece_by_detector.csv", index=False)

    rx, ry = bin_curve(raw_all, y_all, n_bins=10)
    ax_x, ax_y = bin_curve(aff_all, y_all, n_bins=10)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 2.85))

    # Panel (a): Reliability diagram
    axA.plot([0, 1], [0, 1], "--", color=PAL["dark"], lw=0.8, alpha=0.7, label="Perfect calibration")
    axA.plot(rx, ry, "-s", color=PAL["red"], lw=1.4, ms=4,
             label=f"Uncalibrated output (ECE = {ece_raw_means[-1]:.3f})")
    axA.plot(ax_x, ax_y, "-o", color=PAL["blue"], lw=1.5, ms=4,
             label=f"Affine recalibrated (ECE = {ece_aff_means[-1]:.3f})")
    axA.axhline(0.0853, ls=":", lw=0.8, color=PAL["grey"])
    axA.text(0.98, 0.10, "Native prevalence (8.5%)", ha="right", va="bottom", fontsize=6.2, color=PAL["dark"])
    axA.set_xlim(0, 1.0)
    axA.set_ylim(0, 1.0)
    axA.set_xlabel("Mean predicted theft probability")
    axA.set_ylabel("Empirical theft fraction")
    axA.set_title("(a) MAMRD reliability curve")
    axA.legend(loc="upper left", fontsize=6.3)

    # Panel (b): Expected calibration error before and after affine recalibration
    x = np.arange(len(det_names))
    w = 0.36
    axB.bar(x - w / 2, ece_raw_means, w, color=PAL["lred"], edgecolor=PAL["red"], lw=0.6,
            label="Before recalibration (raw)")
    axB.bar(x + w / 2, ece_aff_means, w, color=PAL["blue"], edgecolor=PAL["dark"], lw=0.6,
            label="After two-parameter affine recalibration")
    for xi, (er, ea) in enumerate(zip(ece_raw_means, ece_aff_means)):
        axB.text(xi - w / 2, er + 0.01, f"{er:.2f}", ha="center", va="bottom", fontsize=5.8, color=PAL["red"])
        axB.text(xi + w / 2, ea + 0.01, f"{ea:.2f}", ha="center", va="bottom", fontsize=5.8, color=PAL["blue"])
    axB.set_xticks(x)
    axB.set_xticklabels(det_names, fontsize=6.3, rotation=15)
    axB.set_ylim(0, 0.68)
    axB.set_ylabel("Expected Calibration Error (ECE)")
    axB.set_title("(b) ECE before & after affine recalibration")
    axB.legend(loc="upper right", fontsize=6.3)

    fig.tight_layout()
    fig.savefig(FIG / "fig4_calibration_reliability.pdf")
    fig.savefig(FIG / "fig4_calibration_reliability.png")
    plt.close(fig)
    print("  wrote fig4_calibration_reliability.pdf")


# =========================================================================== #
# Fig. 5: Conformal Tradeoffs (Exact Order Statistic k = ceil((n+1)(1-alpha)))
# =========================================================================== #
def make_fig5_conformal_tradeoffs():
    emp_df = pd.read_csv(N / "trust/empirical_comparator_sgcc.csv")
    dv_df = pd.read_csv(N / "trust/decision_value_sgcc.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.85))

    # Panel (a): Realized group recall on held-out evaluation splits (all 300 MAMRD resplits, alpha=0.20)
    groups = ["Heavy", "Light", "Flat", "Overall"]
    erow = emp_df.iloc[0]
    emp_rec = [
        100.0 * float(erow["emp_rec_heavy_mean"]),
        100.0 * float(erow["emp_rec_light_mean"]),
        100.0 * float(erow["emp_rec_flat_mean"]),
        100.0 * float(erow["emp_rec_overall_mean"]),
    ]
    conf_rec = [
        100.0 * float(erow["conf_rec_heavy_mean"]),
        100.0 * float(erow["conf_rec_light_mean"]),
        100.0 * float(erow["conf_rec_flat_mean"]),
        100.0 * float(erow["conf_rec_overall_mean"]),
    ]

    x = np.arange(len(groups))
    w = 0.35
    axA.bar(x - w / 2, emp_rec, w, color=PAL["lred"], edgecolor=PAL["red"], lw=0.6,
            label=r"Empirical ($\hat{q}_{1-\alpha}$)")
    axA.bar(x + w / 2, conf_rec, w, color=PAL["blue"], edgecolor=PAL["dark"], lw=0.6,
            label=r"Mondrian ($R_{(k_\phi)}$)")
    axA.axhline(80.0, ls="--", lw=1.0, color=PAL["dark"], zorder=1, label="80% Target")
    for xi, (er, cr) in enumerate(zip(emp_rec, conf_rec)):
        axA.text(xi - w / 2, er + 1.0, f"{er:.1f}%", ha="center", va="bottom", fontsize=6.0, color=PAL["red"])
        axA.text(xi + w / 2, cr + 1.0, f"{cr:.1f}%", ha="center", va="bottom", fontsize=6.0, color=PAL["blue"], weight="bold")
    axA.set_xticks(x)
    axA.set_xticklabels(groups, fontsize=7.0)
    axA.set_ylim(58.0, 96.0)
    axA.set_ylabel("Mean realized theft recall (%)")
    axA.set_title("(a) Realized recall vs. 80% target marginal recall")
    axA.legend(loc="upper center", ncol=3, fontsize=5.8, columnspacing=0.7, handletextpad=0.3, framealpha=0.9, edgecolor="#ccc")

    # Panel (b): Inspected population fraction (%) at alpha=0.20 across detectors (run_decision_value.py)
    order = [
        ("GRU-D", "GRU-D"),
        ("HybridAttn", "Self-Attn"),
        ("DKGT", "DKGT-Net"),
        ("WDCNN", "WDCNN"),
        ("MSDM", "MSDM"),
        ("HORLN", "HORLN"),
        ("a34_NaME", "MAMRD"),
    ]
    dv80 = dv_df[np.isclose(dv_df["guarantee"], 0.80)].set_index("arm")
    single_b = [100.0 * float(dv80.loc[a, "audit_single"]) for a, _ in order]
    group_b = [100.0 * float(dv80.loc[a, "audit_cond"]) for a, _ in order]
    red_pct = [100.0 * float(dv80.loc[a, "reduction"]) for a, _ in order]

    xb = np.arange(len(order))
    wb = 0.36
    axB.bar(xb - wb / 2, single_b, wb, color=PAL["lgrey"], edgecolor=PAL["dark"], lw=0.6,
            label=r"Single-threshold ($\min_\phi \hat{q}_\alpha^\phi$)")
    axB.bar(xb + wb / 2, group_b, wb, color=PAL["green"], edgecolor=PAL["dark"], lw=0.6,
            label="Group-conditional Mondrian")
    for xi, (sb, gb, rp) in enumerate(zip(single_b, group_b, red_pct)):
        top = sb + 1.6
        # Differential comparison bracket linking single-threshold to Mondrian
        axB.plot([xi - wb / 2, xi - wb / 2, xi + wb / 2, xi + wb / 2],
                 [sb + 0.5, top, top, gb + 0.5], color=PAL["dgreen"], lw=0.6)
        # Delta reduction percentage centered cleanly above the horizontal bracket
        axB.text(xi, top + 0.5, f"−{rp:.1f}%", ha="center", va="bottom",
                 fontsize=4.6, color=PAL["dgreen"], weight="bold",
                 path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
    axB.set_xticks(xb)
    axB.set_xticklabels([lab for _, lab in order], fontsize=6.3, rotation=15)
    axB.set_xlim(-0.6, len(order) - 0.4)
    axB.set_ylim(0, 118.0)
    axB.set_yticks([0, 20, 40, 60, 80, 100])
    axB.set_ylabel("Inspected consumer fraction (%)")
    axB.set_title(r"(b) Inspection burden reduction at $\alpha = 0.20$")
    axB.legend(loc="upper center", ncol=2, fontsize=5.8, columnspacing=0.8, handletextpad=0.3, framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    fig.savefig(FIG / "fig5_conformal_tradeoffs.pdf")
    fig.savefig(FIG / "fig5_conformal_tradeoffs.png")
    plt.close(fig)
    print("  wrote fig5_conformal_tradeoffs.pdf")


if __name__ == "__main__":
    print("[make_manuscript_figures] generating empirical publication figures (Figures 2, 3, 4, 5)...")
    make_fig2_performance_curves()
    make_fig3_ablation()
    make_fig4_calibration()
    make_fig5_conformal_tradeoffs()
    print(f"[done] all empirical publication figures saved to {FIG}")
