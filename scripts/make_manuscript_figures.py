"""Generate empirical data-driven figures (Fig. 3, Fig. 4, Fig. 5) for MAMRD manuscript.

Synchronizes all empirical data-driven vector PDFs directly with the released codebase and
cached phase_n evaluation records under the corrected exact conformal order
statistic k = ceil((n + 1)(1 - alpha)):
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

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
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
    "lgreen": "#d4efdf",
    "amber": "#b7950b",
    "lamber": "#fcf3cf",
    "purple": "#6c3483",
    "lpurple": "#ebdef0",
    "grey": "#7f8c8d",
    "lgrey": "#f2f4f4",
    "dark": "#1c2833",
}


def bca_ci(x, B=4000, seed=12345):
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
    axA.set_ylim(0.46, 0.615)
    axA.set_ylabel("PR-AUC (Average Precision)")
    axA.set_title("(a) Sequential build-up ($n=9$ cells)")

    # Panel (b): Paired ablation contrasts (Delta AP + 95% BCa CI)
    contrasts = [
        ("Full Stack vs. S0 Core ($S_4 - S_0$)", [s4[k][0] - s0[k][0] for k in common]),
        ("Drop Both Missingness ($\\mathcal{A}_{\\mathrm{no\\text{-}miss}} - S_4$)", [a32[k][0] - s4[k][0] for k in common if k in a32]),
        ("Drop Diff. & Periodicity ($S_2 - S_4$)", [s2[k][0] - s4[k][0] for k in common]),
        ("Step +Fixed Diff. ($S_3 - S_2$)", [s3[k][0] - s2[k][0] for k in common]),
        ("Step +CC-MVI ($S_1 - S_0$)", [s1[k][0] - s0[k][0] for k in common]),
        ("Step +2D Mask CNN ($S_2 - S_1$)", [s2[k][0] - s1[k][0] for k in common]),
        ("Step +Periodicity ($S_4 - S_3$)", [s4[k][0] - s3[k][0] for k in common]),
    ]
    yy = np.arange(len(contrasts))
    for i, (lab, diffs) in enumerate(contrasts):
        m = float(np.mean(diffs))
        lo, hi = bca_ci(diffs)
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
    axB.set_xlim(-0.062, 0.068)
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

    def ece_equal_mass(p, y, n_bins=15):
        p = np.clip(p, 0.0, 1.0)
        y = y.astype(float)
        order = np.argsort(p)
        p, yy = p[order], y[order]
        ece = 0.0
        n = len(p)
        for b in np.array_split(np.arange(n), n_bins):
            if len(b):
                ece += (len(b) / n) * abs(p[b].mean() - yy[b].mean())
        return float(ece)

    det_names, ece_raw_means, ece_aff_means = [], [], []
    mamrd_raw_p, mamrd_aff_p, mamrd_y = [], [], []

    for name, pat, seeds, folds in arms:
        er_list, ea_list = [], []
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
                er_list.append(ece_equal_mass(sc[ti], y[ti]))
                ea_list.append(ece_equal_mass(p_aff, y[ti]))
                if name == "MAMRD":
                    mamrd_raw_p.append(sc[ti])
                    mamrd_aff_p.append(p_aff)
                    mamrd_y.append(y[ti])
        det_names.append(name)
    # Load canonical ECE metrics from calibration_sgcc.csv (generated by run_trust_layer.py)
    cal_csv = N / "trust/calibration_sgcc.csv"
    if cal_csv.exists():
        cdf = pd.read_csv(cal_csv)
        arm_map = {
            "GRU-D": "GRU-D",
            "Self-Attn": "HybridAttn",
            "DKGT-Net": "DKGT",
            "WDCNN": "WDCNN",
            "MSDM": "MSDM",
            "HORLN": "HORLN",
            "MAMRD": "a34_NaME",
        }
        ece_raw_means = [float(cdf[cdf.arm == arm_map[name]]["ece_raw"].mean()) for name in det_names]
        ece_aff_means = [float(cdf[cdf.arm == arm_map[name]]["ece_affine"].mean()) for name in det_names]
    else:
        ece_raw_means = [float(np.mean(er)) for er in [er_list]]
        ece_aff_means = [float(np.mean(ea)) for ea in [ea_list]]

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

    # Panel (a): Realized group recall on held-out evaluation splits (281 valid resplits, alpha=0.20)
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
            label="Empirical group-quantile ($\\hat{q}_{1-\\alpha}^{\\mathrm{emp}}$)")
    axA.bar(x + w / 2, conf_rec, w, color=PAL["blue"], edgecolor=PAL["dark"], lw=0.6,
            label="Mondrian conformal ($R_{(k_\\phi)}^\\phi$, $k_\\phi=\\lceil(n_\\phi+1)(1-\\alpha)\\rceil$)")
    axA.axhline(80.0, ls="--", lw=1.0, color=PAL["dark"], label="Target marginal recall ($1-\\alpha = 80\\%$)")
    for xi, (er, cr) in enumerate(zip(emp_rec, conf_rec)):
        axA.text(xi - w / 2, er + 0.8, f"{er:.1f}%", ha="center", va="bottom", fontsize=6.0, color=PAL["red"])
        axA.text(xi + w / 2, cr + 0.8, f"{cr:.1f}%", ha="center", va="bottom", fontsize=6.0, color=PAL["blue"], weight="bold")
    axA.set_xticks(x)
    axA.set_xticklabels(groups, fontsize=7.0)
    axA.set_ylim(60.0, 93.0)
    axA.set_ylabel("Mean realized theft recall (%)")
    axA.set_title("(a) Realized recall vs. 80% target marginal recall")
    axA.legend(loc="lower right", fontsize=5.9)

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
    axB.bar(xb - w / 2, single_b, w, color=PAL["lgrey"], edgecolor=PAL["dark"], lw=0.6,
            label="Single-threshold conformal ($\\min_\\phi \\hat{q}_\\alpha^\\phi$)")
    axB.bar(xb + w / 2, group_b, w, color=PAL["green"], edgecolor=PAL["dark"], lw=0.6,
            label="Group-conditional Mondrian conformal")
    for xi, (sb, gb, rp) in enumerate(zip(single_b, group_b, red_pct)):
        axB.text(xi + w / 2, gb + 1.2, f"−{rp:.1f}%", ha="center", va="bottom",
                 fontsize=5.8, color=PAL["green"], weight="bold")
    axB.set_xticks(xb)
    axB.set_xticklabels([lab for _, lab in order], fontsize=6.3, rotation=15)
    axB.set_ylim(0, 92.0)
    axB.set_ylabel("Inspected consumer fraction (%)")
    axB.set_title("(b) Inspection burden reduction at $\\alpha = 0.20$")
    axB.legend(loc="upper right", fontsize=6.1)

    fig.tight_layout()
    fig.savefig(FIG / "fig5_conformal_tradeoffs.pdf")
    fig.savefig(FIG / "fig5_conformal_tradeoffs.png")
    plt.close(fig)
    print("  wrote fig5_conformal_tradeoffs.pdf")


if __name__ == "__main__":
    print("[make_manuscript_figures] generating empirical figures (Figures 3, 4, 5)...")
    make_fig3_ablation()
    make_fig4_calibration()
    make_fig5_conformal_tradeoffs()
    print(f"[done] all empirical figures saved to {FIG}")
