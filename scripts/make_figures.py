"""Publication-quality figures for Paper 3 (Engineering, Elsevier).

Senior-researcher figure set carrying the three contributions:
  Fig 1  metric_inflation   — C3: ROC/MAP@100 high, full PR-AUC collapses (per model)
  Fig 2  two_dataset_bars   — protocol-matched 5-baseline x 2-dataset PR-AUC (+CI)
  Fig 3  phenotype          — C1: per-phenotype PR-AUC varies 0.36-0.65 while ROC ~flat
  Fig 4  ceiling            — C2: (a) cumulative build-up plateau (b) lever forest plot
  Fig 5  pr_roc_curves      — pooled PR & ROC curves, a34 vs baselines (SGCC)

Reads locked CSVs from results/phase_n/. Outputs vector PDF + PNG preview to
results/phase_n/figures/. Re-run after SGCC+ECL baseline batches complete.

Style: Tol colourblind-safe palette, embedded fonts (Type-42), IEEE/Elsevier
two-column sizing (single col 3.5in, double 7.16in), CIs everywhere.
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve)

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
from fig_style import C, MODEL_COLOR, MODEL_NAME, apply_style, SINGLE, DOUBLE

N = RESULTS_DIR
B = N / "baselines"
FIG = N / "figures"; FIG.mkdir(exist_ok=True)

# ---- style -----------------------------------------------------------------
apply_style()
# compatibility shim: old rainbow keys -> new semantic palette (color = meaning)
TOL = {"red": C["honest"], "grey": C["base"], "dark": C["ink"], "blue": C["ours"],
       "cyan": C["base_lt"], "green": C["good"], "yellow": C["warn"], "purple": C["warn"]}

# model display order + colours (baselines neutral ramp, MAMRD the lone accent)
MODELS = [(k, MODEL_NAME[k], MODEL_COLOR[k])
          for k in ("grud", "wdcnn", "hybridattn", "horln", "msdm_net", "ours")]
PHENO = [("Heavy_Riser", "Heavy\nRiser"), ("Heavy_Dropper", "Heavy\nDropper"),
         ("Light_Riser", "Light\nRiser"), ("Light_Dropper", "Light\nDropper"),
         ("Flat", "Flat")]


def _bca_ci(x, B=8000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return np.nan, np.nan
    th = x.mean(); rng = np.random.default_rng(0)
    bt = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((bt < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)]); jm = jk.mean()
    dn = 6 * (((jm - jk) ** 2).sum() ** 1.5); a = ((jm - jk) ** 3).sum() / dn if dn else 0
    q = lambda al: np.percentile(bt, 100 * norm.cdf(z0 + (z0 + norm.ppf(al)) / (1 - a * (z0 + norm.ppf(al)))))
    return q(.025), q(.975)


def _mapk(y, s, K):
    o = np.argsort(-s)[:K]; yk = y[o]
    if yk.sum() == 0: return 0.0
    return float((np.cumsum(yk) / (np.arange(len(yk)) + 1) * yk).sum() / yk.sum())


def _load_cells(pat):
    """-> dict[(seed,fold)] = {pr,roc,f1,map100,scores,labels, pr_<ph>, roc_<ph>}"""
    out = {}
    for f in glob.glob(pat):
        for x in csv.DictReader(open(f)):
            if x.get("pr_auc", "") in ("", "nan"): continue
            rec = {"pr": float(x["pr_auc"]), "roc": float(x["roc_auc"]), "f1": float(x["f1"])}
            for ph, _ in PHENO:
                if f"pr_{ph}" in x and x[f"pr_{ph}"] not in ("", "nan"):
                    rec[f"pr_{ph}"] = float(x[f"pr_{ph}"]); rec[f"roc_{ph}"] = float(x[f"roc_{ph}"])
            if x.get("val_scores_json"):
                y = np.array(json.loads(x["val_labels_json"])); s = np.array(json.loads(x["val_scores_json"]))
                rec["labels"], rec["scores"] = y, s
                rec["map100"] = _mapk(y, s, 100)
            out[(int(x["seed"]), int(x["fold"]))] = rec
    return out


def _model_data():
    d = {"ours": _load_cells(str(N / "a34_cell_*.csv"))}
    for k, _, _ in MODELS:
        if k == "ours": continue
        d[k] = _load_cells(str(B / f"{k}_*.csv"))
    return d


def _mean(cells, key):
    v = [c[key] for c in cells.values() if key in c]
    return float(np.mean(v)) if v else np.nan


# ===========================================================================
def fig_metric_inflation(d):
    """C3: per model, ROC vs MAP@100 vs full PR-AUC — easy metrics saturate."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    names = [m[1].replace("\n", " ") for m in MODELS]
    roc = [_mean(d[k], "roc") for k, _, _ in MODELS]
    m100 = [_mean(d[k], "map100") for k, _, _ in MODELS]
    pr = [_mean(d[k], "pr") for k, _, _ in MODELS]
    x = np.arange(len(MODELS)); w = 0.26
    ax.bar(x - w, roc, w, label="ROC-AUC", color=TOL["grey"], edgecolor="k", linewidth=0.4)
    ax.bar(x, m100, w, label="MAP@100", color=TOL["cyan"], edgecolor="k", linewidth=0.4)
    ax.bar(x + w, pr, w, label="full-prevalence PR-AUC", color=TOL["red"], edgecolor="k", linewidth=0.4)
    for xi, (r, m, p) in enumerate(zip(roc, m100, pr)):
        if np.isfinite(m) and np.isfinite(p):
            ax.annotate("", xy=(xi + w, p), xytext=(xi, m),
                        arrowprops=dict(arrowstyle="->", color=TOL["dark"], lw=0.7, alpha=0.7))
            ax.text(xi + 0.02, (m + p) / 2, f"−{m-p:.2f}", fontsize=6, color=TOL["dark"])
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.axhline(0.0853, ls=":", lw=0.8, color="k", alpha=0.6)
    ax.text(len(MODELS) - 0.5, 0.10, "base rate 8.5%", fontsize=6, ha="right", color="k")
    ax.set_title("")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _save(fig, "fig1_metric_inflation")


def fig_phenotype(d):
    """C1: per-phenotype PR-AUC varies (consumption-level axis) while ROC stays flat."""
    cells = d["ours"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    pr = [_mean(cells, f"pr_{ph}") for ph, _ in PHENO]
    roc = [_mean(cells, f"roc_{ph}") for ph, _ in PHENO]
    x = np.arange(len(PHENO)); w = 0.38
    ax.bar(x - w/2, pr, w, label="PR-AUC", color=TOL["red"], edgecolor="k", linewidth=0.4)
    ax.bar(x + w/2, roc, w, label="ROC-AUC", color=TOL["grey"], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in PHENO], fontsize=6.5)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0)
    ax.set_title("")
    ax.legend(loc="upper right")
    if np.isfinite(pr[0]) and np.isfinite(pr[-1]):
        ax.annotate(f"{pr[0]/pr[-1]:.1f}× gap", xy=(0, pr[0]), xytext=(1.4, pr[0] + 0.06),
                    fontsize=7, color=TOL["dark"],
                    arrowprops=dict(arrowstyle="-", color=TOL["dark"], lw=0.6))
    _save(fig, "fig3_phenotype")


def fig_ceiling(d):
    """C2: (a) cumulative build-up plateau; (b) lever forest plot."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8), gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) build-up — paired means on common cells
    stages = [("S0\ncore", N / "bu/a34_bu0_core_*.csv"), ("S1\n+native", N / "bu/a34_bu1_ccname_*.csv"),
              ("S2\n+mask", N / "bu/a34_bu2_mask_*.csv"), ("S3\n+diff", N / "bu/a33_generic_*.csv"),
              ("S4\n+period", N / "a34_cell_*.csv")]
    sd = [(_load_cells(str(p))) for _, p in stages]
    common = set.intersection(*[set(s) for s in sd]) if all(sd) else set()
    means = [np.mean([s[k]["pr"] for k in common]) for s in sd] if common else [np.nan]*5
    xs = np.arange(5)
    axL.plot(xs, means, "-o", color=TOL["blue"], ms=4)
    axL.fill_between(xs, means, min(means), alpha=0.12, color=TOL["blue"])
    axL.set_xticks(xs); axL.set_xticklabels([s[0] for s in stages], fontsize=6.5)
    axL.set_ylabel("PR-AUC"); axL.set_title(f"(a) Cumulative build-up (n={len(common)})")
    if np.isfinite(means[0]):
        axL.text(0.03, 0.97, f"+{means[-1]-means[0]:.3f} total\n(plateau ≈ ceiling)",
                 transform=axL.transAxes, ha="left", va="top", fontsize=6.5, color=TOL["dark"])
    # (b) lever forest plot — locked Δ PR-AUC vs a14/a33 (from CLAIMS_N + build-up)
    levers = [  # (label, delta, lo, hi)
        ("Imputation (token vs 0)", -0.0006, -0.006, 0.005),
        ("Focal loss (γ=2)",        -0.0030, -0.011, 0.004),
        ("Complexity branch (L1)",  -0.0024, -0.011, 0.004),
        ("PhenoMoE (label-free)",   +0.0026, -0.004, 0.010),
        ("Periodicity arm (S4)",    +0.0028, -0.008, 0.013),
        ("Differencing arm (S3)",   +0.0141, +0.004, 0.028),
        ("Full MAMRD stack vs WDCNN",   +0.0135, +0.006, 0.021),
    ]
    yy = np.arange(len(levers))
    for i, (lab, dlt, lo, hi) in enumerate(levers):
        sig = lo > 0 or hi < 0
        c = TOL["red"] if sig else TOL["grey"]
        axR.plot([lo, hi], [i, i], "-", color=c, lw=1.4)
        axR.plot(dlt, i, "o", color=c, ms=4)
    axR.axvline(0, ls="--", lw=0.8, color="k", alpha=0.7)
    axR.axvspan(-0.01, 0.01, color="k", alpha=0.05)
    axR.set_yticks(yy); axR.set_yticklabels([l[0] for l in levers], fontsize=6.5)
    axR.set_xlabel("Δ PR-AUC (95% BCa CI)"); axR.set_title("(b) Interventions vs the ceiling")
    axR.legend(handles=[Patch(color=TOL["red"], label="significant + material"),
                        Patch(color=TOL["grey"], label="null (CI incl. 0)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    _save(fig, "fig4_ceiling")


def fig_pr_roc(d):
    """Cell-averaged PR + ROC curves; legend AP/AUC are means over cells (match tab_baselines)."""
    fig, (axP, axR) = plt.subplots(1, 2, figsize=(DOUBLE, 2.8))
    rgrid = np.linspace(0, 1, 200); fgrid = np.linspace(0, 1, 200)
    for k, lab, col in MODELS:
        cells = d[k]
        prc, rocc, aps, aucs = [], [], [], []
        for c in cells.values():
            if "labels" not in c: continue
            y, s = c["labels"], c["scores"]
            if y.sum() == 0 or y.sum() == len(y): continue
            p_, r_, _ = precision_recall_curve(y, s); o = np.argsort(r_)
            prc.append(np.interp(rgrid, r_[o], p_[o])); aps.append(average_precision_score(y, s))
            fpr, tpr, _ = roc_curve(y, s)
            rocc.append(np.interp(fgrid, fpr, tpr)); aucs.append(roc_auc_score(y, s))
        if not prc: continue
        lw = 1.8 if k == "ours" else 1.0; nm = lab.split(chr(10))[0]
        axP.plot(rgrid, np.mean(prc, axis=0), color=col, lw=lw, label=f"{nm} ({np.mean(aps):.2f})")
        axR.plot(fgrid, np.mean(rocc, axis=0), color=col, lw=lw, label=f"{nm} ({np.mean(aucs):.2f})")
    axP.axhline(0.0853, ls=":", lw=0.8, color="k", alpha=0.6)
    axP.set_xlabel("recall"); axP.set_ylabel("precision"); axP.set_title("(a) Precision–Recall (AP)")
    axP.legend(loc="upper right"); axP.set_ylim(0, 1)
    axR.plot([0, 1], [0, 1], ls="--", lw=0.7, color="k", alpha=0.5)
    axR.set_xlabel("false positive rate"); axR.set_ylabel("true positive rate"); axR.set_title("(b) ROC (AUC)")
    axR.legend(loc="lower right")
    _save(fig, "fig5_pr_roc_curves")


def fig_two_dataset(d, ecl_csv):
    """6-detector x 2-dataset PR-AUC bars (+BCa CI). Both datasets reported at the
    same n=9 cells (3 folds x 3 seeds); the ECL column reads the trust per-cell dump
    so it is the full-prevalence replicated mean, not a single optimistic fold."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.7))
    sg = {k: [c["pr"] for c in d[k].values()] for k, _, _ in MODELS}
    # ECL arm naming in the per-cell dump differs from the SGCC display keys
    ECL_KEY = {"ours": "a34_generic", "msdm_net": "msdm"}  # remaining keys identical
    ecl = {}
    if Path(ecl_csv).exists():
        for x in csv.DictReader(open(ecl_csv)):
            ecl.setdefault(x["arm"], []).append(float(x["ap"]))
    x = np.arange(len(MODELS)); w = 0.38
    sg_m = [np.mean(sg[k]) if sg[k] else np.nan for k, _, _ in MODELS]
    sg_e = [np.subtract(*reversed(_bca_ci(sg[k]))) / 2 if len(sg[k]) > 1 else 0 for k, _, _ in MODELS]
    ec_l = [ecl.get(ECL_KEY.get(k, k), []) for k, _, _ in MODELS]
    ec_m = [np.mean(v) if v else np.nan for v in ec_l]
    ec_e = [np.subtract(*reversed(_bca_ci(v))) / 2 if len(v) > 1 else 0 for v in ec_l]
    ax.bar(x - w/2, sg_m, w, yerr=np.abs(sg_e), capsize=2, label="SGCC (real theft)",
           color=TOL["blue"], edgecolor="k", linewidth=0.4)
    ax.bar(x + w/2, ec_m, w, yerr=np.abs(ec_e), capsize=2, label="ECL (real load, synthetic FDI)",
           color=TOL["green"], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([m[1].replace("\n", " ") for m in MODELS], fontsize=6.5)
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("")
    ax.legend(loc="upper left")
    _save(fig, "fig2_two_dataset")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = _model_data()
    print(f"[figures] models with data: " +
          ", ".join(f"{k}({len(d[k])})" for k in d))
    if which in ("all", "1"): fig_metric_inflation(d)
    if which in ("all", "3"): fig_phenotype(d)
    if which in ("all", "4"): fig_ceiling(d)
    if which in ("all", "5"): fig_pr_roc(d)
    if which in ("all", "2"): fig_two_dataset(d, str(N / "ecl_trust/ecl_trust_percell.csv"))
    print(f"[figures] -> {FIG}")
