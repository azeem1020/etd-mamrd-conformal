#!/usr/bin/env python3
"""Compute all 100% 5-fold verified manuscript tables, statistics, and regenerate
all vector PDF + PNG figures into results/r3_revision/ from saved R2 & R3 outputs.

Outputs (under results/r3_revision/):
  1. r3_manuscript_verified_ledger.json              (single source of truth for R3 text/tables)
  2. tables/tab_strata.tex, tab_matched.tex, tab_perstratum.tex + 7 authoritative 5-fold CSVs
  3. figures/fig2_performance_curves.pdf             (5-fold SGCC PR/ROC)
  4. figures/fig3_ablation_bars.pdf                  (S0-S4 M=15 + C2 3-way & clean, Folds 0-4)
  5. figures/fig4_calibration_reliability.pdf        (5-fold constrained w=exp(eta)>0)
  6. figures/fig5_conformal_tradeoffs.pdf            (5-fold SGCC Mondrian audit)
  7. figures/fig1_metric_inflation.pdf               (5-fold SGCC Fig S2)
  8. figures/fig6_trust_guarantee.pdf                (5-fold SGCC Fig S4)
  9. figures/fig_s6_r3_conformal_robustness.pdf      (5-fold C4 IPW/pooled + C6 schemes)
     (plus preview PNGs in fig_previews/)
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
from collections import defaultdict
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, roc_curve,
)

csv.field_size_limit(10**7)

THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
_REPO_DIR = SCRIPTS_DIR.parent
for p in (str(THIS), str(SCRIPTS_DIR), str(_REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import _ROOT, RESULTS_DIR, SGCC_PATH

ORIG_REPO = _ROOT
ORIG_RES = RESULTS_DIR
R3_RES = _ROOT / "results" / "r3_revision"
RAW34 = R3_RES / "C1_5fold_and_stats" / "raw_folds34"
C3_DIR = R3_RES / "C3_ecl_client_disjoint" / "cells"
TABLES_DIR = R3_RES / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = R3_RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_FIG_PNG = R3_RES / "fig_previews"
SCRATCH_FIG_PNG.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec
from zkmix_lab.metrics import recall_at_fpr, topk_tie_neutral
from run_local_grid import _stratified_three_way_split
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from conformal_common import NREP, n_min, resplit_rng, stratified_half, cc_recall_threshold, cell_bca

N_DIST = NormalDist()
ALL_FOLDS = (0, 1, 2, 3, 4)

# Identical arm ordering to run_decision_value.py & compute_r2_revisions.py
SGCC_ARMS = [
    ("GRU-D",           "GRU-D",      "baselines/grud_{s}_{f}.csv",       range(3), 4),
    ("Self-Attn",       "HybridAttn", "baselines/hybridattn_{s}_{f}.csv", range(3), 5),
    ("DKGT-Net",        "DKGT",       "baselines/dkgt_{s}_{f}.csv",       range(3), 6),
    ("WDCNN",           "WDCNN",      "baselines/wdcnn_{s}_{f}.csv",      range(3), 3),
    ("MSDM",            "MSDM",       "baselines/msdm_net_{s}_{f}.csv",   range(3), 2),
    ("HORLN",           "HORLN",      "baselines/horln_{s}_{f}.csv",      range(3), 1),
    ("MAMRD (matched)", "a34_NaME",   "a34_cell_{s}_{f}.csv",             range(3), 0),
    ("MAMRD (5-seed)",  "a34_NaME",   "a34_cell_{s}_{f}.csv",             range(5), 0),
]

# SGCC Table 4 Note shares SGCC_ARMS Table 6 resplits; ECL Table 4 Note uses independent seeded
# 50/50 stratified calibration/evaluation draws across the 15 client-disjoint cells.
ECL_C3_ARMS = [
    ("GRU-D",     "grud",        4),
    ("Self-Attn", "hybridattn",  5),
    ("WDCNN",     "wdcnn",       3),
    ("MSDM",      "msdm",        2),
    ("HORLN",     "horln",       1),
    ("MAMRD",     "a34_generic", 0),
]

PAL = {
    "blue": "#1F5A8A", "lblue": "#AEC7E8", "orange": "#D95F02",
    "red": "#C0392B", "lred": "#F5B7B1", "green": "#1E8449",
    "dgreen": "#145A32", "purple": "#6C3483", "grey": "#7F8C8D",
    "lgrey": "#D5DBDB", "dark": "#2C3E50",
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


def bca_ci(x, B: int = 2000, alpha: float = 0.05, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(arr)
    if n < 2 or np.allclose(arr, arr[0]):
        m = float(np.mean(arr)) if n else float("nan")
        return m, m
    th = float(arr.mean())
    rng = np.random.default_rng(seed)
    bt = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(B)])
    p_lt = min(max(float(np.mean(bt < th)), 1e-6), 1.0 - 1e-6)
    z0 = N_DIST.inv_cdf(p_lt)
    jk = np.array([np.delete(arr, i).mean() for i in range(n)])
    jm = jk.mean()
    den = 6.0 * (np.sum((jm - jk) ** 2) ** 1.5)
    a = float(np.sum((jm - jk) ** 3) / den) if den else 0.0
    def adj(q):
        z = N_DIST.inv_cdf(q)
        denom = max(1.0 - a * (z0 + z), 1e-6)
        return N_DIST.cdf(z0 + (z0 + z) / denom)
    return float(np.quantile(bt, np.clip(adj(alpha / 2), 0, 1))), float(np.quantile(bt, np.clip(adj(1 - alpha / 2), 0, 1)))


def cell_path(rel_pat: str, s: int, f: int) -> Path:
    sub = rel_pat.format(s=s, f=f)
    if f in (3, 4):
        return RAW34 / Path(sub).name
    return ORIG_RES / sub


def load_csv_cell(fp: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    r = next(csv.DictReader(open(fp)))
    return r, np.array(json.loads(r["val_labels_json"]), int), np.array(json.loads(r["val_scores_json"]), float)


def collapse_grp(ph: np.ndarray) -> np.ndarray:
    out = np.empty(len(ph), dtype=object)
    for i, p in enumerate(ph):
        out[i] = "Heavy" if p in ("Heavy_Riser", "Heavy_Dropper") else ("Light" if p in ("Light_Riser", "Light_Dropper") else "Flat")
    return out


def stable_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_positive_exp(z: np.ndarray, y: np.ndarray, C: float = 1e4) -> tuple[float, float, bool]:
    """Minimize Eq. (9) via BFGS with scale-invariant mean normalization."""
    n = float(len(y))
    def loss_grad(th):
        eta, b = float(th[0]), float(th[1])
        w = np.exp(np.clip(eta, -20.0, 20.0))
        lin = w * z + b
        nll = (np.sum(np.logaddexp(0.0, lin) - y * lin) + 0.5 * (w ** 2) / C) / n
        resid = expit(lin) - y
        return float(nll), np.array([((float(np.sum(resid * z)) + w / C) * w) / n, float(np.sum(resid)) / n])
    res = minimize(loss_grad, np.array([0.0, 0.0]), method="BFGS", jac=True, options={"gtol": 1e-9, "maxiter": 500})
    return float(np.exp(res.x[0])), float(res.x[1]), bool(res.success)


def fit_positive_lbfgsb(z: np.ndarray, y: np.ndarray, C: float = 1e4) -> tuple[float, float, bool]:
    """Cross-check Eq. (9) directly via box-constrained L-BFGS-B (w >= 1e-8)."""
    n = float(len(y))
    def loss_grad(wb):
        w, b = float(wb[0]), float(wb[1])
        lin = w * z + b
        nll = (np.sum(np.logaddexp(0.0, lin) - y * lin) + 0.5 * (w ** 2) / C) / n
        resid = expit(lin) - y
        return float(nll), np.array([(float(np.sum(resid * z)) + w / C) / n, float(np.sum(resid)) / n])
    res = minimize(
        loss_grad, np.array([1.0, 0.0]), method="L-BFGS-B", jac=True,
        bounds=[(1e-8, None), (None, None)], options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
    )
    return float(res.x[0]), float(res.x[1]), bool(res.success)


def _fmt_sci_tex(val: float) -> str:
    """Format floating-point value as LaTeX scientific notation (e.g. 3.89 \\times 10^{-8})."""
    s = f"{val:.2e}"
    mant, exp_s = s.split("e")
    return f"{mant} \\times 10^{{{int(exp_s)}}}"


def ece_binned(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.clip(p, 0.0, 1.0); y = y.astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0; n = len(p)
    for i in range(n_bins):
        msk = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if msk.sum() > 0:
            ece += (msk.sum() / n) * abs(p[msk].mean() - y[msk].mean())
    return float(ece)


def save_fig(fig, stem: str, copy_to_root: bool = False) -> None:
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = SCRATCH_FIG_PNG / f"{stem}.png"
    fig.savefig(pdf_p)
    fig.savefig(png_p, dpi=220)
    plt.close(fig)
    print(f"  wrote {pdf_p.name} (+ preview PNG)")


def nanmean_zero(arr2d: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(arr2d), axis=1)
    return np.where(cnt > 0, np.nansum(arr2d, axis=1) / np.maximum(cnt, 1), 0.0)


def nanstd_zero(arr2d: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(arr2d), axis=1)
    m = nanmean_zero(arr2d)[:, None]
    sq = np.where(~np.isnan(arr2d), (arr2d - m) ** 2, 0.0).sum(axis=1)
    return np.where(cnt > 1, np.sqrt(sq / np.maximum(cnt, 1)), 0.0)


def stratified_frac(y: np.ndarray, f_cal: float, rng: np.random.Generator):
    idx = np.arange(len(y)); cal = []
    for c in (0, 1):
        ic = idx[y == c].copy(); rng.shuffle(ic)
        k = min(max(1, int(round(f_cal * len(ic)))), len(ic) - 1)
        cal.append(ic[:k])
    cal_idx = np.concatenate(cal); mask = np.zeros(len(y), bool); mask[cal_idx] = True
    return np.where(mask)[0], np.where(~mask)[0]


def weighted_conformal_flags(s_cal_pos, w_cal_pos, s_eval, w_eval, alpha: float) -> np.ndarray:
    n = len(s_cal_pos)
    if n == 0:
        return np.ones(len(s_eval), dtype=bool)
    r_cal = 1.0 - np.asarray(s_cal_pos, float)
    w_cal = np.asarray(w_cal_pos, float)
    order = np.argsort(r_cal, kind="mergesort")
    r_sorted = r_cal[order]; w_cum = np.cumsum(w_cal[order]); W_tot = float(w_cum[-1])
    r_eval = 1.0 - np.asarray(s_eval, float)
    needed = (1.0 - alpha) * (W_tot + np.asarray(w_eval, float))
    pos_idx = np.searchsorted(w_cum, needed, side="left")
    flags = np.empty(len(s_eval), dtype=bool)
    inf_m = pos_idx >= n; flags[inf_m] = True
    fin_m = ~inf_m
    if fin_m.any():
        flags[fin_m] = r_eval[fin_m] <= r_sorted[pos_idx[fin_m]]
    return flags


def ci_str(lo: float, hi: float, fmt: str = ".3f", signed: bool = False) -> str:
    s = "+" if signed else ""
    return f"[\\,{lo:{s}{fmt}},\\ {hi:{s}{fmt}}\\,]"


def main() -> None:
    ledger: dict = {}

    # ------------------------------------------------------------------
    # 1. Load SGCC raw matrix & 5-fold splits + router groups
    # ------------------------------------------------------------------
    print("[1/7] Computing 5-fold SGCC partitions & router statistics...", flush=True)
    if not SGCC_PATH.exists():
        raise FileNotFoundError(
            f"Missing SGCC raw dataset at {SGCC_PATH}. Obtain and prepare data/SGCC.csv "
            "per data/README.md (scripts/prepare_sgcc_dataset.py) or set MAMRD_SGCC."
        )
    df_sgcc = pd.read_csv(SGCC_PATH)
    y_all = df_sgcc["FLAG"].values.astype(np.int64)
    date_cols = list(df_sgcc.columns[2:1031])
    dow = pd.to_datetime(date_cols, format="%Y-%m-%d").dayofweek.to_numpy()
    wkday_mask, wkend_mask = (dow < 5), (dow >= 5)

    raw = df_sgcc.drop(columns=["CONS_NO", "FLAG"]).values[:, :1029].astype(np.float64)
    miss_rate_all = np.isnan(raw).mean(axis=1)
    daily_kwh_all = nanmean_zero(df_sgcc.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64))
    g_all, m_all = gap_magnitude(raw.reshape(len(df_sgcc), 147, 7))
    g_cal_all = (nanmean_zero(raw[:, wkday_mask]) - nanmean_zero(raw[:, wkend_mask])).astype(np.float64)
    sd_all = nanstd_zero(raw).astype(np.float64)

    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_all))
    fold_info = {}
    router_cutoffs = []
    t2_acc = defaultdict(list)

    for f in ALL_FOLDS:
        tr, val = folds[f]
        es, cal, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        params = fit_router(g_all[tr], m_all[tr])
        router_cutoffs.append(params)
        ph_te = assign(g_all[te], m_all[te], params)
        grp_te = collapse_grp(ph_te)
        fold_info[f] = {"tr": tr, "es": es, "cal": cal, "te": te, "ph": ph_te, "grp": grp_te}
        t2_acc["train_n"].append(len(tr)); t2_acc["train_pos"].append(int(y_all[tr].sum()))
        t2_acc["es_n"].append(len(es)); t2_acc["es_pos"].append(int(y_all[es].sum()))
        t2_acc["cal_n"].append(len(cal)); t2_acc["cal_pos"].append(int(y_all[cal].sum()))
        t2_acc["te_n"].append(len(te)); t2_acc["te_pos"].append(int(y_all[te].sum()))
        for gn in ("Heavy", "Light", "Flat"):
            t2_acc[f"te_{gn}_n"].append(int((grp_te == gn).sum()))
            t2_acc[f"te_{gn}_pos"].append(int(((grp_te == gn) & (y_all[te] == 1)).sum()))

    ledger["sgcc_router_cutoffs_5fold"] = {
        "tau_f_mean": float(np.mean([p["tau"] for p in router_cutoffs])),
        "tau_f_sd": float(np.std([p["tau"] for p in router_cutoffs], ddof=1)),
        "mu_plus_mean": float(np.mean([p["med_drop"] for p in router_cutoffs])),
        "mu_plus_sd": float(np.std([p["med_drop"] for p in router_cutoffs], ddof=1)),
        "mu_minus_mean": float(np.mean([p["med_rise"] for p in router_cutoffs])),
        "mu_minus_sd": float(np.std([p["med_rise"] for p in router_cutoffs], ddof=1)),
    }

    # Verify disjoint union of 5 test folds
    all_te_idx = np.concatenate([fold_info[f]["te"] for f in ALL_FOLDS])
    assert len(all_te_idx) == len(set(all_te_idx.tolist())) == 14420
    # Aggregate consumer & kWh profile across all 14,420 unique test consumers
    df_te_all = pd.DataFrame({
        "group": np.concatenate([fold_info[f]["grp"] for f in ALL_FOLDS]),
        "y": y_all[all_te_idx],
        "kwh": daily_kwh_all[all_te_idx],
    })
    df_th = df_te_all[df_te_all["y"] == 1]
    sgcc_grp_rates = {gn: float(df_te_all[df_te_all["group"] == gn]["y"].mean()) for gn in ("Heavy", "Light", "Flat")}
    sgcc_grp_shares = {
        gn: {
            "test_pop_pct": 100.0 * float((df_te_all["group"] == gn).mean()),
            "test_pos_count": int((df_th["group"] == gn).sum()),
            "theft_case_share_pct": 100.0 * float((df_th["group"] == gn).mean()),
            "theft_kwh_share_pct": 100.0 * float(df_th[df_th["group"] == gn]["kwh"].sum() / df_th["kwh"].sum()),
            "theft_median_daily_kwh": float(df_th[df_th["group"] == gn]["kwh"].median()),
            "all_median_daily_kwh": float(df_te_all[df_te_all["group"] == gn]["kwh"].median()),
            "base_rate_pct": 100.0 * sgcc_grp_rates[gn],
        }
        for gn in ("Heavy", "Light", "Flat")
    }
    ledger["sgcc_group_profile_5fold"] = sgcc_grp_shares

    # Load all SGCC cell CSVs once
    sgcc_cells = {}
    for det, dv_arm, pat, seeds, arm_idx in SGCC_ARMS:
        for s in seeds:
            for f in ALL_FOLDS:
                if (pat, s, f) not in sgcc_cells:
                    _, yt, sc = load_csv_cell(cell_path(pat, s, f))
                    assert len(yt) == len(fold_info[f]["te"])
                    assert np.array_equal(yt, y_all[fold_info[f]["te"]])
                    sgcc_cells[(pat, s, f)] = (yt, sc)

    # ------------------------------------------------------------------
    # 2. 5-fold SGCC Conformal Split Ledger, Unadjusted vs Order Statistic & Light Hit/Lift
    # ------------------------------------------------------------------
    print("[2/7] Computing 5-fold SGCC conformal splits & Light/Unadjusted stats...", flush=True)
    flat_ncal_all = []
    heavy_ncal_all = []
    light_ncal_all = []
    conformal_5fold_full = []
    emp_vs_conf_mamrd25 = []
    half_cal_stats = defaultdict(list)

    for det, dv_arm, pat, seeds, arm_idx in SGCC_ARMS:
        cell_metrics = []
        cell_gap = []
        for s in seeds:
            for f in ALL_FOLDS:
                yt, st = sgcc_cells[(pat, s, f)]
                grp = fold_info[f]["grp"]
                res_rows = []
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(arm_idx, s, f, rep))
                    if det != "MAMRD (matched)":
                        half_cal_stats["cal_n"].append(len(ci)); half_cal_stats["cal_pos"].append(int(yt[ci].sum()))
                        half_cal_stats["eval_n"].append(len(ti)); half_cal_stats["eval_pos"].append(int(yt[ti].sum()))
                        for gn in ("Heavy", "Light", "Flat"):
                            half_cal_stats[f"cal_{gn}_pos"].append(int(((grp[ci] == gn) & (yt[ci] == 1)).sum()))
                            half_cal_stats[f"eval_{gn}_pos"].append(int(((grp[ti] == gn) & (yt[ti] == 1)).sum()))
                        heavy_ncal_all.append(int(((grp[ci] == "Heavy") & (yt[ci] == 1)).sum()))
                        light_ncal_all.append(int(((grp[ci] == "Light") & (yt[ci] == 1)).sum()))
                        flat_ncal_all.append(int(((grp[ci] == "Flat") & (yt[ci] == 1)).sum()))

                    thr_c, triv = {}, {}
                    for gn in ("Heavy", "Light", "Flat"):
                        m_c = grp[ci] == gn
                        thr_c[gn], triv[gn] = cc_recall_threshold(st[ci][m_c], yt[ci][m_c], 0.20)
                    f_cond = st[ti] >= np.array([thr_c[g] for g in grp[ti]])
                    f_sing = st[ti] >= min(thr_c.values())

                    row = {"cond_insp": float(f_cond.mean()), "single_insp": float(f_sing.mean()),
                           "red": 1.0 - float(f_cond.mean()) / float(f_sing.mean()),
                           "flat_trivial": bool(triv["Flat"])}
                    for gn in ("Heavy", "Light", "Flat"):
                        m_e = grp[ti] == gn
                        pos_e = m_e & (yt[ti] == 1)
                        fl_e = m_e & f_cond
                        row[f"{gn}_rec"] = float(f_cond[pos_e].mean()) if pos_e.any() else np.nan
                        row[f"{gn}_hit"] = float(yt[ti][fl_e].mean()) if fl_e.any() else np.nan
                        row[f"{gn}_lift"] = row[f"{gn}_hit"] / sgcc_grp_rates[gn] if np.isfinite(row[f"{gn}_hit"]) else np.nan

                    if det == "MAMRD (5-seed)":
                        # Also evaluate unadjusted empirical (1-0.20) quantile on exact same resplits
                        thr_emp = {}
                        for gn in ("Heavy", "Light", "Flat"):
                            pos_scores = st[ci][(grp[ci] == gn) & (yt[ci] == 1)]
                            thr_emp[gn] = float(np.quantile(pos_scores, 0.20)) if len(pos_scores) > 0 else 0.0
                        f_emp = st[ti] >= np.array([thr_emp[g] for g in grp[ti]])
                        er = {"emp_insp": float(f_emp.mean()), "conf_insp": float(f_cond.mean()),
                              "emp_rec_all": float(f_emp[yt[ti] == 1].mean()),
                              "conf_rec_all": float(f_cond[yt[ti] == 1].mean())}
                        for gn in ("Heavy", "Light", "Flat"):
                            pos_e = (grp[ti] == gn) & (yt[ti] == 1)
                            er[f"emp_rec_{gn}"] = float(f_emp[pos_e].mean()) if pos_e.any() else np.nan
                            er[f"conf_rec_{gn}"] = float(f_cond[pos_e].mean()) if pos_e.any() else np.nan
                        emp_vs_conf_mamrd25.append(er)

                    res_rows.append(row)
                rdf = pd.DataFrame(res_rows)
                cmean = {k: float(rdf[k].mean()) for k in rdf.columns}
                cmean["seed"] = s; cmean["fold"] = f
                cell_metrics.append(cmean)
                cell_gap.append(100.0 * (cmean["Heavy_hit"] - cmean["Flat_hit"]))

        cdf = pd.DataFrame(cell_metrics)
        r_lo, r_hi = bca_ci(100.0 * cdf["red"].values)
        g_lo, g_hi = bca_ci(np.array(cell_gap))
        fold_reds = 100.0 * cdf.groupby("fold")["red"].mean().values
        conformal_5fold_full.append({
            "detector": det,
            "n_cells": len(cdf),
            "n_splits": len(cdf) * NREP,
            "Heavy_hit_pct": 100.0 * float(cdf["Heavy_hit"].mean()),
            "Light_hit_pct": 100.0 * float(cdf["Light_hit"].mean()),
            "Flat_hit_pct": 100.0 * float(cdf["Flat_hit"].mean()),
            "Heavy_lift": float(cdf["Heavy_lift"].mean()),
            "Light_lift": float(cdf["Light_lift"].mean()),
            "Flat_lift": float(cdf["Flat_lift"].mean()),
            "Heavy_minus_Flat_pp": float(np.mean(cell_gap)),
            "Heavy_minus_Flat_ci_lo": g_lo,
            "Heavy_minus_Flat_ci_hi": g_hi,
            "cond_insp_pct": 100.0 * float(cdf["cond_insp"].mean()),
            "single_insp_pct": 100.0 * float(cdf["single_insp"].mean()),
            "workload_red_pct": 100.0 * float(cdf["red"].mean()),
            "workload_red_ci_lo": r_lo,
            "workload_red_ci_hi": r_hi,
            "fold_red_min_pct": float(fold_reds.min()),
            "fold_red_max_pct": float(fold_reds.max()),
            "flat_trivial_pct": 100.0 * float(cdf["flat_trivial"].mean()),
        })

    # Per-fold ratios (n_cal / 5 folds)
    ledger["conformal_5fold_full"] = conformal_5fold_full
    edf = pd.DataFrame(emp_vs_conf_mamrd25)
    ledger["mamrd_5fold_emp_vs_conf"] = {
        "n_splits": len(edf),
        "emp_insp_pct": 100.0 * float(edf["emp_insp"].mean()),
        "emp_insp_p025": 100.0 * float(np.quantile(edf["emp_insp"], 0.025)),
        "emp_insp_p975": 100.0 * float(np.quantile(edf["emp_insp"], 0.975)),
        "conf_insp_pct": 100.0 * float(edf["conf_insp"].mean()),
        "conf_insp_p025": 100.0 * float(np.quantile(edf["conf_insp"], 0.025)),
        "conf_insp_p975": 100.0 * float(np.quantile(edf["conf_insp"], 0.975)),
        "emp_rec_Heavy_pct": 100.0 * float(edf["emp_rec_Heavy"].mean()),
        "emp_rec_Light_pct": 100.0 * float(edf["emp_rec_Light"].mean()),
        "emp_rec_Flat_pct": 100.0 * float(edf["emp_rec_Flat"].mean()),
        "emp_rec_Overall_pct": 100.0 * float(edf["emp_rec_all"].mean()),
        "emp_Flat_viol_pct": 100.0 * float((edf["emp_rec_Flat"] < 0.80).mean()),
        "conf_rec_Heavy_pct": 100.0 * float(edf["conf_rec_Heavy"].mean()),
        "conf_rec_Light_pct": 100.0 * float(edf["conf_rec_Light"].mean()),
        "conf_rec_Flat_pct": 100.0 * float(edf["conf_rec_Flat"].mean()),
        "conf_rec_Overall_pct": 100.0 * float(edf["conf_rec_all"].mean()),
        "conf_Flat_viol_pct": 100.0 * float((edf["conf_rec_Flat"] < 0.80).mean()),
    }

    # Finite eligibility across all 5-fold resplits
    fn_arr = np.array(flat_ncal_all)
    hn_arr = np.array(heavy_ncal_all)
    ln_arr = np.array(light_ncal_all)
    ledger["conformal_5fold_eligibility"] = {
        "total_resplits_all_detectors": len(fn_arr),
        "n_heavy_cal_mean": float(hn_arr.mean()), "n_heavy_cal_min": int(hn_arr.min()), "n_heavy_cal_max": int(hn_arr.max()),
        "n_light_cal_mean": float(ln_arr.mean()), "n_light_cal_min": int(ln_arr.min()), "n_light_cal_max": int(ln_arr.max()),
        "n_flat_cal_mean": float(fn_arr.mean()), "n_flat_cal_min": int(fn_arr.min()), "n_flat_cal_max": int(fn_arr.max()),
        "flat_finite_80_pct": 100.0 * float((fn_arr >= n_min(0.20)).mean()),
        "flat_finite_90_pct": 100.0 * float((fn_arr >= n_min(0.10)).mean()),
        "flat_finite_95_pct": 100.0 * float((fn_arr >= n_min(0.05)).mean()),
    }
    # MAMRD 500 splits per-target lift curve for Fig S4a (fig6_trust_guarantee)
    mamrd_lifts_by_target = []
    for g_tgt in (0.80, 0.90, 0.95):
        alpha = round(1.0 - g_tgt, 2)
        h_hits, l_hits, f_hits = [], [], []
        for s in range(5):
            for f in ALL_FOLDS:
                yt, st = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)]
                grp = fold_info[f]["grp"]
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(0, s, f, rep))
                    thr_c = {gn: cc_recall_threshold(st[ci][grp[ci] == gn], yt[ci][grp[ci] == gn], alpha)[0]
                             for gn in ("Heavy", "Light", "Flat")}
                    f_cond = st[ti] >= np.array([thr_c[g] for g in grp[ti]])
                    for arr_h, gn in ((h_hits, "Heavy"), (l_hits, "Light"), (f_hits, "Flat")):
                        fl = (grp[ti] == gn) & f_cond
                        if fl.any():
                            arr_h.append(float(yt[ti][fl].mean()))
        mamrd_lifts_by_target.append({
            "guarantee": g_tgt,
            "lift_heavy": float(np.mean(h_hits)) / sgcc_grp_rates["Heavy"],
            "lift_light": float(np.mean(l_hits)) / sgcc_grp_rates["Light"],
            "lift_flat": float(np.mean(f_hits)) / sgcc_grp_rates["Flat"],
        })
    ledger["mamrd_5fold_lifts_by_target"] = mamrd_lifts_by_target

    # Partition ledger means across all 5 SGCC folds + Window-disjoint ECL + Client-disjoint 5-fold ECL
    ledger["table2_sgcc_5fold"] = {
        k: float(np.mean(v)) for k, v in t2_acc.items()
    }
    ledger["table2_sgcc_5fold"].update({
        f"conf_{k}": float(np.mean(v)) for k, v in half_cal_stats.items()
    })
    df_ecl_c3_part = pd.read_csv(R3_RES / "C3_ecl_client_disjoint" / "c3_ecl_client_disjoint_partitions.csv")
    ledger["table2_ecl_client_disjoint_5fold"] = {
        "train_win": float(df_ecl_c3_part["train_win"].mean()),
        "train_pos": float(df_ecl_c3_part["train_pos"].mean()),
        "train_clients": float(df_ecl_c3_part["train_clients"].mean()),
        "es_win": float(df_ecl_c3_part["es_win"].mean()),
        "es_pos": float(df_ecl_c3_part["es_pos"].mean()),
        "es_clients": float(df_ecl_c3_part["es_clients"].mean()),
        "te_win": float(df_ecl_c3_part["te_win"].mean()),
        "te_pos": float(df_ecl_c3_part["te_pos"].mean()),
        "te_clients": float(df_ecl_c3_part["te_clients"].mean()),
    }

    # ------------------------------------------------------------------
    # 3. 5-fold Prospective & Retrospective Table 4 (SGCC 5-fold + ECL Client-Disjoint 5-fold)
    # ------------------------------------------------------------------
    print("[3/7] Computing 5-fold Table 4 prospective & retrospective operating points...", flush=True)
    t4_5fold = []
    for det, dv_arm, pat, seeds, arm_idx in SGCC_ARMS:
        cells = [(s, f, *sgcc_cells[(pat, s, f)]) for s in seeds for f in ALL_FOLDS]
        row = {"dataset": "SGCC (5-fold)", "detector": det, "n_cells": len(cells)}
        for b in (0.01, 0.05):
            r_list, i_list, pr_list, pf_list, pi_list = [], [], [], [], []
            for s, f, yt, sc in cells:
                r_val = float(recall_at_fpr(yt, sc, b))
                i_val = float(np.mean(yt)) * r_val + (1.0 - float(np.mean(yt))) * b
                r_list.append(r_val); i_list.append(100.0 * i_val)
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(arm_idx, s, f, rep))
                    thr = np.quantile(sc[ci][yt[ci] == 0], 1.0 - b)
                    ye, se = yt[ti], sc[ti]
                    pr_list.append(float((se[ye == 1] > thr).mean()))
                    pf_list.append(100.0 * float((se[ye == 0] > thr).mean()))
                    pi_list.append(100.0 * float((se > thr).mean()))
            tag = f"{int(b*100)}pct"
            row[f"retro_rec_{tag}"] = float(np.mean(r_list))
            row[f"retro_insp_{tag}"] = float(np.mean(i_list))
            row[f"prosp_rec_{tag}"] = float(np.mean(pr_list))
            row[f"prosp_fpr_{tag}"] = float(np.mean(pf_list))
            row[f"prosp_insp_{tag}"] = float(np.mean(pi_list))
        t4_5fold.append(row)

    for det, c3_arm, arm_idx in ECL_C3_ARMS:
        cells = []
        for s in range(3):
            for f in ALL_FOLDS:
                _, yt, sc = load_csv_cell(C3_DIR / f"{c3_arm}_{s}_{f}.csv")
                cells.append((s, f, yt, sc))
        row = {"dataset": "ECL Client-Disjoint (5-fold)", "detector": det, "n_cells": len(cells)}
        for b in (0.01, 0.05):
            r_list, i_list, pr_list, pf_list, pi_list = [], [], [], [], []
            for s, f, yt, sc in cells:
                r_val = float(recall_at_fpr(yt, sc, b))
                i_val = float(np.mean(yt)) * r_val + (1.0 - float(np.mean(yt))) * b
                r_list.append(r_val); i_list.append(100.0 * i_val)
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(arm_idx, s, f, rep))
                    thr = np.quantile(sc[ci][yt[ci] == 0], 1.0 - b)
                    ye, se = yt[ti], sc[ti]
                    pr_list.append(float((se[ye == 1] > thr).mean()))
                    pf_list.append(100.0 * float((se[ye == 0] > thr).mean()))
                    pi_list.append(100.0 * float((se > thr).mean()))
            tag = f"{int(b*100)}pct"
            row[f"retro_rec_{tag}"] = float(np.mean(r_list))
            row[f"retro_insp_{tag}"] = float(np.mean(i_list))
            row[f"prosp_rec_{tag}"] = float(np.mean(pr_list))
            row[f"prosp_fpr_{tag}"] = float(np.mean(pf_list))
            row[f"prosp_insp_{tag}"] = float(np.mean(pi_list))
        t4_5fold.append(row)
    ledger["table4_5fold_prospective"] = t4_5fold

    # ------------------------------------------------------------------
    # 4. 5-fold C6 Alternative Groupings (all 25 MAMRD cells, 500 splits) & 5-fold C4
    # ------------------------------------------------------------------
    print("[4/7] Computing 5-fold C6 (4 groupings) & 5-fold C4 conformal sensitivity...", flush=True)
    # Build per-fold assignments for all 4 schemes on Folds 0..4
    c6_scheme_rows = []
    for scheme in ("original_block_contrast", "calendar_weekday_weekend", "magnitude_tertiles", "magnitude_x_variability"):
        grp_map_f = {}
        for f in ALL_FOLDS:
            tr, te = fold_info[f]["tr"], fold_info[f]["te"]
            if scheme == "original_block_contrast":
                grp_map_f[f] = collapse_grp(assign(g_all[te], m_all[te], fit_router(g_all[tr], m_all[tr])))
            elif scheme == "calendar_weekday_weekend":
                grp_map_f[f] = collapse_grp(assign(g_cal_all[te], m_all[te], fit_router(g_cal_all[tr], m_all[tr])))
            elif scheme == "magnitude_tertiles":
                q1, q2 = np.quantile(m_all[tr], [1/3, 2/3])
                g_te = np.where(m_all[te] > q2, "High_kWh", np.where(m_all[te] > q1, "Mid_kWh", "Low_kWh"))
                grp_map_f[f] = g_te
            else:
                med_m = float(np.median(m_all[tr])); med_s = float(np.median(sd_all[tr]))
                hi_m = m_all[te] > med_m; hi_s = sd_all[te] > med_s
                g_te = np.empty(len(te), dtype=object)
                g_te[hi_m & hi_s] = "HH"; g_te[hi_m & (~hi_s)] = "HL"
                g_te[(~hi_m) & hi_s] = "LH"; g_te[(~hi_m) & (~hi_s)] = "LL"
                grp_map_f[f] = g_te
        gnames = sorted(set(grp_map_f[0].tolist()))
        for g_tgt in (0.80, 0.90, 0.95):
            alpha = round(1.0 - g_tgt, 2); req = n_min(alpha)
            c_red, c_cond, c_sing, c_rec, c_fin, c_min_ncal = [], [], [], [], [], []
            for s in range(5):
                for f in ALL_FOLDS:
                    yt, st = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)]
                    grp = grp_map_f[f]
                    rr, rc, rs, rrec, rfin, rmin = [], [], [], [], [], []
                    for rep in range(NREP):
                        ci, ti = stratified_half(yt, resplit_rng(0, s, f, rep))
                        thr_c = {}; min_n = 10**9; all_fin = True
                        for gn in gnames:
                            m_c = grp[ci] == gn
                            nc = int(yt[ci][m_c].sum())
                            min_n = min(min_n, nc)
                            if nc < req:
                                all_fin = False
                            thr_c[gn], _ = cc_recall_threshold(st[ci][m_c], yt[ci][m_c], alpha)
                        fc = st[ti] >= np.array([thr_c[g] for g in grp[ti]])
                        fs = st[ti] >= min(thr_c.values())
                        rc.append(float(fc.mean())); rs.append(float(fs.mean()))
                        rr.append(1.0 - rc[-1] / rs[-1])
                        rrec.append(float(fc[yt[ti] == 1].mean()))
                        rfin.append(float(all_fin)); rmin.append(min_n)
                    c_red.append(float(np.mean(rr))); c_cond.append(float(np.mean(rc)))
                    c_sing.append(float(np.mean(rs))); c_rec.append(float(np.mean(rrec)))
                    c_fin.append(float(np.mean(rfin))); c_min_ncal.append(int(min(rmin)))
            r_lo, r_hi = bca_ci(100.0 * np.array(c_red))
            c6_scheme_rows.append({
                "scheme": scheme, "guarantee": g_tgt, "n_groups": len(gnames), "n_cells": 25, "n_splits": 500,
                "cond_insp_pct": 100.0 * float(np.mean(c_cond)),
                "single_insp_pct": 100.0 * float(np.mean(c_sing)),
                "workload_red_pct": 100.0 * float(np.mean(c_red)),
                "workload_red_ci_lo": r_lo, "workload_red_ci_hi": r_hi,
                "realized_recall_pct": 100.0 * float(np.mean(c_rec)),
                "all_groups_finite_pct": 100.0 * float(np.mean(c_fin)),
                "min_ncal_across_groups": int(min(c_min_ncal)),
            })
    ledger["c6_alt_groupings_5fold"] = c6_scheme_rows

    # 5-fold C4: Pooled 5-fold Flat sample size + Selective-Inspection IPW + Distribution Shift
    c4_pooled_5fold = []
    for g_tgt in (0.80, 0.90, 0.95):
        alpha = round(1.0 - g_tgt, 2)
        for f_cal in (0.15, 0.25, 0.40, 0.50, 0.65):
            recs, viols, ncs, fins, conds, sings, reds = [], [], [], [], [], [], []
            for s in range(5):
                yt_p = np.concatenate([sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)][0] for f in ALL_FOLDS])
                st_p = np.concatenate([sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)][1] for f in ALL_FOLDS])
                grp_p = np.concatenate([fold_info[f]["grp"] for f in ALL_FOLDS])
                for rep in range(NREP):
                    rng = np.random.default_rng(100000 * int(round(g_tgt * 100)) + 1000 * s + 50 * int(round(f_cal * 100)) + rep)
                    ci, ti = stratified_frac(yt_p, f_cal, rng)
                    thr_c = {}; triv = {}
                    for gn in ("Heavy", "Light", "Flat"):
                        m_c = grp_p[ci] == gn
                        thr_c[gn], triv[gn] = cc_recall_threshold(st_p[ci][m_c], yt_p[ci][m_c], alpha)
                        if gn == "Flat":
                            ncs.append(int(yt_p[ci][m_c].sum())); fins.append(float(not triv[gn]))
                    fc = st_p[ti] >= np.array([thr_c[g] for g in grp_p[ti]])
                    fs = st_p[ti] >= min(thr_c.values())
                    fl_pos = (grp_p[ti] == "Flat") & (yt_p[ti] == 1)
                    fr = float(fc[fl_pos].mean())
                    recs.append(fr); viols.append(float(fr < g_tgt))
                    conds.append(float(fc.mean())); sings.append(float(fs.mean()))
                    reds.append(1.0 - conds[-1] / sings[-1])
            c4_pooled_5fold.append({
                "target": g_tgt, "f_cal": f_cal, "n_evals": len(recs),
                "flat_ncal_mean": float(np.mean(ncs)), "flat_ncal_min": int(np.min(ncs)),
                "flat_finite_pct": 100.0 * float(np.mean(fins)),
                "flat_recall_pct": 100.0 * float(np.mean(recs)),
                "flat_recall_sd_pct": 100.0 * float(np.std(recs, ddof=1)),
                "flat_viol_pct": 100.0 * float(np.mean(viols)),
                "cond_insp_pct": 100.0 * float(np.mean(conds)),
                "workload_red_pct": 100.0 * float(np.mean(reds)),
            })
    ledger["c4_pooled_5fold_flat"] = c4_pooled_5fold

    # 5-fold Selective inspection IPW
    c4_ipw_5fold = []
    for scen, gamma, trunc_top50 in [
        ("uniform_e1", 0.0, False), ("selective_gamma0.5", 0.5, False),
        ("selective_gamma1.0", 1.0, False), ("selective_gamma1.5", 1.5, False),
        ("selective_gamma2.0", 2.0, False), ("truncated_top50", 0.0, True),
    ]:
        for method in ("naive_unweighted", "tibshirani_ipw"):
            recs, insps, h_rec, f_rec, h_viol, f_viol = [], [], [], [], [], []
            for s in range(5):
                for f in ALL_FOLDS:
                    yt, st = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)]
                    grp = fold_info[f]["grp"]
                    for rep in range(NREP):
                        ci, ti = stratified_half(yt, resplit_rng(0, s, f, rep))
                        rng = np.random.default_rng(70000 + 1000 * s + 100 * f + rep + int(round(gamma * 10)))
                        e_cal = np.full(len(ci), 0.5) if trunc_top50 else np.clip(np.power(np.clip(st[ci], 1e-4, 1.0), gamma), 0.05, 1.0)
                        e_ev = np.full(len(ti), 0.5) if trunc_top50 else np.clip(np.power(np.clip(st[ti], 1e-4, 1.0), gamma), 0.05, 1.0)
                        if trunc_top50:
                            med_cal = float(np.median(st[ci]))
                            aud_cal = st[ci] >= med_cal
                            e_cal = np.where(aud_cal, 1.0, 0.05)
                            e_ev = np.where(st[ti] >= med_cal, 1.0, 0.05)
                        else:
                            aud_cal = rng.random(len(ci)) < e_cal
                        fc = np.zeros(len(ti), dtype=bool)
                        for gn in ("Heavy", "Light", "Flat"):
                            m_c_pos = aud_cal & (grp[ci] == gn) & (yt[ci] == 1)
                            m_e = grp[ti] == gn
                            if method == "naive_unweighted":
                                thr_g, _ = cc_recall_threshold(st[ci][m_c_pos], np.ones(int(m_c_pos.sum()), int), 0.20)
                                fc[m_e] = st[ti][m_e] >= thr_g
                            else:
                                w_c = 1.0 / e_cal[m_c_pos]; w_e = 1.0 / e_ev[m_e]
                                fc[m_e] = weighted_conformal_flags(st[ci][m_c_pos], w_c, st[ti][m_e], w_e, 0.20)
                        recs.append(float(fc[yt[ti] == 1].mean()))
                        insps.append(float(fc.mean()))
                        hp = (grp[ti] == "Heavy") & (yt[ti] == 1); fp_m = (grp[ti] == "Flat") & (yt[ti] == 1)
                        hr = float(fc[hp].mean()); fr = float(fc[fp_m].mean())
                        h_rec.append(hr); f_rec.append(fr)
                        h_viol.append(float(hr < 0.80)); f_viol.append(float(fr < 0.80))
            c4_ipw_5fold.append({
                "scenario": scen, "gamma": gamma, "method": method, "n_splits": len(recs),
                "overall_recall_pct": 100.0 * float(np.mean(recs)),
                "overall_insp_pct": 100.0 * float(np.mean(insps)),
                "heavy_recall_pct": 100.0 * float(np.mean(h_rec)), "heavy_viol_pct": 100.0 * float(np.mean(h_viol)),
                "flat_recall_pct": 100.0 * float(np.mean(f_rec)), "flat_viol_pct": 100.0 * float(np.mean(f_viol)),
            })
    ledger["c4_ipw_5fold"] = c4_ipw_5fold

    # 5-fold C4 distribution shifts
    c4_shift_5fold = []
    # Cross-fold f1 != f2 (5 seeds x 20 pairs = 100 evals)
    cf_rec, cf_h, cf_l, cf_f, cf_insp = [], [], [], [], []
    for s in range(5):
        for f1 in ALL_FOLDS:
            y1, s1 = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f1)]
            g1 = fold_info[f1]["grp"]
            thr_c = {gn: cc_recall_threshold(s1[g1 == gn], y1[g1 == gn], 0.20)[0] for gn in ("Heavy", "Light", "Flat")}
            for f2 in ALL_FOLDS:
                if f1 == f2:
                    continue
                y2, s2 = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f2)]
                g2 = fold_info[f2]["grp"]
                fc = s2 >= np.array([thr_c[g] for g in g2])
                cf_rec.append(float(fc[y2 == 1].mean())); cf_insp.append(float(fc.mean()))
                cf_h.append(float(fc[(g2 == "Heavy") & (y2 == 1)].mean()))
                cf_l.append(float(fc[(g2 == "Light") & (y2 == 1)].mean()))
                cf_f.append(float(fc[(g2 == "Flat") & (y2 == 1)].mean()))
    c4_shift_5fold.append({
        "regime": "cross_fold_transfer (f1 != f2)", "n_evals": len(cf_rec),
        "overall_recall_pct": 100.0 * float(np.mean(cf_rec)),
        "heavy_recall_pct": 100.0 * float(np.mean(cf_h)),
        "light_recall_pct": 100.0 * float(np.mean(cf_l)),
        "flat_recall_pct": 100.0 * float(np.mean(cf_f)),
        "overall_insp_pct": 100.0 * float(np.mean(cf_insp)),
    })
    for regime, low_to_high in [("low_miss_to_high_miss", True), ("high_miss_to_low_miss", False)]:
        recs, hs, ls, fs, insps = [], [], [], [], []
        for s in range(5):
            for f in ALL_FOLDS:
                yt, st = sgcc_cells[("a34_cell_{s}_{f}.csv", s, f)]
                grp = fold_info[f]["grp"]
                mr = miss_rate_all[fold_info[f]["te"]]
                med = float(np.median(mr))
                ci_m = mr <= med if low_to_high else mr > med
                ti_m = ~ci_m
                thr_c = {gn: cc_recall_threshold(st[ci_m & (grp == gn)], yt[ci_m & (grp == gn)], 0.20)[0]
                         for gn in ("Heavy", "Light", "Flat")}
                fc = st[ti_m] >= np.array([thr_c[g] for g in grp[ti_m]])
                recs.append(float(fc[yt[ti_m] == 1].mean())); insps.append(float(fc.mean()))
                hs.append(float(fc[(grp[ti_m] == "Heavy") & (yt[ti_m] == 1)].mean()))
                ls.append(float(fc[(grp[ti_m] == "Light") & (yt[ti_m] == 1)].mean()))
                fs.append(float(fc[(grp[ti_m] == "Flat") & (yt[ti_m] == 1)].mean()))
        c4_shift_5fold.append({
            "regime": regime, "n_evals": len(recs),
            "overall_recall_pct": 100.0 * float(np.mean(recs)),
            "heavy_recall_pct": 100.0 * float(np.mean(hs)),
            "light_recall_pct": 100.0 * float(np.mean(ls)),
            "flat_recall_pct": 100.0 * float(np.mean(fs)),
            "overall_insp_pct": 100.0 * float(np.mean(insps)),
        })
    ledger["c4_shift_5fold"] = c4_shift_5fold

    # ------------------------------------------------------------------
    # 5. 5-fold C8 Constrained Positive Calibration (Verify all 115 fits)
    # ------------------------------------------------------------------
    print("[5/7] Verifying 5-fold C8 constrained positive calibration across 115 fits...", flush=True)
    c8_5fold_rows = []
    rel_raw_all, rel_aff_all, rel_y_all = [], [], []
    for det, _, pat, seeds, _ in SGCC_ARMS:
        er_l, ec_l, br_l, bc_l, w_l, b_l, dap_l, rho_l, dw_lb_l = [], [], [], [], [], [], [], [], []
        for s in seeds:
            for f in ALL_FOLDS:
                yt, sc = sgcc_cells[(pat, s, f)]
                sc = np.clip(sc, 1e-6, 1.0 - 1e-6)
                ci, ti = np.arange(0, len(yt), 2), np.arange(1, len(yt), 2)
                z_cal = stable_logit(sc[ci])
                w_pos, b_pos, ok_bfgs = fit_positive_exp(z_cal, yt[ci])
                w_lb, b_lb, ok_lb = fit_positive_lbfgsb(z_cal, yt[ci])
                assert ok_lb, f"L-BFGS-B did not converge on {det} s={s} f={f}"
                if not ok_bfgs:
                    w_pos, b_pos = w_lb, b_lb
                dw_lb = abs(w_pos - w_lb)
                db_lb = abs(b_pos - b_lb)
                assert max(dw_lb, db_lb) < 1e-4, f"BFGS vs L-BFGS-B diff on {det} s={s} f={f}: {dw_lb}, {db_lb}"
                assert w_pos > 0.0
                p_ev = expit(w_pos * stable_logit(sc[ti]) + b_pos)
                dap = abs(float(average_precision_score(yt[ti], sc[ti])) - float(average_precision_score(yt[ti], p_ev)))
                rho = float(spearmanr(sc[ti], p_ev).statistic)
                assert dap < 1e-12 and np.isclose(rho, 1.0, atol=1e-12)
                er_l.append(ece_binned(sc[ti], yt[ti])); ec_l.append(ece_binned(p_ev, yt[ti]))
                br_l.append(float(np.mean((sc[ti] - yt[ti]) ** 2)))
                bc_l.append(float(np.mean((p_ev - yt[ti]) ** 2)))
                w_l.append(w_pos); b_l.append(b_pos); dap_l.append(dap); rho_l.append(rho); dw_lb_l.append(dw_lb)
                if det == "MAMRD (5-seed)":
                    rel_raw_all.append(sc[ti]); rel_aff_all.append(p_ev); rel_y_all.append(yt[ti])
        c8_5fold_rows.append({
            "detector": det, "n_cells": len(w_l),
            "ece_raw": float(np.mean(er_l)), "ece_cal": float(np.mean(ec_l)),
            "brier_raw": float(np.mean(br_l)), "brier_cal": float(np.mean(bc_l)),
            "w_mean": float(np.mean(w_l)), "w_sd": float(np.std(w_l, ddof=1)), "w_min": float(np.min(w_l)),
            "b_mean": float(np.mean(b_l)), "b_sd": float(np.std(b_l, ddof=1)),
            "max_abs_diff_ap": float(np.max(dap_l)), "min_spearman_rho": float(np.min(rho_l)),
            "max_abs_diff_w_lbfgsb": float(np.max(dw_lb_l)), "all_lbfgsb_converged": True,
        })
    ledger["c8_5fold_constrained_cal"] = c8_5fold_rows

    # ------------------------------------------------------------------
    # 6. 5-fold Supplementary Tables S2 (tab_strata), S3 (tab_matched), S4 (tab_perstratum)
    # ------------------------------------------------------------------
    print("[6/7] Updating 5-fold Supplementary Tables S2, S3, S4...", flush=True)
    FIVE_PH = ["Heavy_Dropper", "Heavy_Riser", "Light_Dropper", "Light_Riser", "Flat"]
    PH_DISP = {"Heavy_Dropper": "Heavy (+)", "Heavy_Riser": "Heavy ($-$)",
               "Light_Dropper": "Light (+)", "Light_Riser": "Light ($-$)", "Flat": "Flat"}

    def ap_safe(y, s):
        return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")

    def matched_ap_draws(pos, neg, target, reps=100, seed=777):
        P = len(pos); keep = min(int(round(P * (1.0 - target) / target)), len(neg))
        rng = np.random.default_rng(seed); vals = []
        for _ in range(reps):
            nn = neg[rng.choice(len(neg), keep, replace=False)] if keep < len(neg) else neg
            vals.append(average_precision_score(np.r_[np.ones(P), np.zeros(len(nn))], np.r_[pos, nn]))
        return float(np.mean(vals))

    # Collect per-stratum metrics across Folds 0-4
    strata_per_det = defaultdict(lambda: defaultdict(list))
    mamrd25_matched = []
    mamrd_9v_cells = {}
    base15_agg = {}
    det_map_s4 = [
        ("WDCNN", "baselines/wdcnn_{s}_{f}.csv", range(3)),
        ("HORLN", "baselines/horln_{s}_{f}.csv", range(3)),
        ("Self-Attn", "baselines/hybridattn_{s}_{f}.csv", range(3)),
        ("GRU-D", "baselines/grud_{s}_{f}.csv", range(3)),
        ("MSDM", "baselines/msdm_net_{s}_{f}.csv", range(3)),
        ("DKGT", "baselines/dkgt_{s}_{f}.csv", range(3)),
        ("MAMRD25", "a34_cell_{s}_{f}.csv", range(5)),
        ("MAMRD15", "a34_cell_{s}_{f}.csv", range(3)),
    ]
    for dname, pat, seeds in det_map_s4:
        for s in seeds:
            for f in ALL_FOLDS:
                yt, sc = sgcc_cells[(pat, s, f)]
                ph = fold_info[f]["ph"]
                if dname == "MAMRD25":
                    # Max per-stratum prevalence across 5 phenotypes in fold f test set
                    targ = max(float(yt[ph == p].mean()) for p in FIVE_PH)
                    m_row = {"seed": s, "fold": f}
                    for p in FIVE_PH:
                        mk = ph == p
                        m_row[p] = matched_ap_draws(sc[mk & (yt == 1)], sc[mk & (yt == 0)], targ)
                    # Pooled Heavy vs Light matched
                    mh = np.isin(ph, ["Heavy_Riser", "Heavy_Dropper"])
                    ml = np.isin(ph, ["Light_Riser", "Light_Dropper"])
                    m_row["Heavy_pooled"] = matched_ap_draws(sc[mh & (yt == 1)], sc[mh & (yt == 0)], targ)
                    m_row["Light_pooled"] = matched_ap_draws(sc[ml & (yt == 1)], sc[ml & (yt == 0)], targ)
                    mamrd25_matched.append(m_row)
                for p in FIVE_PH:
                    mk = ph == p
                    strata_per_det[dname][f"{p}_ap"].append(ap_safe(yt[mk], sc[mk]))
                    strata_per_det[dname][f"{p}_roc"].append(float(roc_auc_score(yt[mk], sc[mk])))
                    # Threshold chosen on cal using F1
                    # Keep F1 consistent with main CSV row
                    r_meta, _, _ = load_csv_cell(cell_path(pat, s, f))
                    strata_per_det[dname][f"{p}_f1"].append(float(r_meta[f"f1_{p}"]))
                strata_per_det[dname]["agg_ap"].append(ap_safe(yt, sc))
                strata_per_det[dname]["agg_roc"].append(float(roc_auc_score(yt, sc)))
                r_meta, _, _ = load_csv_cell(cell_path(pat, s, f))
                strata_per_det[dname]["agg_f1"].append(float(r_meta["f1"]))

    # Write 5-fold tab_strata.tex (S2)
    frag_hdr = "% LaTeX table fragment for \\input{} in manuscript/supplementary; requires \\revthree, \\name, \\prauc, \\rocauc, \\ci macros."
    s2_lines = [
        frag_hdr,
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{\revthree{Per-group detectability of \name{} on SGCC across all five folds ($n=25$ cells, BCa $95\,\%$ CI).} (+)/($-$): positive ($g_i > \tau_f$) or negative ($g_i < -\tau_f$) intra-block column contrast; Flat: $|g_i| \le \tau_f$ (\revthree{Section~3.8} of the main article).}",
        r"\label{tab:strata}",
        r"\begin{tabular}{lccccc}",
        r"\toprule Group & Pop.\ (\%) & Theft (\%) & $\prauc$~(95\,\% CI) & $\rocauc$ & $F_1$ \\",
        r"\midrule",
    ]
    ph_all_5 = np.concatenate([fold_info[f]["ph"] for f in ALL_FOLDS])
    s2_rows = {}
    for p in FIVE_PH:
        mk = ph_all_5 == p
        pop_pct = 100.0 * float(mk.mean())
        th_pct = 100.0 * float(y_all[all_te_idx][mk].mean())
        aps = strata_per_det["MAMRD25"][f"{p}_ap"]
        rocs = strata_per_det["MAMRD25"][f"{p}_roc"]
        f1s = strata_per_det["MAMRD25"][f"{p}_f1"]
        lo, hi = bca_ci(aps)
        s2_rows[p] = {"pop": pop_pct, "theft": th_pct, "ap": float(np.mean(aps)), "lo": lo, "hi": hi,
                      "roc": float(np.mean(rocs)), "f1": float(np.mean(f1s))}
        s2_lines.append(
            f"{PH_DISP[p]} & \\revthree{{{pop_pct:.1f}}} & \\revthree{{{th_pct:.1f}}} & "
            f"\\revthree{{${np.mean(aps):.3f}$~\\ci{{{lo:.3f}}}{{{hi:.3f}}}}} & "
            f"\\revthree{{{np.mean(rocs):.3f}}} & \\revthree{{{np.mean(f1s):.3f}}} \\\\"
        )
    agg_aps = strata_per_det["MAMRD25"]["agg_ap"]
    alo, ahi = bca_ci(agg_aps)
    agg_roc = float(np.mean(strata_per_det["MAMRD25"]["agg_roc"]))
    agg_f1 = float(np.mean(strata_per_det["MAMRD25"]["agg_f1"]))
    s2_lines += [
        r"\midrule",
        f"Aggregate & 100.0 & \\revthree{{{100.0*float(y_all[all_te_idx].mean()):.1f}}} & "
        f"\\revthree{{${np.mean(agg_aps):.3f}$~\\ci{{{alo:.3f}}}{{{ahi:.3f}}}}} & "
        f"\\revthree{{{agg_roc:.3f}}} & \\revthree{{{agg_f1:.3f}}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ]
    (TABLES_DIR / "tab_strata.tex").write_text("\n".join(s2_lines) + "\n", encoding="utf-8")
    ledger["tab_strata_5fold"] = s2_rows

    # Write 5-fold tab_matched.tex (S3)
    mdf = pd.DataFrame(mamrd25_matched)
    chi2_m, p_m = friedmanchisquare(*[mdf[p].values for p in FIVE_PH])
    s3_contrasts = [
        ("Heavy ($-$) $-$ Flat", mdf["Heavy_Riser"].values - mdf["Flat"].values),
        ("Heavy (+) $-$ Flat", mdf["Heavy_Dropper"].values - mdf["Flat"].values),
        ("Flat $-$ Light (+)", mdf["Flat"].values - mdf["Light_Dropper"].values),
        ("Flat $-$ Light ($-$)", mdf["Flat"].values - mdf["Light_Riser"].values),
        ("Heavy ($-$) $-$ Light (+)", mdf["Heavy_Riser"].values - mdf["Light_Dropper"].values),
        ("Heavy (pooled) $-$ Light (pooled)", mdf["Heavy_pooled"].values - mdf["Light_pooled"].values),
    ]
    s3_lines = [
        frag_hdr,
        r"\begin{table*}[!t]",
        r"\centering",
        f"\\caption{{\\revthree{{Prevalence-matched per-group detectability of \\name{{}} on SGCC across all five folds:}} negatives subsampled within each group to a common prevalence, $100$ draws averaged per cell, contrasts paired over \\revthree{{$n=25$ cells (BCa $95\\,\\%$ CI). The coarse consumption hierarchy is resolved (Friedman $\\chi^2_4 = {chi2_m:.2f}$, $p = {_fmt_sci_tex(p_m)}$).}}}}",
        r"\label{tab:matched}",
        r"\begin{tabular}{lcc}",
        r"\toprule Contrast & $\Delta\prauc$~(95\,\% CI) & Resolved \\",
        r"\midrule",
    ]
    s3_json = {"friedman_chi2": float(chi2_m), "friedman_p": float(p_m), "contrasts": []}
    for lab, arr in s3_contrasts:
        mu = float(np.mean(arr)); lo, hi = bca_ci(arr)
        res = "yes" if (lo > 0 or hi < 0) else "no"
        s3_lines.append(f"{lab:33s} & \\revthree{{${mu:+.3f}$~\\ci{{{lo:+.3f}}}{{{hi:+.3f}}}}} & \\revthree{{{res}}} \\\\")
        s3_json["contrasts"].append({"contrast": lab, "mean": mu, "lo": lo, "hi": hi, "resolved": res == "yes"})
    s3_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "tab_matched.tex").write_text("\n".join(s3_lines) + "\n", encoding="utf-8")
    ledger["tab_matched_5fold"] = s3_json

    # Write 5-fold tab_perstratum.tex (S4)
    cols_s4 = ["WDCNN", "HORLN", "Self-Attn", "GRU-D", "MSDM", "DKGT"]
    s4_lines = [
        frag_hdr,
        r"\begin{table*}[!t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
        r"\caption{\revthree{Cross-detector per-group $\prauc$ on SGCC across all five evaluation folds (means; \name{} $n=25$, each baseline $n=15$, paired \name{}$-$best-baseline gap over the $15$ matched cells).} Best values per row in bold.}",
        r"\label{tab:perstratum}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}",
        r"\toprule Group & WDCNN & HORLN & Self-Attn & GRU-D & MSDM & DKGT & \name{} ($n{=}25$) & Gap vs.\ Best ($n{=}15$) \\",
        r"\midrule",
    ]
    for p in FIVE_PH:
        bvals = [float(np.mean(strata_per_det[c][f"{p}_ap"])) for c in cols_s4]
        best_idx = int(np.argmax(bvals))
        best_c = cols_s4[best_idx]
        m25 = float(np.mean(strata_per_det["MAMRD25"][f"{p}_ap"]))
        m15_arr = np.array(strata_per_det["MAMRD15"][f"{p}_ap"])
        diff = m15_arr - np.array(strata_per_det[best_c][f"{p}_ap"])
        d_m = float(diff.mean()); d_lo, d_hi = bca_ci(diff)
        row_max = max(max(bvals), m25)
        cells_str = " & ".join(
            f"\\revthree{{\\textbf{{{v:.3f}}}}}" if abs(v - row_max) < 1e-12 else f"\\revthree{{{v:.3f}}}"
            for v in bvals
        )
        m_str = f"\\revthree{{\\textbf{{{m25:.3f}}}}}" if abs(m25 - row_max) < 1e-12 else f"\\revthree{{{m25:.3f}}}"
        s4_lines.append(f"{PH_DISP[p]} & {cells_str} & {m_str} & \\revthree{{${d_m:+.3f}$~\\ci{{{d_lo:+.3f}}}{{{d_hi:+.3f}}}}} \\\\")
    # Aggregate row
    b_agg = [float(np.mean(strata_per_det[c]["agg_ap"])) for c in cols_s4]
    m25_agg = float(np.mean(strata_per_det["MAMRD25"]["agg_ap"]))
    diff_agg = np.array(strata_per_det["MAMRD15"]["agg_ap"]) - np.array(strata_per_det["HORLN"]["agg_ap"])
    da_m = float(diff_agg.mean()); da_lo, da_hi = bca_ci(diff_agg)
    agg_str = " & ".join(f"\\revthree{{{v:.3f}}}" for v in b_agg)
    m15_grp_note = ", ".join(f"{PH_DISP[p]} {np.mean(strata_per_det['MAMRD15'][f'{p}_ap']):.3f}" for p in FIVE_PH)
    s4_lines += [
        r"\midrule",
        f"Aggregate & {agg_str} & \\revthree{{\\textbf{{{m25_agg:.3f}}}}} & \\revthree{{${da_m:+.3f}$~\\ci{{{da_lo:+.3f}}}{{{da_hi:+.3f}}}}} \\\\",
        r"\bottomrule", r"\end{tabular*}",
        f"\\par\\vspace{{2pt}}\\raggedright\\scriptsize\\textit{{Note:}} \\revthree{{Primary model comparisons and paired gaps use the 15 matched seed--fold cells (Folds 0--4 $\\times$ Seeds 0--2) common to all architectures (MAMRD 15-cell matched means are: {m15_grp_note}, Aggregate {np.mean(strata_per_det['MAMRD15']['agg_ap']):.3f}). The MAMRD column reports all 25 evaluated cells (Seeds 0--4 $\\times$ Folds 0--4).}}",
        r"\end{table*}",
    ]
    (TABLES_DIR / "tab_perstratum.tex").write_text("\n".join(s4_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 7. Regenerate ALL 7 Matplotlib Vector Figures + Previews
    # ------------------------------------------------------------------
    print("[7/7] Regenerating all 5-fold vector PDF figures...", flush=True)

    # --- Figure 2: 5-Fold SGCC PR & ROC Curves ---
    fig2_cfg = [
        ("GRU-D",     "baselines/grud_{s}_{f}.csv",       range(3), {"color": "#2D3436", "lw": 1.3, "ls": (0, (3, 2, 1.5, 2)), "zorder": 1}),
        ("DKGT-Net",  "baselines/dkgt_{s}_{f}.csv",       range(3), {"color": "#8C6D31", "lw": 1.3, "ls": (0, (1, 1)), "zorder": 2}),
        ("WDCNN",     "baselines/wdcnn_{s}_{f}.csv",      range(3), {"color": "#C97A10", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2, 1.5, 2)), "zorder": 3}),
        ("MSDM",      "baselines/msdm_net_{s}_{f}.csv",   range(3), {"color": "#6E388C", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2)), "zorder": 4}),
        ("Self-Attn", "baselines/hybridattn_{s}_{f}.csv", range(3), {"color": "#207D53", "lw": 1.5, "ls": (0, (1.5, 2)), "zorder": 5}),
        ("HORLN",     "baselines/horln_{s}_{f}.csv",      range(3), {"color": "#D33F23", "lw": 1.4, "ls": (0, (5, 2)), "zorder": 6}),
        ("MAMRD",     "a34_cell_{s}_{f}.csv",             range(5), {"color": "#1B5E94", "lw": 2.2, "ls": "-", "zorder": 7}),
    ]
    fig, (axP, axR) = plt.subplots(1, 2, figsize=(7.16, 2.85), layout="constrained")
    grid = np.linspace(0, 1, 200)
    for name, pat, seeds, style in fig2_cfg:
        prc_l, roc_l, ap_l, auc_l = [], [], [], []
        for s in seeds:
            for f in ALL_FOLDS:
                yt, sc = sgcc_cells[(pat, s, f)]
                p_, r_, _ = precision_recall_curve(yt, sc); o = np.argsort(r_)
                prc_l.append(np.interp(grid, r_[o], p_[o])); ap_l.append(average_precision_score(yt, sc))
                fpr, tpr, _ = roc_curve(yt, sc)
                roc_l.append(np.interp(grid, fpr, tpr)); auc_l.append(roc_auc_score(yt, sc))
        axP.plot(grid, np.mean(prc_l, 0), label=f"{name} ({np.mean(ap_l):.3f})", **{k: v for k, v in style.items()})
        axR.plot(grid, np.mean(roc_l, 0), label=f"{name} ({np.mean(auc_l):.3f})", **{k: v for k, v in style.items()})
    axP.axhline(0.0853, ls=":", lw=0.9, color="#636E72", alpha=0.8, label="Base rate (0.085)", zorder=0)
    axP.set_xlim(-0.02, 1.02); axP.set_ylim(-0.02, 1.02); axP.set_xlabel("Recall"); axP.set_ylabel("Precision")
    axP.set_title("(a) Precision–Recall (5-fold AP)"); axP.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DCE1E7", fontsize=6.8)
    axR.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#636E72", alpha=0.7, label="Chance (0.500)", zorder=0)
    axR.set_xlim(-0.02, 1.02); axR.set_ylim(-0.02, 1.02); axR.set_xlabel("False Positive Rate"); axR.set_ylabel("True Positive Rate")
    axR.set_title("(b) ROC (5-fold AUC)"); axR.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DCE1E7", fontsize=6.8)
    save_fig(fig, "fig2_performance_curves")

    # --- Figure 3 / Table 5: Controlled ablation on all five folds (Folds 0-4 x Seeds 0-2, M=15) ---
    # Folds 0-2: Revision 2 cells (results/phase_n_retrained/bu, a32_sgcc.csv, a34_cell);
    # Folds 3-4: Revision 3 extension (results/r3_revision/C9_ablation_folds34/bu, raw_folds34/a34_cell).
    ABL34 = R3_RES / "C9_ablation_folds34"
    def load_ab_dict(pat):
        d = {}
        for s in (0, 1, 2):
            for f in ALL_FOLDS:
                fp = (ORIG_RES if f < 3 else ABL34) / pat.format(s=s, f=f)
                d[(s, f)] = float(next(csv.DictReader(open(fp)))["pr_auc"])
        return d
    s0 = load_ab_dict("bu/a34_bu0_core_{s}_{f}.csv")
    s1 = load_ab_dict("bu/a34_bu1_ccname_{s}_{f}.csv")
    s2 = load_ab_dict("bu/a34_bu2_mask_{s}_{f}.csv")
    s3 = load_ab_dict("bu/a33_generic_{s}_{f}.csv")
    s4 = {(s, f): float(next(csv.DictReader(open((ORIG_RES if f < 3 else RAW34) / f"a34_cell_{s}_{f}.csv")))["pr_auc"])
          for s in range(3) for f in ALL_FOLDS}
    s4_nm = load_ab_dict("bu/a34_no_missingness_{s}_{f}.csv")
    a32_d = {(int(r["seed"]), int(r["fold"])): float(r["pr_auc"]) for r in csv.DictReader(open(ORIG_RES / "a32_sgcc.csv"))}
    a32_d.update({(s, f): float(next(csv.DictReader(open(ABL34 / f"bu/a32_generic_{s}_{f}.csv")))["pr_auc"])
                  for s in range(3) for f in ALL_FOLDS if f >= 3})
    assert all(len(d) == 15 for d in (s0, s1, s2, s3, s4, s4_nm, a32_d)), "Table 5 needs 15 cells per arm"

    # Table 5 Panels A-B (five-fold, paired on identical seed-fold cells)
    keys15 = sorted(s0.keys())
    def _ap(d): return np.array([d[k] for k in keys15])
    def _pair(a, b):
        diff = _ap(a) - _ap(b)
        lo, hi = bca_ci(diff)
        _, pw = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="exact")
        return {"delta": float(diff.mean()), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": float(pw),
                "wins": int(np.sum(diff > 0)), "rel_pct": 100.0 * float(diff.mean() / _ap(b).mean()), "n": len(diff)}
    t5_rows = []
    stage_list = [("S0", s0), ("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)]
    for i, (nm, d) in enumerate(stage_list):
        lo, hi = bca_ci(_ap(d))
        row = {"panel": "A", "row": nm, "ap": float(_ap(d).mean()), "ap_ci_lo": lo, "ap_ci_hi": hi}
        if i:
            st = _pair(d, stage_list[i - 1][1]); cu = _pair(d, s0)
            row.update({f"step_{k}": v for k, v in st.items()}); row.update({f"cum_{k}": v for k, v in cu.items()})
        t5_rows.append(row)
    for nm, a, b in [("S3 - S4", s3, s4), ("S2 - S4", s2, s4), ("S1 - S2", s1, s2),
                     ("A34_no-miss - S4", s4_nm, s4), ("A32 - S4", a32_d, s4)]:
        lo, hi = bca_ci(_ap(a))
        t5_rows.append({"panel": "B", "row": nm, "ap": float(_ap(a).mean()), "ap_ci_lo": lo, "ap_ci_hi": hi,
                        "ap_ref": float(_ap(b).mean()), **{f"pair_{k}": v for k, v in _pair(a, b).items()}})
    ledger["table5_ablation_5fold"] = t5_rows
    df_c2p = pd.read_csv(R3_RES / "C2_missingness_sensitivity" / "c2_paired_contrasts.csv").set_index("contrast")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.05), gridspec_kw={"width_ratios": [1.0, 1.28]})
    stages = [("S0\nCore", s0), ("S1\n+CC-MVI", s1), ("S2\n+Mask", s2), ("S3\n+Diff", s3), ("S4\nMAMRD", s4)]
    common = sorted(s0.keys())
    means_a = [np.mean([st[k] for k in common]) for _, st in stages]
    cis_a = [bca_ci([st[k] for k in common]) for _, st in stages]
    yerr_a = np.array([[m - lo, hi - m] for m, (lo, hi) in zip(means_a, cis_a)]).T
    cols_a = [PAL["grey"], PAL["lblue"], PAL["lblue"], PAL["lblue"], PAL["blue"]]
    axA.bar(np.arange(5), means_a, width=0.52, yerr=yerr_a, capsize=3, color=cols_a, edgecolor=PAL["dark"], lw=0.6)
    for xi, (m, (lo, hi)) in enumerate(zip(means_a, cis_a)):
        axA.text(xi, hi + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4, weight="bold")
    axA.set_xticks(np.arange(5)); axA.set_xticklabels([l for l, _ in stages], fontsize=6.4)
    axA.set_ylim(0.45, 0.64); axA.set_ylabel("PR-AUC (Average Precision)")
    axA.set_title("(a) Sequential addition (Folds 0–4, $M=15$)")

    def _tup(diffs):
        return (float(np.mean(diffs)), *bca_ci(diffs))
    def _c2tup(key):
        r = df_c2p.loc[key]
        return (float(r["delta_ap"]), float(r["ci_lo"]), float(r["ci_hi"]))

    contrasts_fig3 = [
        ("Combined vs. Linear Interp.", _c2tup("3Way: Combined MAMRD - ConsumptionOnly (LinearInterp)")),
        ("Combined vs. Mask-Only Neural", _c2tup("3Way: Combined MAMRD - MaskOnlyNeural")),
        ("Clean Cohort: MAMRD vs. No-Miss", _c2tup("CleanCohort: MAMRD - A34_no_miss")),
        ("Full Model vs. S0 Core ($S_4 - S_0$)", _tup([s4[k] - s0[k] for k in common])),
        ("Without Missingness Pathways", _tup([s4_nm[k] - s4[k] for k in common])),
        ("Without Missingness + Periodicity", _tup([a32_d[k] - s4[k] for k in common])),
        ("Step +CC-MVI ($S_1 - S_0$)", _tup([s1[k] - s0[k] for k in common])),
        ("Step +2-D Mask CNN ($S_2 - S_1$)", _tup([s2[k] - s1[k] for k in common])),
        ("Step +Periodicity ($S_4 - S_3$)", _tup([s4[k] - s3[k] for k in common])),
    ]
    yy = np.arange(len(contrasts_fig3))
    for i, (lab, (m, lo, hi)) in enumerate(contrasts_fig3):
        sig = (lo > 0) or (hi < 0)
        c = PAL["red"] if (sig and m < 0) else (PAL["blue"] if sig else PAL["grey"])
        axB.plot([lo, hi], [i, i], "-", color=c, lw=1.8); axB.plot(m, i, "o", color=c, ms=4.3)
        axB.text(hi + 0.006 if m >= 0 else lo - 0.006, i, f"{m:+.3f}", va="center", ha="left" if m >= 0 else "right", fontsize=6.0, color=PAL["dark"])
    axB.axvline(0.0, ls="--", lw=0.8, color=PAL["dark"], alpha=0.8)
    axB.set_yticks(yy); axB.set_yticklabels([c[0] for c in contrasts_fig3], fontsize=5.9)
    axB.set_xlim(-0.115, 0.310); axB.set_xlabel("Paired $\\Delta$ PR-AUC (95% BCa CI)")
    axB.set_title("(b) Controlled ablation & 3-way neural contrasts")
    fig.tight_layout()
    save_fig(fig, "fig3_ablation_bars")

    # --- Supplementary Figure S3 (fig4_ceiling): five-fold cumulative build-up and branch increments ---
    # Same layout as the Revision 2 make_figures.fig_ceiling, now read from the five-fold Table 5 rows above.
    from matplotlib.patches import Patch
    t5a = {r["row"]: r for r in t5_rows if r["panel"] == "A"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 2.8), gridspec_kw={"width_ratios": [1, 1.15]})
    xs = np.arange(5); means_c = [t5a[k]["ap"] for k in ("S0", "S1", "S2", "S3", "S4")]
    axL.plot(xs, means_c, "-o", color=PAL["blue"], ms=4)
    axL.fill_between(xs, means_c, min(means_c), alpha=0.12, color=PAL["blue"])
    axL.set_xticks(xs); axL.set_xticklabels(["S0\ncore", "S1\n+CC-MVI", "S2\n+mask", "S3\n+diff", "S4\n+period"], fontsize=6.5)
    axL.set_ylabel("PR-AUC"); axL.set_title(f"(a) Cumulative build-up (n={len(keys15)})")
    axL.text(0.03, 0.97, f"+{means_c[-1] - means_c[0]:.3f} total\n(plateau ≈ bottleneck)",
             transform=axL.transAxes, ha="left", va="top", fontsize=6.5, color=PAL["dark"])
    levers = [("CC-MVI front-end (S0→S1)", "S1", "step"), ("2-D mask branch (S1→S2)", "S2", "step"),
              ("Differencing branch (S2→S3)", "S3", "step"), ("Periodicity branch (S3→S4)", "S4", "step"),
              ("Full MAMRD vs core (S0→S4)", "S4", "cum")]
    for i, (lab, v, kind) in enumerate(levers):
        dlt, lo, hi = (t5a[v][f"{kind}_{c}"] for c in ("delta", "ci_lo", "ci_hi"))
        c = PAL["red"] if (lo > 0 or hi < 0) else PAL["grey"]
        axR.plot([lo, hi], [i, i], "-", color=c, lw=1.4); axR.plot(dlt, i, "o", color=c, ms=4)
    axR.axvline(0, ls="--", lw=0.8, color="k", alpha=0.7); axR.axvspan(-0.01, 0.01, color="k", alpha=0.05)
    axR.set_yticks(np.arange(len(levers))); axR.set_yticklabels([l[0] for l in levers], fontsize=6.5); axR.invert_yaxis()
    axR.set_xlabel("Δ PR-AUC (95% BCa CI)"); axR.set_title(f"(b) Branch increments (n={len(keys15)})")
    axR.legend(handles=[Patch(color=PAL["red"], label="interval excludes 0"), Patch(color=PAL["grey"], label="interval includes 0"),
                        Patch(color="k", alpha=0.05, label="±0.01 materiality band")],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=6.3)
    fig.tight_layout()
    save_fig(fig, "fig4_ceiling")

    # --- Figure 4: 5-Fold Constrained Positive Calibration (w = exp(eta) > 0) ---
    raw_all_p = np.concatenate(rel_raw_all); aff_all_p = np.concatenate(rel_aff_all); y_rel = np.concatenate(rel_y_all)
    def bin_curve(p, y, nb=10):
        edges = np.linspace(0.0, 1.0, nb + 1); px, py = [], []
        for i in range(nb):
            msk = (p >= edges[i]) & (p < edges[i + 1] if i < nb - 1 else p <= edges[i + 1])
            if msk.sum() >= 20:
                px.append(float(p[msk].mean())); py.append(float(y[msk].mean()))
        return np.array(px), np.array(py)
    rx, ry = bin_curve(raw_all_p, y_rel); ax_x, ax_y = bin_curve(aff_all_p, y_rel)
    c8_plot_rows = [r for r in c8_5fold_rows if r["detector"] != "MAMRD (matched)"]
    c8_labels = [r["detector"].replace(" (5-seed)", "") for r in c8_plot_rows]
    ece_r = [r["ece_raw"] for r in c8_plot_rows]; ece_c = [r["ece_cal"] for r in c8_plot_rows]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 2.85))
    axA.plot([0, 1], [0, 1], "--", color=PAL["dark"], lw=0.8, alpha=0.7, label="Perfect calibration")
    axA.plot(rx, ry, "-s", color=PAL["red"], lw=1.4, ms=4, label=f"Uncalibrated output (ECE = {ece_r[-1]:.3f})")
    axA.plot(ax_x, ax_y, "-o", color=PAL["blue"], lw=1.5, ms=4, label=f"Constrained $w=\\exp(\\eta)$ (ECE = {ece_c[-1]:.3f})")
    axA.axhline(0.0853, ls=":", lw=0.8, color=PAL["grey"])
    axA.text(0.98, 0.10, "Native prevalence (8.5%)", ha="right", va="bottom", fontsize=6.2, color=PAL["dark"])
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.set_xlabel("Mean predicted theft probability"); axA.set_ylabel("Empirical theft fraction")
    axA.set_title("(a) 5-fold MAMRD reliability curve"); axA.legend(loc="upper left", fontsize=6.2)
    xb = np.arange(len(c8_labels)); wb = 0.36
    axB.bar(xb - wb/2, ece_r, wb, color=PAL["lred"], edgecolor=PAL["red"], lw=0.6, label="Before recalibration (raw)")
    axB.bar(xb + wb/2, ece_c, wb, color=PAL["blue"], edgecolor=PAL["dark"], lw=0.6, label="After constrained recalibration")
    for xi, (er, ea) in enumerate(zip(ece_r, ece_c)):
        axB.text(xi - wb/2, er + 0.01, f"{er:.2f}", ha="center", va="bottom", fontsize=5.8, color=PAL["red"])
        axB.text(xi + wb/2, ea + 0.01, f"{ea:.2f}", ha="center", va="bottom", fontsize=5.8, color=PAL["blue"])
    axB.set_xticks(xb); axB.set_xticklabels(c8_labels, fontsize=6.3, rotation=15); axB.set_ylim(0, 0.48)
    axB.set_ylabel("Expected Calibration Error (ECE)"); axB.set_title("(b) 5-fold ECE before & after recalibration")
    axB.legend(loc="upper right", fontsize=6.2)
    fig.tight_layout()
    save_fig(fig, "fig4_calibration_reliability")

    # --- Figure 5: 5-Fold SGCC Conformal Tradeoffs ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.85))
    groups = ["Heavy", "Light", "Flat", "Overall"]
    ec = ledger["mamrd_5fold_emp_vs_conf"]
    emp_rec = [ec["emp_rec_Heavy_pct"], ec["emp_rec_Light_pct"], ec["emp_rec_Flat_pct"], ec["emp_rec_Overall_pct"]]
    conf_rec = [ec["conf_rec_Heavy_pct"], ec["conf_rec_Light_pct"], ec["conf_rec_Flat_pct"], ec["conf_rec_Overall_pct"]]
    xa = np.arange(4); wa = 0.35
    axA.bar(xa - wa/2, emp_rec, wa, color=PAL["lred"], edgecolor=PAL["red"], lw=0.6, label=r"Empirical ($\hat{q}_{\alpha}^{\mathrm{emp}}$)")
    axA.bar(xa + wa/2, conf_rec, wa, color=PAL["blue"], edgecolor=PAL["dark"], lw=0.6, label=r"Mondrian ($R_{(k_\phi)}$)")
    axA.axhline(80.0, ls="--", lw=1.0, color=PAL["dark"], zorder=1, label="80% Target")
    for xi, (er, cr) in enumerate(zip(emp_rec, conf_rec)):
        axA.text(xi - wa/2, er - 1.2, f"{er:.1f}%", ha="center", va="top", fontsize=5.4, color=PAL["red"], weight="bold")
        axA.text(xi + wa/2, cr + 1.0, f"{cr:.1f}%", ha="center", va="bottom", fontsize=5.8, color=PAL["blue"], weight="bold")
    axA.set_xticks(xa); axA.set_xticklabels(groups, fontsize=7.0); axA.set_ylim(58.0, 96.0)
    axA.set_ylabel("Mean realized theft recall (%)"); axA.set_title("(a) 5-fold realized recall vs. 80% target")
    axA.legend(loc="upper center", ncol=3, fontsize=5.8, framealpha=0.9, edgecolor="#ccc")

    c5_order = [r for r in conformal_5fold_full if r["detector"] != "MAMRD (matched)"]
    c5_labels = [r["detector"].replace(" (5-seed)", "") for r in c5_order]
    single_b = [r["single_insp_pct"] for r in c5_order]
    group_b = [r["cond_insp_pct"] for r in c5_order]
    red_b = [r["workload_red_pct"] for r in c5_order]
    xb = np.arange(len(c5_order)); wb = 0.36
    axB.bar(xb - wb/2, single_b, wb, color=PAL["lgrey"], edgecolor=PAL["dark"], lw=0.6, label=r"Single-threshold ($\min_\phi \hat{q}_\alpha^\phi$)")
    axB.bar(xb + wb/2, group_b, wb, color=PAL["green"], edgecolor=PAL["dark"], lw=0.6, label="Group-conditional Mondrian")
    for xi, (sb, gb, rp) in enumerate(zip(single_b, group_b, red_b)):
        top = sb + 1.8
        axB.plot([xi - wb/2, xi - wb/2, xi + wb/2, xi + wb/2], [sb + 0.5, top, top, gb + 0.5], color=PAL["dgreen"], lw=0.6)
        axB.text(xi, top + 0.6, f"−{rp:.1f}%", ha="center", va="bottom", fontsize=4.8, color=PAL["dgreen"], weight="bold",
                 path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
    axB.set_xticks(xb); axB.set_xticklabels(c5_labels, fontsize=6.3, rotation=15); axB.set_xlim(-0.6, len(c5_order) - 0.4)
    axB.set_ylim(0, 118.0); axB.set_ylabel("Inspected consumer fraction (%)"); axB.set_title(r"(b) 5-fold workload reduction at $\alpha = 0.20$")
    axB.legend(loc="upper center", ncol=2, fontsize=5.8, framealpha=0.9, edgecolor="#ccc")
    fig.tight_layout()
    save_fig(fig, "fig5_conformal_tradeoffs")

    # --- Fig. S2 (fig1_metric_inflation.pdf): 5-Fold Metric Inflation ---
    s2_det = [
        ("GRU-D", "baselines/grud_{s}_{f}.csv", range(3)),
        ("WDCNN", "baselines/wdcnn_{s}_{f}.csv", range(3)),
        ("Self-Attn", "baselines/hybridattn_{s}_{f}.csv", range(3)),
        ("HORLN", "baselines/horln_{s}_{f}.csv", range(3)),
        ("MSDM", "baselines/msdm_net_{s}_{f}.csv", range(3)),
        ("MAMRD", "a34_cell_{s}_{f}.csv", range(5)),
    ]
    roc_m, m100_m, pr_m = [], [], []
    for _, pat, seeds in s2_det:
        rr, mm, pp = [], [], []
        for s in seeds:
            for f in ALL_FOLDS:
                yt, sc = sgcc_cells[(pat, s, f)]
                rr.append(roc_auc_score(yt, sc)); pp.append(average_precision_score(yt, sc))
                mm.append(topk_tie_neutral(yt, sc, 100)[0])
        roc_m.append(float(np.mean(rr))); m100_m.append(float(np.mean(mm))); pr_m.append(float(np.mean(pp)))
    fig, ax = plt.subplots(figsize=(7.16, 2.85))
    x = np.arange(len(s2_det)); w = 0.25
    ax.bar(x - w, roc_m, w, label="ROC-AUC", color=PAL["lgrey"], edgecolor="k", lw=0.5)
    ax.bar(x, m100_m, w, label="MAP@100", color=PAL["lblue"], edgecolor="k", lw=0.5)
    ax.bar(x + w, pr_m, w, label="Full-prevalence PR-AUC", color=PAL["blue"], edgecolor="k", lw=0.5)
    for xi, (r, m, p) in enumerate(zip(roc_m, m100_m, pr_m)):
        drop = m - p
        ax.annotate("", xy=(xi + w, p + 0.015), xytext=(xi, m + 0.015),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8, connectionstyle="arc3,rad=-0.22"))
        ax.text(xi + w / 2, max(m, p) + 0.065, f"−{drop:.2f}", ha="center", va="bottom", fontsize=6.5, color="#900c3f", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in s2_det], fontsize=7.5); ax.set_ylabel("Score", fontsize=8); ax.set_ylim(0, 1.16)
    ax.axhline(0.0853, ls=":", lw=0.8, color="k", alpha=0.6)
    ax.text(len(s2_det) - 0.5, 0.105, "base rate 8.53%", fontsize=6.5, ha="right", color="k")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7.2, framealpha=0.9, edgecolor="#ccc")
    save_fig(fig, "fig1_metric_inflation", copy_to_root=True)

    # --- Fig. S4 (fig6_trust_guarantee.pdf): 5-Fold SGCC Audit Lift & Hit-Rate Gap ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.75))
    gx = [r["guarantee"] for r in mamrd_lifts_by_target]
    for col, lab, c, mk, ms in [
        ("lift_heavy", f"Heavy ({sgcc_grp_shares['Heavy']['base_rate_pct']:.1f}% base)", PAL["red"], "o", 6.2),
        ("lift_light", f"Light ({sgcc_grp_shares['Light']['base_rate_pct']:.1f}% base)", PAL["blue"], "s", 4.2),
        ("lift_flat", f"Flat ({sgcc_grp_shares['Flat']['base_rate_pct']:.1f}% base)", PAL["grey"], "^", 5.0),
    ]:
        axA.plot(gx, [r[col] for r in mamrd_lifts_by_target], marker=mk, ms=ms, color=c, label=lab)
    axA.axhline(1.0, color=PAL["dark"], lw=0.8, ls=":")
    axA.text(0.88, 1.08, "random audit (1.0×)", fontsize=6, color=PAL["dark"], ha="center")
    axA.set_xticks([0.80, 0.90, 0.95]); axA.invert_xaxis()
    axA.set_xlabel("Target marginal theft recall $1-\\alpha$"); axA.set_ylabel("Audit-efficiency lift over random")
    axA.set_title("(a) 5-fold MAMRD lift by consumption group"); axA.legend(loc="upper left", fontsize=6.5)

    c6_sorted = sorted(c5_order, key=lambda r: r["Heavy_minus_Flat_pp"])
    c6_sorted_labels = [r["detector"].replace(" (5-seed)", "") for r in c6_sorted]
    ys = np.arange(len(c6_sorted)); xmax = 0.0
    for i, r in enumerate(c6_sorted):
        m = r["Heavy_minus_Flat_pp"]; lo = r["Heavy_minus_Flat_ci_lo"]; hi = r["Heavy_minus_Flat_ci_hi"]
        c = PAL["red"] if "MAMRD" in r["detector"] else PAL["dark"]
        axB.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=c, capsize=2.5, markersize=4.5)
        xmax = max(xmax, hi)
        axB.text(hi + 0.6, i, f"{m:+.1f} pp", va="center", ha="left", fontsize=6.2, color=c)
    axB.axvline(0.0, color=PAL["dark"], lw=0.8, ls=":"); axB.set_yticks(ys); axB.set_yticklabels(c6_sorted_labels)
    axB.set_xlabel("Heavy$-$Flat audit hit-rate gap (percentage points, 95% CI)")
    axB.set_title("(b) 5-fold Heavy vs. Flat gap at $\\alpha=0.20$"); axB.set_xlim(-2.0, xmax + 7.5)
    fig.tight_layout()
    save_fig(fig, "fig6_trust_guarantee", copy_to_root=True)

    # --- Fig. S6 (fig_s6_r3_conformal_robustness.pdf): NEW 5-Fold C4 IPW + C6 Groupings ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 2.85))
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
    naive_r = []
    ipw_r = []
    for g in gammas:
        scen = "uniform_e1" if g == 0.0 else f"selective_gamma{g:.1f}"
        naive_r.append(next(r["overall_recall_pct"] for r in c4_ipw_5fold if r["scenario"] == scen and r["method"] == "naive_unweighted"))
        ipw_r.append(next(r["overall_recall_pct"] for r in c4_ipw_5fold if r["scenario"] == scen and r["method"] == "tibshirani_ipw"))
    axA.plot(gammas, naive_r, "-s", color=PAL["red"], lw=1.5, ms=4.5, label="Naive unweighted")
    axA.plot(gammas, ipw_r, "-o", color=PAL["blue"], lw=1.5, ms=4.5, label="Weighted conformal (IPW)")
    axA.axhline(80.0, ls="--", lw=0.9, color=PAL["dark"], label="80% marginal target")
    axA.set_xlabel(r"Selection-bias exponent $\gamma$ ($e(x)=\max(0.05, s^\gamma)$)")
    axA.set_ylabel("Mean realized recall (%)")
    axA.set_ylim(45.0, 100.0); axA.set_title("(a) 5-fold selective inspection & IPW recovery")
    axA.legend(loc="lower left", fontsize=6.3)

    sch_labels = [
        ("original_block_contrast", "Block Contrast\n(Original)"),
        ("calendar_weekday_weekend", "Weekday vs.\nWeekend"),
        ("magnitude_tertiles", "Magnitude\nTertiles"),
        ("magnitude_x_variability", "Magnitude x\nVariability"),
    ]
    c6_80 = {r["scheme"]: r for r in c6_scheme_rows if np.isclose(r["guarantee"], 0.80)}
    x6 = np.arange(len(sch_labels))
    reds6 = [c6_80[k]["workload_red_pct"] for k, _ in sch_labels]
    lo6 = [c6_80[k]["workload_red_ci_lo"] for k, _ in sch_labels]
    hi6 = [c6_80[k]["workload_red_ci_hi"] for k, _ in sch_labels]
    yerr6 = np.array([[m - l, h - m] for m, l, h in zip(reds6, lo6, hi6)]).T
    axB.bar(x6, reds6, width=0.52, yerr=yerr6, capsize=3, color=PAL["green"], edgecolor=PAL["dark"], lw=0.6)
    for xi, (m, _, h) in enumerate(zip(reds6, lo6, hi6)):
        axB.text(xi, h + 1.0, f"{m:.1f}%", ha="center", va="bottom", fontsize=6.4, weight="bold", color=PAL["dgreen"])
    axB.set_xticks(x6); axB.set_xticklabels([l for _, l in sch_labels], fontsize=6.4)
    axB.set_ylim(0, 56.0); axB.set_ylabel("Inspection workload reduction (%)")
    axB.set_title(r"(b) 5-fold grouping schemes at $\alpha=0.20$")
    fig.tight_layout()
    save_fig(fig, "fig_s6_r3_conformal_robustness", copy_to_root=True)

    # Sensitivity suites produced by their own drivers (C2, C5, C7): read the 5-fold summaries
    # into the ledger so every manuscript number has one authoritative source.
    c2_dir, c5_dir, c7_dir = (R3_RES / "C2_missingness_sensitivity", R3_RES / "C5_preprocessing_sensitivity",
                              R3_RES / "C7_baseline_hpo")
    ledger["c2_missingness_5fold"] = pd.read_csv(c2_dir / "c2_paired_contrasts.csv").to_dict("records")
    ledger["c5_preprocessing_5fold"] = {
        "summary": pd.read_csv(c5_dir / "c5_sensitivity_summary.csv").to_dict("records"),
        "contrasts": pd.read_csv(c5_dir / "c5_paired_contrasts.csv").to_dict("records")}
    c7_tuned = pd.read_csv(c7_dir / "c7_tuned_baselines_summary.csv").to_dict("records")
    c7_mamrd = json.loads((c7_dir / "c7_mamrd_tuning_summary.json").read_text())
    ledger["c7_tuning_5fold"] = {
        "des_grid": pd.concat([pd.read_csv(c7_dir / "c7_hpo_validation_grid.csv")[["backbone", "lr", "wd", "best_val_ap"]],
                               pd.read_csv(c7_dir / "c7_mamrd_tuning_grid.csv")[["backbone", "lr", "wd", "best_val_ap"]]]
                              ).to_dict("records"),
        "tuned_baselines": c7_tuned, "mamrd_tuning": c7_mamrd,
        "original_preprocessing": (pd.read_csv(c7_dir / "c7_wdcnn_interp_summary.csv").to_dict("records")
                                   + pd.read_csv(c7_dir / "c7_finardi_native_summary.csv").to_dict("records"))}
    s13_rows = ([{"detector": "MAMRD", "selected_lr": c7_mamrd["selected_lr"], "selected_wd": c7_mamrd["selected_wd"],
                  "shared_schedule_ap": c7_mamrd["default_mamrd_ap"], "tuned_ap": c7_mamrd["tuned_mamrd_ap"],
                  "tuned_ci_lo": c7_mamrd["tuned_mamrd_ci"][0], "tuned_ci_hi": c7_mamrd["tuned_mamrd_ci"][1]}]
                + [{"detector": r["backbone"], "selected_lr": r["lr_selected"], "selected_wd": r["wd_selected"],
                    "shared_schedule_ap": None, "tuned_ap": r["ap_mean"], "tuned_ci_lo": r["ap_ci_lo"],
                    "tuned_ci_hi": r["ap_ci_hi"]} for r in c7_tuned])

    # Export authoritative 5-fold flat CSV tables
    csv_exports = [
        ("table_s13_tuning_5fold.csv", s13_rows),
        ("table4_5fold_prospective.csv", ledger["table4_5fold_prospective"]),
        ("table6_5fold_conformal.csv", ledger["conformal_5fold_full"]),
        ("c4_pooled_5fold_flat.csv", ledger["c4_pooled_5fold_flat"]),
        ("c4_ipw_5fold.csv", ledger["c4_ipw_5fold"]),
        ("c4_shift_5fold.csv", ledger["c4_shift_5fold"]),
        ("c6_alt_groupings_5fold.csv", ledger["c6_alt_groupings_5fold"]),
        ("c8_5fold_constrained_cal.csv", ledger["c8_5fold_constrained_cal"]),
        ("table5_ablation_5fold.csv", ledger["table5_ablation_5fold"]),
    ]
    for csv_name, rows in csv_exports:
        pd.DataFrame(rows).to_csv(TABLES_DIR / csv_name, index=False)
        print(f"  wrote tables/{csv_name}")

    # Save master ledger JSON
    ledger_path = R3_RES / "r3_manuscript_verified_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[DONE] Verified 5-fold ledger saved -> {ledger_path.name}")


if __name__ == "__main__":
    main()
