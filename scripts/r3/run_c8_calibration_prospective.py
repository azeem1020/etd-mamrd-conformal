#!/usr/bin/env python3
"""Comment 8: Constrained Positive-Slope Affine Recalibration & Prospective Table 4.

Note on Folds 0-2 diagnostic vs. authoritative 5-fold manuscript results:
  This standalone script evaluates the Folds 0-2 (range(3)) diagnostic grid (including
  the BFGS vs. box-constrained L-BFGS-B numerical cross-check) into
  results/r3_revision/_3fold_diagnostics/C8_calibration_and_prospective/.
  The authoritative 5-fold Revision 3 calibration (115 fits) and prospective Table 4
  manuscript results are computed by scripts/r3/build_r3_5fold_tables_and_figures.py
  into results/r3_revision/r3_manuscript_verified_ledger.json and results/r3_revision/tables/.

1. Compares unconstrained LogisticRegression(C=1e4) against strictly positive
   slope parameterization w = exp(eta) > 0 (solved via BFGS + analytic grad,
   and cross-checked via L-BFGS-B with bounds [(1e-8, None), (None, None)])
   across all 7 SGCC detectors (78 cell fits across Folds 0-2: 15 MAMRD 5-seed + 9 MAMRD 3-seed + 54 baseline cells).
   Verifies exact rank preservation (Spearman rho = 1.0, max |dAP| < 1e-12),
   positive w_min > 0, and numerical equivalence to unconstrained estimates.
2. Produces a consolidated Prospective vs. Retrospective Table 4 comparison CSV
   joining r2_audit/table4_fpr_budgets.csv and r2_audit/table4_prospective.csv.
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

csv.field_size_limit(10**7)

THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
_REPO_DIR = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(_REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import _ROOT, RESULTS_DIR, scrub_paths

ORIG_REPO = _ROOT
ORIG_RES = RESULTS_DIR
R3_RES = _ROOT / "results" / "r3_revision"
OUT = R3_RES / "_3fold_diagnostics" / "C8_calibration_and_prospective"
OUT.mkdir(parents=True, exist_ok=True)


def stable_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def ece_binned(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
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


def fit_positive_exp(z: np.ndarray, y: np.ndarray, C: float = 1e4) -> tuple[float, float, bool, object]:
    """Minimize sum_i [log(1 + exp(exp(eta)*z_i + b)) - y_i*(exp(eta)*z_i + b)] + exp(2*eta)/(2*C)."""
    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta, b = float(theta[0]), float(theta[1])
        w = np.exp(np.clip(eta, -20.0, 20.0))
        lin = w * z + b
        # softplus(lin) - y * lin == logaddexp(0, lin) - y * lin
        nll = np.sum(np.logaddexp(0.0, lin) - y * lin) + 0.5 * (w ** 2) / C
        prob = expit(lin)
        resid = prob - y
        d_b = float(np.sum(resid))
        d_w = float(np.sum(resid * z)) + w / C
        d_eta = d_w * w
        return float(nll), np.array([d_eta, d_b], dtype=float)

    res = minimize(loss_and_grad, np.array([0.0, 0.0]), method="BFGS", jac=True,
                   options={"gtol": 1e-9, "maxiter": 500})
    w_opt = float(np.exp(res.x[0]))
    b_opt = float(res.x[1])
    return w_opt, b_opt, bool(res.success), loss_and_grad


def fit_positive_lbfgsb(z: np.ndarray, y: np.ndarray, C: float = 1e4) -> tuple[float, float, bool]:
    """Cross-check via box-constrained L-BFGS-B: w >= 1e-8, b unconstrained."""
    def loss_and_grad(wb: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = float(wb[0]), float(wb[1])
        lin = w * z + b
        nll = np.sum(np.logaddexp(0.0, lin) - y * lin) + 0.5 * (w ** 2) / C
        resid = expit(lin) - y
        return float(nll), np.array([float(np.sum(resid * z)) + w / C, float(np.sum(resid))], dtype=float)

    res = minimize(loss_and_grad, np.array([1.0, 0.0]), method="L-BFGS-B", jac=True,
                   bounds=[(1e-8, None), (None, None)], options={"ftol": 1e-13, "gtol": 1e-9})
    return float(res.x[0]), float(res.x[1]), bool(res.success)


def main() -> None:
    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log("=== C8: CONSTRAINED POSITIVE-SLOPE CALIBRATION & PROSPECTIVE TABLE 4 ===")
    arms = [
        ("GRU-D", ORIG_RES / "baselines/grud_{s}_{f}.csv", range(3), range(3)),
        ("DKGT-Net", ORIG_RES / "baselines/dkgt_{s}_{f}.csv", range(3), range(3)),
        ("WDCNN", ORIG_RES / "baselines/wdcnn_{s}_{f}.csv", range(3), range(3)),
        ("Self-Attn", ORIG_RES / "baselines/hybridattn_{s}_{f}.csv", range(3), range(3)),
        ("MSDM", ORIG_RES / "baselines/msdm_net_{s}_{f}.csv", range(3), range(3)),
        ("HORLN", ORIG_RES / "baselines/horln_{s}_{f}.csv", range(3), range(3)),
        ("MAMRD (9-cell)", ORIG_RES / "a34_cell_{s}_{f}.csv", range(3), range(3)),
        ("MAMRD (15-cell)", ORIG_RES / "a34_cell_{s}_{f}.csv", range(5), range(3)),
    ]

    rows = []
    grad_checked = False
    for det_name, pat, seeds, folds in arms:
        for s in seeds:
            for f in folds:
                fp = Path(str(pat).format(s=s, f=f))
                r = next(csv.DictReader(open(fp)))
                y = np.array(json.loads(r["val_labels_json"]), dtype=int)
                sc = np.clip(np.array(json.loads(r["val_scores_json"]), dtype=float), 1e-6, 1.0 - 1e-6)
                ci, ti = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
                z_cal = stable_logit(sc[ci])
                z_ev = stable_logit(sc[ti])
                y_cal, y_ev = y[ci], y[ti]

                # 1. Unconstrained sklearn LogisticRegression(C=1e4)
                lr = LogisticRegression(C=1e4).fit(z_cal.reshape(-1, 1), y_cal)
                w_unc = float(lr.coef_[0, 0])
                b_unc = float(lr.intercept_[0])
                p_unc = lr.predict_proba(z_ev.reshape(-1, 1))[:, 1]

                # 2. Strictly positive exponential parameterization w = exp(eta) > 0
                w_exp, b_exp, ok_exp, fn_loss = fit_positive_exp(z_cal, y_cal, C=1e4)
                if not grad_checked:
                    err = check_grad(lambda t: fn_loss(t)[0], lambda t: fn_loss(t)[1], np.array([0.0, -1.5]))
                    assert err < 1e-4, f"Analytic gradient check failed: {err}"
                    log(f"[check_grad] analytic gradient error = {err:.3e} (< 1e-4 PASS)")
                    grad_checked = True
                p_exp = expit(w_exp * z_ev + b_exp)

                # 3. Cross-check via box-constrained L-BFGS-B
                w_lb, b_lb, ok_lb = fit_positive_lbfgsb(z_cal, y_cal, C=1e4)

                assert w_exp > 0.0 and w_lb > 0.0
                ap_raw = float(average_precision_score(y_ev, sc[ti]))
                ap_exp = float(average_precision_score(y_ev, p_exp))
                rho_sp = float(spearmanr(sc[ti], p_exp).statistic)
                assert abs(ap_raw - ap_exp) < 1e-12, f"AP changed: {ap_raw} vs {ap_exp}"
                assert np.isclose(rho_sp, 1.0, atol=1e-12), f"Spearman rho != 1: {rho_sp}"

                rows.append({
                    "detector": det_name,
                    "seed": s,
                    "fold": f,
                    "w_unconstrained": w_unc,
                    "b_unconstrained": b_unc,
                    "w_exp_pos": w_exp,
                    "b_exp_pos": b_exp,
                    "w_lbfgsb_pos": w_lb,
                    "b_lbfgsb_pos": b_lb,
                    "abs_diff_w_exp_vs_unc": abs(w_exp - w_unc),
                    "abs_diff_b_exp_vs_unc": abs(b_exp - b_unc),
                    "ece_raw": ece_binned(sc[ti], y_ev, 10),
                    "ece_unconstrained": ece_binned(p_unc, y_ev, 10),
                    "ece_constrained": ece_binned(p_exp, y_ev, 10),
                    "brier_raw": float(np.mean((sc[ti] - y_ev) ** 2)),
                    "brier_unconstrained": float(np.mean((p_unc - y_ev) ** 2)),
                    "brier_constrained": float(np.mean((p_exp - y_ev) ** 2)),
                    "ap_raw": ap_raw,
                    "ap_constrained": ap_exp,
                    "abs_diff_ap": abs(ap_raw - ap_exp),
                    "spearman_rho": rho_sp,
                    "ok_exp": ok_exp,
                    "ok_lbfgsb": ok_lb,
                })

    df_cell = pd.DataFrame(rows)
    df_cell.to_csv(OUT / "constrained_calibration_per_cell.csv", index=False)

    # Detector summary
    det_order = [a[0] for a in arms]
    sum_rows = []
    for det_name in det_order:
        sub = df_cell[df_cell["detector"] == det_name]
        sum_rows.append({
            "detector": det_name,
            "n_cells": len(sub),
            "ece_raw": sub["ece_raw"].mean(),
            "ece_unconstrained": sub["ece_unconstrained"].mean(),
            "ece_constrained": sub["ece_constrained"].mean(),
            "ece_rel_change_pct": 100.0 * (sub["ece_constrained"].mean() - sub["ece_raw"].mean()) / sub["ece_raw"].mean(),
            "brier_raw": sub["brier_raw"].mean(),
            "brier_constrained": sub["brier_constrained"].mean(),
            "brier_rel_change_pct": 100.0 * (sub["brier_constrained"].mean() - sub["brier_raw"].mean()) / sub["brier_raw"].mean(),
            "w_mean": sub["w_exp_pos"].mean(),
            "w_sd": sub["w_exp_pos"].std(ddof=1),
            "w_min": sub["w_exp_pos"].min(),
            "b_mean": sub["b_exp_pos"].mean(),
            "b_sd": sub["b_exp_pos"].std(ddof=1),
            "max_abs_diff_w_vs_unc": sub["abs_diff_w_exp_vs_unc"].max(),
            "max_abs_diff_b_vs_unc": sub["abs_diff_b_exp_vs_unc"].max(),
            "max_abs_diff_ap": sub["abs_diff_ap"].max(),
            "min_spearman_rho": sub["spearman_rho"].min(),
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(OUT / "constrained_calibration_summary.csv", index=False)
    log("\n--- Constrained Positive Parameterization (w = exp(eta) > 0) Summary ---")
    log(df_sum[["detector", "n_cells", "ece_raw", "ece_constrained", "brier_raw", "brier_constrained",
                "w_mean", "w_min", "max_abs_diff_w_vs_unc", "max_abs_diff_ap"]].to_string(index=False))

    # Build consolidated Table 4 (Retrospective vs Prospective FPR operating points)
    df_ret = pd.read_csv(ORIG_RES / "r2_audit" / "table4_fpr_budgets.csv")
    df_pro = pd.read_csv(ORIG_RES / "r2_audit" / "table4_prospective.csv")

    pro_rows = []
    seen = set()
    for _, r in df_ret.iterrows():
        key = (r["dataset"], r["detector"])
        if key in seen:
            continue
        seen.add(key)
        ds, det = key
        r1 = df_ret[(df_ret["dataset"] == ds) & (df_ret["detector"] == det) & (df_ret["budget_fpr"] == 0.01)].iloc[0]
        r5 = df_ret[(df_ret["dataset"] == ds) & (df_ret["detector"] == det) & (df_ret["budget_fpr"] == 0.05)].iloc[0]
        p1 = df_pro[(df_pro["dataset"] == ds) & (df_pro["detector"] == det) & (df_pro["budget_fpr"] == 0.01)].iloc[0]
        p5 = df_pro[(df_pro["dataset"] == ds) & (df_pro["detector"] == det) & (df_pro["budget_fpr"] == 0.05)].iloc[0]
        pro_rows.append({
            "dataset": ds,
            "detector": det,
            "n_cells": int(r1["n_cells"]),
            "retro_rec_1pct": float(r1["recall"]),
            "retro_insp_1pct": float(r1["insp_pct"]),
            "prosp_rec_1pct": float(p1["recall"]),
            "prosp_fpr_1pct": float(p1["fpr_pct"]),
            "prosp_insp_1pct": float(p1["insp_pct"]),
            "retro_rec_5pct": float(r5["recall"]),
            "retro_insp_5pct": float(r5["insp_pct"]),
            "prosp_rec_5pct": float(p5["recall"]),
            "prosp_fpr_5pct": float(p5["fpr_pct"]),
            "prosp_insp_5pct": float(p5["insp_pct"]),
            "n_prosp_resplits": int(p1["n_resplits"]),
        })
    df_t4 = pd.DataFrame(pro_rows)
    df_t4.to_csv(OUT / "table4_prospective_vs_retrospective.csv", index=False)
    log("\n--- Table 4 Retrospective vs. Prospective Operating Points ---")
    log(df_t4.to_string(index=False))

    summary_json = {
        "n_cell_fits": int(len(df_cell)),
        "min_w_constrained": float(df_cell["w_exp_pos"].min()),
        "max_abs_diff_w_exp_vs_unconstrained": float(df_cell["abs_diff_w_exp_vs_unc"].max()),
        "max_abs_diff_b_exp_vs_unconstrained": float(df_cell["abs_diff_b_exp_vs_unc"].max()),
        "max_abs_diff_ece": float(np.max(np.abs(df_cell["ece_constrained"] - df_cell["ece_unconstrained"]))),
        "max_abs_diff_ap": float(df_cell["abs_diff_ap"].max()),
        "min_spearman_rho": float(df_cell["spearman_rho"].min()),
        "detector_summary": sum_rows,
        "table4_prospective_vs_retrospective": pro_rows,
    }
    (OUT / "c8_summary.json").write_text(json.dumps(summary_json, indent=2))
    log(f"\n[DONE] Saved C8 artifacts to {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
