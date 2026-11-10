"""Generate the Single Source of Truth (SSOT) Retrained Numerical Ledger.

Ingests all raw CSV records from RESULTS_DIR (default: results/phase_n_retrained/),
computes grand means, 95% BCa confidence intervals, paired ablation contrasts,
conformal audit metrics, and ECE recalibration values, and saves them into:
    retrained_metrics_ledger.json
"""
from __future__ import annotations
import sys
import os
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro_config import RESULTS_DIR, SGCC_PATH, ECL_PATH

RES = RESULTS_DIR
OUT_JSON = RES / "retrained_metrics_ledger.json"


def bca_ci(data: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Compute Bias-Corrected and Accelerated (BCa) bootstrap confidence interval."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 3:
        return float(np.mean(data)), float(np.mean(data))
    rng = np.random.default_rng(seed)
    theta_hat = float(np.mean(data))
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot_means[b] = float(np.mean(rng.choice(data, size=n, replace=True)))

    # Bias correction z0
    prop = float(np.mean(boot_means < theta_hat))
    prop = max(1e-6, min(1.0 - 1e-6, prop))
    from scipy.stats import norm
    z0 = float(norm.ppf(prop))

    # Acceleration factor a (jackknife)
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(np.mean(np.delete(data, i)))
    jmean = float(np.mean(jack))
    num = float(np.sum((jmean - jack) ** 3))
    den = float(6.0 * (np.sum((jmean - jack) ** 2) ** 1.5))
    a = (num / den) if abs(den) > 1e-12 else 0.0

    z_alpha = float(norm.ppf(alpha / 2.0))
    z_1alpha = float(norm.ppf(1.0 - alpha / 2.0))

    def _perc(z):
        p = norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
        return float(np.clip(p * 100.0, 0.0, 100.0))

    p_lo = _perc(z_alpha)
    p_hi = _perc(z_1alpha)
    lo = float(np.percentile(boot_means, p_lo))
    hi = float(np.percentile(boot_means, p_hi))
    return lo, hi


def extract_mamrd_metrics():
    ledger = {}
    rows = []
    for s in range(5):
        for f in range(3):
            fp = RES / f"a34_cell_{s}_{f}.csv"
            if fp.exists():
                r = pd.read_csv(fp).iloc[0]
                rows.append({
                    "seed": s, "fold": f,
                    "ap": float(r["pr_auc"]), "roc": float(r["roc_auc"]), "f1": float(r["f1"]),
                    "p100": float(r["p_at_100"]), "p200": float(r["p_at_200"]),
                    "hd_pr": float(r["pr_Heavy_Dropper"]), "ld_pr": float(r["pr_Light_Dropper"]),
                    "hr_pr": float(r["pr_Heavy_Riser"]), "lr_pr": float(r["pr_Light_Riser"]),
                    "flat_pr": float(r["pr_Flat"]),
                })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return ledger

    # 15-cell metrics
    for m in ["ap", "roc", "f1", "p100", "p200", "hd_pr", "ld_pr", "hr_pr", "lr_pr", "flat_pr"]:
        vals = df[m].values
        lo, hi = bca_ci(vals)
        ledger[f"MAMRD_15CELL_{m.upper()}_MEAN"] = round(float(np.mean(vals)), 4)
        ledger[f"MAMRD_15CELL_{m.upper()}_CILO"] = round(lo, 4)
        ledger[f"MAMRD_15CELL_{m.upper()}_CIHI"] = round(hi, 4)

    # 9-cell matched metrics (seeds 0..2)
    df9 = df[df.seed.isin([0, 1, 2])]
    for m in ["ap", "roc", "f1", "p100", "p200", "hd_pr", "ld_pr", "hr_pr", "lr_pr", "flat_pr"]:
        vals = df9[m].values
        lo, hi = bca_ci(vals)
        ledger[f"MAMRD_9CELL_{m.upper()}_MEAN"] = round(float(np.mean(vals)), 4)
        ledger[f"MAMRD_9CELL_{m.upper()}_CILO"] = round(lo, 4)
        ledger[f"MAMRD_9CELL_{m.upper()}_CIHI"] = round(hi, 4)

    return ledger


def extract_baseline_metrics():
    ledger = {}
    baselines = [
        ("WDCNN", "baselines/wdcnn_{s}_{f}.csv"),
        ("HORLN", "baselines/horln_{s}_{f}.csv"),
        ("HYBRIDATTN", "baselines/hybridattn_{s}_{f}.csv"),
        ("GRUD", "baselines/grud_{s}_{f}.csv"),
        ("MSDM", "baselines/msdm_net_{s}_{f}.csv"),
        ("DKGT", "baselines/dkgt_{s}_{f}.csv"),
    ]
    for bname, pat in baselines:
        rows = []
        for s in range(3):
            for f in range(3):
                fp = RES / pat.format(s=s, f=f)
                if fp.exists():
                    r = pd.read_csv(fp).iloc[0]
                    rows.append({
                        "ap": float(r["pr_auc"]), "roc": float(r["roc_auc"]), "f1": float(r["f1"]),
                        "p100": float(r["p_at_100"]), "p200": float(r["p_at_200"]),
                    })
        df = pd.DataFrame(rows)
        if len(df) == 9:
            for m in ["ap", "roc", "f1", "p100", "p200"]:
                vals = df[m].values
                lo, hi = bca_ci(vals)
                ledger[f"{bname}_{m.upper()}_MEAN"] = round(float(np.mean(vals)), 4)
                ledger[f"{bname}_{m.upper()}_CILO"] = round(lo, 4)
                ledger[f"{bname}_{m.upper()}_CIHI"] = round(hi, 4)
    return ledger


def extract_ablation_metrics():
    ledger = {}
    stages = [
        ("S0", "bu/a34_bu0_core_{s}_{f}.csv"),
        ("S1", "bu/a34_bu1_ccname_{s}_{f}.csv"),
        ("S2", "bu/a34_bu2_mask_{s}_{f}.csv"),
        ("S3", "bu/a33_generic_{s}_{f}.csv"),
        ("S4", "a34_cell_{s}_{f}.csv"),
        ("A34_NOMISS", "bu/a34_no_missingness_{s}_{f}.csv"),
    ]
    data = {}
    for code, pat in stages:
        rows = []
        for s in range(3):
            for f in range(3):
                fp = RES / pat.format(s=s, f=f)
                if fp.exists():
                    r = pd.read_csv(fp).iloc[0]
                    rows.append({"seed": s, "fold": f, "ap": float(r["pr_auc"]), "roc": float(r["roc_auc"]), "f1": float(r["f1"])})
        if len(rows) == 9:
            df = pd.DataFrame(rows).set_index(["seed", "fold"])
            data[code] = df
            vals = df.ap.values
            lo, hi = bca_ci(vals)
            ledger[f"ABLATION_{code}_AP_MEAN"] = round(float(np.mean(vals)), 4)
            ledger[f"ABLATION_{code}_AP_CILO"] = round(lo, 4)
            ledger[f"ABLATION_{code}_AP_CIHI"] = round(hi, 4)

    # A32
    a32_fp = RES / "a32_sgcc.csv"
    if a32_fp.exists():
        df32 = pd.read_csv(a32_fp).set_index(["seed", "fold"])
        data["A32"] = df32
        vals = df32["pr_auc"].values
        lo, hi = bca_ci(vals)
        ledger["ABLATION_A32_AP_MEAN"] = round(float(np.mean(vals)), 4)
        ledger["ABLATION_A32_AP_CILO"] = round(lo, 4)
        ledger["ABLATION_A32_AP_CIHI"] = round(hi, 4)

    # Paired Contrasts
    if "S4" in data and "S0" in data:
        diff = (data["S4"].ap - data["S0"].ap).values
        lo, hi = bca_ci(diff)
        ledger["CONTRAST_S4_MINUS_S0_DIFF"] = round(float(np.mean(diff)), 4)
        ledger["CONTRAST_S4_MINUS_S0_CILO"] = round(lo, 4)
        ledger["CONTRAST_S4_MINUS_S0_CIHI"] = round(hi, 4)

    if "S4" in data and "A34_NOMISS" in data:
        diff = (data["S4"].ap - data["A34_NOMISS"].ap).values
        lo, hi = bca_ci(diff)
        ledger["CONTRAST_S4_MINUS_NOMISS_DIFF"] = round(float(np.mean(diff)), 4)
        ledger["CONTRAST_S4_MINUS_NOMISS_CILO"] = round(lo, 4)
        ledger["CONTRAST_S4_MINUS_NOMISS_CIHI"] = round(hi, 4)

    return ledger


def extract_ecl_metrics():
    ledger = {}
    ecl_fp = RES / "ecl_trust" / "ecl_trust_percell.csv"
    if ecl_fp.exists():
        df = pd.read_csv(ecl_fp)
        for arm in df.arm.unique():
            sub = df[df.arm == arm]
            vals = sub["ap"].values
            lo, hi = bca_ci(vals)
            prefix = "MAMRD" if arm == "a34_generic" else arm.upper()
            ledger[f"ECL_{prefix}_AP_MEAN"] = round(float(np.mean(vals)), 4)
            ledger[f"ECL_{prefix}_AP_CILO"] = round(lo, 4)
            ledger[f"ECL_{prefix}_AP_CIHI"] = round(hi, 4)
    return ledger


def extract_calibration_metrics():
    ledger = {}
    cal_fp = RES / "trust" / "calibration_sgcc.csv"
    if cal_fp.exists():
        df = pd.read_csv(cal_fp)
        for arm, grp in df.groupby("arm"):
            det = str(arm).upper().replace("-", "_")
            ledger[f"ECE_RAW_{det}"] = round(float(grp["ece_raw"].mean()), 4)
            ledger[f"ECE_AFF_{det}"] = round(float(grp["ece_affine"].mean()), 4)
            ledger[f"ECE_CAL_{det}"] = round(float(grp["ece_cal"].mean()), 4)
            ledger[f"BRIER_RAW_{det}"] = round(float(grp["brier_raw"].mean()), 4)
            ledger[f"BRIER_AFF_{det}"] = round(float(grp["brier_affine"].mean()), 4)
            ledger[f"BRIER_CAL_{det}"] = round(float(grp["brier_cal"].mean()), 4)
    return ledger


def main():
    ledger = {}
    ledger.update(extract_mamrd_metrics())
    ledger.update(extract_baseline_metrics())
    ledger.update(extract_ablation_metrics())
    ledger.update(extract_ecl_metrics())
    ledger.update(extract_calibration_metrics())

    with open(OUT_JSON, "w") as f:
        json.dump(ledger, f, indent=2)

    print(f"[done] Compiled {len(ledger)} SSOT parameters into {OUT_JSON}")
    for k, v in sorted(ledger.items())[:20]:
        print(f"  {k:36s} = {v}")
    if len(ledger) > 20:
        print(f"  ... and {len(ledger)-20} more entries.")


if __name__ == "__main__":
    main()

