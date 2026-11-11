"""Leakage-free behavioral phenotype router.

Replicates the Paper-4 (ADA) phenotype-assignment algorithm exactly, but applied
to ALL customers (not just theft) so it can (a) stratify detection performance and
(b) drive a label-free MoE router.

Rule (consumption-only, label-free):
    g_i = mean(observed, block columns 0-4) - mean(observed, block columns 5-6)  # signed gap
    m_i = mean(observed consumption)                            # magnitude
    if |g_i| <= tau:           phenotype = Flat
    elif g_i > 0:  dir=Dropper (columns 0-4 heavier); Heavy if m_i > median{m: droppers} else Light
    else:          dir=Riser   (columns 5-6 heavier); Heavy if m_i > median{m: risers}   else Light

Column convention: on the chronologically ordered SGCC grid every 7-day block starts on a
Wednesday (2014-01-01), so columns 0-4 are Wednesday-Sunday and columns 5-6 are
Monday-Tuesday; the gap is therefore not a weekday/weekend contrast. Gaps/magnitudes are computed on RAW
kWh (pre-normalisation), NaN-aware, matching Paper-4's kWh semantics.

LEAKAGE FIREWALL: tau and the within-direction medians are FIT ON TRAIN CUSTOMERS
ONLY and applied to test. The gap/magnitude features themselves are per-customer
(each customer's own series) and label-free. Labels never enter assignment.
"""
from __future__ import annotations
import warnings
import numpy as np

PHENOTYPES = ("Heavy_Dropper", "Light_Dropper", "Heavy_Riser", "Light_Riser", "Flat")
PH_INDEX = {p: i for i, p in enumerate(PHENOTYPES)}


def gap_magnitude(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed switching gap g and magnitude m per customer (NaN-aware).

    X_raw : (N, W, 7) raw kWh grid, NaN at missing. Returns (g, m), each (N,).
    """
    n = X_raw.shape[0]
    wk = X_raw[:, :, 0:5].reshape(n, -1)   # columns 0-4 of each 7-day block (Wed-Sun on the chronological SGCC grid)
    we = X_raw[:, :, 5:7].reshape(n, -1)   # columns 5-6 (Mon-Tue; 2014-01-01 is a Wednesday)
    allc = X_raw.reshape(n, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid = np.nanmean(wk, axis=1)
        wke = np.nanmean(we, axis=1)
        mag = np.nanmean(allc, axis=1)
    mid = np.where(np.isfinite(mid), mid, 0.0)
    wke = np.where(np.isfinite(wke), wke, 0.0)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    g = (mid - wke).astype(np.float64)     # >0 -> dropper (columns 0-4 heavier)
    return g, mag.astype(np.float64)


def fit_router(g_train: np.ndarray, m_train: np.ndarray, flat_frac: float = 0.165):
    """Fit (tau, median_dropper, median_riser) on TRAIN customers only.

    tau = the flat_frac quantile of |g| over train (label-free); default 0.165
    matches Paper-4's Flat share. Returns a params dict for assign().
    """
    tau = float(np.quantile(np.abs(g_train), flat_frac))
    drop = g_train > tau
    rise = g_train < -tau
    med_d = float(np.median(m_train[drop])) if drop.any() else 0.0
    med_r = float(np.median(m_train[rise])) if rise.any() else 0.0
    return {"tau": tau, "med_drop": med_d, "med_rise": med_r}


def assign(g: np.ndarray, m: np.ndarray, params: dict) -> np.ndarray:
    """Assign phenotype labels (str array) given fitted params. Vectorised."""
    tau, med_d, med_r = params["tau"], params["med_drop"], params["med_rise"]
    out = np.empty(len(g), dtype=object)
    flat = np.abs(g) <= tau
    drop = (~flat) & (g > 0)
    rise = (~flat) & (g <= 0)
    out[flat] = "Flat"
    out[drop & (m > med_d)] = "Heavy_Dropper"
    out[drop & (m <= med_d)] = "Light_Dropper"
    out[rise & (m > med_r)] = "Heavy_Riser"
    out[rise & (m <= med_r)] = "Light_Riser"
    return out


def assign_train_test(X_raw: np.ndarray, train_idx: np.ndarray,
                      eval_idx: np.ndarray, flat_frac: float = 0.165) -> np.ndarray:
    """Convenience: fit on train, assign for eval_idx. Returns (len(eval_idx),) str."""
    g, m = gap_magnitude(X_raw)
    params = fit_router(g[train_idx], m[train_idx], flat_frac=flat_frac)
    return assign(g[eval_idx], m[eval_idx], params), params


__all__ = ["PHENOTYPES", "PH_INDEX", "gap_magnitude", "fit_router",
           "assign", "assign_train_test"]
