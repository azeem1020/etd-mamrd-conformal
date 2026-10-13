"""Data loaders, preprocessing, FDI injection, and splits for MSDM reproduction.

Preprocessing (paper §V-A): keep customers with ≥1 valid record; outlier handling;
min–max normalize (Eq. 13); missing→0 with an indicator channel (ref [20]).
SGCC uses native labels; ISET/LCL/ECL inject FDI (msdm/fdi.py).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .fdi import inject

SGCC_CSV = Path(os.environ.get("MAMRD_SGCC", Path(__file__).resolve().parent.parent / "data" / "SGCC.csv"))


# --------------------------- preprocessing --------------------------- #
def minmax_normalize(X, mode="global"):
    """Eq. 13 f(x)=(x−min)/(max−min). mode='global' (paper) or 'per_customer'."""
    if mode == "global":
        lo, hi = np.nanmin(X), np.nanmax(X)
        d = (hi - lo) or 1.0
        return (X - lo) / d
    lo = np.nanmin(X, axis=1, keepdims=True)
    hi = np.nanmax(X, axis=1, keepdims=True)
    d = hi - lo; d[d == 0] = 1.0
    return (X - lo) / d


def clip_outliers(X, n_sigma=3.0):
    """[ASSUMPTION] 3σ cap per customer (NaN-aware), standard ETD outlier handling."""
    m = np.nanmean(X, axis=1, keepdims=True)
    s = np.nanstd(X, axis=1, keepdims=True)
    hi = m + n_sigma * s
    return np.where(np.isfinite(X) & (X > hi), hi, X)


def to_channels(X):
    """Return (N, 2, T): channel0 = value (NaN→0), channel1 = observed-indicator."""
    mask = (~np.isnan(X)).astype(np.float32)
    val = np.where(np.isnan(X), 0.0, X).astype(np.float32)
    return np.stack([val, mask], axis=1)


def preprocess(X, norm_mode="global", with_mask=True):
    X = clip_outliers(X)
    X = minmax_normalize(X, norm_mode)
    return to_channels(X) if with_mask else np.where(np.isnan(X), 0.0, X)[:, None, :].astype(np.float32)


# --------------------------- loaders --------------------------- #
def load_sgcc(csv=SGCC_CSV):
    """SGCC native: returns (X daily (N,T) with NaN, y, cons_no)."""
    df = pd.read_csv(csv)
    y = df["FLAG"].values.astype(np.int64)
    cons = df["CONS_NO"].values
    X = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    return X, y, cons


def _load_wide_csv(path, id_col=0, resample=None):
    """Generic loader for ISET/LCL/ECL once provided: wide CSV (customers × time).
    `resample` (e.g. 'D','H') aggregates sub-daily columns if a datetime index exists.
    [ASSUMPTION] adapt column parsing per actual file format on arrival."""
    df = pd.read_csv(path)
    raise NotImplementedError(
        "Point this at the provided ISET/LCL/ECL file and adapt parsing. "
        "Target output: X (N_customers, T) float with NaN at missing.")


# --------------------------- splits --------------------------- #
def fdi_dataset(X, rng, anomaly_frac=0.10):
    """Inject FDI theft (cross-scenario datasets). Returns (X_theft_mixed, y)."""
    return inject(X, rng, anomaly_frac=anomaly_frac)


def sgcc_ratio_split(N, y, train_ratio, seed=0):
    """Random train/test split at given training ratio (paper Table III: 50–80%)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    ntr = int(round(train_ratio * N))
    return idx[:ntr], idx[ntr:]


def cross_scenario_indices(meta):
    """Build SSD (source) + TSD1–5 (targets) per paper Table I from a metadata frame.
    meta columns: dataset∈{ISET,LCL,ECL}, type∈{Residential,SME,Industrial}.
    [ASSUMPTION] SSD = subset of ISET residential; TSD1/2 ISET res (disjoint),
    TSD3 ISET SME, TSD4 LCL res, TSD5 ECL industrial. Returns dict name→row indices."""
    raise NotImplementedError("Wire once ISET/LCL/ECL provided; see README Table I.")


__all__ = ["minmax_normalize", "clip_outliers", "to_channels", "preprocess",
           "load_sgcc", "fdi_dataset", "sgcc_ratio_split", "cross_scenario_indices"]
