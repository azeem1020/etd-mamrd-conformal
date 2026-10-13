"""Phase L — Hand-crafted engineered features for the auxiliary branch (A24).

Computes 12 features per customer that have empirical r=0.07–0.10 correlation
with theft, sourced from ``research_outputs/tables/04_feature_target_correlation.csv``.
These features are *not* learned by the existing A9-lean backbone from raw
input; injecting them as a side branch closes the gap.

Feature list (in order)
-----------------------
1.  max_consec_nan      — longest consecutive NaN run  (r=0.102)
2.  lag1_autocorr       — lag-1 Pearson autocorrelation (r=0.097)
3.  missing_frac        — fraction of NaN cells        (r=0.095)
4.  roll_std_max        — max 30-day rolling std       (r=0.084) — burst tampering
5.  roll_std_range      — range of 30-day rolling std  (r=0.084)
6.  consumption_entropy — Shannon entropy of binned values (r=0.066)
7.  wed_sat_gap         — signed Wed-Sat gap            (clustering axis)
8.  dow_profile_var     — variance of 7-day mean profile (r=−0.07 inverted)
9.  cv_global           — coefficient of variation     (theft has lower CV)
10. iqr                 — inter-quartile range          (r=0.086)
11. p90_p10_range       — 90th − 10th percentile range  (r=0.071)
12. mean_dow_5          — mean Saturday consumption     (r=0.091)

All features computed with NaN-aware reducers; NaN-only series are mapped to 0.
Returned tensor shape: (N, 12) float32, **standardised** (zero-mean, unit-std)
across the population so all features land on a comparable scale.

NaN safety
----------
- np.nanmean / np.nanstd / np.nanmax / np.nanmin with all-NaN inputs are
  caught and replaced with 0.
- The standardisation step uses population mean/std (not per-fold), which
  is FINE because it does NOT use labels — only the raw consumption
  distribution. No label leakage.

Public API
----------
``compute_eng_features(X_raw) -> (N, 12) float32`` (un-standardised)
``compute_eng_features_standardised(X_raw) -> ((N, 12), (mean, std)) float32``
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


N_FEATURES: int = 12
FEATURE_NAMES: tuple[str, ...] = (
    "max_consec_nan",
    "lag1_autocorr",
    "missing_frac",
    "roll_std_max",
    "roll_std_range",
    "consumption_entropy",
    "wed_sat_gap",
    "dow_profile_var",
    "cv_global",
    "iqr",
    "p90_p10_range",
    "mean_dow_5",
)


def _max_consec_nan(daily: np.ndarray) -> np.ndarray:
    """Longest consecutive NaN run per customer.

    daily : (N, T) flattened daily series.
    Returns (N,) float32.
    """
    N, T = daily.shape
    is_nan = np.isnan(daily).astype(np.int32)
    out = np.zeros(N, dtype=np.int32)
    # Vectorised running streak (efficient via cumulative sums of breaks).
    for i in range(N):
        # find max consecutive 1s in is_nan[i]
        m = 0
        c = 0
        row = is_nan[i]
        for v in row:
            if v:
                c += 1
                if c > m:
                    m = c
            else:
                c = 0
        out[i] = m
    return out.astype(np.float32)


def _lag1_autocorr(daily: np.ndarray) -> np.ndarray:
    """Lag-1 Pearson autocorrelation per row, NaN-aware via pairwise deletion."""
    N, T = daily.shape
    out = np.zeros(N, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(N):
            x  = daily[i, :-1]
            y  = daily[i, 1:]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            xm = x[mask]
            ym = y[mask]
            xc = xm - xm.mean()
            yc = ym - ym.mean()
            denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
            if denom == 0:
                continue
            out[i] = float((xc * yc).sum() / denom)
    return out


def _missing_frac(daily: np.ndarray) -> np.ndarray:
    return np.isnan(daily).mean(axis=1).astype(np.float32)


def _roll_std_features(daily: np.ndarray, win: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """30-day rolling std → return (max, range) per customer."""
    N, T = daily.shape
    if T < win:
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
    # Slide window manually (NaN-aware).
    n_wins = T - win + 1
    stds = np.zeros((N, n_wins), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w in range(n_wins):
            stds[:, w] = np.nanstd(daily[:, w:w + win], axis=1)
    stds = np.where(np.isfinite(stds), stds, 0.0)
    rmax = stds.max(axis=1)
    rmin = stds.min(axis=1)
    return rmax.astype(np.float32), (rmax - rmin).astype(np.float32)


def _consumption_entropy(daily: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Shannon entropy of binned daily values per customer.

    Bins use the same 0..max range per customer (each customer is normalised
    independently → already on 0..1 scale post-Phase-C preprocessing).
    """
    N, T = daily.shape
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        x = daily[i]
        x = x[~np.isnan(x)]
        if len(x) < 5:
            continue
        # Histogram with fixed bins on the customer's own range.
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo + 1e-9:
            continue
        hist, _ = np.histogram(x, bins=n_bins, range=(lo, hi))
        p = hist / hist.sum()
        p = p[p > 0]
        out[i] = float(-(p * np.log(p)).sum())
    return out


def _wed_sat_gap(X_raw: np.ndarray) -> np.ndarray:
    """Signed Saturday-minus-Wednesday gap, NaN-aware mean over weeks."""
    wed = X_raw[:, :, 2]
    sat = X_raw[:, :, 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wm = np.nanmean(wed, axis=1)
        sm = np.nanmean(sat, axis=1)
    wm = np.where(np.isfinite(wm), wm, 0.0)
    sm = np.where(np.isfinite(sm), sm, 0.0)
    return (sm - wm).astype(np.float32)


def _dow_profile_var(X_raw: np.ndarray) -> np.ndarray:
    """Variance of the 7-day mean profile per customer.

    Computes the mean of each day-of-week across all weeks, then the variance
    of that 7-vector. NaN-aware. Customers with all-NaN rows → 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dow_means = np.nanmean(X_raw, axis=1)             # (N, 7)
    dow_means = np.where(np.isfinite(dow_means), dow_means, 0.0)
    return dow_means.var(axis=1).astype(np.float32)


def _cv_global(daily: np.ndarray) -> np.ndarray:
    """Coefficient of variation = std / mean per customer (NaN-aware)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(daily, axis=1)
        s = np.nanstd(daily, axis=1)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.where(np.isfinite(s), s, 0.0)
    eps = 1e-6
    return (s / (np.abs(m) + eps)).astype(np.float32)


def _iqr_and_range(daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IQR = p75 − p25, and p90 − p10 per customer (NaN-aware)."""
    N, T = daily.shape
    iqr = np.zeros(N, dtype=np.float32)
    rng90_10 = np.zeros(N, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(N):
            x = daily[i]
            x = x[~np.isnan(x)]
            if len(x) < 5:
                continue
            q = np.percentile(x, [10, 25, 75, 90])
            iqr[i]      = float(q[2] - q[1])
            rng90_10[i] = float(q[3] - q[0])
    return iqr, rng90_10


def _mean_dow_5(X_raw: np.ndarray) -> np.ndarray:
    """Mean Saturday consumption per customer (NaN-aware)."""
    sat = X_raw[:, :, 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sm = np.nanmean(sat, axis=1)
    return np.where(np.isfinite(sm), sm, 0.0).astype(np.float32)


def compute_eng_features(X_raw: np.ndarray) -> np.ndarray:
    """Compute the 12-feature engineered vector per customer.

    Parameters
    ----------
    X_raw : (N, W, D) float32 — daily-grid (NaN allowed).

    Returns
    -------
    feats : (N, 12) float32, un-standardised.
    """
    if X_raw.ndim != 3:
        raise ValueError(f"X_raw must be (N, W, D); got {X_raw.shape}")
    N, W, D = X_raw.shape
    daily = X_raw.reshape(N, W * D)                          # (N, W*D)

    f1  = _max_consec_nan(daily)
    f2  = _lag1_autocorr(daily)
    f3  = _missing_frac(daily)
    f4, f5 = _roll_std_features(daily, win=30)
    f6  = _consumption_entropy(daily)
    f7  = _wed_sat_gap(X_raw)
    f8  = _dow_profile_var(X_raw)
    f9  = _cv_global(daily)
    f10, f11 = _iqr_and_range(daily)
    f12 = _mean_dow_5(X_raw)

    out = np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], axis=1)
    # Final NaN/Inf scrub.
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32, copy=False)


def compute_eng_features_standardised(
    X_raw: np.ndarray,
    mean: np.ndarray | None = None,
    std:  np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute features and standardise to zero-mean unit-std across customers.

    Standardisation parameters are POPULATION-level (no label use → no leakage).
    If ``mean`` / ``std`` are passed, they are used directly; otherwise computed
    from this batch. This enables a single-pass population standardisation that
    is then applied identically across folds.

    Returns (standardised_feats, mean, std) — pass mean/std to subsequent calls
    if needed (e.g., test-time evaluation).
    """
    feats = compute_eng_features(X_raw)                                  # (N, 12)
    if mean is None:
        mean = feats.mean(axis=0)
    if std is None:
        std = feats.std(axis=0)
    std_safe = np.where(std > 1e-6, std, 1.0)
    standardised = (feats - mean) / std_safe
    # Winsorise to ±5 σ. Several features (e.g. dow_profile_var, std≈1e-3) are
    # heavy-tailed near-constants whose rare outliers z-score to >100σ. Fed into
    # the branch MLP these produce a gradient that, under the global grad-clip
    # (norm=1.0), consumes the entire budget and starves the backbone — collapsing
    # PR-AUC from ~0.55 to ~0.30. Clipping to ±5 is a fixed, label-free, causal
    # transform (no leakage; same constant for every row).
    standardised = np.clip(standardised, -5.0, 5.0)
    return standardised.astype(np.float32, copy=False), mean.astype(np.float32), std.astype(np.float32)


__all__ = [
    "N_FEATURES",
    "FEATURE_NAMES",
    "compute_eng_features",
    "compute_eng_features_standardised",
]
