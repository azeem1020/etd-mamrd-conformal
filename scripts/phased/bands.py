"""Consumption-band taxonomy — the group function for Mondrian conformal prediction.

Replaces phased/phenotype.py for this manuscript.

WHY THE PHENOTYPE ROUTER WAS RETIRED
------------------------------------
phenotype.py splits consumers on a signed "mid-week minus weekend" switching gap g,
computed by reshaping the consumption matrix to (weeks, 7) and treating column slots 0-4
as weekdays and 5-6 as weekend. The SGCC CSV header is ordered lexicographically
(2014/1/1, 2014/1/10, 2014/1/11, ..., 2014/1/2, 2014/1/20, ...) and no stage of the
pipeline sorts it, so slot index mod 7 is not the day of the week: each slot holds a
near-uniform mixture of all seven days, and only 45.2% of the days landing in the two
"weekend" slots are weekends. The gap therefore does not measure switching behaviour, and
the Flat group it induces is not a behavioural class -- it is the near-zero-consumption
tail, which reaches |g| ~ 0 by arithmetic degeneracy under any column ordering.

Magnitude is a mean over ALL columns and is order-invariant, so it is untouched by the
ordering defect. It is the only axis this module uses.

WHAT A TAXONOMY FUNCTION MUST SATISFY
-------------------------------------
Mondrian / group-conditional conformal prediction requires the taxonomy function to be a
deterministic function of information available at prediction time, applied identically to
calibration and test points (Vovk et al. 2005). Consumption level is measured from the
consumer's own metered history, so it qualifies directly. It is not predicted, and no
label enters it.

LEAKAGE FIREWALL
----------------
Band edges are fitted on TRAIN consumers only and applied to test, exactly as
phenotype.fit_router / phenotype.assign did. Labels never enter band assignment.
"""
from __future__ import annotations

import warnings

import numpy as np

# Consumption quartiles. run_band_frontier.py measures the frontier over K = 2..6 on the
# eligible population: K = 5 is the largest K every band of which clears the n_min = 11
# threshold of the (0.80, 0.90) training-conditional certificate in >= 90% of calibration
# draws, but it does so only at 100% commitment of the confirmed-theft library, leaving no
# reserve. K = 4 certifies at 65% commitment. The operating point is K = 4 so the selection
# sweep has room to show the lowest band degrading as gamma rises rather than starting
# already at the edge of feasibility; K = 5 and K = 6 are reported as the frontier.
K_DEFAULT = 4


def band_names(K: int = K_DEFAULT) -> tuple[str, ...]:
    """Compact table labels. C1 is the lowest-consumption band."""
    return tuple(f"C{i + 1}" for i in range(K))


def mean_consumption(X_raw: np.ndarray) -> np.ndarray:
    """Mean observed daily consumption per consumer, NaN-aware.

    X_raw : (N, D) or (N, W, 7) raw consumption, NaN at missing readings.

    Mean rather than total: SGCC carries ~25% missing readings, and a total would score a
    consumer with many unread days as a low consumer. That would make the taxonomy partly
    a function of meter-reading coverage, which is plausibly correlated with tampering --
    exactly the confound the group function must not contain. The mean is invariant to how
    many days were observed.

    Identical to phenotype.gap_magnitude's second return value, so any comparison against
    the retired partition is on the same statistic.
    """
    flat = X_raw.reshape(len(X_raw), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN rows
        m = np.nanmean(flat, axis=1)
    return np.where(np.isfinite(m), m, 0.0).astype(np.float64)


MIN_OBSERVED_DAYS = 100        # ~10% of the 1,034-day window


def eligible(X_raw: np.ndarray) -> np.ndarray:
    """Boolean mask of consumers the banded certificate covers. Label-free.

    Two exclusions, both scope conditions on the guarantee rather than data cleaning,
    and both declared in the manuscript because each removes a distinctive subpopulation:

      (1) Meters reading identically zero (4.99% of SGCC). A premises drawing no power is
          vacant or disconnected, not a low-consumption consumer, and it enters the lowest
          band by arithmetic rather than by consumption. Theft prevalence among them is
          1.70% against 8.53% overall. Excluded on author instruction, 2026-08-14.

      (2) Meters with fewer than MIN_OBSERVED_DAYS readings (0.18%). A band assignment
          computed from under a tenth of the window is not a consumption level. These are
          NOT a nuisance stratum: they carry 25.0% theft prevalence, roughly three times
          the base rate, and consumers under 200 readings are enriched about 2.4x. That is
          consistent with tampering suppressing reporting as well as recorded volume, so
          they are better treated as a separate high-priority inspection class than folded
          into a consumption band. The banded certificate does not cover them and the
          manuscript says so.

    Above the threshold the mean is precise enough that coverage is not a concern: at the
    1st percentile of coverage (261 readings) the standard error of a consumer's mean is
    0.062 of their own standard deviation.
    """
    flat = X_raw.reshape(len(X_raw), -1)
    obs = np.isfinite(flat).sum(axis=1)
    return (mean_consumption(X_raw) > 0) & (obs >= MIN_OBSERVED_DAYS)


def fit_bands(m_train: np.ndarray, K: int = K_DEFAULT) -> dict:
    """Fit equal-population band edges on TRAIN consumers only. Label-free.

    Equal-population rather than equal-width: consumption is heavy-tailed (SGCC p50 = 5.7
    against p100 = 13,601), so equal-width bands would place almost every consumer in the
    lowest band and leave the rest with no confirmed-theft library at all.
    """
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    qs = np.linspace(0.0, 1.0, K + 1)[1:-1]                  # K-1 interior cut points
    edges = np.quantile(np.asarray(m_train, float), qs)
    # Ties at zero (SGCC has ~5% exactly-zero consumers) can collapse adjacent edges,
    # which would silently empty a band. Nudge duplicates apart so searchsorted stays
    # monotone and any resulting imbalance is visible in the composition table instead.
    edges = np.maximum.accumulate(edges + np.arange(len(edges)) * 1e-12)
    return {"edges": edges.astype(np.float64), "K": int(K)}


def assign_bands(m: np.ndarray, params: dict) -> np.ndarray:
    """Assign band labels (str array) given fitted edges. Vectorised."""
    idx = np.searchsorted(params["edges"], np.asarray(m, float), side="right")
    idx = np.clip(idx, 0, params["K"] - 1)
    return np.asarray(band_names(params["K"]), dtype=object)[idx]


def assign_train_test(X_raw: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
                      K: int = K_DEFAULT) -> tuple[np.ndarray, dict]:
    """Convenience: fit edges on train, assign for eval_idx."""
    m = mean_consumption(X_raw)
    params = fit_bands(m[train_idx], K=K)
    return assign_bands(m[eval_idx], params), params


__all__ = ["K_DEFAULT", "MIN_OBSERVED_DAYS", "band_names", "mean_consumption",
           "eligible", "fit_bands", "assign_bands", "assign_train_test"]
