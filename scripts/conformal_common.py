"""Shared split-conformal audit protocol for every SGCC/ECL conformal analysis.

Single source of truth for the resplit generator, the class-conditional (Mondrian)
recall threshold, and the resampling estimators used by
  run_decision_value.py     (Table 6, Fig. 5b)
  run_empirical_comparator.py (Fig. 5a, Discussion Item 3)
  run_trust_layer.py        (Supplementary fig:trust, RESULTS_TRUST.md)
  run_ecl_trust_analysis.py (Supplementary Table S6)
so that all reported conformal quantities are computed on identical resplits.

Small-sample handling (no eligibility gate, no exclusion, no fallback)
--------------------------------------------------------------------
For n calibration thefts in a group and target recall 1-alpha, the split-conformal
threshold uses the k-th order statistic, k = ceil((n+1)(1-alpha)). When k > n
(equivalently n < ceil((1-alpha)/alpha)), the conformal quantile is q_hat = +inf and
the prediction set is the whole label space, so coverage holds trivially
(Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2023,
proof of Theorem 1). For a recall threshold this means: flag EVERY consumer of that
group. The audit cost of a small group is therefore charged to the inspection
workload instead of the resplit being discarded, which keeps every resplit in the
analysis (balanced design, no selection on the calibration draw).
"""
from __future__ import annotations
import math
import numpy as np

NREP = 20  # repeated stratified 50/50 conformal calibration/evaluation resplits per cell


def n_min(alpha: float) -> int:
    """Smallest n with k = ceil((n+1)(1-alpha)) <= n, i.e. ceil((1-alpha)/alpha)."""
    n = 0
    while int(np.ceil((n + 1) * (1.0 - alpha))) > n:
        n += 1
    return n


def resplit_rng(arm_index: int, seed: int, fold: int, rep: int) -> np.random.Generator:
    """Deterministic, process-independent RNG for resplit `rep` of cell (arm, seed, fold)."""
    return np.random.default_rng(((arm_index * 97 + int(seed) * 31 + int(fold)) * 23 + rep) & 0xFFFFFFFF)


def stratified_half(y, rng):
    """Stratified 50/50 split of indices preserving class balance -> (calib_idx, eval_idx)."""
    idx = np.arange(len(y)); cal = []
    for c in (0, 1):
        ic = idx[y == c]; rng.shuffle(ic); cal.append(ic[: len(ic) // 2])
    cal = np.concatenate(cal)
    mask = np.zeros(len(y), bool); mask[cal] = True
    return np.where(mask)[0], np.where(~mask)[0]


def cc_recall_threshold(s_cal, y_cal, alpha):
    """Class-conditional (Mondrian) conformal threshold on the THEFT class.

    Nonconformity of a calibration theft is (1 - s). With n thefts and
    k = ceil((n+1)(1-alpha)): tau = k-th smallest nonconformity if k <= n, else
    tau = 1 (q_hat = +inf -> flag every member of the group). Flag iff s >= 1 - tau.
    Guarantees P(flagged | theft, group) >= 1 - alpha under within-group
    exchangeability (Sadinle et al. 2019; Angelopoulos & Bates 2023, Thm 1).

    Returns (threshold, trivial) where trivial=True marks the q_hat = +inf case.
    """
    pos = np.asarray(s_cal)[np.asarray(y_cal) == 1]
    n = len(pos)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 0.0, True
    tau = float(np.sort(1.0 - pos)[k - 1])
    return 1.0 - tau, False


def bca_ci(x, B=2000, alpha=0.05, seed=0):
    """BCa CI for the mean of a 1-D sample x (Efron 1987; DiCiccio & Efron 1996)."""
    from scipy.stats import norm
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); th = x.mean()
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((boot < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jk.mean(); num = ((jbar - jk) ** 3).sum(); den = 6.0 * ((jbar - jk) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0.0
    def adj(q):
        z = norm.ppf(q); return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, [adj(alpha / 2), adj(1 - alpha / 2)])
    return (float(lo), float(hi))


def cell_summary(cell_value_lists):
    """Grand mean over cells of per-cell resplit means, plus the pooled resplit mean.

    With no eligibility gate every cell contributes NREP resplits, so both are equal
    (balanced design); the pooled value is returned only to assert that equality.
    Values that are undefined for a resplit (e.g. precision with nothing flagged)
    are dropped before averaging. Returns (grand_mean, pooled_mean, n_cells, n_values).
    """
    cells = [np.asarray(c, float) for c in cell_value_lists]
    cells = [c[np.isfinite(c)] for c in cells]
    cells = [c for c in cells if c.size]
    if not cells:
        return float("nan"), float("nan"), 0, 0
    grand = float(np.mean([c.mean() for c in cells]))
    pooled = float(np.concatenate(cells).mean())
    return grand, pooled, len(cells), int(sum(c.size for c in cells))


def cell_bca(cell_value_lists, B=2000, alpha=0.05, seed=0):
    """Cluster BCa: resampling unit = seed-fold cell (resplits within a cell reuse the
    same detector scores and are not independent). Applied to per-cell means."""
    means = [np.nanmean(c) for c in cell_value_lists if np.isfinite(np.asarray(c, float)).any()]
    return bca_ci(means, B=B, alpha=alpha, seed=seed)
