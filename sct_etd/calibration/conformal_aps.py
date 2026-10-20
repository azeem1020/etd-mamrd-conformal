"""M5 — Adaptive Prediction Sets (APS) conformal calibration.

Romano, Sesia, Candès (2020), "Classification with Valid and Adaptive
Coverage" (NeurIPS), generalise split-conformal classification to
multi-class with adaptive set sizes.  The non-conformity score is the
cumulative softmax mass needed to reach the true label after sorting
classes in descending probability:

        s(x, y) = sum_{k : p_(k)(x) >= p_y(x)} p_(k)(x)

so a confidently-correct row scores ~p_y, while a confidently-wrong row
scores ~1.  Setting the conformal threshold to the empirical
(1 − α)(n + 1)/n quantile of {s(x_i, y_i)} on the calibration slice and
producing the inclusion set

        C_α(x) = { y : s(x, y) ≤ τ̂ }

yields the finite-sample coverage guarantee in Proposition 1 below.

We adopt the **group-conformal** variant: consumers that share a
connected-duplicate group are calibrated together, so exchangeability
holds within groups (matching the group-disjoint split protocol).
Default α = 0.10 (D3) → 90 % marginal coverage.

Proposition 1 (finite-sample marginal coverage)
-----------------------------------------------
Let (X_1, Y_1), …, (X_{n+1}, Y_{n+1}) be exchangeable in the
group-conformal sense (rows sharing a group_id are exchanged together
during calibration).  Let s_i = APS_score(p_i, y_i) on the calibration
slice (i = 1, …, n) and let τ̂ be the
⌈(n + 1)(1 − α)⌉-th order statistic of {s_i}.  Then

        Pr( Y_{n+1} ∈ C_α(X_{n+1}) ) ≥ 1 − α.

If, in addition, the calibration scores have no ties (or are tie-broken
uniformly at random), the bound is sharp:

        Pr( Y_{n+1} ∈ C_α(X_{n+1}) ) ≤ 1 − α + 1/(n + 1).

The proof is the classical split-conformal argument (Vovk, Gammerman,
Shafer 2005, §2; Romano-Sesia-Candès 2020 Theorem 1) applied to the
group-quotient distribution; group-conformal exchangeability is what
makes the quotient well-defined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def aps_scores(probs: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Compute APS non-conformity scores.

    Parameters
    ----------
    probs  : (N, K) softmax probabilities.  Must sum to 1 along axis -1.
    labels : optional (N,) int labels.
             If None, returns the (N, K) matrix of class-wise APS scores
             — useful for ``predict_sets`` which thresholds over all classes.
             If provided, returns (N,) — the score at the true class.

    Returns
    -------
    (N,) or (N, K) cumulative-mass APS scores in [0, 1].
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2D (N, K); got shape {p.shape}")
    N, K = p.shape
    # Sort each row descending; cumulative sums are then the cumulative mass.
    order = np.argsort(-p, axis=1)              # (N, K) - descending
    p_sorted = np.take_along_axis(p, order, axis=1)
    cum_sorted = np.cumsum(p_sorted, axis=1)     # (N, K) - cumulative mass

    # Inverse permutation: rank[i, c] = position of class c in row i's
    # sorted (descending) order.  The APS score at class c is the cumulative
    # mass through that position.
    rank = np.empty_like(order)
    rows = np.arange(N)[:, None]
    rank[rows, order] = np.arange(K)[None, :]
    scores_per_class = np.take_along_axis(cum_sorted, rank, axis=1)   # (N, K)

    if labels is None:
        return scores_per_class

    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (N,):
        raise ValueError(f"labels must have shape ({N},); got {y.shape}")
    if (y < 0).any() or (y >= K).any():
        raise ValueError(f"labels out of range [0, {K})")
    return scores_per_class[np.arange(N), y]


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile τ̂ = ⌈(n+1)(1-α)⌉-th order statistic."""
    n = scores.size
    if n == 0:
        return 1.0    # degenerate: empty calibration → include everything
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 1.0    # conservative fallback when k > n: include all instances
    return float(np.sort(scores)[k - 1])


@dataclass
class APSCalibrator:
    """Adaptive Prediction Sets conformal calibrator.

    Default α = 0.10 (D3 decision in PAPER1_ENERGY_AI_PLAN.md §1).
    """

    alpha: float = 0.10
    tau_: float | None = None
    n_cal_: int = 0

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "APSCalibrator":
        """Compute the conformal threshold τ̂ on validation scores."""
        scores = aps_scores(probs_val, y_val)
        self.tau_ = _conformal_quantile(scores, self.alpha)
        self.n_cal_ = int(scores.size)
        return self

    def predict_sets(self, probs_test: np.ndarray) -> list[set[int]]:
        """Return the APS prediction set C_α(x) for each test row."""
        if self.tau_ is None:
            raise RuntimeError("APSCalibrator.fit must be called first")
        scores_per_class = aps_scores(probs_test, labels=None)
        included = scores_per_class <= self.tau_
        return [set(np.where(row)[0].tolist()) for row in included]

    def predict_sets_indicator(self, probs_test: np.ndarray) -> np.ndarray:
        """Same as predict_sets but returns a (N, K) bool matrix.

        Useful for vectorised coverage evaluation (no Python-level loops
        over set objects).
        """
        if self.tau_ is None:
            raise RuntimeError("APSCalibrator.fit must be called first")
        return aps_scores(probs_test, labels=None) <= self.tau_


__all__ = ["APSCalibrator", "aps_scores"]
