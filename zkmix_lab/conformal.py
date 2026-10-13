"""APS conformal calibration — Romano, Sesia, Candès (NeurIPS 2020).

For a multi-class classifier with calibrated probability vector
``p ∈ Δ^K``, the *adaptive prediction set* score is the cumulative
softmax mass needed to reach the true label after sorting classes in
descending probability::

    s(p, y) = sum_{k : p_(k) >= p_y} p_(k)

A confidently-correct example scores ≈ p_y; a confidently-wrong example
scores ≈ 1.  The conformal threshold ``τ̂`` is the
⌈(n + 1)(1 − α)⌉-th order statistic of the calibration scores; the
prediction set on a new ``x`` is

    C_α(x) = { y : s(p(x), y) ≤ τ̂ }.

Proposition 1 of Romano-Sesia-Candès guarantees marginal coverage
``Pr(Y_{n+1} ∈ C_α(X_{n+1})) ≥ 1 − α`` under exchangeability.

Binary specialisation
---------------------
For binary classification with sigmoid output ``p̂``, we form the
two-vector ``p = (1 − p̂, p̂)`` and apply the K-class machinery with
K = 2.  The four-fold possibility for binary prediction sets is::

    {0}       — confident normal
    {1}       — confident theft
    {0, 1}    — abstain (model uncertain)
    ∅         — never produced (APS cannot exclude the true class
                  while including the wrong one when p̂ ∈ [0, 1]).

The empty set IS possible in principle when τ̂ < min_k p_(k); we guard
against that case by reporting it as ``abstain`` (the safer action on
imbalanced ETD data).

Why this matters for the trustworthy-AI framing
-----------------------------------------------
APS is the cheapest *post-hoc* mechanism that gives an explicit
coverage guarantee — every backbone in the lab can be wrapped with the
same calibrator without retraining.  The framework-level claim is then
backed by an *operational* abstention rule, not just a calibration
plot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """APS non-conformity scores ``s(p, y)``.

    Parameters
    ----------
    probs  : (N, K) softmax probabilities.
    labels : (N,)   integer class indices.

    Returns
    -------
    (N,) cumulative-mass scores.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be (N, K).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must align with probs.")
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)             # high to low
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    # rank_of_label[i] = position of labels[i] in the sorted order.
    inv_order = np.argsort(order, axis=1)
    rank_of_label = inv_order[np.arange(n), labels]
    return cum[np.arange(n), rank_of_label]


def aps_inclusion(probs: np.ndarray, tau: float) -> np.ndarray:
    """Indicator matrix ``C_α(x)`` for every example × class.

    Returns
    -------
    (N, K) bool — ``True`` where ``s(p, y) ≤ tau``.
    """
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    # For each example × class, find its rank then the cumulative score
    # at that rank.  Vectorised via argsort-of-argsort.
    inv_order = np.argsort(order, axis=1)          # rank of each class
    scores = np.take_along_axis(cum, inv_order, axis=1)
    return scores <= tau


@dataclass(frozen=True)
class APSResult:
    """Result of an APS calibration + evaluation.

    Coverage attributes
    -------------------
    coverage_marginal
        ``P(Y ∈ C(X))`` averaged over the test set.  Romano-Sesia-Candès
        (2020) Proposition 1 guarantees this is ≥ 1 − α under exchangeability.
    coverage_pos / coverage_neg
        ``P(Y ∈ C(X) | Y = 1)`` and ``P(Y ∈ C(X) | Y = 0)``.  Marginal-only
        APS gives no guarantee on these; Mondrian APS does.  We report
        them on every cell so the asymmetry is visible.
    """

    alpha: float
    tau: float
    coverage_marginal: float
    coverage_pos: float
    coverage_neg: float
    mean_set_size: float
    abstain_rate: float
    empty_rate: float
    n_cal: int
    n_eval: int
    mode: str = "marginal"    # "marginal" or "mondrian"
    tau_per_class: tuple[float, ...] = ()  # only populated for mondrian


def _class_conditional_coverage(
    sets: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """``(coverage_pos, coverage_neg)`` from a boolean (N, K) inclusion matrix."""
    n = sets.shape[0]
    in_set = sets[np.arange(n), labels]
    mask_pos = labels == 1
    mask_neg = labels == 0
    cov_pos = float(in_set[mask_pos].mean()) if mask_pos.any() else float("nan")
    cov_neg = float(in_set[mask_neg].mean()) if mask_neg.any() else float("nan")
    return cov_pos, cov_neg


class APSCalibrator:
    """Split-conformal APS calibrator (marginal variant).

    Usage
    -----
    >>> cal = APSCalibrator(alpha=0.10)
    >>> cal.fit(cal_probs, cal_labels)
    >>> result = cal.evaluate(test_probs, test_labels)
    >>> sets = cal.predict_sets(test_probs)
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha
        self.tau: float | None = None
        self.n_cal: int = 0

    @staticmethod
    def to_binary_probs(p_pos: np.ndarray) -> np.ndarray:
        if p_pos.ndim != 1:
            raise ValueError("p_pos must be 1-D.")
        return np.stack([1.0 - p_pos, p_pos], axis=1)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "APSCalibrator":
        scores = aps_scores(probs, labels)
        n = scores.shape[0]
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), n)
        self.tau = float(np.partition(scores, k - 1)[k - 1])
        self.n_cal = n
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        if self.tau is None:
            raise RuntimeError("Call fit() before predict_sets().")
        return aps_inclusion(probs, self.tau)

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> APSResult:
        if self.tau is None:
            raise RuntimeError("Call fit() before evaluate().")
        sets = self.predict_sets(probs)
        n = sets.shape[0]
        in_set = sets[np.arange(n), labels]
        cov_pos, cov_neg = _class_conditional_coverage(sets, labels)
        set_sizes = sets.sum(axis=1)
        return APSResult(
            alpha=self.alpha,
            tau=self.tau,
            coverage_marginal=float(in_set.mean()),
            coverage_pos=cov_pos,
            coverage_neg=cov_neg,
            mean_set_size=float(set_sizes.mean()),
            abstain_rate=float((set_sizes >= 2).mean()),
            empty_rate=float((set_sizes == 0).mean()),
            n_cal=self.n_cal,
            n_eval=n,
            mode="marginal",
        )


# ---------------------------------------------------------------------------
# Mondrian (class-conditional) APS — Vovk 2003; Romano et al. 2020 §4
# ---------------------------------------------------------------------------


def _aps_inclusion_per_class_tau(
    probs: np.ndarray, tau_per_class: np.ndarray
) -> np.ndarray:
    """Per-class-τ̂ inclusion: include class y iff ``s(p, y) ≤ τ̂_y``.

    Returns
    -------
    (N, K) bool — ``True`` where ``cumulative_score_to_y ≤ tau_per_class[y]``.
    """
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    inv_order = np.argsort(order, axis=1)
    scores = np.take_along_axis(cum, inv_order, axis=1)
    # Broadcast per-class thresholds over rows.
    return scores <= tau_per_class[None, :]


class MondrianAPSCalibrator:
    """Class-conditional APS — calibrate τ̂_k separately per class k.

    Construction (Vovk 2003 ``Mondrian conformal predictors``; Romano-Sesia-
    Candès 2020 §4 specialise to APS).  Let

        S_k = { s(p_i, y_i) : y_i = k }   (calibration scores in class k)
        n_k = |S_k|
        τ̂_k = ⌈(n_k + 1)(1 − α)⌉-th order statistic of S_k.

    At test time the prediction set is

        C_α(x) = { y' : s(p(x), y') ≤ τ̂_{y'} }.

    Guarantee
    ---------
    Under exchangeability *within each class*,

        Pr( Y_{n+1} ∈ C_α(X_{n+1}) | Y_{n+1} = k ) ≥ 1 − α   ∀ k.

    This is the stronger class-conditional bound; marginal APS gives only
    the unconditional version.  For imbalanced ETD data (8.5 % theft
    prior), the class-conditional bound is the operationally meaningful
    one — it forbids the calibrator from "achieving" 90 % coverage by
    abstaining on every theft case.
    """

    def __init__(self, alpha: float = 0.10, n_classes: int = 2) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if n_classes < 2:
            raise ValueError("n_classes must be ≥ 2.")
        self.alpha = alpha
        self.n_classes = n_classes
        self.tau_per_class: np.ndarray | None = None
        self.n_cal_per_class: np.ndarray | None = None

    @staticmethod
    def to_binary_probs(p_pos: np.ndarray) -> np.ndarray:
        return APSCalibrator.to_binary_probs(p_pos)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "MondrianAPSCalibrator":
        scores = aps_scores(probs, labels)
        tau = np.empty(self.n_classes, dtype=np.float64)
        n_per = np.zeros(self.n_classes, dtype=np.int64)
        for k in range(self.n_classes):
            mask = labels == k
            n_k = int(mask.sum())
            if n_k == 0:
                # Degenerate: no calibration data for class k.  Set τ̂_k = 1
                # so every score ≤ τ̂_k → class k is always included
                # (conservative — favours over-coverage, never under).
                tau[k] = 1.0
                n_per[k] = 0
                continue
            cls_scores = scores[mask]
            kth = int(np.ceil((n_k + 1) * (1.0 - self.alpha)))
            kth = min(max(kth, 1), n_k)
            tau[k] = float(np.partition(cls_scores, kth - 1)[kth - 1])
            n_per[k] = n_k
        self.tau_per_class = tau
        self.n_cal_per_class = n_per
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        if self.tau_per_class is None:
            raise RuntimeError("Call fit() before predict_sets().")
        return _aps_inclusion_per_class_tau(probs, self.tau_per_class)

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> APSResult:
        if self.tau_per_class is None or self.n_cal_per_class is None:
            raise RuntimeError("Call fit() before evaluate().")
        sets = self.predict_sets(probs)
        n = sets.shape[0]
        in_set = sets[np.arange(n), labels]
        cov_pos, cov_neg = _class_conditional_coverage(sets, labels)
        set_sizes = sets.sum(axis=1)
        return APSResult(
            alpha=self.alpha,
            tau=float(self.tau_per_class.mean()),  # representative scalar
            coverage_marginal=float(in_set.mean()),
            coverage_pos=cov_pos,
            coverage_neg=cov_neg,
            mean_set_size=float(set_sizes.mean()),
            abstain_rate=float((set_sizes >= 2).mean()),
            empty_rate=float((set_sizes == 0).mean()),
            n_cal=int(self.n_cal_per_class.sum()),
            n_eval=n,
            mode="mondrian",
            tau_per_class=tuple(float(t) for t in self.tau_per_class),
        )


# ---------------------------------------------------------------------------
# Calibration error (ECE) — useful regardless of conformal step.
# ---------------------------------------------------------------------------


def expected_calibration_error_with_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """ECE plus per-bin breakdown for reliability-diagram plotting.

    Returns
    -------
    ece : float
    bins : list of dicts with keys
        ``edge_lo`` / ``edge_hi`` : bin boundaries on [0, 1].
        ``n``                     : examples falling in the bin.
        ``mean_prob``             : mean predicted probability in the bin.
        ``empirical_rate``        : empirical positive rate in the bin.
        ``gap``                   : |mean_prob − empirical_rate|.

    Bin convention: equal-width on [0, 1], left-closed / right-open
    except the rightmost bin which is right-closed (so prob=1.0 lands
    in the last bin rather than nowhere).
    """
    if probs.ndim != 1 or labels.ndim != 1 or probs.shape != labels.shape:
        raise ValueError("probs and labels must both be 1-D with the same length.")
    n = probs.shape[0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    out: list[dict[str, float]] = []
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (probs >= bins[b]) & (probs < bins[b + 1])
        else:
            mask = (probs >= bins[b]) & (probs <= bins[b + 1])
        n_b = int(mask.sum())
        if n_b == 0:
            out.append({
                "edge_lo": float(bins[b]),
                "edge_hi": float(bins[b + 1]),
                "n": 0,
                "mean_prob": float("nan"),
                "empirical_rate": float("nan"),
                "gap": float("nan"),
            })
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        gap = abs(acc - conf)
        ece += (n_b / n) * gap
        out.append({
            "edge_lo": float(bins[b]),
            "edge_hi": float(bins[b + 1]),
            "n": n_b,
            "mean_prob": conf,
            "empirical_rate": acc,
            "gap": gap,
        })
    return float(ece), out


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Scalar ECE — thin wrapper over :func:`expected_calibration_error_with_bins`."""
    ece, _ = expected_calibration_error_with_bins(probs, labels, n_bins=n_bins)
    return ece


__all__ = [
    "aps_scores",
    "aps_inclusion",
    "APSCalibrator",
    "MondrianAPSCalibrator",
    "APSResult",
    "expected_calibration_error",
    "expected_calibration_error_with_bins",
]
