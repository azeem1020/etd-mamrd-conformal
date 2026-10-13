"""Evaluation metrics and paired-bootstrap confidence intervals.

For ETD with ~9 % theft prior, PR-AUC is the canonical metric (AUROC
is overly optimistic under imbalance; see Saito & Rehmsmeier 2015,
"The Precision-Recall Plot Is More Informative than the ROC Plot
When Evaluating Binary Classifiers on Imbalanced Datasets").  We
report all three (PR-AUC, AUROC, MAP@K) but make decisions on PR-AUC.

The bootstrap design is *paired*: for each replicate we resample
indices with replacement once and score *both* candidate models on
that same resample.  This factors out the dataset variance and tests
only the architecture-driven difference — appropriate for the
head-to-head comparison framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve.

    Equivalent to scikit-learn's ``average_precision_score`` — the
    weighted-by-recall-delta sum of precisions, NOT the trapezoid AUC
    of the PR curve, which can over-estimate near the corners.
    """
    return float(average_precision_score(y_true, y_score))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """MAP@k as defined in Zheng et al. 2018 §V-A-3 Eq. 14.

    Order samples by descending score, take the top-k, then average
    Precision@k_i over the positions of true positives in the top-k.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    order = np.argsort(-y_score, kind="stable")[:k]
    top_labels = y_true[order]
    if top_labels.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(top_labels)
    positions = np.arange(1, k + 1)
    precisions = cum_hits / positions
    return float(precisions[top_labels == 1].mean())


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, recall_target: float = 0.90
) -> tuple[float, float]:
    """Operating-point precision when recall is at least ``recall_target``.

    Picks the threshold along the PR curve that maximises precision
    subject to recall ≥ ``recall_target``.  Returns
    ``(precision_at_R, threshold)``.  If no threshold satisfies the
    recall floor (impossible at recall_target ≤ 1.0 since recall=1.0 is
    always achievable by predicting everything), the highest-recall
    threshold is returned.

    Why this metric: PR-AUC is a curve; operators need a point.  For
    fraud detection, a fixed recall floor with maximised precision is
    the canonical operating choice — it caps "missed thefts" and
    maximises "review productivity" simultaneously.
    """
    if not (0.0 < recall_target <= 1.0):
        raise ValueError("recall_target must be in (0, 1].")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall arrays have length n + 1; thresholds has length n.
    # Walk left → right (decreasing recall, increasing precision usually).
    mask = recall[:-1] >= recall_target
    if not mask.any():
        # Every threshold gave recall < target — return the lowest threshold.
        return float(precision[0]), float(thresholds[0])
    idx = np.argmax(precision[:-1][mask])
    valid_thresh = thresholds[mask]
    return float(precision[:-1][mask][idx]), float(valid_thresh[idx])


def f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Max F1 over all thresholds + the achieving threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # Avoid div-by-zero — sklearn already handles it but we mirror.
    f1 = 2 * precision * recall / np.clip(precision + recall, a_min=1e-12, a_max=None)
    # Drop the rightmost point (recall=0, precision=1, no threshold).
    f1_t = f1[:-1]
    if len(f1_t) == 0:
        return 0.0, 0.5
    idx = int(np.argmax(f1_t))
    return float(f1_t[idx]), float(thresholds[idx])


def accuracy_at_prior(
    y_true: np.ndarray, y_score: np.ndarray, prior: float | None = None
) -> tuple[float, float]:
    """Accuracy with threshold set so that fraction-flagged matches the prior.

    If ``prior`` is None, use the empirical positive rate of ``y_true``.
    This isolates the model's *ranking* quality from threshold tuning —
    if the model ranks correctly, top-``prior`` examples are positives.
    """
    if prior is None:
        prior = float(y_true.mean())
    if not (0.0 < prior < 1.0):
        raise ValueError("prior must be in (0, 1).")
    n = len(y_true)
    n_flag = max(1, int(round(n * prior)))
    # Threshold = score at rank n_flag from the top.
    sorted_scores = np.sort(y_score)[::-1]
    thresh = float(sorted_scores[n_flag - 1])
    y_pred = (y_score >= thresh).astype(int)
    return float((y_pred == y_true).mean()), thresh


def summarise(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: Sequence[int] = (100, 200),
    recall_target: float = 0.90,
) -> dict[str, float]:
    """Bundle every per-cell metric we report into one dict."""
    out: dict[str, float] = {
        "pr_auc": pr_auc(y_true, y_score),
        "auroc": auroc(y_true, y_score),
    }
    for k in k_values:
        if len(y_true) >= k:
            out[f"map@{k}"] = map_at_k(y_true, y_score, k)
    # Operational metrics.
    out["precision_at_recall"] = precision_at_recall(y_true, y_score, recall_target)[0]
    out["f1_optimal"] = f1_optimal(y_true, y_score)[0]
    out["accuracy_at_prior"] = accuracy_at_prior(y_true, y_score)[0]
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Result of a paired bootstrap on PR-AUC margin.

    Attributes
    ----------
    delta_mean
        Mean of (score_b - score_a) across bootstrap replicates.
    ci_low, ci_high
        Lower and upper bounds of the (1 - alpha) percentile CI.
    alpha
        Two-sided significance level (e.g. 0.05 for a 95 % CI).
    n_replicates
        Number of bootstrap resamples performed.
    margin_point
        Point estimate of (score_b - score_a) on the full dataset
        (no resampling) — anchors the CI to a single number.
    """

    delta_mean: float
    ci_low: float
    ci_high: float
    alpha: float
    n_replicates: int
    margin_point: float


def paired_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_replicates: int = 2000,
    alpha: float = 0.05,
    metric: str = "pr_auc",
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap CI for (metric_b − metric_a).

    Both score vectors must be aligned to the same ``y_true`` — the
    same resample of indices is scored under both, so the variance of
    label assignment cancels out.

    Returns the CI for *b minus a*: positive lower bound means b wins,
    negative upper bound means a wins, CI straddling zero means
    inconclusive.
    """
    if score_a.shape != score_b.shape or score_a.shape != y_true.shape:
        raise ValueError("score_a, score_b, y_true must be the same length.")
    if metric not in {"pr_auc", "auroc"}:
        raise ValueError(f"Unknown metric {metric!r}; use 'pr_auc' or 'auroc'.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    fn = pr_auc if metric == "pr_auc" else auroc

    margin_point = fn(y_true, score_b) - fn(y_true, score_a)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_replicates, dtype=np.float64)
    valid = 0
    for r in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        y_r = y_true[idx]
        # If a resample happens to be all-negative or all-positive,
        # PR-AUC is undefined.  Skip and resample.
        if y_r.sum() == 0 or y_r.sum() == n:
            continue
        deltas[valid] = fn(y_r, score_b[idx]) - fn(y_r, score_a[idx])
        valid += 1
    if valid < n_replicates * 0.9:
        raise RuntimeError(
            f"Bootstrap discarded {n_replicates - valid}/{n_replicates} "
            "replicates — dataset too small or too imbalanced."
        )
    deltas = deltas[:valid]
    return PairedBootstrapResult(
        delta_mean=float(deltas.mean()),
        ci_low=float(np.quantile(deltas, alpha / 2)),
        ci_high=float(np.quantile(deltas, 1 - alpha / 2)),
        alpha=alpha,
        n_replicates=valid,
        margin_point=float(margin_point),
    )


def lock_verdict(
    result: PairedBootstrapResult,
    breakthrough_threshold: float = 0.02,
    mechanism_threshold: float = -0.02,
    direction: str = "headline",
) -> str:
    """Human-readable lock verdict for the paper-narrative gates.

    Parameters
    ----------
    direction
        ``"headline"`` checks ``ci_low ≥ breakthrough_threshold``
        (claim: b beats a by at least the threshold).
        ``"mechanism"`` checks ``ci_high ≤ mechanism_threshold``
        (claim: removing the mechanism *hurts* b by at least the
        absolute threshold).
    """
    if direction == "headline":
        if result.ci_low >= breakthrough_threshold:
            return "LOCK"
        if result.ci_high < breakthrough_threshold:
            return "FAIL"
        return "INCONCLUSIVE"
    if direction == "mechanism":
        if result.ci_high <= mechanism_threshold:
            return "LOCK"
        if result.ci_low > mechanism_threshold:
            return "FAIL"
        return "INCONCLUSIVE"
    raise ValueError(f"Unknown direction {direction!r}.")


__all__ = [
    "pr_auc",
    "auroc",
    "map_at_k",
    "precision_at_recall",
    "f1_optimal",
    "accuracy_at_prior",
    "summarise",
    "paired_bootstrap",
    "PairedBootstrapResult",
    "lock_verdict",
]
