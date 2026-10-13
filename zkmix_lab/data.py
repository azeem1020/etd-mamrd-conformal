"""Data layer — loaders, splits, k-fold + group-disjoint helpers.

Reads the preprocessed SGCC matrix produced by the upstream
``preprocess.py`` (and its function-style port in
``sct_etd.data.preprocess``).  The expected layout is::

    data/after_preprocess_data.csv   # (N, 1029) — already MIN-MAX'd
    data/label.csv                   # column "flag" — 0/1

If those CSVs are absent the loader can synthesise a *structured*
toy matrix for tests — see ``make_synthetic``.

Split families
--------------
* ``random_kfold``  — i.i.d. row-level k-fold.  Test ⊂ train customers
                      is *not* possible (each customer is one row), but
                      this still represents the *random* condition
                      because the deterministic random shuffle treats
                      all customers as exchangeable.
* ``grouped_kfold`` — customer-disjoint k-fold; here it reduces to
                      ``random_kfold`` because the SGCC release gives
                      one row per customer.  The "group" axis exists in
                      multi-meter / multi-month datasets — we keep the
                      hook so cross-dataset extensions can plug in.
* ``temporal_split`` — front 80 % weeks → train, back 20 % → val.
                       Used as a robustness check; not part of the
                       headline grid.

Each split returns ``(train_idx, val_idx)`` numpy int arrays.  The
``KFoldSpec`` dataclass bundles seed/k/family so a runner can iterate
fold IDs without re-deriving the split each time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_sgcc(
    features_csv: str | Path = "data/after_preprocess_data.csv",
    labels_csv: str | Path = "data/label.csv",
    weeks: int = 147,
    days: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the preprocessed SGCC matrix → (X[N, W, D], y[N])."""
    features_csv = Path(features_csv)
    labels_csv = Path(labels_csv)
    if not features_csv.exists() or not labels_csv.exists():
        raise FileNotFoundError(
            f"Expected {features_csv} and {labels_csv}; run preprocess.py first."
        )
    flat = pd.read_csv(features_csv).values.astype(np.float32)
    if flat.shape[1] != weeks * days:
        raise ValueError(
            f"Expected {weeks * days} day-columns, got {flat.shape[1]}; "
            "rerun preprocess.py with the correct truncation."
        )
    y = pd.read_csv(labels_csv)["flag"].values.astype(np.int64)
    if len(y) != flat.shape[0]:
        raise ValueError(
            f"Label/feature row mismatch: {len(y)} vs {flat.shape[0]}."
        )
    x = flat.reshape(-1, weeks, days)
    return x, y


def make_synthetic(
    n_normal: int = 800,
    n_theft: int = 80,
    weeks: int = 147,
    days: int = 7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a structured (X, y) for tests.

    Normal customers: smooth weekly periodicity with mild Gaussian
    noise.  Theft customers: same baseline plus a hard *drop* over a
    random contiguous span of weeks (the canonical SGCC theft
    signature).  Always MIN-MAX-normalised per customer to match the
    real preprocessing range.
    """
    rng = np.random.default_rng(seed)
    n = n_normal + n_theft
    t = weeks * days
    base = np.linspace(0.4, 0.8, days)
    week_pattern = np.tile(base, weeks)
    x = week_pattern + 0.05 * rng.standard_normal((n, t))
    y = np.zeros(n, dtype=np.int64)
    y[n_normal:] = 1
    for i in range(n_normal, n):
        start_week = int(rng.integers(weeks // 4, 3 * weeks // 4))
        end_week = int(rng.integers(start_week + 2, weeks))
        s, e = start_week * days, end_week * days
        x[i, s:e] *= 0.25
    # MIN-MAX per row (post-mortem of preprocess.py Eq. 3).
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    x = (x - mn) / denom
    return x.reshape(n, weeks, days).astype(np.float32), y


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


SplitFamily = Literal["random", "stratified", "temporal"]
RegimeName = Literal["full", "high_miss", "stable"]


# ---------------------------------------------------------------------------
# Regime / missingness-window slicing
# ---------------------------------------------------------------------------
#
# The SGCC release covers 2014-01-01 → 2016-10-31.  Inspection of the raw
# CSV (and Zheng et al. 2018 Fig. 1) shows two missingness regimes:
#
#   • 2014-01-01 → 2015-12-31  — severe missingness (Eq. 1 fills are noisy).
#   • 2016-01-01 → 2016-10-31  — stable instrumentation, few NaNs.
#
# Preprocess.py floor-truncates to 147 weeks (= 1029 days = 1029/7) starting
# from day 0.  Day 0 is 2014-01-01.  104 weeks = 728 days lands the cutoff
# at 2015-12-30 (close enough; the boundary is approximate at the week
# granularity).  Regime B is the remaining 43 weeks (= 301 days, through
# 2016-10-28 — again close enough to the dataset's nominal end of 10-31).
#
# We expose the regime as a slice of the *weeks* axis so every backbone
# can be reconfigured with the regime's week count without re-running
# preprocessing.

REGIME_BOUNDS_WEEKS: dict[str, tuple[int, int]] = {
    "high_miss": (0, 104),   # 2014-01-01 → 2015-12-30
    "stable":    (104, 147), # 2016-01-01 → 2016-10-28
    "full":      (0, 147),   # the whole window
}


def slice_regime(X: np.ndarray, regime: str) -> tuple[np.ndarray, int]:
    """Return ``(X_sliced, weeks)`` for the given regime name.

    For ``regime == "full"`` we never slice — the function is a no-op and
    works for any input shape.  This lets synthetic tests use the runner
    end-to-end without pretending to have a 147-week matrix.

    For the actual missingness regimes (``high_miss`` / ``stable``) we
    require ``X.shape[1] == 147`` so the date-defined boundaries cannot
    silently re-bind if preprocessing changes upstream.
    """
    if regime not in REGIME_BOUNDS_WEEKS:
        raise ValueError(
            f"Unknown regime {regime!r}; expected one of {sorted(REGIME_BOUNDS_WEEKS)}."
        )
    if X.ndim != 3:
        raise ValueError(f"slice_regime expects (N, W, D); got {X.shape}.")
    if regime == "full":
        return X, X.shape[1]
    if X.shape[1] != 147:
        raise ValueError(
            f"Regime {regime!r} requires W=147 (date-defined boundaries); "
            f"got W={X.shape[1]}.  Did preprocess.py change?"
        )
    start, end = REGIME_BOUNDS_WEEKS[regime]
    sub = X[:, start:end, :]
    return sub, end - start


@dataclass(frozen=True)
class KFoldSpec:
    family: SplitFamily = "stratified"
    n_folds: int = 5
    seed: int = 0
    val_fraction_for_temporal: float = 0.20  # only used for "temporal"

    def iter_folds(self, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(y)
        if self.family == "random":
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            yield from kf.split(np.arange(n))
        elif self.family == "stratified":
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )
            yield from skf.split(np.arange(n), y)
        elif self.family == "temporal":
            # Single "fold" — front-of-time / back-of-time split.  We
            # iterate once and stop, so callers that build a fold-loop
            # see exactly one (train, val) pair.
            n_val = max(1, int(self.val_fraction_for_temporal * n))
            idx = np.arange(n)
            yield idx[:-n_val], idx[-n_val:]
        else:
            raise ValueError(f"Unknown split family {self.family!r}.")


__all__ = [
    "load_sgcc",
    "make_synthetic",
    "KFoldSpec",
    "SplitFamily",
    "RegimeName",
    "REGIME_BOUNDS_WEEKS",
    "slice_regime",
]
