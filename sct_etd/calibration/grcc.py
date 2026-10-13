"""GRCC — Group-Regime Conditional Conformal prediction sets.

Application of Gibbs, Cherian & Candès (2023), "Conformal Prediction
With Conditional Guarantees" (JRSS-B; arXiv:2305.12616) to electricity-
theft detection.

The Gibbs-Cherian-Candès framework picks a finite class of test
functions F ⊂ L^2 and produces a prediction set Ĉ_α satisfying

    E[ f(X) · (1{Y ∈ Ĉ_α(X)} − (1 − α)) ] = 0    for all f ∈ F.

When F is the linear span of indicators {1_{S_i}} for a partition
S_1, …, S_k of the covariate space, the GCC condition collapses to
**per-stratum** marginal coverage on each S_i.  GRCC implements that
discrete-partition special case for the SCT-ETD stratum covariate

    Z = (regime, group_cap, audit_tier).

Proposition 2 (per-stratum coverage)
------------------------------------
Let (X_i, Y_i, Z_i) be exchangeable within each stratum
{i : Z_i = S} (group-conformal exchangeability inside the stratum;
holds by construction because Z includes the cap'd group id).  Let
n_S = |{i : Z_i = S, i ≤ n}| be the per-stratum calibration count and

    τ̂_S = ⌈(n_S + 1)(1 − α)⌉ / n_S -th order statistic of {s_i : Z_i = S}.

Then for any stratum S with n_S ≥ 1,

    Pr( Y_{n+1} ∈ Ĉ_α(X_{n+1}) | Z_{n+1} = S ) ≥ 1 − α − 1/(n_S + 1).

Below a configurable ``min_stratum_size`` (default 30) GRCC falls back to
the marginal APS threshold for that stratum — and the fallback is
recorded so the audit can report which strata degraded.

Inputs
------
* val_probs : (N_val, K) softmax probabilities on the val slice.
* val_labels: (N_val,) int labels.
* val_strata: (N_val,) integer stratum ids, or a Python sequence of
  hashables (we convert to dense ids internally).

Outputs
-------
* ``predict_sets_conditional(test_probs, test_strata)`` → list[set[int]]
* ``predict_sets_indicator(test_probs, test_strata)`` → (N, K) bool array

Stratum id assignment for SGCC
------------------------------
See :func:`build_strata` for the recommended encoding of
(regime × group_cap × audit_tier).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from sct_etd.calibration.conformal_aps import _conformal_quantile, aps_scores


def _densify(stratum_ids: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary hashables → dense int ids; return (ids, lookup_keys)."""
    arr = np.asarray(stratum_ids)
    if arr.dtype.kind in ("i", "u"):
        keys, dense = np.unique(arr, return_inverse=True)
    else:
        keys, dense = np.unique(arr.astype(object), return_inverse=True)
    return dense.astype(np.int64), keys


def build_strata(regime: np.ndarray,
                  group_cap: np.ndarray,
                  audit_tier: np.ndarray | None = None,
                  ) -> np.ndarray:
    """Stack (regime, group_cap, audit_tier) into a single per-row int id.

    Parameters
    ----------
    regime     : (N,) int in {0, 1} — pre/post 2016-01 majority for each consumer.
    group_cap  : (N,) int in {1, 2+} — 1 = singleton after cap; 2 = cohort.
                  We accept any positive integer and bin into {1, 2}.
    audit_tier : optional (N,) int in {1, 2, 3} — audit confidence.

    Returns
    -------
    (N,) int64 array of dense stratum ids.
    """
    n = len(regime)
    r = np.asarray(regime, dtype=np.int64).reshape(-1)
    g = (np.asarray(group_cap, dtype=np.int64).reshape(-1) >= 2).astype(np.int64)
    if audit_tier is None:
        t = np.zeros(n, dtype=np.int64)
    else:
        t = np.asarray(audit_tier, dtype=np.int64).reshape(-1)
    # Composite key.
    composite = r * 1000 + g * 100 + t
    dense, _keys = _densify(composite)
    return dense


@dataclass
class GRCCalibrator:
    """Group-Regime Conditional Conformal calibrator.

    Parameters
    ----------
    alpha : float, default 0.10 (target 90 % coverage; D3).
    min_stratum_size : int, default 30.
        Strata with fewer val rows fall back to the marginal APS τ̂.
    """

    alpha: float = 0.10
    min_stratum_size: int = 30

    # Fit state
    tau_per_stratum_: dict[int, float] = field(default_factory=dict)
    tau_marginal_: float | None = None
    n_per_stratum_: dict[int, int] = field(default_factory=dict)
    fallback_strata_: set[int] = field(default_factory=set)

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray,
            strata_val: np.ndarray) -> "GRCCalibrator":
        scores = aps_scores(probs_val, y_val)
        self.tau_marginal_ = _conformal_quantile(scores, self.alpha)
        s_ids = np.asarray(strata_val, dtype=np.int64)
        self.tau_per_stratum_ = {}
        self.n_per_stratum_ = {}
        self.fallback_strata_ = set()
        for sid in np.unique(s_ids):
            sel = s_ids == sid
            n_s = int(sel.sum())
            self.n_per_stratum_[int(sid)] = n_s
            if n_s < self.min_stratum_size:
                self.tau_per_stratum_[int(sid)] = self.tau_marginal_
                self.fallback_strata_.add(int(sid))
            else:
                self.tau_per_stratum_[int(sid)] = _conformal_quantile(
                    scores[sel], self.alpha)
        return self

    def _tau_vector(self, strata: np.ndarray) -> np.ndarray:
        if self.tau_marginal_ is None:
            raise RuntimeError("GRCCalibrator.fit must be called first")
        s_ids = np.asarray(strata, dtype=np.int64)
        out = np.empty(s_ids.shape, dtype=np.float64)
        for i, sid in enumerate(s_ids):
            out[i] = self.tau_per_stratum_.get(int(sid), self.tau_marginal_)
        return out

    def predict_sets_indicator(self, probs_test: np.ndarray,
                                 strata_test: np.ndarray) -> np.ndarray:
        scores_per_class = aps_scores(probs_test, labels=None)   # (N, K)
        tau = self._tau_vector(strata_test)[:, None]              # (N, 1)
        return scores_per_class <= tau

    def predict_sets_conditional(self, probs_test: np.ndarray,
                                   strata_test: np.ndarray) -> list[set[int]]:
        included = self.predict_sets_indicator(probs_test, strata_test)
        return [set(np.where(row)[0].tolist()) for row in included]

    def degraded_strata(self) -> list[tuple[int, int]]:
        """List (stratum_id, n_cal) pairs that fell back to marginal."""
        return sorted((s, self.n_per_stratum_.get(s, 0))
                      for s in self.fallback_strata_)


__all__ = ["GRCCalibrator", "build_strata"]
