r"""Multi-class isotonic recalibration (per-class one-vs-rest + renormalise).

Procedure (Zadrozny & Elkan 2002, multi-class extension):

    For each class k = 0..K-1 fit an isotonic regression
        f_k : p_k → \hat p_k
    on the val slice using one-vs-rest binary labels.  At inference,
    stack the per-class predictions and renormalise to a probability
    simplex.

This is a simple post-hoc calibrator with no shared parameters between
classes; sklearn's IsotonicRegression handles ties and boundary cases.
Falls back to identity if a class has no positive val samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False


@dataclass
class IsotonicCalibrator:
    """Per-class isotonic regression + softmax renormalisation."""

    n_classes: int = 5
    _fits: List[object] = field(default_factory=list)

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        if not _SKLEARN_OK:
            raise ImportError("IsotonicCalibrator requires scikit-learn")
        p = np.asarray(probs_val, dtype=np.float64)
        y = np.asarray(y_val, dtype=np.int64)
        if p.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {p.shape[1]} columns; expected {self.n_classes}")

        self._fits = []
        for k in range(self.n_classes):
            yk = (y == k).astype(np.float64)
            if yk.min() == yk.max():
                # All same value — identity calibrator suffices.
                self._fits.append(None)
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p[:, k], yk)
            self._fits.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fits:
            raise RuntimeError("IsotonicCalibrator.fit must be called first")
        p = np.asarray(probs, dtype=np.float64)
        out = np.zeros_like(p)
        for k in range(self.n_classes):
            iso = self._fits[k]
            out[:, k] = iso.predict(p[:, k]) if iso is not None else p[:, k]
        # Renormalise to a simplex.  Rows that collapse to all-zero are split
        # uniformly so the result is still a probability distribution.
        s = out.sum(axis=1, keepdims=True)
        zero_rows = (s == 0).squeeze(-1)
        out[zero_rows] = 1.0 / self.n_classes
        s[zero_rows] = 1.0
        return out / s


__all__ = ["IsotonicCalibrator"]
