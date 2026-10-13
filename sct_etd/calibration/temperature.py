"""Temperature scaling for multi-class probabilities.

Guo et al. (2017), "On Calibration of Modern Neural Networks" — fit a
single scalar T > 0 on the val slice by minimising NLL over softmax(z / T)
where z are logits.  Cheap, well-understood, and works as a pre-calibrator
fed into APS / GRCC.

Inputs are logits, not probabilities — if you only have probabilities,
recover logits as ``log(p) - log(p).mean(axis=1, keepdims=True)`` (clip
zeros to a small epsilon first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    p = _softmax(logits / max(T, 1e-6))
    eps = 1e-12
    return float(-np.log(p[np.arange(len(labels)), labels] + eps).mean())


@dataclass
class TemperatureCalibrator:
    """Fit a single scalar temperature on val logits + labels.

    Parameters
    ----------
    T_init : float, default 1.0.  Initial value (>0).
    T_min, T_max : float bounds for the grid search.
    n_grid : int, default 41.  Resolution of the log-space grid.
    """

    T_init: float = 1.0
    T_min: float = 0.1
    T_max: float = 10.0
    n_grid: int = 41
    T_: float | None = None

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray) -> "TemperatureCalibrator":
        """Grid search over log-spaced T; pick the NLL minimiser."""
        z = np.asarray(logits_val, dtype=np.float64)
        y = np.asarray(y_val, dtype=np.int64)
        if z.ndim != 2:
            raise ValueError(f"logits must be 2D (N, K); got {z.shape}")
        if y.shape != (z.shape[0],):
            raise ValueError(f"labels shape {y.shape} != expected ({z.shape[0]},)")
        grid = np.geomspace(self.T_min, self.T_max, self.n_grid)
        losses = np.array([_nll(T, z, y) for T in grid])
        self.T_ = float(grid[int(np.argmin(losses))])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax(z / T̂).  Returns (N, K) probabilities."""
        if self.T_ is None:
            raise RuntimeError("TemperatureCalibrator.fit must be called first")
        return _softmax(np.asarray(logits, dtype=np.float64) / self.T_)


__all__ = ["TemperatureCalibrator"]
