"""FDI (False Data Injection) synthetic theft attacks for MSDM reproduction.

The cross-scenario datasets (ISET/LCL/ECL) carry no native theft labels; MSDM
(paper §V-A, ref [23] Zanetti 2019) injects synthetic anomalies. We implement the
standard 6-attack family (Jokar 2016 / Nabil 2019 h1–h6), the convention used across
the ETD synthetic-attack literature. [ASSUMPTION] exact parameterization follows the
widely-used h1–h6 forms; adjust here if a specific Zanetti variant is required.

All attacks operate on OBSERVED entries only (NaN-aware) so missingness structure is
preserved. A contiguous window of `corrupt_days` (paper: 1–6 days per ET sample) may
optionally be targeted; by default the whole series is attacked.
"""
from __future__ import annotations
import numpy as np


def _obs(x):
    return ~np.isnan(x)


def fdi1(x, rng):  # multiplicative constant scaling
    a = rng.uniform(0.2, 0.8)
    return np.where(_obs(x), a * x, x)


def fdi2(x, rng):  # per-timestep random scaling
    b = rng.uniform(0.2, 0.8, size=x.shape)
    return np.where(_obs(x), b * x, x)


def fdi3(x, rng):  # intermittent on/off (random zeroing)
    on = rng.random(x.shape) < rng.uniform(0.3, 0.7)
    return np.where(_obs(x), np.where(on, x, 0.0), x)


def fdi4(x, rng):  # scaled mean replacement (per-timestep)
    m = np.nanmean(x) if _obs(x).any() else 0.0
    a = rng.uniform(0.2, 0.8, size=x.shape)
    return np.where(_obs(x), a * m, x)


def fdi5(x, rng):  # flatten to mean
    m = np.nanmean(x) if _obs(x).any() else 0.0
    return np.where(_obs(x), m, x)


def fdi6(x, rng):  # time reversal
    xr = x[::-1].copy()
    return np.where(_obs(x), np.where(_obs(xr), xr, np.nanmean(x) if _obs(x).any() else 0.0), x)


ATTACKS = [fdi1, fdi2, fdi3, fdi4, fdi5, fdi6]


def inject(X, rng, anomaly_frac=0.10, attack_ids=None):
    """Inject FDI theft into a fraction of customers.

    X : (N, T) float (NaN allowed). Returns (X_out, y) with y=1 for attacked rows.
    Each attacked customer gets a uniformly-random attack from `attack_ids` (default all 6).
    Reproduces paper's 10% anomaly proportion.
    """
    N = X.shape[0]
    attack_ids = attack_ids if attack_ids is not None else list(range(6))
    y = np.zeros(N, dtype=np.int64)
    Xo = X.copy()
    n_an = int(round(anomaly_frac * N))
    idx = rng.choice(N, size=n_an, replace=False)
    for i in idx:
        atk = ATTACKS[rng.choice(attack_ids)]
        Xo[i] = atk(X[i], rng)
        y[i] = 1
    return Xo, y


__all__ = ["ATTACKS", "inject", "fdi1", "fdi2", "fdi3", "fdi4", "fdi5", "fdi6"]
