"""Phase K/L — Physics-Simulation Theft Augmentation.

Generates synthetic theft examples from real normal customers by applying
physically-grounded manipulation rules. Each manipulation corresponds to one
of the **five** theft strategies identified in
``research_outputs/tables/06_theft_4cluster_topology.csv``.

Empirical mix (Phase L):

    Heavy_Dropper  17.7 % — X[T:] *= (1 − f),  f ∈ U(0.55, 0.85)   [sharp ↓]
    Light_Dropper  17.6 % — X[T:] *= (1 − f),  f ∈ U(0.20, 0.50)   [moderate ↓]
    Flat           16.5 % — X[T:] = clip(low_const + N(0, σ), 0, 1) [pinned]
    Light_Riser    24.1 % — X[T:] *= (1 + f),  f ∈ U(0.10, 0.30)   [moderate ↑]
    Heavy_Riser    24.1 % — X[T:] *= (1 + f),  f ∈ U(0.50, 2.00)   [sharp ↑]

Phase K (a18_physsim_aug, deprecated): Dropper + Flat only. Missed 48 % Risers.
Phase L (a22_sym_aug):              Symmetric — covers all five archetypes.

Design notes
------------
* NaN positions are preserved bit-for-bit: the manipulation is multiplicative
  (Light/Heavy) or replacement-at-observed (Flat). NaN cells are skipped so
  the mask channel sees the same NaN signature as a real series of the same
  customer. This prevents any synthetic-missingness artefact bleeding into
  the A7 mask CNN branch.
* Operates on normalised consumption in [0, 1] (post min-max). Manipulations
  remain in [0, 1].
* Strictly fold-safe: caller passes only the training-fold rows; no leakage
  into validation / calibration / test cells.
* Deterministic for a given numpy Generator.

Public API
----------
* ``synthesize_theft_examples(X_normal, rng, archetype_weights=...)``
* ``augment_training_set(X_train, y_train, rng, p_aug=0.40, ...)``
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Archetype manipulations
# ---------------------------------------------------------------------------

def _apply_light_dropper(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative step-down by 20–50 % from a random week T.

    x shape: (W, D); operates in-place on a copy.
    NaN positions are preserved (multiplicative manipulation on NaN = NaN).
    """
    W = x.shape[0]
    t_start = int(rng.integers(20, max(21, W - 20)))
    f = float(rng.uniform(0.20, 0.50))
    x = x.copy()
    x[t_start:] = x[t_start:] * (1.0 - f)
    return x


def _apply_heavy_dropper(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sharp multiplicative step-down by 55–85 %."""
    W = x.shape[0]
    t_start = int(rng.integers(20, max(21, W - 20)))
    f = float(rng.uniform(0.55, 0.85))
    x = x.copy()
    x[t_start:] = x[t_start:] * (1.0 - f)
    return x


def _apply_flat_stealer(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace observed cells from week T with low constant + small noise.

    NaN cells are left untouched so the mask signature is preserved.
    Result is clipped to [0, 1] to remain in the normalised range.
    """
    W, D = x.shape
    t_start = int(rng.integers(20, max(21, W - 20)))
    low_const = float(rng.uniform(0.02, 0.20))
    sigma = 0.03
    x = x.copy()
    nan_mask = np.isnan(x[t_start:])
    # Sample noise for every cell, then write only where observed.
    noise = rng.standard_normal(size=(W - t_start, D)).astype(x.dtype) * sigma
    replacement = np.clip(low_const + noise, 0.0, 1.0)
    # Keep NaN positions; replace observed positions with low constant.
    block = np.where(nan_mask, x[t_start:], replacement)
    x[t_start:] = block
    return x


def _apply_light_riser(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative step-UP by 10–30 % from a random week T (Light_Riser, 24.1 % of theft).

    Light Risers exhibit moderate consumption increases — e.g. residential
    meter-spoofing that reads higher than actual, or cross-billing fraud.
    The increase is multiplicative and clipped to [0, 1] (preprocessed range).
    """
    W = x.shape[0]
    t_start = int(rng.integers(20, max(21, W - 20)))
    f = float(rng.uniform(0.10, 0.30))
    x = x.copy()
    nan_mask = np.isnan(x[t_start:])
    boosted = np.clip(x[t_start:] * (1.0 + f), 0.0, 1.0)
    x[t_start:] = np.where(nan_mask, x[t_start:], boosted)
    return x


def _apply_heavy_riser(
    x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative step-UP by 50–200 % (Heavy_Riser, 24.1 % of theft).

    Heavy Risers typically map to commercial / industrial customers whose
    metered consumption is augmented by an illegal bypass *adding* load on
    top of their already-large baseline. The increase factor can reach 3×.
    Result is clipped to [0, 1] (we operate post-min-max normalisation).
    """
    W = x.shape[0]
    t_start = int(rng.integers(20, max(21, W - 20)))
    f = float(rng.uniform(0.50, 2.00))
    x = x.copy()
    nan_mask = np.isnan(x[t_start:])
    boosted = np.clip(x[t_start:] * (1.0 + f), 0.0, 1.0)
    x[t_start:] = np.where(nan_mask, x[t_start:], boosted)
    return x


_ARCHETYPE_FNS = {
    # Phase K legacy keys (preserved for backwards compatibility with A18)
    "light":  _apply_light_dropper,
    "heavy":  _apply_heavy_dropper,
    "flat":   _apply_flat_stealer,

    # Phase L symmetric keys — explicit names matching the 5-cluster topology
    "heavy_dropper":  _apply_heavy_dropper,
    "light_dropper":  _apply_light_dropper,
    "flat_stealer":   _apply_flat_stealer,
    "light_riser":    _apply_light_riser,
    "heavy_riser":    _apply_heavy_riser,
}


#: Phase L empirical theft mix (matches research_outputs/tables/06_theft_4cluster_topology.csv).
SYMMETRIC_ARCHETYPE_WEIGHTS: dict[str, float] = {
    "heavy_dropper": 0.177,
    "light_dropper": 0.176,
    "flat_stealer":  0.165,
    "light_riser":   0.241,
    "heavy_riser":   0.241,
}


def synthesize_theft_examples(
    X_normal: np.ndarray,
    rng: np.random.Generator,
    archetype_weights: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a randomly-sampled theft archetype to every customer in X_normal.

    Parameters
    ----------
    X_normal : (M, W, D) float32 — normal customer windows (y=0). NaN allowed.
    rng      : numpy Generator (deterministic given seed).
    archetype_weights : optional dict mapping archetype name → probability mass.
        Default mirrors the empirical theft archetype distribution conditional
        on a customer being a thief (Light 48.3 %, Heavy 17.7 %, Flat 16.5 %,
        Other 17.5 %); we collapse "Other" into a 50/50 mix of Light/Heavy
        since we have no separate generative model for the residual class.

        Effective default proportions: Light=0.57, Heavy=0.26, Flat=0.17.

    Returns
    -------
    X_synth : (M, W, D) float32 — synthetic-theft variants of each input row.
    y_synth : (M,)      int64   — all 1 (theft).
    """
    if X_normal.ndim != 3:
        raise ValueError(f"X_normal must be (M, W, D); got {X_normal.shape}")
    if archetype_weights is None:
        # Empirical theft mix with "Other" collapsed into Light/Heavy.
        archetype_weights = {
            "light":  0.483 + 0.5 * 0.175,   # 0.5705
            "heavy":  0.177 + 0.5 * 0.175,   # 0.2645
            "flat":   0.165,                  # 0.1650
        }
    names   = list(archetype_weights.keys())
    weights = np.asarray([archetype_weights[k] for k in names], dtype=np.float64)
    weights = weights / weights.sum()

    M = X_normal.shape[0]
    pick = rng.choice(len(names), size=M, replace=True, p=weights)

    out = np.empty_like(X_normal)
    for i in range(M):
        out[i] = _ARCHETYPE_FNS[names[pick[i]]](X_normal[i], rng)
    return out.astype(np.float32, copy=False), np.ones(M, dtype=np.int64)


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
    p_aug: float = 0.40,
    archetype_weights: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Append synthetic theft examples to a training set.

    Strategy: for each normal customer (y=0), with probability ``p_aug``
    derive ONE synthetic-theft variant labelled y=1 and append it to the
    training set. Real labels and rows are preserved unchanged.

    Effect on class balance (SGCC theft rate 8.5 %, p_aug=0.40):
        Original normals ≈ 91.5 % · N
        Original thefts  ≈  8.5 % · N
        Synthetic theft  ≈ p_aug · 91.5 % · N  =  36.6 % · N
        New theft frac   ≈ (0.085 + 0.366) / (1 + 0.366)  ≈ 33.0 %

    Parameters
    ----------
    X_train, y_train : (N, W, D) float32 and (N,) int64.
    rng              : numpy Generator.
    p_aug            : probability of synthesising a theft variant per normal row.
    archetype_weights: see ``synthesize_theft_examples``.

    Returns
    -------
    X_aug, y_aug : concatenated arrays (real + synthetic) along axis 0.
        Order: real rows first, synthetic rows appended.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train have mismatched first dim")
    if not 0.0 < p_aug <= 1.0:
        raise ValueError(f"p_aug must be in (0, 1]; got {p_aug}")

    normal_idx = np.where(y_train == 0)[0]
    if len(normal_idx) == 0:
        return X_train, y_train  # no normals to augment from

    n_to_synth = rng.binomial(n=len(normal_idx), p=p_aug)
    if n_to_synth == 0:
        return X_train, y_train

    source_idx = rng.choice(normal_idx, size=n_to_synth, replace=False)
    X_synth, y_synth = synthesize_theft_examples(
        X_train[source_idx], rng, archetype_weights=archetype_weights
    )

    X_out = np.concatenate([X_train, X_synth], axis=0)
    y_out = np.concatenate([y_train, y_synth], axis=0)
    return X_out, y_out


__all__ = [
    "synthesize_theft_examples",
    "augment_training_set",
]
