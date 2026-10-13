"""Post-hoc calibration utilities — prior correction + temperature scaling.

Why this module exists
----------------------
The 540-cell run (2026-05-17) revealed two structural calibration
problems shared by every backbone:

  (1) ``BinaryStratifiedSampler`` trains every model on 50 / 50 balanced
      batches, but the deployment prior on SGCC is ≈ 8.5 % theft.  The
      model learns ``p_train(y | x)`` under a 0.5 prior, not the deploy
      prior.  At inference the sigmoid outputs are systematically too
      high.  ECE ≈ 0.20 – 0.35 across all backbones, APS τ̂ saturates at
      1.0, Mondrian saturates, SelectiveNet gate ≡ 1.0 — every downstream
      trustworthy-AI artefact degenerates from this single root cause.

  (2) Residual miscalibration not explained by (1) — different
      architectures end up at different over-/under-confidence levels.

The fixes are textbook and *post-hoc*:

  - **Prior correction** (Saerens, Latinne & Decaestecker 2002;
    Menon et al. 2021 "Long-Tail Learning via Logit Adjustment", ICLR):
    shift the logit by the log-ratio of priors so the predicted
    probabilities lie on the deploy distribution.  Closed-form; no fit.

  - **Temperature scaling** (Guo, Pleiss, Sun & Weinberger 2017,
    "On Calibration of Modern Neural Networks", ICML): fit a single
    scalar ``T`` on a held-out calibration slice such that ``σ(z / T)``
    minimises NLL on that slice.  Monotone in z so PR-AUC / AUROC are
    preserved exactly; ECE goes down.

Order of operations
-------------------
1. Train with balanced sampler + BCE (or focal) — produces ``z_train``.
2. Prior-correct: ``z_corrected = z_train + log(π_deploy / (1 − π_deploy)) − log(π_train / (1 − π_train))``.
   For the standard balanced sampler (π_train = 0.5) this simplifies to
   ``z_corrected = z_train + log(π_deploy / (1 − π_deploy))``.
3. Fit ``T`` on ``(z_corrected_cal, y_cal)`` and apply to test logits.

Both transforms commute with sigmoid in the sense that they are monotone
on the logit axis, so ranking metrics are invariant under them.  ECE,
APS, Mondrian, SelectiveNet all use the *probability* — they are not
invariant — and these are exactly the metrics we want to fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Logit / sigmoid plumbing
# ---------------------------------------------------------------------------


def _stable_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Numerically stable logit ``log(p / (1 - p))``."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


# ---------------------------------------------------------------------------
# Prior correction (closed form)
# ---------------------------------------------------------------------------


def prior_shift(p_train: float, p_deploy: float) -> float:
    """Constant ``Δ`` to ADD to training logits so they live on the deploy prior.

    Derivation (Bayes ratio, assuming p(x | y) is the same in both regimes):

        z_deploy − z_train
            = log(π_deploy / (1 − π_deploy))  −  log(π_train / (1 − π_train))

    Numerical sanity:
        p_train = 0.5, p_deploy = 0.085  →  Δ ≈ −2.376
        (a balanced-sampler-trained model's logits get pushed DOWN by
         ≈ 2.4 nats to match the 8.5 % deploy prior — i.e. ``σ(0) = 0.5``
         becomes ``σ(−2.376) ≈ 0.085`` as required).
    """
    if not (0.0 < p_train < 1.0):
        raise ValueError(f"p_train must be in (0, 1); got {p_train}.")
    if not (0.0 < p_deploy < 1.0):
        raise ValueError(f"p_deploy must be in (0, 1); got {p_deploy}.")
    return math.log(p_deploy / (1.0 - p_deploy)) - math.log(p_train / (1.0 - p_train))


def prior_correct_logits(
    logits: np.ndarray, p_train: float = 0.5, p_deploy: float = 0.085,
) -> np.ndarray:
    """Add the prior-shift constant to every logit.  Vectorised."""
    return np.asarray(logits, dtype=np.float64) + prior_shift(p_train, p_deploy)


def prior_correct_scores(
    scores: np.ndarray, p_train: float = 0.5, p_deploy: float = 0.085,
) -> np.ndarray:
    """Scores in / scores out — go via logits for numerical stability."""
    z = _stable_logit(np.asarray(scores, dtype=np.float64))
    return _sigmoid(prior_correct_logits(z, p_train=p_train, p_deploy=p_deploy))


# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemperatureResult:
    """Outcome of fitting a single-scalar temperature on a cal slice."""
    T: float
    nll_before: float
    nll_after: float
    n_cal: int


def _binary_nll(logits: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Numerically stable binary NLL on raw logits.

    NLL = mean( -log σ(z) ·    y    + -log(1 − σ(z)) · (1 − y) )
        = mean( softplus(-z)·y + softplus(z)·(1 − y) )
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    # softplus(x) = log(1 + exp(x)); stable via max(x, 0) form.
    sp_neg = np.logaddexp(0.0, -z)   # softplus(-z)
    sp_pos = np.logaddexp(0.0,  z)   # softplus( z)
    return float((y * sp_neg + (1.0 - y) * sp_pos).mean())


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray,
    T_bracket: tuple[float, float] = (0.01, 10.0),
) -> TemperatureResult:
    """Find ``T > 0`` minimising binary NLL of ``σ(z / T)`` on a cal slice.

    Uses Brent's method on the bracket (default [0.01, 10]).  The
    optimum's existence and uniqueness follow from the strict convexity
    of NLL in ``1/T``.  Returns ``T = 1`` if cal slice is degenerate
    (all one class) since NLL is then minimised by any T.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.shape != y.shape or z.ndim != 1:
        raise ValueError("logits and labels must be 1-D arrays of equal length.")
    n = int(z.shape[0])
    if n == 0:
        raise ValueError("Empty calibration slice.")
    n_pos = int(y.sum())
    nll_before = _binary_nll(z, y)
    # Degenerate cal slice — temperature has no effect on a one-class NLL.
    if n_pos == 0 or n_pos == n:
        return TemperatureResult(T=1.0, nll_before=nll_before, nll_after=nll_before, n_cal=n)

    def obj(T: float) -> float:
        return _binary_nll(z / T, y) if T > 0 else float("inf")

    # Brent's method needs a valid 3-point bracket where f(middle) < f(both
    # ends).  On near-flat NLL (degenerate / very small / very clean cal
    # slices) the bracketer can fail.  Fall back to bounded search.
    try:
        res = minimize_scalar(obj, bracket=(T_bracket[0], 1.0, T_bracket[1]), method="brent")
        T_star = float(res.x)
    except (ValueError, RuntimeError):
        res = minimize_scalar(
            obj, bounds=(T_bracket[0], T_bracket[1]), method="bounded",
            options={"xatol": 1e-4},
        )
        T_star = float(res.x)
    # Guard against pathological optimiser output.
    if not (T_bracket[0] <= T_star <= T_bracket[1]) or not np.isfinite(T_star):
        T_star = 1.0
    return TemperatureResult(
        T=T_star,
        nll_before=nll_before,
        nll_after=_binary_nll(z / T_star, y),
        n_cal=n,
    )


def apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    """Apply ``T`` to *score-space* inputs by routing through logits."""
    if T <= 0:
        raise ValueError(f"T must be positive; got {T}.")
    z = _stable_logit(np.asarray(scores, dtype=np.float64))
    return _sigmoid(z / T)


# ---------------------------------------------------------------------------
# Combined helper — what the runner / analyser actually calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationOutcome:
    """End-to-end calibration result for one cell.

    ``test_scores_calibrated`` is the prior-corrected, temperature-scaled
    probability for each test example — what every downstream
    trustworthy-AI artefact (ECE, APS, Mondrian, SelectiveNet) should
    consume instead of the raw sigmoid output.
    """
    prior_shift: float
    T: float
    nll_cal_before: float
    nll_cal_after: float
    cal_scores_calibrated: np.ndarray
    test_scores_calibrated: np.ndarray


def calibrate_cell(
    cal_scores: np.ndarray, cal_labels: np.ndarray,
    test_scores: np.ndarray,
    p_train: float = 0.5, p_deploy: float = 0.085,
) -> CalibrationOutcome:
    """Full calibration pipeline for one (cal, test) pair.

    Steps
    -----
    1. Prior-correct the cal logits and the test logits by the same constant Δ.
    2. Fit ``T`` on the prior-corrected cal logits + labels.
    3. Apply ``T`` to both cal and test.

    The two-step (Δ, T) factorisation is intentional — Δ is a known
    closed-form constant that fixes the prior mismatch, T then cleans up
    residual architecture-dependent miscalibration.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    cal_labels = np.asarray(cal_labels, dtype=np.float64)

    shift = prior_shift(p_train, p_deploy)
    z_cal_pc  = _stable_logit(cal_scores)  + shift
    z_test_pc = _stable_logit(test_scores) + shift

    tr = fit_temperature(z_cal_pc, cal_labels)

    cal_cal  = _sigmoid(z_cal_pc  / tr.T)
    test_cal = _sigmoid(z_test_pc / tr.T)

    return CalibrationOutcome(
        prior_shift=shift,
        T=tr.T,
        nll_cal_before=tr.nll_before,
        nll_cal_after=tr.nll_after,
        cal_scores_calibrated=cal_cal,
        test_scores_calibrated=test_cal,
    )


__all__ = [
    "prior_shift",
    "prior_correct_logits",
    "prior_correct_scores",
    "fit_temperature",
    "apply_temperature",
    "calibrate_cell",
    "TemperatureResult",
    "CalibrationOutcome",
]
