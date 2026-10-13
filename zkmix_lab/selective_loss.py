"""SelectiveNet composite loss for binary classification.

Reference
---------
Geifman & El-Yaniv, *SelectiveNet: A Deep Neural Network with an
Integrated Reject Option*, ICML 2019.

Composite loss (binary specialisation)
--------------------------------------
For a batch with logits ``ℓ``, selector gates ``g ∈ [0, 1]``, and binary
labels ``y ∈ {0, 1}``::

    L_sel(ℓ, g, y) = E[g · BCE(ℓ, y)] / E[g]
                     + λ_cov · max(0, c_target − E[g]) ** 2
    L_aux(ℓ, y)    = E[BCE(ℓ, y)]
    L_total        = α · L_sel + β · L_aux

The hinge term penalises coverage falling below the target ``c_target``;
the quadratic penalty (∝ λ_cov) is the SelectiveNet recipe.  Without
the auxiliary BCE, the model is free to drive g → 0 on examples it
cannot classify confidently — the auxiliary keeps every example's
classifier loss in the gradient signal.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def selective_loss(
    logit: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
    coverage_target: float = 0.90,
    alpha_sel: float = 0.5,
    beta_aux: float = 0.5,
    lambda_cov: float = 32.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite SelectiveNet loss + diagnostic dict.

    Parameters
    ----------
    logit : (N,) classifier logits (pre-sigmoid).
    gate  : (N,) selector probabilities in [0, 1].
    y     : (N,) float labels in {0, 1}.

    Returns
    -------
    loss  : scalar tensor (with grad).
    info  : dict with float metrics — selective_bce, aux_bce, coverage,
            coverage_hinge — for logging.
    """
    if logit.shape != y.shape or logit.shape != gate.shape:
        raise ValueError(
            f"logit/gate/y must all be (N,); got {tuple(logit.shape)}, "
            f"{tuple(gate.shape)}, {tuple(y.shape)}."
        )
    if not (0.0 < coverage_target < 1.0):
        raise ValueError("coverage_target must be in (0, 1).")

    bce_per = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    coverage = gate.mean()
    # Avoid division-by-zero on a pathological all-zero gate batch.
    selective_bce = (gate * bce_per).sum() / (coverage * gate.numel() + eps)
    coverage_hinge = torch.clamp(coverage_target - coverage, min=0.0) ** 2
    L_sel = selective_bce + lambda_cov * coverage_hinge
    L_aux = bce_per.mean()
    loss = alpha_sel * L_sel + beta_aux * L_aux
    info = {
        "selective_bce": float(selective_bce.detach()),
        "aux_bce": float(L_aux.detach()),
        "coverage": float(coverage.detach()),
        "coverage_hinge": float(coverage_hinge.detach()),
    }
    return loss, info


__all__ = ["selective_loss"]
