"""Context-Conditional NaME (CC-NaME) — Phase D.

Theoretical basis
-----------------
Phase C NaME learns a globally optimal scalar τ* that minimises the expected
classification loss across all customers:

    τ* = argmin_τ  E_{n,x}[ L(f(x; τ), y_n) ]

This is a first-order approximation. When the MNAR distribution is
heterogeneous — as it is in SGCC, where five structurally distinct theft
archetypes (Heavy Dropper, Light Dropper, Heavy Riser, Light Riser, Flat)
each generate different missingness patterns — the locally optimal token
τ_n* varies across customers:

    τ_n* = argmin_τ  E_{x∼P_n}[ L(f_n(x; τ), y_n) ]

The constraint τ_n* = τ* for all n is violated iff Var[τ_n*] > 0 across the
population.  Given that Heavy Droppers operate at ~10× higher consumption
baselines than Flat stealers, this variance is structurally non-zero.

CC-NaME relaxes the constraint by learning a lightweight context encoder
g: φ(x_valid,n) → δτ_n, so that:

    τ_n = τ_base + δτ_n = τ_base + g(φ(x_valid,n); θ_g)

Key properties
--------------
1.  **No hardcoded dimensions.**  The encoder operates on DOW-aggregated
    statistics (per-day-of-week valid mean, std, missing fraction) — 3×D
    features — that are computed purely from the data.  The encoder
    discovers which DOW dimensions matter from the training gradient, not
    from manual feature engineering.

2.  **Scale invariance.**  All context features are normalised by the
    customer's own global valid std σ_g, making the encoder response
    invariant to absolute consumption scale.  This prevents the encoder
    from learning "high-consumption customers → certain τ" (spurious
    correlation) instead of "customers with this SHAPE of consumption
    profile → this imputation."

3.  **δτ is an implicit theft indicator.**  At convergence, the encoder
    learns that the optimal correction for theft customers deviates
    systematically from the optimal correction for honest customers.
    The δτ_n distribution therefore contains information about theft
    strategy — it is a latent representation of the MNAR mechanism.

4.  **Reduction to scalar NaME.**  If the encoder is disabled
    (``cc_enabled=False``) or its weights are zeroed, CC-NaME reduces
    exactly to the Phase C scalar NaME-NN.  This is verified by
    ``tests/test_cc_name.py::test_cc_name_reduces_to_scalar``.

5.  **Numerical contract at init.**  With ``base_token_init = -1.0`` and
    the encoder weights at their zero-init state (δτ_n = 0 for all n),
    CC-NaME is bit-identical to the Phase C NaME at initialisation.

Context feature construction
-----------------------------
For input x ∈ ℝ^{N×W×D} with NaN at missing positions:

    mask_{n,t,d}  = 1 if x_{n,t,d} is valid,  0 if NaN
    count_d,n     = Σ_t mask_{n,t,d}                   ∈ ℝ^{N×D}
    μ_dow,n       = (Σ_t x_{n,t,d} · mask) / count_d  ∈ ℝ^{N×D}
    μ_g,n         = (Σ_{t,d} x_{n,t,d} · mask) / (Σ_{t,d} mask)  ∈ ℝ^N
    σ_g,n         = √[Var_{valid}(x_n)]                ∈ ℝ^N

    c_mean_d,n    = (μ_dow,n − μ_g,n) / (σ_g,n + ε)   ∈ ℝ^{N×D}
    σ_dow,n       = std of valid cells per DOW / (σ_g,n + ε)  ∈ ℝ^{N×D}
    f_miss_d,n    = 1 − count_d,n / W                  ∈ ℝ^{N×D}

    context_n     = [c_mean; σ_dow; f_miss]  ∈ ℝ^{N × 3D}

All features are dimensionless.  The LayerNorm inside the encoder further
decorrelates them before the linear layers.

Edge cases
----------
- All cells missing (count_d = 0 for some d): that DOW's stats default to 0.
- All cells valid (no NaN): f_miss = 0, δτ → encoder(0-feature) ≈ 0, which
  is the correct limit (no adjustment needed when there is no missingness).
- σ_g = 0 (constant consumption across all time): clamp prevents division
  by zero; context collapses to [0,...,0] → encoder outputs small δτ ≈ 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """Two-layer MLP with LayerNorm that maps 3*D context features to a
    scalar token correction δτ ∈ ℝ.

    Architecture
    ------------
    Linear(3D, hidden) → LayerNorm(hidden) → ReLU → Dropout(p) → Linear(hidden, 1)

    The LayerNorm stabilises training when customers have very different
    consumption profiles (large variance in context feature magnitudes).

    The final linear layer is zero-initialised so that at model init
    δτ_n = 0 for all n — the model starts equivalent to scalar NaME.
    """

    def __init__(self, days: int = 7, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be >= 1.")
        in_dim = 3 * days
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Zero-init the output layer → δτ = 0 at init → identical to scalar NaME.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context : (N, 3*D) → δτ : (N, 1)."""
        return self.net(context)


class ContextConditionalNaME(nn.Module):
    """CC-NaME: instance-adaptive learnable missing-value embedding.

    Parameters
    ----------
    days : int
        Number of days per week in the input grid (D axis).  Default 7.
    hidden : int
        Hidden dim of the context encoder MLP.  Default 32.
    dropout : float
        Dropout probability inside the encoder.  Default 0.1.
    base_token_init : float
        Initial value of the global base token τ_base.  Set to the Phase C
        converged value (−1.094) for a warm start, or to −1.0 to match the
        Phase C init convention.  Default −1.0.
    cc_enabled : bool
        If False, the context encoder is bypassed (δτ = 0 always) and
        CC-NaME reduces to the scalar Phase C NaME.  Useful for the A4
        ablation and for verifying the reduction in tests.  Default True.
    nan_native : bool
        If True, missing cells are identified via ``torch.isnan(x)``
        (NaN-native path — no preprocessing sentinel required).
        If False, cells equal to ``sentinel_value`` are treated as missing.
        Default True (matching the Phase C NaN-native NaME-NN convention).
    sentinel_value : float
        Value used to mark missing cells in the non-NaN-native path.
        Ignored when nan_native=True.  Default −1.0.
    eps : float
        Numerical floor for σ_g in the normalisation denominator.  Default 1e-6.
    """

    def __init__(
        self,
        days: int = 7,
        hidden: int = 32,
        dropout: float = 0.1,
        base_token_init: float = -1.0,
        cc_enabled: bool = True,
        nan_native: bool = True,
        sentinel_value: float = -1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.days = days
        self.cc_enabled = cc_enabled
        self.nan_native = nan_native
        self.sentinel_value = sentinel_value
        self.eps = eps

        # Global base token (same role as scalar τ in Phase C NaME-NN).
        self.base_token = nn.Parameter(torch.tensor(base_token_init, dtype=torch.float32))

        # Context encoder (zero-inited output → δτ=0 at init).
        if cc_enabled:
            self.encoder = ContextEncoder(days=days, hidden=hidden, dropout=dropout)

    # ------------------------------------------------------------------
    # Context extraction helpers
    # ------------------------------------------------------------------

    def _missing_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a float mask with 1.0 at VALID (non-missing) positions.

        Shape: (N, W, D), same as x.
        """
        if self.nan_native:
            return (~torch.isnan(x)).to(x.dtype)
        return (x != self.sentinel_value).to(x.dtype)

    def _build_context(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Compute the 3D-dimensional, scale-invariant context vector.

        Parameters
        ----------
        x      : (N, W, D)  — raw input (may contain NaN or sentinel).
        valid  : (N, W, D)  — float mask, 1.0 at valid positions.

        Returns
        -------
        context : (N, 3*D)  — dimensionless context features.
        """
        n, w, d = x.shape

        # Replace NaN/sentinel with 0.0 for arithmetic (masked out anyway).
        x_fill = torch.where(valid > 0.5, x, torch.zeros_like(x))

        # --- Per-DOW statistics ---
        count_d = valid.sum(dim=1).clamp(min=1.0)            # (N, D)
        sum_d   = (x_fill * valid).sum(dim=1)                 # (N, D)
        mu_dow  = sum_d / count_d                             # (N, D)

        sum_sq_d = (x_fill ** 2 * valid).sum(dim=1)           # (N, D)
        var_dow  = (sum_sq_d / count_d - mu_dow ** 2).clamp(min=0.0)
        std_dow  = var_dow.sqrt()                             # (N, D)

        f_miss  = 1.0 - valid.sum(dim=1) / float(w)          # (N, D) ∈ [0,1]

        # --- Global valid statistics (for normalisation) ---
        count_g = valid.sum(dim=(1, 2)).clamp(min=1.0)        # (N,)
        mu_g    = (x_fill * valid).sum(dim=(1, 2)) / count_g  # (N,)
        sq_g    = (x_fill ** 2 * valid).sum(dim=(1, 2)) / count_g
        var_g   = (sq_g - mu_g ** 2).clamp(min=0.0)
        sig_g   = var_g.sqrt()                                # (N,)

        # Broadcast σ_g to (N, D) for element-wise normalisation.
        sig_g_d = (sig_g + self.eps).unsqueeze(1).expand(n, d)  # (N, D)

        # --- Normalised features (all dimensionless) ---
        # μ_dow shifted by global mean, then divided by σ_g.
        mu_g_d   = mu_g.unsqueeze(1).expand(n, d)              # (N, D)
        c_mean   = (mu_dow - mu_g_d) / sig_g_d                 # (N, D)
        c_std    = std_dow / sig_g_d                           # (N, D)
        c_fmiss  = f_miss                                      # (N, D) ∈ [0,1], already norm

        # Concatenate → (N, 3D).
        return torch.cat([c_mean, c_std, c_fmiss], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace missing cells with the instance-adaptive token τ_n.

        Parameters
        ----------
        x : (N, W, D)  — weekly consumption grid, NaN or sentinel at missing.

        Returns
        -------
        (N, W, D)  — same grid with missing cells replaced by τ_n.
                     No NaN values remain in the output.
        """
        if x.dim() != 3:
            raise ValueError(f"CC-NaME expects (N, W, D); got {tuple(x.shape)}.")

        valid = self._missing_mask(x)                          # (N, W, D)

        if self.cc_enabled:
            context = self._build_context(x, valid)            # (N, 3D)
            delta   = self.encoder(context)                    # (N, 1)
            # τ_n = τ_base + δτ_n, broadcast to (N, W, D).
            tau_n = (self.base_token + delta).unsqueeze(1).expand_as(x)  # (N, W, D)
        else:
            # Scalar NaME fallback — identical to Phase C NaME-NN behaviour.
            tau_n = self.base_token.expand_as(x)

        # Substitute: keep valid cells, replace missing with τ_n.
        missing = (valid < 0.5)                                # (N, W, D) bool
        return torch.where(missing, tau_n, x)

    # ------------------------------------------------------------------
    # Diagnostic accessor
    # ------------------------------------------------------------------

    def token_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance τ_n values for interpretability / paper figures.

        Returns
        -------
        (N,)  — scalar τ_n for each sample.  Used in Figure 5 of the paper:
                "CC-NaME token distribution per theft archetype."
        """
        valid   = self._missing_mask(x)
        context = self._build_context(x, valid)               # (N, 3D)
        delta   = self.encoder(context)                       # (N, 1)
        return (self.base_token + delta).squeeze(1)            # (N,)


__all__ = ["ContextConditionalNaME", "ContextEncoder"]
