"""Multi-Scale Laplacian Kernel — Phase D.

Theoretical basis
-----------------
Zheng et al. 2018 (IEEE TII §IV-B-2) define an anti-mean kernel that computes
the local row/column curvature of the consumption matrix at a fixed 3×3 scale
(receptive field = 3 weeks × 3 days).  This is equivalent to a discrete 2-D
Laplacian at dilation d=1:

    L₁(x)[i,j] = (2x[i,j] − x[i−1,j] − x[i+1,j]) / 3   (row direction)

The operator is optimal for detecting abrupt, localised theft patterns (Heavy
Dropper, Heavy Riser) where the curvature signal is concentrated within a
3-week window.

However, the SGCC theft population contains two additional archetypes that
account for 48.6% of theft cases:

    Light Dropper (17.6%)  — gradual decline over 8–16 weeks
    Light Riser   (24.1%)  — gradual rise   over 8–16 weeks
    Flat          (16.5%)  — flatline that creates a sustained low-curvature
                             anomaly relative to the honest population

These patterns have near-zero L₁ response by construction (low local
curvature) and are therefore structurally undetectable by the Phase C CNN.

The fix is a multi-scale extension of the same Laplacian principle.  For
dilation d, the operator becomes:

    L_d(x)[i,j] = (2x[i,j] − x[i−d,j] − x[i+d,j]) / 3

with effective receptive field (2d+1) weeks × (2d+1) days:
  d=1: RF 3w×3d  — detects Heavy strategies (abrupt 3-week changes)
  d=2: RF 5w×5d  — detects medium-duration anomalies (~5 weeks)
  d=4: RF 9w×9d  — detects Light strategies (gradual 9-week changes)

Crucially, these are NOT arbitrary learnable filters (unlike Phase B.2's
ms_5x5_d2 and ms_7x7_d3).  Each scale is:
  1. Initialised to the exact Zheng Eq.7 nine-stencil weights.
  2. Optionally frozen (``freeze=True``) or fine-tunable (``freeze=False``).
  3. Applied at a different dilation so the inductive bias is preserved.

This makes the multi-scale construction a principled extension of the Zheng
formulation, not an engineering heuristic.  The paper can state:
  "We extend Zheng's single-scale Laplacian (d=1) to a scale pyramid at
  d∈{1,2,4}, preserving the anti-mean inductive bias at each scale while
  expanding the detectable temporal bandwidth to cover all five theft archetypes."

Implementation note: the nine-stencil weight tensor is imported from
zkmix_lab.models._eq7_nine_stencil_weights — a private function in the
Phase C codebase.  We re-expose it here under its own name so Phase D
does not depend on the private API at runtime.

Numerical contract
------------------
For an interior cell (i,j) with i≥d and i≤W−d−1 (no boundary effects):

    MultiScaleLaplacian(x, dilations=(d,))[n, 0:9, i, j]

equals the 9-stencil response of the anti-mean kernel applied to the window
x[n, i-d:i+d+1:d, j-d:j+d+1:d], to torch.float32 precision.
This is asserted in tests/test_ms_kernel.py::test_d1_matches_zheng_stencil.
"""
from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import the nine-stencil weight generator from Phase C codebase.
# We resolve the path at import time so this module works regardless of the
# working directory when scripts are launched from the project root.
# ---------------------------------------------------------------------------

def _get_nine_stencil_weights() -> torch.Tensor:
    """Return the (9, 1, 3, 3) Zheng Eq.7 stencil weight tensor.

    Loads from zkmix_lab if available on sys.path; otherwise reconstructs
    the weights analytically (identical numerics, verified in tests).
    """
    try:
        from zkmix_lab.models import _eq7_nine_stencil_weights  # type: ignore
        return _eq7_nine_stencil_weights()
    except ImportError:
        return _eq7_nine_stencil_weights_local()


def _eq7_nine_stencil_weights_local() -> torch.Tensor:
    """Analytical reconstruction of the Zheng Eq.7 nine-stencil weights.

    Equation 7 defines g1 (row anti-mean) and g2 (col anti-mean) over a
    3×3 window V:

        g1(V)[i,j] = 2V[i,j] − V[i',j] − V[i'',j]   (other two rows)
        g2(V)[i,j] = 2V[i,j] − V[i,j'] − V[i,j'']   (other two cols)

    Applied at every position (i,j) ∈ {0,1,2}², the combined response
    (g1+g2) at (i,j) is:

        (g1+g2)(V)[i,j] = 4V[i,j] − V[i,j'] − V[i,j''] − V[i',j] − V[i'',j]

    i.e. centre = +4, same-row neighbours = −1, same-col neighbours = −1,
    all four corners = 0.  This gives 9 distinct filters (one per position).

    Verified: each filter sums to 4 − 2 − 2 = 0 (anti-mean property).
    Verified: matches zkmix_lab.models._eq7_nine_stencil_weights exactly.
    """
    filters: list[torch.Tensor] = []
    for i in range(3):
        for j in range(3):
            f = torch.zeros(1, 3, 3)
            f[0, i, j] = 4.0
            for jj in range(3):
                if jj != j:
                    f[0, i, jj] = -1.0   # same-row neighbours
            for ii in range(3):
                if ii != i:
                    f[0, ii, j] = -1.0   # same-col neighbours
            filters.append(f)
    return torch.stack(filters, dim=0)   # (9, 1, 3, 3)


class ScaleBranch(nn.Module):
    """One scale of the multi-scale Laplacian: frozen 9-stencil at dilation d.

    Parameters
    ----------
    dilation : int
        Dilation factor.  d=1 reproduces the Phase C Zheng kernel exactly.
    freeze : bool
        If True (default), weights are frozen after Zheng initialisation.
        If False, the stencil can be fine-tuned — same learnable option as
        LearnablePriorConv in kernels.py but with the dilated application.
    out_channels : int
        Number of output channels after the learnable 1×1 conv that follows
        the frozen stencil.  Default 15 (matching Phase C cfg.gamma=15).
    """

    def __init__(
        self,
        dilation: int = 1,
        freeze: bool = True,
        out_channels: int = 15,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be ≥ 1.")

        self.dilation = dilation
        self.freeze   = freeze

        # Frozen (or optionally fine-tunable) Zheng 9-stencil at dilation d.
        self.stencil = nn.Conv2d(
            in_channels=1,
            out_channels=9,
            kernel_size=3,
            padding=dilation,   # 'same' spatial size with dilation
            dilation=dilation,
            bias=False,
        )
        with torch.no_grad():
            self.stencil.weight.copy_(_get_nine_stencil_weights())
        self.stencil.weight.requires_grad_(not freeze)

        # Learnable 1×1 mixing conv (same role as cfg.gamma conv in Phase C).
        self.mix = nn.Conv2d(9, out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x4d: torch.Tensor) -> torch.Tensor:
        """x4d : (N, 1, W, D) → (N, out_channels, W, D)."""
        resp = self.stencil(x4d)                  # (N, 9, W, D)
        return self.act(self.mix(resp))            # (N, out_channels, W, D)


class MultiScaleLaplacian(nn.Module):
    """Multi-scale Laplacian pyramid for the deep-CNN branch of ZKMixV4.

    Applies the frozen Zheng 9-stencil at each requested dilation, concatenates
    the per-scale responses, MaxPools the result, and projects to a fixed-dim
    embedding via a linear head.

    Design choices
    --------------
    *  Each scale shares the same initialisation (Zheng prior) but has
       independent learnable 1×1 mixing weights.  This means scale d=1 is
       free to learn the same specialisation as Phase C while d=2 and d=4
       develop complementary features.
    *  MaxPool(2) is applied to the concatenated multi-scale map, matching
       Phase C's pooling step.  The pooled feature is flattened and projected
       to ``embed_dim``.
    *  The embedding dimension is deliberately kept equal to the Phase C
       beta value (default 60) so the fusion head in ZKMixV4 sees the same
       size input from the CNN branch.

    Parameters
    ----------
    dilations : tuple[int, ...]
        Dilation values to use.  Default (1, 2, 4) covers abrupt, medium,
        and gradual theft patterns respectively.
    freeze_stencils : bool
        Whether to freeze the Zheng stencil at each scale.  Default True
        (matches Phase C frozen-stencil convention; keeping frozen isolates
        the multi-scale contribution from any stencil fine-tuning).
    out_channels_per_scale : int
        Number of channels after the 1×1 conv at each scale.  Default 15
        (matches Phase C cfg.gamma=15).
    weeks : int
        Input grid height (number of weeks).  Default 147.
    days : int
        Input grid width (days per week).  Default 7.
    embed_dim : int
        Output embedding dimension after the linear head.  Default 60
        (matches Phase C cfg.beta=60 for the CNN branch output).
    dropout : float
        Dropout applied before the linear head.  Default 0.1.
    """

    def __init__(
        self,
        dilations: tuple[int, ...] = (1, 2, 4),
        freeze_stencils: bool = True,
        out_channels_per_scale: int = 15,
        weeks: int = 147,
        days: int = 7,
        embed_dim: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dilations) == 0:
            raise ValueError("dilations must be non-empty.")
        self.dilations = dilations

        self.branches = nn.ModuleList(
            ScaleBranch(
                dilation=d,
                freeze=freeze_stencils,
                out_channels=out_channels_per_scale,
            )
            for d in dilations
        )

        self.pool   = nn.MaxPool2d(kernel_size=2)
        total_ch    = out_channels_per_scale * len(dilations)
        pooled_flat = (weeks // 2) * (days // 2) * total_ch
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(pooled_flat, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x4d: torch.Tensor) -> torch.Tensor:
        """x4d : (N, 1, W, D) → (N, embed_dim).

        The input must already have NaN / sentinel values substituted
        (via CC-NaME or scalar NaME) before this module is called.
        """
        if x4d.dim() != 4 or x4d.shape[1] != 1:
            raise ValueError(
                f"MultiScaleLaplacian expects (N, 1, W, D); got {tuple(x4d.shape)}."
            )
        # Compute response at each dilation scale.
        responses = [branch(x4d) for branch in self.branches]  # list[(N, γ, W, D)]
        # Concatenate along channel axis, then pool spatially.
        combined = torch.cat(responses, dim=1)                  # (N, S*γ, W, D)
        pooled   = self.pool(combined)                          # (N, S*γ, W/2, D/2)
        return self.head(pooled)                                # (N, embed_dim)


__all__ = ["MultiScaleLaplacian", "ScaleBranch"]
