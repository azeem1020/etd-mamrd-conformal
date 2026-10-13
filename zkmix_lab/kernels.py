"""Torch-native anti-mean kernel — Phase A1+A2.

Derivation
----------
Zheng, Yang, Niu, Dai, Zhou (IEEE TII 2018), §IV-B-2 defines two
auxiliary kernel functions over a 3×3 weekly window
``V = v[p:p+3, q:q+3]``::

    g1(V)[i, j] = 2 V[i, j] - V[i', j] - V[i'', j]   ({i,i',i''}={0,1,2})
    g2(V)[i, j] = (g1(V^T))^T

g1 is a *row anti-mean* — the central row of g1 evaluated at the centre
of the window collapses to

    g1[1, 1] = 2 V[1, 1] - V[0, 1] - V[2, 1]
             = 3 · (V[1, 1] - mean(V[:, 1]))                          (★)

i.e., the residual of the current row relative to the three-row mean
at the same column.  By symmetry g2[1, 1] = 3·(V[1, 1] − mean(V[1, :])).

When the kernel is *applied at every location* via a sliding window with
zero-padding-1 stride-1 (Eq. 10 says "the generated feature map having
the same size as the raw matrix because the zero-padding of 1 is used"),
the per-cell central residual (★) becomes a 1-D convolution along the
row axis with the fixed kernel

    K_row = [-1,  2, -1] / 3

and analogously along the column axis (a (1, 3) convolution).

A full Eq. 7 layer would convolve the 3×3 stack ``g1(V) + g2(V)`` with a
*learnable* 3×3 filter ŵ.  Because ``g1 + g2`` is a fixed linear map,
that whole construction is mathematically equivalent to **one 3×3 conv
whose weights are initialised to the anti-mean prior and then optionally
trained**.  That equivalence is what makes the lift in this module both
faithful to the paper and clean to extend.

This module therefore exposes:

* ``AntiMeanKernel`` — the canonical, frozen, GPU-resident operator that
  reproduces (★) and its column dual.  Produces an *(N, 2, W, D)* tensor
  whose channels are ``(row_residual, col_residual)``.

* ``LearnablePriorConv`` — a single 3×3 conv whose weights are
  initialised to the Eq. 7 prior (the row+column anti-mean stencil) and
  optionally frozen via ``freeze=True``.  This is the Phase A2 unit:
  same inductive bias at init, but the network can refine it.

* ``weekly_kernel_stats`` — differentiable per-week (mean, std, max,
  p99) summarisation of the kernel response field, replacing the legacy
  CPU numpy path used in ``sct_etd.models.zkmix._weekly_kernel_stats``.

Numerical contract
------------------
For an interior cell ``(i, j)`` (i.e., 1 ≤ i ≤ W-2 and 1 ≤ j ≤ D-2),
``AntiMeanKernel(x)[n, 0, i, j]`` equals
``(2*x[n, 0, i, j] - x[n, 0, i-1, j] - x[n, 0, i+1, j]) / 3`` to
``torch.float32`` precision.  This is asserted bit-exact (atol=1e-6,
rtol=0) in ``tests/test_kernels.py``.  Boundary cells use zero-padding,
matching Eq. 10's "zero-padding of 1".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Frozen anti-mean operator (Phase A1)
# ---------------------------------------------------------------------------


def _row_antimean_kernel() -> torch.Tensor:
    """(1, 1, 3, 1) tensor implementing K_row = [-1, 2, -1] / 3."""
    k = torch.tensor([[-1.0], [2.0], [-1.0]]) / 3.0
    return k.view(1, 1, 3, 1)


def _col_antimean_kernel() -> torch.Tensor:
    """(1, 1, 1, 3) tensor implementing K_col = [-1, 2, -1] / 3."""
    k = torch.tensor([[-1.0, 2.0, -1.0]]) / 3.0
    return k.view(1, 1, 1, 3)


class AntiMeanKernel(nn.Module):
    """Frozen, GPU-resident, differentiable anti-mean kernel.

    Forward
    -------
    x : (N, 1, W, D)  — single-channel weekly grid (weeks × days).
    returns : (N, 2, W, D)  — channel 0 is row residual, channel 1 is
              column residual.  Each channel is a 2-D field at the same
              spatial resolution as ``x``.

    The output is the cell-wise central residual derived in Eq. (★)
    above; it is the *paper-exact* response field, divided by 3 so that
    the operator has unit infinity-norm gain on a flat input (i.e., a
    constant signal maps to zero exactly).
    """

    def __init__(self) -> None:
        super().__init__()
        # Use buffers (move with .to(device)/.cuda(), but no gradients).
        self.register_buffer("kernel_row", _row_antimean_kernel(), persistent=False)
        self.register_buffer("kernel_col", _col_antimean_kernel(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(
                f"AntiMeanKernel expects (N, 1, W, D); got tuple(x.shape)={tuple(x.shape)}."
            )
        row_res = F.conv2d(x, self.kernel_row, padding=(1, 0))
        col_res = F.conv2d(x, self.kernel_col, padding=(0, 1))
        return torch.cat([row_res, col_res], dim=1)


# ---------------------------------------------------------------------------
# Learnable structural prior (Phase A2)
# ---------------------------------------------------------------------------


def _eq7_prior_weights(out_channels: int = 1) -> torch.Tensor:
    """3×3 weights for Eq. 7's row+column anti-mean stencil at the centre.

    The Eq. 7 layer convolves ``g1(V) + g2(V)`` with a learnable 3×3
    filter.  Substituting g1+g2 evaluated at the *centre* of a 3×3
    window (Eq. ★ and its column dual) gives a single equivalent 3×3
    convolution whose weights are

        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]] / 3

    i.e., the 4-neighbour discrete Laplacian normalised so that a
    constant input maps to zero.  This is the *structural prior* used
    to initialise ``LearnablePriorConv`` — refinement during training
    can recover Eq. 7's optimal ŵ if the data demands it.
    """
    base = torch.tensor(
        [[0.0, -1.0, 0.0],
         [-1.0, 4.0, -1.0],
         [0.0, -1.0, 0.0]]
    ) / 3.0
    # Stack identical priors across out_channels so we can over-parameterise
    # without breaking the initial-output invariant.
    return base.view(1, 1, 3, 3).repeat(out_channels, 1, 1, 1)


class LearnablePriorConv(nn.Module):
    """Single 3×3 conv initialised to the Eq. 7 anti-mean prior.

    Parameters
    ----------
    out_channels : int
        How many filters to learn on top of the prior.  Each filter is
        initialised to the same prior weights; symmetry is broken by
        Gaussian perturbation of std ``init_noise`` (default 1e-3) so
        that gradients flow into distinct directions.
    freeze : bool
        If True, the conv weights are detached from the optimiser
        (``requires_grad_(False)``); the layer then matches the
        frozen-prior behaviour but with the same parameter footprint as
        the learnable variant.  Useful for an apples-to-apples
        ``freeze=True`` vs ``freeze=False`` ablation.
    init_noise : float
        Standard deviation of Gaussian symmetry-breaking noise applied
        on top of the prior weights when ``out_channels > 1``.

    Forward
    -------
    x : (N, 1, W, D)  →  (N, out_channels, W, D)
    """

    def __init__(
        self,
        out_channels: int = 1,
        freeze: bool = False,
        init_noise: float = 1e-3,
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be ≥ 1.")
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        with torch.no_grad():
            prior = _eq7_prior_weights(out_channels=out_channels)
            if out_channels > 1 and init_noise > 0:
                prior = prior + init_noise * torch.randn_like(prior)
            self.conv.weight.copy_(prior)
        if freeze:
            self.conv.weight.requires_grad_(False)
        self.freeze = freeze
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(
                f"LearnablePriorConv expects (N, 1, W, D); got tuple(x.shape)={tuple(x.shape)}."
            )
        return self.conv(x)


# ---------------------------------------------------------------------------
# Weekly stats (replaces sct_etd.models.zkmix._weekly_kernel_stats)
# ---------------------------------------------------------------------------


Stat = Literal["mean", "std", "max", "p99"]
_DEFAULT_STATS: tuple[Stat, ...] = ("mean", "std", "max", "p99")


def weekly_kernel_stats(
    response: torch.Tensor,
    stats: tuple[Stat, ...] = _DEFAULT_STATS,
    p99_q: float = 0.99,
) -> torch.Tensor:
    """Per-week (mean, std, max, p99) of a kernel response field.

    Parameters
    ----------
    response : (N, C, W, D) tensor — output of AntiMeanKernel or
        LearnablePriorConv.  C is treated as the channel axis; the
        statistics are computed independently per channel.
    stats : ordered tuple of statistic names to compute.  Output channel
        ordering follows ``(c, s)`` lexicographic where ``c`` is the
        kernel-response channel and ``s`` indexes ``stats``.
    p99_q : quantile used when ``"p99"`` is requested.  Default 0.99.

    Returns
    -------
    (N, W, C * len(stats)) tensor on the same device as ``response``.
    Fully differentiable: no detach/cpu/numpy round-trips.
    """
    if response.dim() != 4:
        raise ValueError(
            f"weekly_kernel_stats expects (N, C, W, D); got tuple(response.shape)={tuple(response.shape)}."
        )
    n, c, w, d = response.shape

    pieces: list[torch.Tensor] = []
    for name in stats:
        if name == "mean":
            pieces.append(response.mean(dim=-1))
        elif name == "std":
            # unbiased=False to match population-std convention used in
            # the legacy KAGB tabular features (np.std default).
            pieces.append(response.std(dim=-1, unbiased=False))
        elif name == "max":
            pieces.append(response.amax(dim=-1))
        elif name == "p99":
            # torch.quantile is differentiable wrt input.
            pieces.append(torch.quantile(response, p99_q, dim=-1))
        else:  # pragma: no cover - guarded by Literal type
            raise ValueError(f"Unknown stat {name!r}.")
    # Each piece is (N, C, W).  Stack on a new last axis then collapse.
    stacked = torch.stack(pieces, dim=-1)            # (N, C, W, S)
    return stacked.permute(0, 2, 1, 3).reshape(n, w, c * len(stats))


# ---------------------------------------------------------------------------
# Convenience: spec dataclass for runner wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelSpec:
    """Specification consumed by ``models.ZKMix`` to choose its prior."""

    mode: Literal["frozen", "learnable_frozen_init", "learnable_free"] = "frozen"
    learnable_out_channels: int = 2

    def build(self) -> nn.Module:
        if self.mode == "frozen":
            return AntiMeanKernel()
        if self.mode == "learnable_frozen_init":
            return LearnablePriorConv(
                out_channels=self.learnable_out_channels, freeze=True
            )
        if self.mode == "learnable_free":
            return LearnablePriorConv(
                out_channels=self.learnable_out_channels, freeze=False
            )
        raise ValueError(f"Unknown KernelSpec.mode {self.mode!r}.")  # pragma: no cover

    @property
    def out_channels(self) -> int:
        return 2 if self.mode == "frozen" else self.learnable_out_channels


__all__ = [
    "AntiMeanKernel",
    "LearnablePriorConv",
    "KernelSpec",
    "weekly_kernel_stats",
]
