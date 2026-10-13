"""Phase K — Attention-Based Peer Comparison branch (A20).

Replaces the hand-coded PRCE Laplacian-CNN (A17) with a *learned* cross-attention
module that operates jointly on:

    own_consumption : (N, W, D)   — the customer's own series (NaME-imputed)
    peer_residual   : (N, W, D)   — r[c, t] = x̂[c, t] − mean_peer[t]

The cross-attention is a 2-node GNN: each customer is one node, the aggregated
peer is another. Attention over the time dimension lets the network discover
*where* the peer comparison is most informative and *how* it should be weighed
against the own signal.

Why this beats Laplacian-PRCE (J1 K-null observation)
-----------------------------------------------------
Fixed Laplacian responds only to local curvature in peer_res. A Light Dropper
whose k-NN matches them with low-consumption normals has near-zero peer_res
across the entire window — Laplacian → 0. Attention can instead pool the
*small but consistent* negative residual across many weeks (sum, not curvature)
and gate down peer-noisy weeks. It can also learn that a Flat Stealer's peer_res
is constant-negative (DC offset), which Laplacian-of-residual is structurally
blind to.

Architecture
------------
1. Per-stream 1-D conv (k=7, padding=3) projects each stream to ``hidden=64``::

       own_h  ∈ R^{N×W×64}    peer_h ∈ R^{N×W×64}

2. Cross-attention with ``own_h`` as Q and ``peer_h`` as K, V::

       attn_out, _ = MHA(Q=own_h, K=peer_h, V=peer_h)     # (N, W, 64)
       h = LN(own_h + attn_out)                            # residual + norm

3. Learnable temporal gate (sigmoid-normalised softmax over W)::

       g_t = sigmoid(Linear(h_t)) ; α = g / Σ_t g
       pooled = Σ_t α_t · h_t                              # (N, 64)

4. Final projection::

       embed = Dropout(ReLU(Linear(64, embed_dim)))

Hyperparameters
---------------
* hidden       : 64  (4 heads × 16 head_dim — matches MultiheadAttention contract)
* n_heads      : 4
* kernel_size  : 7   (one-week receptive field per stream conv; matches the
                     7-day grid which is the natural ETD periodicity)
* dropout      : 0.10
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeerAttentionBranch(nn.Module):
    """Cross-attention between own consumption and peer residual.

    Input (forward signature): both ``own`` and ``peer`` of shape (N, W, D).
    Output: (N, embed_dim).
    """

    def __init__(
        self,
        days: int,
        weeks: int,
        embed_dim: int,
        hidden: int = 64,
        n_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
            )
        self.weeks = weeks
        self.days  = days
        self.hidden = hidden

        pad = kernel_size // 2

        # Stream projections — 1D conv along time axis at the (N, D, W) view.
        self.own_conv = nn.Sequential(
            nn.Conv1d(days, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )
        self.peer_conv = nn.Sequential(
            nn.Conv1d(days, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )

        # Cross-attention with own as query, peer as K/V.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

        # Learnable temporal gate.
        self.gate = nn.Linear(hidden, 1)

        # Final projection to branch embed_dim.
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, own: torch.Tensor, peer: torch.Tensor) -> torch.Tensor:
        # own, peer : (N, W, D)
        if own.shape != peer.shape:
            raise ValueError(
                f"own and peer must share shape; got {tuple(own.shape)} vs {tuple(peer.shape)}"
            )
        if own.shape[1] != self.weeks or own.shape[2] != self.days:
            raise ValueError(
                f"Expected (N, {self.weeks}, {self.days}); got {tuple(own.shape)}"
            )

        # (N, W, D) → (N, D, W) for Conv1d → back to (N, W, hidden).
        own_h  = self.own_conv(own.transpose(1, 2)).transpose(1, 2)        # (N, W, h)
        peer_h = self.peer_conv(peer.transpose(1, 2)).transpose(1, 2)      # (N, W, h)

        # Cross-attention: own asks peer for context across time.
        attn_out, _ = self.cross_attn(own_h, peer_h, peer_h, need_weights=False)
        h = self.norm(own_h + attn_out)                                    # (N, W, h)

        # Temporal gate-weighted pooling.
        g = torch.sigmoid(self.gate(h)).squeeze(-1)                        # (N, W)
        alpha = g / (g.sum(dim=-1, keepdim=True) + 1e-8)                   # (N, W)
        pooled = (h * alpha.unsqueeze(-1)).sum(dim=1)                      # (N, h)

        return self.proj(pooled)                                           # (N, embed_dim)


__all__ = ["PeerAttentionBranch"]
