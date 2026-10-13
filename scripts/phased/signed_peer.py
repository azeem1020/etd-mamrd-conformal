"""Phase L — Signed-Peer Cross-Attention Branch (A25).

Direction-aware replacement for the Phase K hand-coded PRCE-Laplacian (A17)
and the Phase K attention-only GNN (A20). Uses *three* peer features instead
of one, all of which carry direction:

    signed_residual : (N, W, D)   r[c, w, d] = X_filled[c] − mean(X[k-peers])
                                  preserves sign:  + = Riser,  − = Dropper
    peer_std        : (N, W, D)   std across the k-peer cohort per (week, day)
                                  high = peer cohort is noisy → low confidence
    peer_iqr        : (N, W, D)   p75 − p25 across the cohort
                                  robust spread measure; complements peer_std

These three channels are stacked → (N, W, 3*D) = (N, 147, 21) for the
default D=7 SGCC grid.

Architecture
------------
1.  Per-stream Conv1d (k=7, padding=3) projects own (N, W, D) and peer
    (N, W, 3*D) to a shared ``hidden`` dim:
        own_h    = ReLU(Conv1d(D    → hidden, k=7))
        peer_h   = ReLU(Conv1d(3*D  → hidden, k=7))

2.  Cross-attention with ``own_h`` as Q and ``peer_h`` as K, V (4 heads):
        attn_out, _ = MHA(own_h, peer_h, peer_h)
        h = LayerNorm(own_h + attn_out)

3.  Learnable signed-direction gate — *separate* sigmoid gates for positive
    and negative residual contributions, so Riser-direction and Dropper-direction
    activations are encoded distinctly:
        pos_gate = sigmoid(Linear(h, 1))   # contribution from Riser-like time steps
        neg_gate = sigmoid(Linear(h, 1))   # contribution from Dropper-like time steps
        pos_alpha = pos_gate / (pos_gate.sum + ε)
        neg_alpha = neg_gate / (neg_gate.sum + ε)
        pooled = concat( Σ pos_alpha·h ,  Σ neg_alpha·h )      # (N, 2*hidden)

4.  Output projection:
        embed = Dropout(ReLU(Linear(2*hidden, embed_dim)))

Why this beats A20
------------------
A20 had ONE temporal gate; it could only emphasise weeks where peer comparison
was discriminative but could not encode WHICH direction (up or down) was the
deviation. With separate pos / neg gates, the branch can:
  • light up the pos-gate for Risers (+ residual)
  • light up the neg-gate for Droppers (− residual)
  • combine both for time-varying mixed patterns (e.g. drop then rebound)

Why peer_std + peer_iqr help
----------------------------
A clean peer cohort (low std, low IQR) means the residual is informative; a
noisy cohort means the residual is unreliable. The attention learns this
implicitly through K=V conditioning, down-weighting time steps where the
cohort is noisy.

NaN safety
----------
Input ``own`` comes from the NaME-imputed path (no NaN). Peer features are
computed numpy-level with nan-aware reducers and 0-filled (already done in
``compute_signed_peer_features``).
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pre-compute signed peer features at the data-prep stage (numpy)
# ---------------------------------------------------------------------------

def compute_signed_peer_features(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 20,
) -> np.ndarray:
    """Compute peer features (signed residual, peer std, peer IQR) for all N customers.

    The k-nearest-normal-peer matching uses the same top-quartile monthly nanmean
    feature as Phase J/K (robust to step-down theft). Returns the THREE peer
    features stacked along the day axis so the model can consume a single tensor.

    Output shape: (N, W, 3*D) — own NOT included here.
        out[:, :, 0:D]      = signed residual  =  X_filled − peer_mean
        out[:, :, D:2*D]    = peer std (per week, day) across k peers
        out[:, :, 2*D:3*D]  = peer IQR (per week, day) across k peers

    Same fold-safety guarantees as compute_peer_residuals: peer pool is
    ``train_idx[y[train_idx] == 0]``.
    """
    from sklearn.neighbors import NearestNeighbors                   # local import

    if X_raw.ndim != 3:
        raise ValueError(f"X_raw must be (N, W, D); got {X_raw.shape}")
    N, W, D = X_raw.shape
    nb = W // 7

    # ── 1. monthly nanmean → top-quartile peak feature for kNN ───────────────
    x4 = X_raw[:, :nb * 7, :].reshape(N, nb, 7, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        monthly = np.nanmean(x4.reshape(N, nb, -1), axis=-1)            # (N, nb)
    monthly = np.nan_to_num(monthly, nan=0.0)
    top_q   = max(1, nb // 4)
    monthly_desc = np.sort(monthly, axis=1)[:, ::-1]
    knn_feat = monthly_desc[:, :top_q]                                   # (N, top_q)

    # ── 2. normal-only peer pool from train fold ─────────────────────────────
    normal_idx = train_idx[y[train_idx] == 0]
    k_eff = min(k, max(1, len(normal_idx)))

    # ── 3. kNN lookup ─────────────────────────────────────────────────────────
    nn_model = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn_model.fit(knn_feat[normal_idx])
    _, local_idx = nn_model.kneighbors(knn_feat)                          # (N, k_eff)
    global_idx   = normal_idx[local_idx]                                  # (N, k_eff)

    # ── 4. peer baseline statistics ──────────────────────────────────────────
    # X_raw[global_idx] → (N, k_eff, W, D). For speed: NaN-fill peers with the
    # peer-mean per (N, W, D), then use vectorised percentile. This is a tiny
    # bias (NaN positions contribute the mean to percentile estimate) and is
    # MUCH faster than np.nanpercentile (~50x for our shape). The mean and std
    # use nan-aware reducers since they are SIMD-fast.
    peers = X_raw[global_idx]                                              # (N, k_eff, W, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peer_mean = np.nanmean(peers, axis=1)                              # (N, W, D)
        peer_std  = np.nanstd (peers, axis=1)                              # (N, W, D)
    peer_mean = np.nan_to_num(peer_mean, nan=0.0)
    peer_std  = np.nan_to_num(peer_std,  nan=0.0)
    # Fast IQR: fill NaN in peers with peer_mean then use percentile.
    nan_mask_peers = np.isnan(peers)
    if nan_mask_peers.any():
        peers_filled = np.where(nan_mask_peers, peer_mean[:, None, :, :], peers)
    else:
        peers_filled = peers
    peer_p25 = np.percentile(peers_filled, 25, axis=1)                     # (N, W, D)
    peer_p75 = np.percentile(peers_filled, 75, axis=1)                     # (N, W, D)
    peer_iqr = peer_p75 - peer_p25

    # ── 5. signed residual ───────────────────────────────────────────────────
    X_filled  = np.where(np.isnan(X_raw), 0.0, X_raw)
    signed_residual = X_filled - peer_mean                                 # (N, W, D)
    signed_residual[np.isnan(X_raw)] = 0.0                                 # zero where own is NaN

    out = np.concatenate([signed_residual, peer_std, peer_iqr], axis=-1)   # (N, W, 3*D)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class SignedPeerBranch(nn.Module):
    """Cross-attention with directional gating over signed peer features.

    Parameters
    ----------
    days        : D (7 for SGCC weekly grid).
    weeks       : W (147 for SGCC).
    embed_dim   : output projection target.
    hidden      : per-stream feature width (must be divisible by ``n_heads``).
    n_heads     : multi-head count.
    kernel_size : Conv1d kernel size (in days).
    dropout     : attention + projection dropout.
    """

    def __init__(
        self,
        days: int,
        weeks: int,
        embed_dim: int,
        *,
        hidden: int = 64,
        n_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by n_heads ({n_heads})")
        self.weeks  = weeks
        self.days   = days
        self.hidden = hidden

        pad = kernel_size // 2

        # Own stream conv: D → hidden
        self.own_conv = nn.Sequential(
            nn.Conv1d(days, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )
        # Peer stream conv: 3*D → hidden  (signed residual + peer std + peer IQR)
        self.peer_conv = nn.Sequential(
            nn.Conv1d(3 * days, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

        # Two directional gates — Riser-pos and Dropper-neg
        self.pos_gate = nn.Linear(hidden, 1)
        self.neg_gate = nn.Linear(hidden, 1)

        # Output projection (2*hidden → embed_dim because we concat pos/neg pooled)
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, own: torch.Tensor, peer3: torch.Tensor) -> torch.Tensor:
        """
        own   : (N, W, D)            — NaME-imputed own consumption
        peer3 : (N, W, 3*D)          — stacked (signed_residual, peer_std, peer_iqr)
        """
        if own.shape[0] != peer3.shape[0]:
            raise ValueError("Batch dim mismatch")
        if own.shape[2] != self.days or peer3.shape[2] != 3 * self.days:
            raise ValueError(
                f"Expected own D={self.days}, peer D={3 * self.days}; "
                f"got own D={own.shape[2]}, peer D={peer3.shape[2]}"
            )
        if own.shape[1] != self.weeks:
            raise ValueError(f"Expected W={self.weeks}; got W={own.shape[1]}")

        # Conv1d expects (N, C, L) → transpose to (N, D, W) and back.
        own_h  = self.own_conv (own.transpose(1, 2)).transpose(1, 2)        # (N, W, hidden)
        peer_h = self.peer_conv(peer3.transpose(1, 2)).transpose(1, 2)      # (N, W, hidden)

        attn_out, _ = self.cross_attn(own_h, peer_h, peer_h, need_weights=False)
        h = self.norm(own_h + attn_out)                                     # (N, W, hidden)

        # Directional gates
        pos = torch.sigmoid(self.pos_gate(h)).squeeze(-1)                   # (N, W)
        neg = torch.sigmoid(self.neg_gate(h)).squeeze(-1)                   # (N, W)
        eps = 1e-8
        pos_alpha = pos / (pos.sum(dim=-1, keepdim=True) + eps)
        neg_alpha = neg / (neg.sum(dim=-1, keepdim=True) + eps)

        pos_pool = (h * pos_alpha.unsqueeze(-1)).sum(dim=1)                  # (N, hidden)
        neg_pool = (h * neg_alpha.unsqueeze(-1)).sum(dim=1)                  # (N, hidden)
        pooled   = torch.cat([pos_pool, neg_pool], dim=-1)                   # (N, 2*hidden)

        return self.proj(pooled)                                             # (N, embed_dim)


__all__ = [
    "compute_signed_peer_features",
    "SignedPeerBranch",
]
