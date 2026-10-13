"""MSDM model — kMSD + AQ + MSCNet + classifier (paper §IV; Fig. 5/8).

Implemented from the paper's equations. Ambiguous structural details are marked
[ASSUMPTION] and documented in README.md (honest reproduction).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# kMSD — kernel-based Multi-Scale Differencing  (Eq. 6–8)
# --------------------------------------------------------------------------- #
class KMSD(nn.Module):
    """Multi-scale differencing via a fixed difference kernel at several dilations.

    For each scale d in `scales`, apply a depthwise dilated conv with the static
    kernel K (Σ K = 0). Input (B, T); output (B, n_scales, T) — one differenced
    channel per scale (causal left-padding).
    """
    def __init__(self, scales, kernel=(-1.0, 1.0, 0.0)):
        super().__init__()
        self.scales = list(scales)
        k = torch.tensor(kernel, dtype=torch.float32)
        assert abs(float(k.sum())) < 1e-6, "difference kernel must satisfy ΣK=0 (Eq. 8)"
        # Eq. 7 is a CONVOLUTION (D(j)=Σ_t x[j−d·t]·K_t, backward index); F.conv1d is
        # cross-correlation, so we store the FLIPPED kernel to compute Eq. 7 exactly.
        # With K=(-1,1,0)→flip (0,1,-1) and left-pad (ksize-1)·d, out[j]=x[j-d]-x[j]
        # = -Δ_d x[j], matching Eq. 7 verbatim.
        kf = torch.flip(k, dims=[0]).contiguous()
        self.register_buffer("kernel", kf.view(1, 1, -1))  # (1,1,ksize)
        self.ksize = len(kernel)

    def forward(self, x):                                   # x: (B, T)
        b, t = x.shape
        xin = x.unsqueeze(1)                                # (B,1,T)
        outs = []
        for d in self.scales:
            pad = (self.ksize - 1) * d                      # causal left pad
            xp = F.pad(xin, (pad, 0))
            outs.append(F.conv1d(xp, self.kernel, dilation=d))  # (B,1,T)
        return torch.cat(outs, dim=1)                       # (B, n_scales, T)


# --------------------------------------------------------------------------- #
# AQ — Adaptive Quantization  (Eq. 9):  Q_β(x) = tanh(β x), β learnable per scale
# --------------------------------------------------------------------------- #
class AdaptiveQuant(nn.Module):
    def __init__(self, n_scales, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.full((1, n_scales, 1), float(beta_init)))

    def forward(self, x):                                   # (B, n_scales, T)
        return torch.tanh(self.beta * x)


# --------------------------------------------------------------------------- #
# MSCNet — Intra/Inter-Period Convolution (IIPC), residual MSCBlocks (Eq. 10–11)
# --------------------------------------------------------------------------- #
class IIPC(nn.Module):
    """Parallel intra-period (small dilation) + inter-period (period-multiple
    dilation) dilated 1-D convs, concatenated → BN → ReLU. (Fig. 8 / Eq. 11)"""
    def __init__(self, cin, filters, d_intra, d_inter, k=3):
        super().__init__()
        p_i = (k - 1) // 2 * d_intra
        p_e = (k - 1) // 2 * d_inter
        self.intra = nn.Conv1d(cin, filters, k, padding=p_i, dilation=d_intra)
        self.inter = nn.Conv1d(cin, filters, k, padding=p_e, dilation=d_inter)
        self.bn = nn.BatchNorm1d(2 * filters)

    def forward(self, x):
        h = torch.cat([self.intra(x), self.inter(x)], dim=1)   # (B, 2F, T)
        return F.relu(self.bn(h))


class MSCBlock(nn.Module):
    """Two stacked IIPC layers with period-scaled dilations + residual add.
    [ASSUMPTION] IIPC-1 dilations {1, C}; IIPC-2 {2, 2C} (Fig. 8); 1×1 projection
    matches channels for the residual connection (Eq. 10: X^l = MSCBlock(X^{l-1})+X^{l-1})."""
    def __init__(self, cin, filters, period):
        super().__init__()
        self.iipc1 = IIPC(cin, filters, d_intra=1, d_inter=period)
        self.iipc2 = IIPC(2 * filters, filters, d_intra=2, d_inter=2 * period)
        self.out_ch = 2 * filters
        self.proj = nn.Conv1d(cin, self.out_ch, 1) if cin != self.out_ch else nn.Identity()

    def forward(self, x):
        h = self.iipc2(self.iipc1(x))
        return F.relu(h + self.proj(x))


class MSCNet(nn.Module):
    """Residual stack of MSCBlocks over the (quantized) multi-scale differential
    sequence. Input (B, n_scales, T) → feature (B, feat_dim)."""
    def __init__(self, n_scales, period=7, filters=48, n_blocks=2, feat_dim=128):
        super().__init__()
        self.stem = nn.Conv1d(n_scales, feat_dim, 1)
        cin = feat_dim
        blocks = []
        for _ in range(n_blocks):
            blk = MSCBlock(cin, filters, period)
            blocks.append(blk)
            cin = blk.out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head_proj = nn.Conv1d(cin, feat_dim, 1)
        self.feat_dim = feat_dim

    def forward(self, x):                                   # (B, n_scales, T)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.head_proj(h)                               # (B, feat_dim, T)
        return torch.amax(h, dim=-1)                        # global max-pool → (B, feat_dim)


# --------------------------------------------------------------------------- #
# Full MSDM  (Eq. 12 classifier)
# --------------------------------------------------------------------------- #
class MSDM(nn.Module):
    """Full pipeline: optional missing-indicator channel is concatenated to the
    raw series before kMSD (each input channel gets its own differencing).
    forward(x) where x is (B, C_in, T); C_in=1 (value) or 2 (value+mask)."""
    def __init__(self, scales, period=7, kernel=(-1.0, 1.0, 0.0),
                 filters=48, n_blocks=2, feat_dim=128, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.kmsd = KMSD(scales, kernel)
        n_scales = len(scales) * in_channels
        self.aq = AdaptiveQuant(n_scales)
        self.mscnet = MSCNet(n_scales, period, filters, n_blocks, feat_dim)
        self.mlp1 = nn.Linear(feat_dim, 4 * feat_dim)       # MLP1 width = 4× (Eq. 12)
        self.mlp2 = nn.Linear(4 * feat_dim, 1)

    def forward(self, x):                                   # (B, C_in, T) or (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        chans = [self.kmsd(x[:, c, :]) for c in range(x.shape[1])]  # each (B, n_sc, T)
        msd = torch.cat(chans, dim=1)                       # (B, n_sc*C_in, T)
        z = self.mscnet(self.aq(msd))                       # (B, feat_dim)
        return self.mlp2(F.relu(self.mlp1(z))).squeeze(-1)  # logit (B,)


__all__ = ["KMSD", "AdaptiveQuant", "IIPC", "MSCBlock", "MSCNet", "MSDM"]
