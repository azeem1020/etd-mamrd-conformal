"""Faithful re-implementations of published ETD baselines, wired to the SAME
harness contract as our arms so they run under the IDENTICAL honest protocol
(same customer-disjoint split, same per-user min-max preprocessing, same
PR-AUC/ROC/F1/P@k + per-phenotype metrics in run_n_clean.py).

Contract: every module takes x of shape (B, W, D) float32 with NaN cells
preserved (NaN-native, as load_and_preprocess emits) and returns a raw logit
of shape (B,). NaN handling and any extra input channels (missing-mask) are
done INSIDE each module. We deliberately hold the protocol/preprocessing fixed
across all models and vary ONLY the architecture — each paper's bespoke
preprocessing (e.g. Finardi's quantile transform) is replaced by our shared
pipeline so the comparison is controlled.

Models:
  * WDCNN     — Zheng et al., IEEE TII 2018 (Wide & Deep CNN + self-defined
                periodicity kernel). Ported 1:1 from the user's validated Keras
                deployment (TII_Wide-Deep_..., reproduces AUC 0.794): wide
                Dense(90) ∥ deep[Conv2D(15,3x3,stride3)->MaxPool3->Dense(60)]
                -> concat -> Dense(64) -> Dense(1). The torch kernel transform
                is unit-tested bit-equivalent to the numpy reference.
  * HORLN     — Zhu et al., IEEE TII 2023. First-order (IIWC dilated inter/intra
                week conv, dims 32->16, k=3 + HWGConv g=1 -> C1st=200) +
                second-order SDM (multihead self-dependency -> C2nd=300) +
                2-FC classifier. Input C=2 (value + observed-mask).
  * HybridAttn— Finardi et al., arXiv 2020. Two hybrid layers of
                [multi-head self-attention (Cbar=16) || dilated conv (d=1,2 ->
                32ch)] unified by 1x1 conv + LayerNorm + PReLU, then dense(1024)
                -> logit. Input C=2 (value + missing-mask, mask=1 where missing).
  * GRUD      — Che et al., Sci. Rep. 2018. Decay-based GRU for missing data;
                W=147 weeks as time steps, D=7 day-features (multivariate,
                per-feature input decay + hidden decay). Leakage-safe per-sample
                observed mean as the empirical mean fallback.
  * MSDMNet   — Wang et al., IEEE TSG 2025, wrapping our faithful repro
                (msdm_reproduction/msdm/model.py) with the paper's SGCC config
                (scales [1,2,7,28,91,182], period 7, in_channels=2).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_MSDM_DIR = Path(__file__).resolve().parent.parent.parent / "msdm_reproduction"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _value_and_mask(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,W,D) NaN-native -> (value with NaN->0, observed-mask 1.0 where present)."""
    mask = (~torch.isnan(x)).float()
    val = torch.nan_to_num(x, nan=0.0)
    return val, mask


# ===========================================================================
# 1. WDCNN  (Zheng et al., IEEE TII 2018)
# ===========================================================================
def wdcnn_kernel_transform(grid: torch.Tensor) -> torch.Tensor:
    """Torch port of the validated numpy `self_define_cnn_kernel_process`
    (TII_Wide-Deep_.../function.py).  grid: (B, W, D) -> (B, 1, 3W, 3D).

    The numpy reference does: 3x-overlapping-triplet expand on both axes, then
    `preprocess_kernel` whose second (column) loop fully overwrites the first,
    so the EFFECTIVE transform is the per-3-column anti-mean
    [2*c0-c1-c2, 2*c1-c0-c2, 2*c2-c0-c1] applied to the column-expanded grid.
    We reproduce that exact effective behaviour (verified in __main__).
    """
    B, W, D = grid.shape
    dev, dt = grid.device, grid.dtype
    # ---- row expand (axis-0): out row 3i+t = grid[i+t], boundary tail -> 0 ----
    e0 = torch.zeros(B, 3 * W, D, device=dev, dtype=dt)
    for t in range(3):
        idx = torch.arange(W, device=dev) + t          # source rows i+t
        valid = idx < W
        rows = torch.arange(W, device=dev) * 3 + t      # dest rows 3i+t
        if valid.any():
            e0[:, rows[valid], :] = grid[:, idx[valid], :]
    # ---- col expand (axis-1): out col 3j+t = e0[:,:,j+t], boundary tail -> 0 ----
    e = torch.zeros(B, 3 * W, 3 * D, device=dev, dtype=dt)
    for t in range(3):
        idx = torch.arange(D, device=dev) + t
        valid = idx < D
        cols = torch.arange(D, device=dev) * 3 + t
        if valid.any():
            e[:, :, cols[valid]] = e0[:, :, idx[valid]]
    # ---- effective kernel = per-3-column anti-mean (col loop overwrites row loop) ----
    c0, c1, c2 = e[:, :, 0::3], e[:, :, 1::3], e[:, :, 2::3]   # each (B,3W,D)
    out = torch.empty_like(e)
    out[:, :, 0::3] = 2 * c0 - c1 - c2
    out[:, :, 1::3] = 2 * c1 - c0 - c2
    out[:, :, 2::3] = 2 * c2 - c0 - c1
    return out.unsqueeze(1)                              # (B,1,3W,3D)


class WDCNN(nn.Module):
    def __init__(self, weeks: int = 147, days: int = 7,
                 gamma: int = 15, alpha: int = 90, beta: int = 60, fuse: int = 64):
        super().__init__()
        self.weeks, self.days = weeks, days
        # deep branch: stride-3 conv on the 3x-expanded kernel grid -> (W,D)
        self.conv = nn.Conv2d(1, gamma, kernel_size=3, stride=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        with torch.no_grad():                           # infer flat dim once
            d = self.pool(self.conv(torch.zeros(1, 1, 3 * weeks, 3 * days)))
            self._deep_flat = d.numel()
        self.deep_fc = nn.Linear(self._deep_flat, beta)
        self.wide_fc = nn.Linear(weeks * days, alpha)
        self.fuse_fc = nn.Linear(alpha + beta, fuse)
        self.head = nn.Linear(fuse, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, _ = _value_and_mask(x)                     # NaN->0 (Zheng step-1 tail rule)
        wide = F.relu(self.wide_fc(val.flatten(1)))     # (B, alpha)
        deep_in = wdcnn_kernel_transform(val)           # (B,1,3W,3D)
        h = F.relu(self.conv(deep_in))
        h = self.pool(h).flatten(1)
        deep = F.relu(self.deep_fc(h))                  # (B, beta)
        f = F.relu(self.fuse_fc(torch.cat([wide, deep], dim=1)))
        return self.head(f).squeeze(-1)


# ===========================================================================
# 2. HORLN  (Zhu et al., IEEE TII 2023)
# ===========================================================================
class IIWCBlock(nn.Module):
    """Inter-and-Intra Week Convolution block. Parallel dilated convs (d=1..r):
    InterWeekConv kernel (3,1) along the week axis, IntraWeekConv kernel (1,3)
    along the day axis; all outputs concatenated then fused (1x1) + PReLU + BN.
    """
    def __init__(self, cin: int, cout: int, r: int = 3, k: int = 3):
        super().__init__()
        pad_off = k // 2
        self.inter = nn.ModuleList([
            nn.Conv2d(cin, cout, (k, 1), padding=(d * pad_off, 0), dilation=(d, 1))
            for d in range(1, r + 1)])
        self.intra = nn.ModuleList([
            nn.Conv2d(cin, cout, (1, k), padding=(0, d * pad_off), dilation=(1, d))
            for d in range(1, r + 1)])
        self.fuse = nn.Conv2d(2 * r * cout, cout, kernel_size=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU()

    def forward(self, x):
        parts = [c(x) for c in self.inter] + [c(x) for c in self.intra]
        return self.act(self.bn(self.fuse(torch.cat(parts, dim=1))))


class HWGConv(nn.Module):
    """H x W group convolution: a spatial-sized kernel collapsing (H,W)->(1,1),
    groups g (paper sets g=1). Output (B, cout)."""
    def __init__(self, cin: int, cout: int, h: int, w: int, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(h, w), groups=groups)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.conv(x).flatten(1))        # (B, cout)


class SDM(nn.Module):
    """Self-Dependency Modeling: multihead day-day correlation (second order).
    F^uiw (B,Cmid,H,W) -> reshape M (B,h,W,Dh), Dh=H*Cmid/h -> F^sdm=M @ M^T
    (B,h,W,W) -> HWGConv -> F^2nd (B,C2nd)."""
    def __init__(self, cmid: int, h_weeks: int, w_days: int, heads: int, c2nd: int):
        super().__init__()
        assert (h_weeks * cmid) % heads == 0, "heads must divide H*Cmid"
        self.heads = heads
        self.w = w_days
        self.hwg = HWGConv(heads, c2nd, w_days, w_days, groups=1)

    def forward(self, x):                               # x: (B,Cmid,H,W)
        B, C, H, W = x.shape
        Dh = H * C // self.heads
        m = x.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B,W,C*H)
        m = m.reshape(B, W, self.heads, Dh).permute(0, 2, 1, 3)  # (B,heads,W,Dh)
        sdm = torch.matmul(m, m.transpose(-1, -2))      # (B,heads,W,W)
        return self.hwg(sdm)                            # (B,c2nd)


class HORLN(nn.Module):
    def __init__(self, weeks: int = 147, days: int = 7,
                 iiwc_dims=(32, 16), c1st: int = 200, c2nd: int = 300,
                 heads: int = 8, r: int = 3, fc_hidden: int = 128):
        super().__init__()
        self.weeks, self.days = weeks, days
        cin = 2                                         # value + observed-mask
        self.block1 = IIWCBlock(cin, iiwc_dims[0], r=r)
        self.block2 = IIWCBlock(iiwc_dims[0], iiwc_dims[1], r=r)
        cmid = iiwc_dims[1]
        self.hwg_first = HWGConv(cmid, c1st, weeks, days, groups=1)   # F^1st
        self.sdm = SDM(cmid, weeks, days, heads=heads, c2nd=c2nd)     # F^2nd
        self.fc1 = nn.Linear(c1st + c2nd, fc_hidden)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        val, mask = _value_and_mask(x)                  # (B,W,D) each
        h = torch.stack([val, mask], dim=1)             # (B,2,W,D)
        f_uiw = self.block2(self.block1(h))             # (B,cmid,W,D)
        f1 = self.hwg_first(f_uiw)                       # (B,c1st)
        f2 = self.sdm(f_uiw)                             # (B,c2nd)
        z = self.act(self.fc1(torch.cat([f1, f2], dim=1)))
        return self.fc2(z).squeeze(-1)


# ===========================================================================
# 3. HybridAttn  (Finardi et al., arXiv 2020)
# ===========================================================================
class MultiHeadSelfAttention2D(nn.Module):
    """Finardi eq.(1)-(2): input (B,C,L,D) -> flatten (B,L,C*D) -> q,k,v linear
    to (B,L,Cbar*D) -> reshape (B,Cbar,L,D) -> Attn = softmax(q k^T/sqrt(D)) v
    along the sequence axis L -> (B,Cbar,L,D)."""
    def __init__(self, cin: int, cbar: int, d: int):
        super().__init__()
        self.cbar, self.d = cbar, d
        self.wq = nn.Linear(cin * d, cbar * d)
        self.wk = nn.Linear(cin * d, cbar * d)
        self.wv = nn.Linear(cin * d, cbar * d)

    def _proj(self, lin, xf, B, L):
        return lin(xf).view(B, L, self.cbar, self.d).permute(0, 2, 1, 3)  # (B,Cbar,L,D)

    def forward(self, x):                               # (B,C,L,D)
        B, C, L, D = x.shape
        xf = x.permute(0, 2, 1, 3).reshape(B, L, C * D)
        q, k, v = self._proj(self.wq, xf, B, L), self._proj(self.wk, xf, B, L), self._proj(self.wv, xf, B, L)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.d ** 0.5), dim=-1)
        return torch.matmul(att, v)                     # (B,Cbar,L,D)


class HybridLayer(nn.Module):
    """[self-attn (Cbar) || dilated conv (d=1,2, each conv_each ch -> 2*conv_each)]
    concatenated, unified by 1x1 conv -> GroupNorm(1) (LayerNorm-equiv) -> PReLU."""
    def __init__(self, cin: int, cbar: int, conv_each: int, cout: int, d_days: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention2D(cin, cbar, d_days)
        self.conv1 = nn.Conv2d(cin, conv_each, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(cin, conv_each, 3, padding=2, dilation=2)
        self.unify = nn.Conv2d(cbar + 2 * conv_each, cout, kernel_size=1)
        self.norm = nn.GroupNorm(1, cout)
        self.act = nn.PReLU()

    def forward(self, x):                               # (B,C,L,D)
        a = self.attn(x)
        c = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        return self.act(self.norm(self.unify(torch.cat([a, c], dim=1))))


class HybridAttn(nn.Module):
    def __init__(self, weeks: int = 147, days: int = 7,
                 cbar: int = 16, conv_each: int = 16, cout: int = 32, fc_hidden: int = 1024):
        super().__init__()
        self.weeks, self.days = weeks, days
        cin = 2                                         # value + missing-mask
        self.layer1 = HybridLayer(cin, cbar, conv_each, cout, days)
        self.layer2 = HybridLayer(cout, cbar, conv_each, cout, days)
        self.fc1 = nn.Linear(cout * weeks * days, fc_hidden)
        self.act = nn.PReLU()
        self.head = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        val, miss = _value_and_mask(x)
        miss = 1.0 - miss                               # Finardi: mask=1 where MISSING
        h = torch.stack([val, miss], dim=1)             # (B,2,W,D); L=weeks, D=days
        h = self.layer2(self.layer1(h))                 # (B,cout,W,D)
        z = self.act(self.fc1(h.flatten(1)))
        return self.head(z).squeeze(-1)


# ===========================================================================
# 4. GRU-D  (Che et al., Scientific Reports 2018)
# ===========================================================================
class GRUD(nn.Module):
    """Decay-based GRU for missing data. Time axis = weeks (W steps), features =
    days (D). Per-feature input decay gamma_x and full hidden decay gamma_h on
    interval delta (steps since last observation). Empirical mean is the
    leakage-safe per-sample observed mean per feature."""
    def __init__(self, weeks: int = 147, days: int = 7, hidden: int = 64):
        super().__init__()
        self.weeks, self.days, self.hidden = weeks, days, hidden
        D, H = days, hidden
        self.w_dg_x = nn.Linear(D, D)                   # gamma_x (diagonal-style, per-feature)
        self.w_dg_h = nn.Linear(D, H)                   # gamma_h (full)
        # GRU gates take [x_hat (D), mask (D)] and decayed hidden (H)
        self.x2h = nn.Linear(2 * D, 3 * H)
        self.h2h = nn.Linear(H, 3 * H)
        self.head = nn.Linear(H, 1)

    def forward(self, x):                               # (B,W,D) NaN-native
        B, W, D = x.shape
        dev = x.device
        mask = (~torch.isnan(x)).float()                # (B,W,D)
        val = torch.nan_to_num(x, nan=0.0)
        # per-sample observed mean per feature (fallback 0 if a feature all-missing)
        denom = mask.sum(dim=1).clamp_min(1.0)          # (B,D)
        x_mean = (val * mask).sum(dim=1) / denom        # (B,D)
        # delta (steps since last obs), per Che: 0 at t=0; else 1 (+prev if prev missing)
        delta = torch.zeros(B, W, D, device=dev)
        for t in range(1, W):
            prev_m = mask[:, t - 1, :]
            delta[:, t, :] = 1.0 + (1.0 - prev_m) * delta[:, t - 1, :]
        h = torch.zeros(B, self.hidden, device=dev)
        x_last = x_mean.clone()                          # last observed value, init to mean
        for t in range(W):
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]
            x_t = val[:, t, :]
            g_x = torch.exp(-F.relu(self.w_dg_x(d_t)))   # (B,D)
            g_h = torch.exp(-F.relu(self.w_dg_h(d_t)))   # (B,H)
            x_hat = m_t * x_t + (1 - m_t) * (g_x * x_last + (1 - g_x) * x_mean)
            h_dec = g_h * h
            gates_x = self.x2h(torch.cat([x_hat, m_t], dim=1))
            gates_h = self.h2h(h_dec)
            xr, xz, xn = gates_x.chunk(3, dim=1)
            hr, hz, hn = gates_h.chunk(3, dim=1)
            r = torch.sigmoid(xr + hr)
            z = torch.sigmoid(xz + hz)
            n = torch.tanh(xn + r * hn)
            h = (1 - z) * h_dec + z * n
            x_last = m_t * x_t + (1 - m_t) * x_last       # carry last observed
        return self.head(h).squeeze(-1)


# ===========================================================================
# 5. MSDM  (Wang et al., IEEE TSG 2025) — wrap faithful repro on (B,W,D)
# ===========================================================================
class MSDMNet(nn.Module):
    def __init__(self, weeks: int = 147, days: int = 7):
        super().__init__()
        if str(_MSDM_DIR) not in sys.path:
            sys.path.insert(0, str(_MSDM_DIR))
        from msdm.model import MSDM                       # noqa: E402
        self.net = MSDM(scales=[1, 2, 7, 28, 91, 182], period=7,
                        kernel=(-1.0, 1.0, 0.0), filters=48, n_blocks=2,
                        feat_dim=128, in_channels=2)

    def forward(self, x):                               # (B,W,D)
        B = x.shape[0]
        val, mask = _value_and_mask(x)
        seq = val.reshape(B, -1)                        # (B,T)
        msk = mask.reshape(B, -1)
        return self.net(torch.stack([seq, msk], dim=1)) # (B,2,T) -> logit (B,)


# ===========================================================================
# 6. DKGT-Net  (Tian et al., IEEE TII 2026) — faithful KAN+GRU dual-branch core
# ---------------------------------------------------------------------------
# CONTROLLED COMPARISON (state in paper): the published DKGT-Net pairs this
# KAN+GRU dual-branch detector with (i) WGAN-GP theft augmentation, (ii) a CSSI
# consistency-driven training loss, and (iii) reports F1/ACC/AUC on an enhanced
# 73,900-user set whose theft class is ~90% GAN-SYNTHETIC (32,143 generated vs
# ~3,600 real) at a 45.3% base rate. Evaluating on synthetic theft at an inflated
# base rate is exactly the measurement artifact this paper studies (C3); we
# re-implement the ARCHITECTURE under the identical honest protocol (per-user
# min-max, leakage-free split, real 8.5% prevalence, PR-AUC primary) with the
# standard training recipe used for every other baseline, holding the augmentation
# and the CSSI loss off. The reported score is the architecture under a common
# protocol, NOT a refutation of the full DKGT-Net method (whose CSSI ablation,
# F1 0.95->0.62, is itself measured on that synthetic/balanced set).
# Simplifications (documented): KANLinear uses efficient B-spline edges (Liu 2024);
# the ECA channel-attention + "spreading" layer are realised as a tanh-sigmoid
# fusion gate. Input grid (B,W,D).
# ===========================================================================
def _stencil_3x3(grid: torch.Tensor) -> torch.Tensor:
    """3x overlapping-triplet expansion on BOTH axes (no anti-mean): (B,W,D)->(B,3W,3D).
    Realises DKGT-Net's KAN input shape (weeks x 3, days x 3) (Tian 2026 Fig.3, Eq.10).
    Reuses the expansion half of the WDCNN reference transform; boundary tail -> 0."""
    B, W, D = grid.shape
    dev, dt = grid.device, grid.dtype
    e0 = torch.zeros(B, 3 * W, D, device=dev, dtype=dt)
    for t in range(3):
        idx = torch.arange(W, device=dev) + t; valid = idx < W
        rows = torch.arange(W, device=dev) * 3 + t
        if valid.any():
            e0[:, rows[valid], :] = grid[:, idx[valid], :]
    e = torch.zeros(B, 3 * W, 3 * D, device=dev, dtype=dt)
    for t in range(3):
        idx = torch.arange(D, device=dev) + t; valid = idx < D
        cols = torch.arange(D, device=dev) * 3 + t
        if valid.any():
            e[:, :, cols[valid]] = e0[:, :, idx[valid]]
    return e                                                       # (B,3W,3D)


class KANLinear(nn.Module):
    """B-spline parametrised KAN-Linear edge (Liu et al. 2024; Tian 2026 Eq.7-9,19).
    O = O_base + O_offset = base(silu(x)) + spline(x); spline via uniform B-spline
    basis of `order` over `grid` intervals on [-1,1] (partition of unity). Spline
    weights init std = in_f^-0.5 so the summed spline output stays O(1) for large in_f."""
    def __init__(self, in_f: int, out_f: int, grid: int = 5, order: int = 3):
        super().__init__()
        self.in_f, self.out_f, self.grid, self.order = in_f, out_f, grid, order
        h = 2.0 / grid
        knots = torch.arange(-1 - order * h, 1 + order * h + 1e-6, h)
        self.register_buffer("knots", knots)
        self.base_w = nn.Linear(in_f, out_f)
        n_basis = grid + order
        self.spline_w = nn.Parameter(torch.randn(out_f, in_f, n_basis) * (in_f ** -0.5))

    def _bspline(self, x):                                          # (B,in_f)->(B,in_f,n_basis)
        x = torch.tanh(x).unsqueeze(-1)                            # squash to [-1,1]
        t = self.knots
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)             # order-0
        for k in range(1, self.order + 1):
            lo = (x - t[:-(k+1)]) / (t[k:-1] - t[:-(k+1)] + 1e-8)
            hi = (t[k+1:] - x) / (t[k+1:] - t[1:-k] + 1e-8)
            b = lo * b[..., :-1] + hi * b[..., 1:]
        return b                                                   # (B,in_f,grid+order)

    def forward(self, x):
        base = self.base_w(torch.nn.functional.silu(x))
        spline = torch.einsum("bik,oik->bo", self._bspline(x), self.spline_w)
        return base + spline


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020; Tian 2026 fusion module):
    1-D conv over the channel axis -> sigmoid gate -> channel-wise reweight."""
    def __init__(self, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):                                          # (B,C)
        a = self.conv(x.unsqueeze(1)).squeeze(1)                   # (B,C)
        return x * torch.sigmoid(a)


class DKGTNet(nn.Module):
    """Faithful KAN+GRU dual-branch core (Tian 2026, IEEE TII). KAN branch: 3-layer
    KAN(x)=(Phi3 o Phi2 o Phi1)(x) on the (weeks x 3, days x 3) stencil -> 128, dropout
    0.5 (Eq.10,20). GRU branch: 2-layer GRU over (weeks, days) -> 128 (Eq.12,17).
    Fusion: F_fusion=FC(concat) (Eq.13); A=sigmoid(tanh(F_fusion)) (Eq.14);
    F_att=ECA(F_gru * A) (Eq.15 + ECA module); head on concat(F_fusion,F_att,F_kan) (Eq.16).
    WGAN augmentation + CSSI consistency-training are the controlled-OFF inflation variables."""
    def __init__(self, weeks: int = 147, days: int = 7, kan_hidden: int = 128,
                 gru_hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.weeks, self.days = weeks, days
        flat3 = 9 * weeks * days                                   # (W*3)(D*3)
        self.kan1 = KANLinear(flat3, kan_hidden)                  # Phi1
        self.kan2 = KANLinear(kan_hidden, kan_hidden)            # Phi2
        self.kan3 = KANLinear(kan_hidden, kan_hidden)            # Phi3
        self.kan_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(days, gru_hidden, num_layers=2, batch_first=True,
                          dropout=dropout)
        self.fuse = nn.Linear(kan_hidden + gru_hidden, gru_hidden)
        self.eca = ECA(k=3)
        self.head = nn.Sequential(
            nn.Linear(kan_hidden + 2 * gru_hidden, 128), nn.PReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):                                          # (B,W,D)
        v = torch.nan_to_num(x, 0.0)
        stencil = _stencil_3x3(v).reshape(v.shape[0], -1)          # (B, 9WD)
        fk = self.kan_drop(self.kan3(self.kan2(self.kan1(stencil))))  # (B,128)
        _, h = self.gru(v)                                         # h (2,B,128)
        fg = h[-1]                                                 # (B,128)
        ffus = self.fuse(torch.cat([fk, fg], dim=1))               # (B,128)  Eq.13
        a = torch.sigmoid(torch.tanh(ffus))                        # Eq.14
        fatt = self.eca(fg * a)                                    # Eq.15 + ECA
        return self.head(torch.cat([ffus, fatt, fk], dim=1)).squeeze(-1)  # Eq.16


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_wdcnn(weeks=147, days=7):       return WDCNN(weeks, days)
def build_horln(weeks=147, days=7):       return HORLN(weeks, days)
def build_hybridattn(weeks=147, days=7):  return HybridAttn(weeks, days)
def build_grud(weeks=147, days=7):        return GRUD(weeks, days)
def build_msdm(weeks=147, days=7):        return MSDMNet(weeks, days)
def build_dkgt(weeks=147, days=7):        return DKGTNet(weeks, days)

BASELINE_BUILDERS = {
    "wdcnn": build_wdcnn,
    "horln": build_horln,
    "hybridattn": build_hybridattn,
    "grud": build_grud,
    "msdm_net": build_msdm,
    "dkgt": build_dkgt,
}


# ---------------------------------------------------------------------------
# self-test: WDCNN kernel bit-equivalence vs numpy reference + forward shapes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    ref_dir = Path("reference_wdcnn")  # external WDCNN reference for the __main__ self-test only
    sys.path.insert(0, str(ref_dir))
    from function import self_define_cnn_kernel_process  # numpy reference

    rng = np.random.default_rng(0)
    W, D = 5, 4                                          # small grid for the check
    g = rng.standard_normal((3, W, D)).astype(np.float32)
    ref = self_define_cnn_kernel_process(g.reshape(3, W, D, 1))     # (3,3W,3D,1)
    ours = wdcnn_kernel_transform(torch.from_numpy(g))             # (3,1,3W,3D)
    ref_t = torch.from_numpy(ref[..., 0])                          # (3,3W,3D)
    max_err = (ours[:, 0] - ref_t).abs().max().item()
    print(f"[wdcnn kernel] max|torch-numpy| = {max_err:.3e}  ->",
          "OK" if max_err < 1e-5 else "MISMATCH")

    xb = torch.randn(8, 147, 7)
    xb[xb > 1.5] = float("nan")                          # inject NaN
    for name, b in BASELINE_BUILDERS.items():
        m = b()
        out = m(xb)
        n_par = sum(p.numel() for p in m.parameters()) / 1e6
        assert out.shape == (8,), f"{name} bad shape {out.shape}"
        assert torch.isfinite(out).all(), f"{name} non-finite logits"
        print(f"[{name:11s}] out {tuple(out.shape)}  params={n_par:.3f}M  OK")
