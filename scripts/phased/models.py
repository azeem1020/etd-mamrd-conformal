"""ZKMixV4 — Context-Conditional NaME + Multi-Scale Laplacian + Phase E extensions.

Phase D model.  Separate from zkmix_lab; does NOT modify any Phase C file.

Architecture
------------
ZKMixV4 is ZKMixV3 with two targeted changes in Phase D:

1.  **CC-NaME replaces scalar NaME.**  The instance-adaptive token τ_n
    (computed from valid-cell context) is applied at the front of every
    branch — identical pass-through point as the scalar NaME in ZKMixV3.

2.  **Multi-scale Laplacian replaces single-scale CNN branch.**
    The frozen Zheng 9-stencil at d=1 is extended to a three-scale pyramid
    at d∈{1,2,4}.  The branch embedding dimension is unchanged (embed_dim=60)
    so the fusion head is unaffected.

Phase E adds two further optional branches (independently switchable):

3.  **Explicit missing-mask channel (A7).**  The binary validity mask M
    (1.0 = observed, 0.0 = NaN) is passed through a second Laplacian CNN
    parallel to the consumption CNN.  M contains theft-specific temporal
    structure (meter tampering fingerprints) that is DESTROYED by NaME
    substitution in the standard path.  The Laplacian of M detects NaN
    boundary patterns: periodic gaps from meter tampering produce periodic
    spikes in L(M).

4.  **Global temporal trend branch (A8).**  A 3-layer MLP processes 21
    monthly summary statistics (mean, std, nan_frac per 7-week block) over
    the full 3-year trajectory.  This is the only branch with sufficient
    temporal receptive field to detect Light Dropper/Riser archetypes
    (gradual drift over 6–24 months, SNR ≈ 0.01 in any 9-week window).
    The monthly nan_frac sequence also encodes the Jan-2016 MNAR regime
    shift implicitly, providing regime-conditional imputation without
    requiring explicit regime labels.

Everything else is identical to ZKMixV3:
  - Wide branch: Flatten → Linear(W*D, α) → ReLU → Dropout
  - Mixer branch: patch_proj + kernel-stat injection → MixerBlock2D×2 →
                  multi-pool → Linear(3*mixer_hidden, mixer_out_dim)
  - Fusion: concat(wide, cnn_ms, mixer, [mask_cnn], [temporal]) →
            Linear(embed_dim, fuse_hidden) → ReLU → Dropout → Linear(fuse_hidden, 1)
  - Optimizer/schedule/loss: held constant from Phase C (TrainConfig defaults)

Ablation switches
-----------------
Both Phase D and Phase E components can be independently disabled:

    Phase D arms (PLAN_D_2026-05-27.md):
    use_cc_name=True,  use_ms_kernel=True  → A3 (Phase D headline)
    use_cc_name=True,  use_ms_kernel=False → A1 (CC-NaME only)
    use_cc_name=False, use_ms_kernel=True  → A2 (multi-scale only)
    use_cc_name=False, use_ms_kernel=False → equivalent to Phase C NaME-NN

    Phase E arms (PLAN_E_2026-05-27.md):
    A3 + use_mask_channel=True             → A7 (mask channel)
    A3 + use_temporal_branch=True          → A8 (temporal branch)
    A3 + use_mask_channel + use_temporal   → A9 (full Phase E headline candidate)

    Phase I arms (PLAN_I_2026-05-27.md — ablation of A12):
    A12 + use_cnn_consump=False            → A13 (drop CNN-consump, embed=216)
    A12 + use_mixer=False                  → A14 (drop Mixer, embed=212)
    A12 + use_mask_channel=False           → A15 (drop CNN-mask, embed=246)
    A12 + use_temporal_branch=False        → A16 (drop Temporal MLP, embed=244)
    A9 (linear Wide)  already registered  → A12 − MambaWide (embed=276)
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve zkmix_lab on sys.path (supports running from any CWD).
# ---------------------------------------------------------------------------

def _ensure_zkmix_on_path() -> None:
    lab = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "zkmix_lab")
    )
    if lab not in sys.path:
        sys.path.insert(0, lab)

_ensure_zkmix_on_path()

from zkmix_lab.kernels import KernelSpec, weekly_kernel_stats   # type: ignore
from zkmix_lab.models import MixerBlock2D, _multi_pool          # type: ignore

from .cc_name import ContextConditionalNaME
from .ms_kernel import MultiScaleLaplacian


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZKMixV4Config:
    """Hyper-parameters for ZKMixV4.

    Defaults are kept identical to ZKMixV3Config so that the Phase C
    training recipe applies without modification.
    """
    weeks: int = 147
    days:  int = 7

    # ----- Wide branch (unchanged from Phase C) -----
    alpha: int = 90

    # ----- Deep-CNN branch (multi-scale Laplacian) -----
    beta:  int = 60          # embed_dim for the consumption CNN branch
    gamma: int = 15          # out_channels_per_scale in MultiScaleLaplacian

    # Dilation scales for the multi-scale Laplacian pyramid.
    # (1,) reproduces the single-scale Phase C CNN branch.
    ms_dilations: tuple[int, ...] = (1, 2, 4)
    ms_freeze_stencils: bool = True   # freeze Zheng prior at each scale

    # ----- Mixer branch (unchanged from Phase C) -----
    mixer_hidden:      int = 96
    mixer_token_mlp:   int = 192
    mixer_channel_mlp: int = 192
    mixer_n_blocks:    int = 2
    mixer_out_dim:     int = 64
    kernel: KernelSpec = field(default_factory=KernelSpec)
    stats: tuple[str, ...] = ("mean", "std", "max", "p99")
    inject_kernel_stats: bool = True

    # ----- Fusion head (unchanged from Phase C) -----
    fuse_hidden: int = 96
    dropout: float = 0.10

    # ----- CC-NaME settings -----
    use_cc_name: bool = True        # True → CC-NaME;  False → scalar NaME (A0-equivalent)
    cc_hidden:   int = 32           # context encoder hidden dim
    cc_dropout:  float = 0.1        # context encoder dropout
    base_token_init: float = -1.0   # matches Phase C init convention

    # Which branches receive CC-NaME imputation.  Default = all three (A1/A2/A3 behaviour).
    # Set to ("wide", "mixer") for A6: CNN branch receives scalar NaME instead.
    cc_name_branches: tuple[str, ...] = ("wide", "mixer", "cnn")

    # ----- Multi-scale kernel switch -----
    use_ms_kernel: bool = True      # True → pyramid at ms_dilations;  False → d=1 only

    # NaN-native path (consistent with Phase C NaME-NN arm).
    nan_native: bool = True

    # ----- Phase E: A7 — explicit missing-mask channel -----
    # The binary validity mask M is processed through a second Laplacian CNN
    # (same dilations as the consumption CNN).  The Laplacian of M detects
    # NaN boundary patterns: the spatial-temporal fingerprint of meter
    # tampering that NaME substitution actively destroys.
    use_mask_channel: bool = False
    mask_beta: int = 30             # embed dim for the mask CNN branch (½ of beta)

    # ----- Phase E: A8 — global temporal trend branch -----
    # 21 monthly blocks (7 weeks each) × 3 statistics (mean, std, nan_frac)
    # → MLP → temporal_dim.  Targets Light Dropper/Riser archetypes that are
    # structurally undetectable at any local receptive field (SNR ≈ 0.01 in
    # a 9-week window).  The nan_frac sequence also encodes the Jan-2016 MNAR
    # regime shift, providing implicit regime conditioning.
    use_temporal_branch: bool = False
    temporal_dim: int = 32          # output dim of temporal MLP
    temporal_mlp_hidden: int = 64   # hidden dim of temporal MLP

    # ----- Phase G: A11 — BiLSTM wide branch (K1 triggered — null result) -----
    # Replaces the linear Wide branch with a 2-layer BiLSTM + temporal attention.
    # K1 triggered: all three folds negative (Δ=−0.0066, CI=[−0.0146,+0.0020]).
    # Root cause: uniform LSTM gating dilutes step-change signal across 147 steps.
    use_bilstm_wide: bool = False
    bilstm_hidden: int = 32         # hidden per direction; concat → 64 → proj to alpha
    bilstm_layers: int = 2

    # ----- Phase H: A12 — Mamba2 wide branch -----
    # Replaces the linear Wide branch with a BiMamba2 (bidirectional Mamba2) +
    # temporal attention pooling.  Unlike BiLSTM, Mamba2's input-dependent Δ
    # (step-size parameter) amplifies state transitions at discriminative time
    # steps (Heavy Dropper week T: large Δ → large hidden-state jump) while
    # smoothly integrating quiet periods (Light Dropper drift: small Δ per step,
    # cumulative SNR ≈ 1.9 over 147 weeks).  The MNAR regime boundary at week
    # ~104 produces a large Δ transition, giving implicit regime conditioning.
    # chunk_size=21 divides seq_len=147 cleanly (7 chunks of 21 weeks each).
    # Requires mamba_ssm 2.3.2: conda run -n mamba_gpu python run_local_grid.py
    use_mamba_wide: bool = False
    mamba_d_model: int = 32         # SSM projection dim; inner_dim = d_model × expand = 64
    mamba_d_state: int = 128        # SSM state dimension (rich memory for 147-week trajectories)
    mamba_d_conv: int = 4           # local convolution context (4 weeks ≈ 1 month)
    mamba_expand: int = 2           # inner expansion factor
    mamba_headdim: int = 32         # SSM head dim; num_heads = inner_dim / headdim = 2

    # ----- Phase I: ablation branch flags (PLAN_I_2026-05-27.md) -----
    # Set to False to drop the named branch from the model.  Used for the
    # A12 ablation study to identify which branches contribute incrementally.
    use_cnn_consump: bool = True    # False → omit CNN-consump (Deep-CNN) branch
    use_mixer: bool = True          # False → omit Mixer branch

    # ----- Phase J: Peer-Relative Consumption Encoding (PLAN_J_2026-05-27.md) -----
    # When True the model expects input (N, W, D*2): the first D columns are the
    # standard imputed consumption series; the last D columns are the peer-relative
    # residual r[c,t] = x̂[c,t] − mean(x̂[k-nearest normal peers, t]).
    # The residual is processed by a second MultiScaleLaplacian CNN (PRCE-CNN).
    # Peer groups are computed in runner_d.compute_peer_residuals() using only
    # normal customers in the current fold's training set (leakage-safe).
    use_prce: bool = False
    prce_beta: int = 60             # embed dim for PRCE-CNN (= beta by default)

    # ----- Phase K: Attention-Based Peer Comparison (PLAN_K_2026-05-27.md) -----
    # When True the PRCE input (N, W, D*2) is consumed by a learned cross-attention
    # branch (PeerAttentionBranch) instead of the hand-coded Laplacian-CNN. Replaces
    # the prce_cnn_branch when use_prce=True and use_peer_gnn=True. Output dim is
    # ``peer_gnn_dim`` (default 60 — matches prce_beta for clean A17 / A20 contrast).
    use_peer_gnn: bool = False
    peer_gnn_dim: int = 60
    peer_gnn_hidden: int = 64
    peer_gnn_heads: int = 4
    peer_gnn_kernel: int = 7

    # ----- Phase L: Signed-Peer Cross-Attention (PLAN_L) -----
    # When True the runner stacks (own, signed_residual, peer_std, peer_iqr) into
    # an (N, W, 4*D) tensor.  Input layout is:
    #   x[:, :, 0:D]           = own consumption (NaN allowed; goes through CC-NaME)
    #   x[:, :, D:4*D]         = pre-computed (signed_residual | peer_std | peer_iqr)
    # The SignedPeerBranch (phased.signed_peer.SignedPeerBranch) handles the
    # peer portion; embed_dim contribution = ``signed_peer_dim`` (default 60).
    # Mutually exclusive with use_peer_gnn (which uses 2*D layout).
    use_signed_peer: bool = False
    signed_peer_dim: int = 60
    signed_peer_hidden: int = 64
    signed_peer_heads: int = 4
    signed_peer_kernel: int = 7

    # ----- Phase L: Engineered-Features Branch (A24, PLAN_L) -----
    # When True, a small MLP processes 12 pre-computed engineered features
    # (max_consec_nan, lag1_autocorr, missing_frac, roll_std_max, roll_std_range,
    #  consumption_entropy, wed_sat_gap, dow_profile_var, cv_global, iqr,
    #  p90_p10_range, mean_dow_5).  Features are appended to the input tensor by
    # the runner along the LAST AXIS as an additional channel block of width 12,
    # but ZKMixV4 splits internally — see ``embed()``.
    # When this branch is active the model expects:
    #     x[:, :, 0:D]                     own consumption
    #     [if PRCE/signed_peer:]           peer block (D*1, D*3, or D*4)
    # PLUS a SEPARATE eng_feat tensor of shape (N, 12) is passed via a model
    # method (see set_eng_features), NOT stacked along the day axis (the
    # engineered features are time-invariant per customer).
    use_eng_features: bool = False
    eng_features_dim: int = 30
    eng_features_hidden: int = 64
    eng_features_n: int = 12

    # ----- Phase N: multi-scale DIFFERENCING branch (MSDM, Wang TSG 2025) -----
    # Captures rate-of-change rather than level: Delta_d x[t] = x_imp[t]-x_imp[t-d]
    # at several dilations, adaptive-quantised tanh(beta*Delta) (beta learnable),
    # then a 1-D CNN -> pooled -> Linear (zero-init: no-op at init, clean a31 vs a14).
    use_diff_branch: bool = False
    diff_scales: tuple[int, ...] = (1, 7, 30)
    diff_dim: int = 32
    diff_hidden: int = 48

    # ----- MSDM inter/intra-period convolution branch (MSCNet IIPC, generic) -----
    # Dilated convs on the imputed daily sequence at intra-period {1,2} and
    # inter-period {C,2C} dilations, where the period C = `days` (one cycle:
    # week=7 for SGCC daily, day=24/48 for hourly/30-min). Resolution-agnostic:
    # C adapts to the grid. Captures cross-cycle structure (theft alters the
    # periodic rhythm). Zero-init output → no-op at init (clean ablation).
    use_interperiod_branch: bool = False
    ip_dim: int = 32
    ip_hidden: int = 32


# ---------------------------------------------------------------------------
# Phase G: BiLSTM Wide branch
# ---------------------------------------------------------------------------

class BiLSTMWide(nn.Module):
    """Wide branch: 2-layer BiLSTM with temporal attention pooling.

    Replaces the linear Wide branch (Flatten→Linear) to capture sequential
    temporal dynamics invisible to a bag-of-features projection:

    - Light Dropper/Riser drift: forward LSTM cell state integrates the
      cumulative slope over 147 weeks (SNR ≈ 1.9 cumulative vs 0.013 local).
    - Heavy Dropper step change: bidirectional hidden state captures
      asymmetric pre/post-drop context.
    - MNAR regime transition (week ~104): backward LSTM traverses the
      boundary producing a large hidden-state transition at the regime edge.

    Temporal attention learns which time steps are most theft-discriminative
    (Heavy Dropper: step-change week; Light Dropper: full drift window).

    Input  : (N, W, D) — NaME-processed, no NaN
    Output : (N, embed_dim)
    """

    def __init__(
        self,
        days: int,
        hidden: int,
        layers: int,
        dropout: float,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=days,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        lstm_out = hidden * 2  # bidirectional concat

        # Learnable attention query — small init so attention starts near-uniform
        self.attn_query = nn.Parameter(torch.randn(lstm_out) * 0.01)

        self.proj = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                                    # (N, W, lstm_out)
        scores = torch.einsum("d,nwd->nw", self.attn_query, out) # (N, W)
        alpha  = torch.softmax(scores / (out.shape[-1] ** 0.5), dim=-1)  # (N, W)
        ctx    = torch.einsum("nw,nwd->nd", alpha, out)          # (N, lstm_out)
        return self.proj(ctx)                                     # (N, embed_dim)


# ---------------------------------------------------------------------------
# Phase H: Mamba2 Wide branch
# ---------------------------------------------------------------------------

class MambaWide(nn.Module):
    """Wide branch: bidirectional Mamba2 with temporal attention pooling.

    Mechanistic motivation vs BiLSTM (Phase G, K1 triggered):
    BiLSTM gates are sigmoid functions of (x_t, h_{t-1}); ALL 147 time steps
    contribute uniformly to every gate — step-change signal at Heavy Dropper
    week T is diluted by 146 other updates.  Mamba2's input-dependent Δ_t
    (step size) is a monotone function of x_t alone, so the model learns to
    assign large Δ_t at discriminative weeks and small Δ_t elsewhere:

    - Heavy Dropper (step ↓ at T): large Δ_T → large SSM state jump → encodes
      the transition prominently; subsequent steps see a "reset" hidden state.
    - Light Dropper/Riser (drift ±0.13 kWh/wk): small Δ at every step → the
      SSM integrates the cumulative slope; SNR 0.013 local → 1.9 cumulative.
    - MNAR boundary (week ~104): imputation regime shift produces unusual x̂
      statistics → large Δ → implicit regime boundary encoded in hidden state.

    Bidirectional design:
    - Forward (L→R): integrates pre-theft context leading up to week T.
    - Backward (R→L): integrates post-theft validation — "sustained absence"
      distinguishes genuine theft onset from transient measurement anomaly.

    chunk_size=21 divides seq_len=147 exactly (7 chunks of 21 weeks each;
    ≈ 5-month seasonal blocks, aligned with SGCC seasonal consumption patterns).

    Requires mamba_ssm 2.3.2 (CUDA kernels): use conda env mamba_gpu.

    Input  : (N, W, D) — NaME-processed, no NaN; W=147, D=7
    Output : (N, embed_dim)
    """

    # Chunk size that cleanly divides SGCC seq_len=147.
    # 147 = 3 × 7²; divisors include 1,3,7,21,49,147. We use 21.
    _CHUNK_SIZE = 21

    def __init__(
        self,
        n_features: int,        # D = 7 (days per week)
        d_model: int,           # SSM projection dim (inner_dim = d_model × expand)
        d_state: int,           # SSM state dimension
        d_conv: int,            # local convolution context (weeks)
        expand: int,            # inner expansion factor
        headdim: int,           # SSM head dim (num_heads = inner_dim / headdim)
        embed_dim: int,         # output projection target (= alpha = 90)
        dropout: float,
    ) -> None:
        super().__init__()

        # Lazy import: keeps module importable in tgpu env (no mamba_ssm);
        # only fails at instantiation time if mamba_ssm is absent.
        try:
            from mamba_ssm import Mamba2 as _Mamba2
        except ImportError as exc:
            raise ImportError(
                "MambaWide (A12) requires mamba_ssm 2.3.2.  "
                "Run with: conda run -n mamba_gpu python run_local_grid.py --arms a12_mamba_wide"
            ) from exc

        inner_dim = d_model * expand
        if inner_dim % headdim != 0:
            raise ValueError(
                f"inner_dim ({inner_dim}) must be divisible by headdim ({headdim}); "
                f"got d_model={d_model}, expand={expand}, headdim={headdim}."
            )

        self.input_proj = nn.Linear(n_features, d_model)

        _mamba_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=self._CHUNK_SIZE,
            rmsnorm=True,
            norm_before_gate=False,
            use_mem_eff_path=True,
        )
        self.mamba_fwd = _Mamba2(**_mamba_kwargs)
        self.mamba_bwd = _Mamba2(**_mamba_kwargs)

        bidi_dim = d_model * 2   # forward + backward concatenated

        # Temporal attention — learns WHICH weeks carry theft signal:
        #   Heavy Dropper: concentrates at step-change week T
        #   Light Dropper: spreads across the full drift window
        #   MNAR boundary: concentrates near week ~104
        self.attn_query = nn.Parameter(torch.randn(bidi_dim) * 0.01)

        self.proj = nn.Sequential(
            nn.LayerNorm(bidi_dim),
            nn.Linear(bidi_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, W, D) — NaME-imputed weekly grid, no NaN
        h = self.input_proj(x)                          # (N, W, d_model)
        h_fwd = self.mamba_fwd(h)                       # (N, W, d_model) — forward SSM
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)       # (N, W, d_model) — backward SSM
        h_bi  = torch.cat([h_fwd, h_bwd], dim=-1)      # (N, W, 2*d_model)

        # Scaled dot-product attention over time axis
        scores = torch.einsum("d,nwd->nw", self.attn_query, h_bi)
        alpha  = torch.softmax(scores / (h_bi.shape[-1] ** 0.5), dim=-1)  # (N, W)
        ctx    = torch.einsum("nw,nwd->nd", alpha, h_bi)                   # (N, 2*d_model)
        return self.proj(ctx)                           # (N, embed_dim)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ZKMixV4(nn.Module):
    """Wide-and-Deep MIXER with CC-NaME, Multi-Scale Laplacian, and Phase E
    extensions (mask channel + temporal branch).

    Forward
    -------
    x : (N, W, D)   — weekly consumption grid, NaN at missing positions
                       (nan_native=True) or sentinel=-1.0 (nan_native=False).

    Returns
    -------
    logit : (N,)    — pre-sigmoid classification score.

    Equivalence guarantees
    ----------------------
    G1.  With ``use_cc_name=False, use_ms_kernel=False, use_mask_channel=False,
         use_temporal_branch=False``, this model is functionally equivalent to
         Phase C ``zkmix_v3_name_nn`` (scalar NaME, d=1 Laplacian).

    G2.  With ``use_cc_name=True`` and the context encoder weights at init
         (all-zero output layer), the model is bit-identical to G1.

    G3.  With ``use_mask_channel=True`` and the mask CNN head weights at their
         random initialisation, the model adds a mask branch to G1/G2.  The
         mask branch produces non-zero output at init, so G3 does NOT reduce
         to G1 exactly — but the imputation path is unchanged.

    G4.  With ``use_temporal_branch=True`` and the temporal MLP output layer
         zero-initialised, the temporal branch contributes zero at init,
         reducing to the model without it.
    """

    def __init__(self, cfg: ZKMixV4Config | None = None) -> None:
        super().__init__()
        cfg = cfg or ZKMixV4Config()
        self.cfg = cfg

        # ============= CC-NaME (or scalar NaME fallback) =============
        self.name_module = ContextConditionalNaME(
            days=cfg.days,
            hidden=cfg.cc_hidden,
            dropout=cfg.cc_dropout,
            base_token_init=cfg.base_token_init,
            cc_enabled=cfg.use_cc_name,
            nan_native=cfg.nan_native,
        )
        # A6 branch-selective NaME: CNN branch uses scalar NaME while wide+mixer
        # get CC-NaME. Only relevant when use_cnn_consump=True.
        _cnn_needs_scalar = "cnn" not in cfg.cc_name_branches
        if _cnn_needs_scalar and cfg.use_cc_name and cfg.use_cnn_consump:
            self.name_module_scalar: ContextConditionalNaME | None = ContextConditionalNaME(
                days=cfg.days,
                hidden=cfg.cc_hidden,
                dropout=cfg.cc_dropout,
                base_token_init=cfg.base_token_init,
                cc_enabled=False,          # scalar only
                nan_native=cfg.nan_native,
            )
        else:
            self.name_module_scalar = None

        # ============= WIDE branch =============
        # Priority: MambaWide (A12) > BiLSTMWide (A11, K1) > linear (A9 default)
        if cfg.use_mamba_wide and cfg.use_bilstm_wide:
            raise ValueError(
                "use_mamba_wide and use_bilstm_wide are mutually exclusive; "
                "set exactly one to True."
            )
        if cfg.use_mamba_wide:
            self.wide_fc: nn.Module = MambaWide(
                n_features=cfg.days,
                d_model=cfg.mamba_d_model,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim,
                embed_dim=cfg.alpha,
                dropout=cfg.dropout,
            )
        elif cfg.use_bilstm_wide:
            self.wide_fc = BiLSTMWide(
                days=cfg.days,
                hidden=cfg.bilstm_hidden,
                layers=cfg.bilstm_layers,
                dropout=cfg.dropout,
                embed_dim=cfg.alpha,
            )
        else:
            self.wide_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cfg.weeks * cfg.days, cfg.alpha),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # effective_dilations used by both CNN-consump and mask CNN branches
        effective_dilations = cfg.ms_dilations if cfg.use_ms_kernel else (1,)

        # ============= DEEP-CNN branch (consumption, multi-scale or single-scale) =============
        if cfg.use_cnn_consump:
            self.cnn_branch: MultiScaleLaplacian | None = MultiScaleLaplacian(
                dilations=effective_dilations,
                freeze_stencils=cfg.ms_freeze_stencils,
                out_channels_per_scale=cfg.gamma,
                weeks=cfg.weeks,
                days=cfg.days,
                embed_dim=cfg.beta,
                dropout=cfg.dropout,
            )
        else:
            self.cnn_branch = None

        # ============= DEEP-MIXER branch (unchanged from Phase C) =============
        if cfg.use_mixer:
            self._kernel_module = cfg.kernel.build()
            self._kernel_out_ch  = cfg.kernel.out_channels
            self.patch_proj  = nn.Linear(cfg.days, cfg.mixer_hidden)
            stat_dim         = self._kernel_out_ch * len(cfg.stats)
            self.stat_proj   = nn.Linear(stat_dim, cfg.mixer_hidden)
            self.stat_norm   = nn.LayerNorm(cfg.mixer_hidden)
            self.mixer_blocks = nn.ModuleList(
                MixerBlock2D(
                    n_tokens=cfg.weeks,
                    hidden=cfg.mixer_hidden,
                    token_mlp=cfg.mixer_token_mlp,
                    channel_mlp=cfg.mixer_channel_mlp,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.mixer_n_blocks)
            )
            self.mixer_norm = nn.LayerNorm(cfg.mixer_hidden)
            self.mixer_fc   = nn.Sequential(
                nn.Linear(3 * cfg.mixer_hidden, cfg.mixer_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # ============= Phase E: A7 — mask Laplacian branch =============
        # Processes the binary validity mask M through the SAME Laplacian pyramid
        # as the consumption CNN.  L(M) is nonzero only at NaN boundaries, encoding
        # the temporal fingerprint of meter tampering events.
        if cfg.use_mask_channel:
            self.mask_cnn_branch: MultiScaleLaplacian | None = MultiScaleLaplacian(
                dilations=effective_dilations,
                freeze_stencils=False,          # mask responses differ from consumption; allow learning
                out_channels_per_scale=cfg.gamma,
                weeks=cfg.weeks,
                days=cfg.days,
                embed_dim=cfg.mask_beta,
                dropout=cfg.dropout,
            )
        else:
            self.mask_cnn_branch = None

        # ============= Phase J: A17 — peer-relative residual branch (PRCE-CNN) =============
        # Processes the peer-relative residual r = x̂[c] − mean(x̂[k-peers]) through
        # a MultiScaleLaplacian CNN identical in structure to the CNN-consump branch.
        # Laplacian of r detects the STEP CHANGE in peer-relative consumption at the
        # theft week for Light Dropper/Riser archetypes (48.3% of theft customers).
        # Input channel: X_aug[:, :, days:] when use_prce=True (pre-computed by runner).
        #
        # Phase K (A20+): when ``use_peer_gnn=True`` (and ``use_prce=True``) the
        # Laplacian branch is replaced by a learned cross-attention branch
        # (PeerAttentionBranch). Input shape contract is unchanged.
        if cfg.use_prce and not cfg.use_peer_gnn:
            self.prce_cnn_branch: MultiScaleLaplacian | None = MultiScaleLaplacian(
                dilations=effective_dilations,
                freeze_stencils=False,
                out_channels_per_scale=cfg.gamma,
                weeks=cfg.weeks,
                days=cfg.days,
                embed_dim=cfg.prce_beta,
                dropout=cfg.dropout,
            )
        else:
            self.prce_cnn_branch = None

        if cfg.use_prce and cfg.use_peer_gnn:
            from .gnn import PeerAttentionBranch
            self.peer_gnn_branch = PeerAttentionBranch(
                days=cfg.days,
                weeks=cfg.weeks,
                embed_dim=cfg.peer_gnn_dim,
                hidden=cfg.peer_gnn_hidden,
                n_heads=cfg.peer_gnn_heads,
                kernel_size=cfg.peer_gnn_kernel,
                dropout=cfg.dropout,
            )
        else:
            self.peer_gnn_branch = None

        # ============= Phase L: A25 — Signed-Peer cross-attention branch =============
        # Consumes (signed_residual, peer_std, peer_iqr) stacked along day-axis.
        # Mutually exclusive with use_peer_gnn — both can't be on simultaneously.
        if cfg.use_signed_peer and cfg.use_peer_gnn:
            raise ValueError("use_signed_peer and use_peer_gnn are mutually exclusive.")
        if cfg.use_signed_peer:
            from .signed_peer import SignedPeerBranch
            self.signed_peer_branch = SignedPeerBranch(
                days=cfg.days,
                weeks=cfg.weeks,
                embed_dim=cfg.signed_peer_dim,
                hidden=cfg.signed_peer_hidden,
                n_heads=cfg.signed_peer_heads,
                kernel_size=cfg.signed_peer_kernel,
                dropout=cfg.dropout,
            )
        else:
            self.signed_peer_branch = None

        # ============= Phase L: A24 — Engineered-features MLP branch ==============
        # Time-invariant per-customer scalars (12 features). MLP → ``eng_features_dim``
        # output appended to the fusion concat. Output of last layer zero-init so
        # the branch contributes zero at init (preserves G4 invariance).
        if cfg.use_eng_features:
            self.eng_features_mlp = nn.Sequential(
                nn.Linear(cfg.eng_features_n, cfg.eng_features_hidden),
                nn.LayerNorm(cfg.eng_features_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.eng_features_hidden, cfg.eng_features_dim),
                nn.ReLU(inplace=True),
            )
            nn.init.zeros_(self.eng_features_mlp[-2].weight)
            nn.init.zeros_(self.eng_features_mlp[-2].bias)
        else:
            self.eng_features_mlp = None

        # ============= Phase N: multi-scale differencing branch (MSDM) =========
        # Input: imputed daily sequence (N, T). Multi-scale causal differences at
        # cfg.diff_scales, adaptive-quantised tanh(beta*Delta), 1-D CNN -> pool ->
        # Linear(diff_dim). Output layer zero-init -> branch contributes 0 at init.
        if cfg.use_diff_branch:
            self.diff_beta = nn.Parameter(torch.tensor(1.0))     # learnable quant sharpness
            n_sc = len(cfg.diff_scales)
            self.diff_cnn = nn.Sequential(
                nn.Conv1d(n_sc, cfg.diff_hidden, kernel_size=7, padding=3),
                nn.BatchNorm1d(cfg.diff_hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(cfg.diff_hidden, cfg.diff_hidden, kernel_size=7, padding=3, dilation=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(1),
            )
            self.diff_head = nn.Linear(cfg.diff_hidden, cfg.diff_dim)
            nn.init.zeros_(self.diff_head.weight)
            nn.init.zeros_(self.diff_head.bias)
        else:
            self.diff_cnn = None

        # ============= MSDM inter/intra-period convolution branch (IIPC) =========
        if cfg.use_interperiod_branch:
            C = cfg.days                                   # period (one cycle in steps)
            h = cfg.ip_hidden
            mk = lambda d: nn.Conv1d(1, h, kernel_size=3, padding=d, dilation=d)
            self.ip_convs = nn.ModuleList([mk(1), mk(2), mk(C), mk(2 * C)])  # intra{1,2}+inter{C,2C}
            self.ip_bn = nn.BatchNorm1d(4 * h)
            self.ip_head = nn.Linear(4 * h, cfg.ip_dim)
            nn.init.zeros_(self.ip_head.weight)            # zero-init → no-op at init
            nn.init.zeros_(self.ip_head.bias)
        else:
            self.ip_convs = None

        # ============= Phase E: A8 — global temporal trend branch =============
        # 21 monthly blocks (7 weeks × 7 days) → (mean, std, nan_frac) per block
        # → (N, 21*3=63) → MLP → (N, temporal_dim).
        # The nan_frac sequence encodes the MNAR regime shift without explicit labels.
        # Output layer zero-initialised: branch contributes 0 at init (G4).
        self._n_blocks = cfg.weeks // 7   # 147 // 7 = 21
        if cfg.use_temporal_branch:
            in_dim = self._n_blocks * 3   # 63
            self.temporal_mlp: nn.Sequential | None = nn.Sequential(
                nn.Linear(in_dim, cfg.temporal_mlp_hidden),
                nn.LayerNorm(cfg.temporal_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.temporal_mlp_hidden, cfg.temporal_dim),
                nn.ReLU(inplace=True),
            )
            # Zero-init last Linear so temporal branch ≡ 0 at initialisation.
            nn.init.zeros_(self.temporal_mlp[-2].weight)
            nn.init.zeros_(self.temporal_mlp[-2].bias)
        else:
            self.temporal_mlp = None

        # ============= FUSION =============
        embed_dim = cfg.alpha                           # wide always present
        if cfg.use_cnn_consump:
            embed_dim += cfg.beta
        if cfg.use_mixer:
            embed_dim += cfg.mixer_out_dim
        if cfg.use_mask_channel:
            embed_dim += cfg.mask_beta
        if cfg.use_temporal_branch:
            embed_dim += cfg.temporal_dim
        if cfg.use_prce and not cfg.use_peer_gnn and not cfg.use_signed_peer:
            embed_dim += cfg.prce_beta
        if cfg.use_prce and cfg.use_peer_gnn:
            embed_dim += cfg.peer_gnn_dim
        if cfg.use_signed_peer:
            embed_dim += cfg.signed_peer_dim
        if cfg.use_eng_features:
            embed_dim += cfg.eng_features_dim
        if cfg.use_diff_branch:
            embed_dim += cfg.diff_dim
        if cfg.use_interperiod_branch:
            embed_dim += cfg.ip_dim
        self.embed_dim = embed_dim
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim, cfg.fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.fuse_hidden, 1)

    # ------------------------------------------------------------------
    # Branch helpers
    # ------------------------------------------------------------------

    def _wide(self, x: torch.Tensor) -> torch.Tensor:
        return self.wide_fc(x)                                 # (N, alpha)

    def _deep_cnn(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn_branch(x.unsqueeze(1))                 # (N, beta)

    def _deep_mixer(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_proj(x)                            # (N, W, mixer_hidden)
        if self.cfg.inject_kernel_stats:
            resp  = self._kernel_module(x.unsqueeze(1))        # (N, kC, W, D)
            stats = weekly_kernel_stats(resp, stats=self.cfg.stats)  # (N, W, kC*S)
            tokens = tokens + self.stat_norm(self.stat_proj(stats))
        for blk in self.mixer_blocks:
            tokens = blk(tokens)
        tokens = self.mixer_norm(tokens)
        pooled = _multi_pool(tokens, dim=1)                    # (N, 3*mixer_hidden)
        return self.mixer_fc(pooled)                           # (N, mixer_out_dim)

    def _temporal_features(
        self, x_raw: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute 21 monthly (mean, std, nan_frac) blocks from the raw input.

        Parameters
        ----------
        x_raw : (N, W, D)  with NaN at missing positions (original input, pre-NaME)
        mask  : (N, W, D)  float, 1.0 = valid, 0.0 = NaN

        Returns (N, temporal_dim).  The 21-block nan_frac sequence encodes the
        Jan-2016 MNAR regime shift (blocks 0-14 ≈ 2014-2015 at ~38% NaN;
        blocks 15-20 ≈ 2016 at ~1% NaN), providing implicit regime conditioning.
        """
        n, w, d = x_raw.shape
        nb  = self._n_blocks       # 21
        w_u = nb * 7               # 147 (W is exactly divisible)

        # Reshape to (N, nb, 7_weeks, D).
        xb = x_raw[:, :w_u, :].reshape(n, nb, 7, d)      # (N, 21, 7, 7)
        mb = mask[:, :w_u, :].reshape(n, nb, 7, d)        # (N, 21, 7, 7)

        # Per-block statistics over the 7×D = 49 cells.
        count    = mb.sum(dim=(2, 3)).clamp(min=1.0)      # (N, 21)
        xf       = torch.where(mb > 0.5, xb, torch.zeros_like(xb))
        blk_mean = (xf * mb).sum(dim=(2, 3)) / count      # (N, 21)
        blk_sq   = (xf ** 2 * mb).sum(dim=(2, 3)) / count
        blk_std  = (blk_sq - blk_mean ** 2).clamp(min=0.0).sqrt()  # (N, 21)
        blk_nan  = 1.0 - mb.sum(dim=(2, 3)) / float(7 * d)         # (N, 21)

        ctx = torch.stack([blk_mean, blk_std, blk_nan], dim=-1)  # (N, 21, 3)
        return self.temporal_mlp(ctx.flatten(1))                  # (N, temporal_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, x: torch.Tensor, eng_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Return pre-fusion embedding (N, embed_dim). Exposed for SelectiveModel.

        Layout contract by config:
          * Phase D/E (default):                       x is (N, W, D)
          * use_prce=True, use_peer_gnn=False:         x is (N, W, 2*D)
              [:, :, 0:D]      = own consumption (raw NaN-containing)
              [:, :, D:2D]     = peer-relative residual (single channel block)
          * use_prce=True, use_peer_gnn=True:          x is (N, W, 2*D)  (A20)
          * use_signed_peer=True:                      x is (N, W, 4*D)  (A25/A29)
              [:, :, 0:D]      = own consumption
              [:, :, D:4*D]    = (signed_residual | peer_std | peer_iqr) stacked

        ``eng_feat`` (optional): (N, eng_features_n) tensor of engineered
        features. Required only when ``cfg.use_eng_features=True``.  The runner
        passes this through ``model(x, eng_feat=...)`` indirectly via
        ``forward(x, eng_feat=...)``.
        """
        if x.dim() != 3:
            raise ValueError(f"ZKMixV4 expects (N, W, D[*K]); got {tuple(x.shape)}.")
        n, w, d_in = x.shape
        D = self.cfg.days

        # Determine input layout
        if self.cfg.use_signed_peer:
            expected_d = D * 4
            if d_in != expected_d:
                raise ValueError(
                    f"ZKMixV4 use_signed_peer=True expects D={expected_d} "
                    f"(own + signed_residual + peer_std + peer_iqr); got D={d_in}."
                )
            peer_residual_block = None   # not the Laplacian-style peer_res
            signed_peer_block = x[:, :, D:]   # (N, W, 3*D)
            x = x[:, :, :D]
        elif self.cfg.use_prce:
            expected_d = D * 2
            if d_in != expected_d:
                raise ValueError(
                    f"ZKMixV4 use_prce=True expects D={expected_d} "
                    f"(stacked consumption + peer_res); got D={d_in}."
                )
            peer_residual_block = x[:, :, D:]
            signed_peer_block = None
            x = x[:, :, :D]
        else:
            if d_in != D:
                raise ValueError(f"ZKMixV4 configured for D={D}; got D={d_in}.")
            peer_residual_block = None
            signed_peer_block = None
        peer_res = peer_residual_block

        n, w, d = x.shape
        if w != self.cfg.weeks:
            raise ValueError(
                f"ZKMixV4 configured for W={self.cfg.weeks}; got W={w}."
            )

        # Preserve raw NaN positions BEFORE imputation — needed by A7 and A8.
        need_raw = (self.mask_cnn_branch is not None or self.temporal_mlp is not None)
        if need_raw:
            # NaME returns a new tensor; x is not mutated, so x_raw is a safe alias.
            x_raw    = x
            raw_mask = (~torch.isnan(x)).float()               # (N, W, D)

        if self.name_module_scalar is not None:
            # A6 branch-selective: CC-NaME for wide+mixer, scalar NaME for CNN.
            x_cc  = self.name_module(x)                        # (N, W, D), CC-NaME filled
            x_sc  = self.name_module_scalar(x)                 # (N, W, D), scalar filled
            wide  = self._wide(x_cc)                           # (N, alpha)
            parts: list[torch.Tensor] = [wide]
            if self.cnn_branch is not None:
                parts.append(self._deep_cnn(x_sc))             # (N, beta)
            if self.cfg.use_mixer:
                parts.append(self._deep_mixer(x_cc))           # (N, mixer_out_dim)
            own_imputed = x_cc                                  # for downstream peer-GNN
        else:
            # Standard path: single NaME for all branches.
            x_imp = self.name_module(x)                        # (N, W, D), no NaN
            wide  = self._wide(x_imp)                          # (N, alpha)
            parts = [wide]
            if self.cnn_branch is not None:
                parts.append(self._deep_cnn(x_imp))            # (N, beta)
            if self.cfg.use_mixer:
                parts.append(self._deep_mixer(x_imp))          # (N, mixer_out_dim)
            own_imputed = x_imp                                 # for downstream peer-GNN

        if self.mask_cnn_branch is not None:
            # A7: Laplacian of the binary mask — detects NaN boundary fingerprints.
            parts.append(self.mask_cnn_branch(raw_mask.unsqueeze(1)))  # (N, mask_beta)

        if self.temporal_mlp is not None:
            # A8: monthly trend MLP — detects Dropper/Riser drift + regime shift.
            parts.append(self._temporal_features(x_raw, raw_mask))    # (N, temporal_dim)

        if self.prce_cnn_branch is not None:
            # A17: PRCE-CNN — Laplacian of peer-relative residual.
            # Detects step-change in r[c,t] = x̂[c,t] − mean_peer[t] at theft week.
            # Targets Light Dropper / Riser archetypes (48.3 % of theft customers).
            if peer_res is None:
                raise RuntimeError(
                    "ZKMixV4 use_prce=True but embed() received no peer_res channel. "
                    "Pass x of shape (N, W, D*2) with peer_res stacked on days axis."
                )
            parts.append(self.prce_cnn_branch(peer_res.unsqueeze(1)))  # (N, prce_beta)

        if self.peer_gnn_branch is not None:
            # A20: learned cross-attention peer aggregation. ``x`` here is the
            # NaME-imputed own consumption (post-CC-NaME); peer_res is the raw
            # peer-relative residual (numpy-level zeros at NaN positions).
            if peer_res is None:
                raise RuntimeError(
                    "ZKMixV4 use_peer_gnn=True but embed() received no peer_res channel."
                )
            # Use post-imputation own series so the attention input is NaN-free.
            parts.append(self.peer_gnn_branch(own_imputed, peer_res))  # (N, peer_gnn_dim)

        if self.signed_peer_branch is not None:
            # A25: signed cross-attention with peer_std + peer_iqr context.
            if signed_peer_block is None:
                raise RuntimeError(
                    "ZKMixV4 use_signed_peer=True but embed() received no peer block."
                )
            parts.append(self.signed_peer_branch(own_imputed, signed_peer_block))  # (N, signed_peer_dim)

        if self.eng_features_mlp is not None:
            # A24: engineered-feature MLP branch (time-invariant per customer).
            if eng_feat is None:
                raise RuntimeError(
                    "ZKMixV4 use_eng_features=True but embed() received no eng_feat tensor. "
                    "Pass via forward(x, eng_feat=...)."
                )
            if eng_feat.dim() != 2 or eng_feat.size(1) != self.cfg.eng_features_n:
                raise ValueError(
                    f"eng_feat must be (N, {self.cfg.eng_features_n}); got {tuple(eng_feat.shape)}"
                )
            parts.append(self.eng_features_mlp(eng_feat))     # (N, eng_features_dim)

        if self.diff_cnn is not None:
            # Phase N: multi-scale differencing on the imputed daily sequence.
            n_, w_, d_ = own_imputed.shape
            seq = own_imputed.reshape(n_, w_ * d_)             # (N, T) daily, no NaN
            diffs = []
            for dsc in self.cfg.diff_scales:
                pad = torch.zeros(n_, dsc, device=seq.device, dtype=seq.dtype)
                diffs.append(torch.cat([pad, seq[:, dsc:] - seq[:, :-dsc]], dim=1))  # causal
            dstack = torch.tanh(self.diff_beta * torch.stack(diffs, dim=1))  # (N, n_sc, T)
            h = self.diff_cnn(dstack).squeeze(-1)              # (N, diff_hidden)
            parts.append(self.diff_head(h))                    # (N, diff_dim)

        if self.ip_convs is not None:
            # MSDM IIPC: intra{1,2} + inter{C,2C} dilated convs on imputed daily seq.
            n_, w_, d_ = own_imputed.shape
            seq = own_imputed.reshape(n_, 1, w_ * d_)          # (N,1,T)
            hcat = torch.cat([torch.relu(c(seq)) for c in self.ip_convs], dim=1)  # (N,4h,T)
            hcat = torch.relu(self.ip_bn(hcat))
            pooled = torch.amax(hcat, dim=-1)                  # (N, 4h)
            parts.append(self.ip_head(pooled))                 # (N, ip_dim)

        return torch.cat(parts, dim=-1)                        # (N, embed_dim)

    def forward(self, x: torch.Tensor, eng_feat: torch.Tensor | None = None) -> torch.Tensor:
        emb   = self.embed(x, eng_feat=eng_feat)               # (N, embed_dim)
        fused = self.fuse(emb)                                 # (N, fuse_hidden)
        return self.head(fused).squeeze(-1)                    # (N,)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def cc_name_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance τ_n values for Figure 5 of the paper.

        Returns
        -------
        (N,)  — instance-adaptive NaME token value for each sample.
        """
        if not self.cfg.use_cc_name:
            return self.name_module.base_token.expand(x.shape[0])
        return self.name_module.token_values(x)


# ---------------------------------------------------------------------------
# Canonical arm factories (matching PLAN_D_2026-05-27.md + PLAN_E_2026-05-27.md)
# ---------------------------------------------------------------------------

def build_a1(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A1 arm: CC-NaME only (single-scale d=1 Laplacian)."""
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=False,          # ms_dilations ignored; uses (1,) internally
    ))


def build_a2(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A2 arm: multi-scale Laplacian only (scalar NaME)."""
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=False,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
    ))


def build_a3(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A3 arm: Phase D headline — CC-NaME + multi-scale Laplacian."""
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
    ))


def build_a4(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A4 arm: CC-NaME + single-scale only (confirms multi-scale is additive)."""
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1,),           # single scale — equivalent to A1 structurally
    ))


def build_a5(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A5 arm: A3 minus wide branch (confirms wide branch necessity)."""
    model = _ZKMixV4NoWide(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
    ))
    return model


class _ZKMixV4NoWide(ZKMixV4):
    """A5: ZKMixV4 without the wide branch.  Used for ablation only."""

    def __init__(self, cfg: ZKMixV4Config) -> None:
        # Build with alpha=90 so tensors are allocated, then override forward.
        super().__init__(cfg)
        # Override fusion head to match reduced embed dim (no wide branch).
        reduced_dim = 0
        if cfg.use_cnn_consump:
            reduced_dim += cfg.beta
        if cfg.use_mixer:
            reduced_dim += cfg.mixer_out_dim
        if cfg.use_mask_channel:
            reduced_dim += cfg.mask_beta
        if cfg.use_temporal_branch:
            reduced_dim += cfg.temporal_dim
        self.fuse = nn.Sequential(
            nn.Linear(reduced_dim, cfg.fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.embed_dim = reduced_dim

    def embed(self, x: torch.Tensor, eng_feat: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"_ZKMixV4NoWide expects (N, W, D); got {tuple(x.shape)}.")
        need_raw = (self.mask_cnn_branch is not None or self.temporal_mlp is not None)
        if need_raw:
            x_raw    = x
            raw_mask = (~torch.isnan(x)).float()
        x_imp = self.name_module(x)
        parts: list[torch.Tensor] = []
        if self.cnn_branch is not None:
            parts.append(self._deep_cnn(x_imp))
        if self.cfg.use_mixer:
            parts.append(self._deep_mixer(x_imp))
        if self.mask_cnn_branch is not None:
            parts.append(self.mask_cnn_branch(raw_mask.unsqueeze(1)))
        if self.temporal_mlp is not None:
            parts.append(self._temporal_features(x_raw, raw_mask))
        return torch.cat(parts, dim=-1)


def build_a6(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A6 arm: branch-selective NaME — CC-NaME for wide+mixer; scalar NaME for CNN.

    Scientific premise (Stage 2 of PLAN_D_LOCAL_GRID_2026-05-27.md):
    The negative synergy in A3 arises because τ_n serves two conflicting roles:
      (a) statistical imputation for wide/mixer  → trained toward customer baseline
      (b) clean Laplacian input for CNN          → must stay in [0,1] valid range
    A6 decouples these: CC-NaME satisfies (a), scalar NaME satisfies (b).
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        cc_name_branches=("wide", "mixer"),   # CNN receives scalar NaME only
    ))


def build_a7(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A7 arm: A3 + explicit missing-mask channel.

    Scientific premise (PLAN_E_2026-05-27.md §A7):
    The binary validity mask M (1=observed, 0=NaN) encodes theft-specific
    temporal patterns (meter tampering fingerprints).  NaME substitution
    DESTROYS this signal by making NaN positions indistinguishable from valid
    cells.  Adding M as a separate input to a second Laplacian CNN preserves
    the spatial-temporal structure of missingness, which is the #1 and #3
    correlate with theft (max_consec_nan r=0.102, missing_frac r=0.095).
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
    ))


def build_a8(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A8 arm: A3 + global temporal trend branch.

    Scientific premise (PLAN_E_2026-05-27.md §A8):
    Light Dropper/Riser archetypes (48.3% of theft) exhibit gradual consumption
    drift over 6-24 months.  SNR in any 9-week window ≈ 0.013-0.027; these
    patterns are structurally undetectable by the Laplacian at any dilation.
    The temporal MLP processes 21 monthly summary statistics over the full
    3-year window, giving the only branch with sufficient temporal coverage.
    The monthly nan_frac sequence also provides implicit MNAR regime conditioning
    (blocks 0-14: ~38% NaN 2014-2015; blocks 15-20: ~1% NaN 2016).
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
    ))


def build_a9(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A9 arm: A3 + mask channel + temporal branch — Phase E headline candidate.

    Combines A7 (missingness fingerprint) and A8 (global trend) on top of
    the best Phase D configuration.  Independent hypotheses; expected additive
    unless there is negative synergy in the mask+temporal interaction.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
    ))


def _build_a0_nanative(weeks: int = 147, days: int = 7):
    from zkmix_lab.models import build_model as _bm  # type: ignore
    return _bm("zkmix_v3_name_nn", weeks=weeks, days=days)


def build_a11(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A11: A9 + BiLSTM Wide branch (replaces linear Wide).

    Identical to A9 except the Wide branch (Flatten→Linear) is replaced by a
    2-layer BiLSTM with temporal attention pooling.  Targets gradual drift
    patterns (Light Dropper/Riser, 48%+ of theft cases) that are structurally
    undetectable by the linear projection but detectable via cumulative slope
    integration over 147 weeks (SNR ≈ 1.9 cumulative vs ≈ 0.013 local).

    Embed dim: 276 (BiLSTM projected to alpha=90, unchanged from A9).
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
        use_bilstm_wide=True,
        bilstm_hidden=32,
        bilstm_layers=2,
    ))


def build_a10(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A10 arm: identical architecture to A9, trained with focal loss (γ=2).

    The loss override is applied in run_local_grid.py via ARM_LOSS_OVERRIDES.
    This factory builds the model only; loss_type="focal" is injected at
    TrainConfig construction time.
    """
    return build_a9(weeks=weeks, days=days)


def build_a12(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A12: A9 + BiMamba2 Wide branch (replaces linear Wide).

    Scientific premise (PLAN_H_2026-05-27.md):
    Phase G (BiLSTM, A11) triggered K1: all folds negative (Δ=−0.0066).
    Diagnosis: uniform LSTM gating distributes all 147 time steps equally
    into the hidden state — step-change signal at Heavy Dropper week T is
    diluted by 146 other LSTM updates.

    Mamba2's input-dependent Δ_t (step size) is a function of x_t alone,
    allowing the model to learn large Δ at discriminative weeks (step changes,
    regime boundaries) and small Δ during quiet periods (gradual integration
    of Light Dropper/Riser drift).  This is a structurally different operator,
    not a stronger version of recurrence.

    Design: BiMamba2 (forward + backward, each d_model=32) + temporal attention
    pooling → Linear(64, 90).  chunk_size=21 divides seq_len=147 cleanly.
    Embed dim: 276 (unchanged from A9).

    Requires mamba_ssm 2.3.2:
        conda run -n mamba_gpu python run_local_grid.py --arms a12_mamba_wide
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mamba_wide=True,
        mamba_d_model=32,
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
    ))


def build_a13(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A13: A12 without CNN-consump branch.

    Ablation (PLAN_I_2026-05-27.md): tests whether the multi-scale Laplacian
    CNN on consumption adds incremental information beyond what MambaWide already
    captures from the same x̂ input.  MambaWide provides global sequential
    structure; CNN-consump provides local multi-scale (d=1,2,4) curvature.
    Embed dim: 276 − 60 = 216.  Requires mamba_gpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_cnn_consump=False,
    ))


def build_a14(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A14: A12 without Mixer branch.

    Ablation: tests whether the MixerBlock2D (patch-token MLP with kernel-stat
    injection) adds information beyond the other 4 branches.  Embed dim: 276 − 64 = 212.
    Requires mamba_gpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_mixer=False,
    ))


def build_a15(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A15: A12 without CNN-mask branch.

    Ablation: tests whether the binary validity mask Laplacian CNN (A7 contribution)
    remains informative in the full A12 model.  Embed dim: 276 − 30 = 246.
    Requires mamba_gpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=False,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
    ))


def build_a16(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A16: A12 without Temporal MLP branch.

    Ablation: tests whether the 21-block monthly MLP (A8 contribution) remains
    informative in the full A12 model — specifically whether MambaWide has
    already captured the global drift and regime-conditioning signals that A8
    was designed to provide.  Embed dim: 276 − 32 = 244.  Requires mamba_gpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
    ))


def build_a9_lean(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A9-lean: A9 with Mixer branch removed (Phase I confirmed Mixer is prunable).

    Phase I ablation (PLAN_I_2026-05-27.md):
    A14 (A12 − Mixer): Δ = −0.0003, CI = [−0.0049, +0.0041] → NULL-prunable.
    Removing Mixer reduces embed_dim 276 → 212 and removes ~210K parameters
    with no expected performance loss.  Used as the baseline for Phase J.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
        use_mixer=False,
    ))


def build_a17(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A17: A9-lean + Peer-Relative Consumption Encoding branch (PRCE-CNN).

    Scientific premise (PLAN_J_2026-05-27.md):
    82.3 % of theft customers are individually undetectable by single-customer
    models.  The peer-relative residual r[c,t] = x̂[c,t] − mean_peer[t] exposes
    the step-change at theft week W_t for Light Droppers (48.3 % of theft),
    which is invisible in the absolute series but significant relative to
    k=20 nearest normal neighbours.

    This model expects x of shape (N, W, 14): x[:, :, :7] is the standard
    raw consumption series and x[:, :, 7:] is the pre-computed peer residual
    (runner_d.compute_peer_residuals, fold-safe, normal-only peer pool).

    Embed dim: 212 (A9-lean) + 60 (PRCE-CNN) = 272.
    No mamba_ssm required — runs in standard tgpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
        use_mixer=False,
        use_prce=True,
        prce_beta=60,
    ))


def build_a18(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A18: A9-lean trained with Physics-Sim Theft Augmentation (PLAN_K).

    Architecture identical to A9-lean. The differentiator is *training-set
    augmentation*: synthetic theft examples synthesised from real normal
    customers via documented theft-strategy manipulations (light / heavy /
    flat archetypes). The augmentation hook lives in ``run_cell``; the model
    here is therefore a vanilla A9-lean. Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a19(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A19: A9-lean initialised from Contrastive SSL pre-training (PLAN_K).

    Architecture identical to A9-lean; the differentiator is *initialisation*.
    Before supervised fine-tuning the runner pre-trains the backbone with an
    NT-Xent contrastive objective on the full unlabeled X (all 42 372
    customers). The pre-trained state-dict is loaded with shape-compatible
    matching before the standard training loop begins.  Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a20(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A20: A9-lean + Attention-Based Peer Comparison branch (PLAN_K).

    Replaces the hand-coded PRCE Laplacian-CNN (A17) with a learned
    cross-attention block (PeerAttentionBranch) jointly processing the
    customer's own NaME-imputed series and the pre-computed peer residual.
    Input shape contract identical to A17: (N, W, 14) — own | peer_res.
    Embed dim: 212 (A9-lean) + 60 (peer-GNN) = 272.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_ms_kernel=True,
        ms_dilations=(1, 2, 4),
        use_mask_channel=True,
        mask_beta=30,
        use_temporal_branch=True,
        temporal_dim=32,
        temporal_mlp_hidden=64,
        use_mixer=False,
        use_prce=True,
        use_peer_gnn=True,
        peer_gnn_dim=60,
        peer_gnn_hidden=64,
        peer_gnn_heads=4,
        peer_gnn_kernel=7,
    ))


def build_a21(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A21: Full theft-strategy mitigation stack — A20 + SSL init + PhysSim aug.

    Architecturally identical to A20 (A9-lean + peer-GNN branch). Stacking is
    realised by the runner:
      - SSL pre-trains the A9-lean backbone on all X; state-dict loaded with
        shape-compat matching (peer-GNN branch + fuse + head remain random).
      - Training set augmented with synthetic theft examples (A18 logic).
      - Standard supervised fine-tune.

    Embed dim 272 (same as A20).
    """
    return build_a20(weeks=weeks, days=days)


# ---------------------------------------------------------------------------
# Phase L (PLAN_L_2026-05-27.md) — TDS-MTN+ v2 component arms
# ---------------------------------------------------------------------------

def build_a22(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A22: A9-lean + symmetric 5-archetype training-set augmentation.

    Architecturally identical to A9-lean. Differentiator: ``run_cell``
    augmentation hook generates synthetic theft examples across all 5
    archetypes (incl. Risers) in empirical proportions.  Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a23(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A23: A9-lean initialised from SupCon (archetype-supervised contrastive).

    Architecture identical to A9-lean.  Runner performs SupCon pre-training
    using archetype labels before fine-tuning; shape-compatible state-dict
    loaded into the fresh model.  Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a24(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A24: A9-lean + engineered-features MLP branch.

    Adds 12 hand-crafted features (max_consec_nan, lag1_autocorr, …) via a
    small MLP (12 → 64 → 30).  Embed dim: 212 + 30 = 242.
    Runner provides the ``eng_feat`` tensor via ``model(x, eng_feat=…)``.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mixer=False,
        use_eng_features=True, eng_features_dim=30, eng_features_hidden=64,
    ))


def build_a25(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A25: A9-lean + Signed-Peer cross-attention branch.

    Replaces A20 (direction-blind) with a branch that ingests
    (signed_residual | peer_std | peer_iqr) and uses TWO directional gates
    (Riser-pos / Dropper-neg).  Input shape: (N, W, 4*D) = (N, 147, 28).
    Embed dim: 212 + 60 = 272.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mixer=False,
        use_signed_peer=True, signed_peer_dim=60,
    ))


def build_a26(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A26: A9-lean + multi-task head (binary theft + 6-way archetype, λ=0.3).

    The architecture of the *backbone* is identical to A9-lean. The runner
    wraps it in ``MultiTaskZKMixV4`` which adds a 6-way archetype head and
    routes through ``train_multitask()``.  Embed dim 212 (backbone).
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a27(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A27: A9-lean + Confident-Learning sample weighting.

    Same architecture; runner computes per-sample CL weights via in-fold CV
    on a tiny scorer model before training, then passes weights to the
    training loop.  Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a28(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A28: A9-lean + SAM optimizer + TTA inference.

    Same architecture; runner enables SAM during training (use_sam=True in
    MTTrainConfig — also applied through multitask hook) and uses TTA at
    inference time.  Embed dim 212.
    """
    return build_a9_lean(weeks=weeks, days=days)


def build_a29(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A29: FULL TDS-MTN+ stack.

    Architecture: A9-lean backbone + signed-peer branch + engineered-features
    branch. Embed dim: 212 + 60 + 30 = 302.

    Runner wraps in MultiTaskZKMixV4, enables:
      * Symmetric augmentation                   (A22)
      * SupCon backbone init                     (A23)
      * Multi-task loss                          (A26)
      * Confident-Learning sample weights        (A27)
      * SAM optimizer                            (A28 train)
      * TTA inference                            (A28 infer)
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mixer=False,
        use_signed_peer=True,  signed_peer_dim=60,
        use_eng_features=True, eng_features_dim=30, eng_features_hidden=64,
    ))


def build_a30(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A30 (Phase L, L1): a14 backbone (BiMamba2 wide, no Mixer) + eng-features branch.

    Representation-completeness arm. The a14 backbone (Mamba-2 wide + mask + temporal,
    Mixer pruned) never computes the temporal-complexity statistics that Paper 4
    identifies as the dominant theft separator (lag-1 autocorrelation KS=0.152,
    Shannon entropy 0.089, CV 0.083). This arm supplies them explicitly via the
    12 hand-crafted features (lag1_autocorr, roll_std_max/range, consumption_entropy,
    cv_global, wed_sat switching gap, dow_profile_var, missingness features) → MLP →
    fuse. Embed dim: 212 + 30 = 242.

    Differs from a14 ONLY by the eng-features branch (clean L1 ablation vs a14).
    Requires mamba_gpu env; runner injects eng_feat via the ENG_FEAT_ARMS path
    (train-fit standardisation, leakage-safe).
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mixer=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_eng_features=True, eng_features_dim=30, eng_features_hidden=64,
    ))


def build_a31(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A31 (Phase N): a14 backbone + multi-scale DIFFERENCING branch (MSDM-style).

    Differs from a14 ONLY by the differencing branch (zero-init -> a31 == a14 at
    init; clean paired ablation). Tests whether a rate-of-change representation
    (the one axis untested by L1/imputation/MoE/focal) lifts the detectable strata.
    Embed dim: 212 + 32 = 244. Requires mamba_gpu env.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=True, temporal_dim=32, temporal_mlp_hidden=64,
        use_mixer=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_diff_branch=True, diff_scales=(1, 7, 30), diff_dim=32, diff_hidden=48,
    ))


def build_a32_generic(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A32 (generic/cross-dataset): Mamba-wide + multi-scale CNN + differencing arm.

    STRIPPED of the SGCC-specific branches (CC-NaME MNAR front-end, mask-channel,
    monthly-block temporal MLP, Mixer) that are inert/mismatched on clean or
    different-resolution datasets. Keeps only resolution/length-agnostic components:
      * Mamba-wide (SSM long-range, any length)
      * multi-scale dilated CNN (local–mid patterns)
      * differencing arm (rate-of-change, scale-robust; MSDM-inspired)
    use_cc_name=False → generic scalar NaN handling (no missingness assumption).
    Embed: alpha(90, mamba) + beta(60, cnn) + diff_dim(32) = 182. Requires mamba_gpu.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=False,                       # generic, no MNAR front-end
        use_cnn_consump=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=False,                  # drop SGCC missingness branch
        use_temporal_branch=False,               # drop SGCC monthly-block branch
        use_mixer=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_diff_branch=True, diff_scales=(1, 7, 30), diff_dim=32, diff_hidden=48,
    ))


def build_a33_generic(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A33 (generic + missingness-adaptive): a32 core + CC-NaME + mask-channel.

    Recovers a14-parity on SGCC WITHOUT SGCC-overfitting. CC-NaME and the mask
    Laplacian are missingness-ADAPTIVE (carry signal on SGCC's 25% missing; ~INERT
    on clean ECL/PRECON — CC-NaME→identity when no NaN, mask Laplacian→~constant).
    Drops ONLY the truly SGCC-structure-specific monthly-temporal MLP (hardcoded
    weeks//7 blocks). Keeps resolution-agnostic Mamba + multi-scale conv + diff arm.
    Embed: alpha90(mamba)+beta60(cnn)+mask30+diff32 = 212. Requires mamba_gpu.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,                        # missingness-adaptive, inert on clean
        use_cnn_consump=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,     # missingness fingerprint, ~inert on clean
        use_temporal_branch=False,               # drop ONLY SGCC-structure-specific branch
        use_mixer=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_diff_branch=True, diff_scales=(1, 7, 30), diff_dim=32, diff_hidden=48,
    ))


def build_a34_generic(weeks: int = 147, days: int = 7) -> ZKMixV4:
    """A34 (generic + periodicity): a33 + MSDM inter/intra-period convolution branch.

    Adds the one unported MSDM/MSCNet component — IIPC dilated convs at intra-period
    {1,2} and inter-period {C,2C} (period C = days, resolution-adaptive). Captures
    cross-cycle rhythm changes (theft alters periodic structure). Generic + zero-init.
    Embed: 212 (a33) + ip_dim 32 = 244. Requires mamba_gpu.
    """
    return ZKMixV4(ZKMixV4Config(
        weeks=weeks, days=days,
        use_cc_name=True,
        use_cnn_consump=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
        use_mask_channel=True, mask_beta=30,
        use_temporal_branch=False,
        use_mixer=False,
        use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=32,
        use_diff_branch=True, diff_scales=(1, 7, 30), diff_dim=32, diff_hidden=48,
        use_interperiod_branch=True, ip_dim=32, ip_hidden=32,
    ))


def _a34_cfg(**over):
    base = dict(use_cc_name=True, use_cnn_consump=True, use_ms_kernel=True, ms_dilations=(1, 2, 4),
                use_mask_channel=True, mask_beta=30, use_temporal_branch=False, use_mixer=False,
                use_mamba_wide=True, mamba_d_model=32, mamba_d_state=128, mamba_d_conv=4,
                mamba_expand=2, mamba_headdim=32,
                use_diff_branch=True, diff_scales=(1, 7, 30), diff_dim=32, diff_hidden=48,
                use_interperiod_branch=True, ip_dim=32, ip_hidden=32)
    base.update(over)
    return base


def build_a34_ablate(drop):
    """a34 drop-one ablation: build a34 with exactly one component disabled."""
    flag = {"diff": "use_diff_branch", "ip": "use_interperiod_branch",
            "mask": "use_mask_channel", "ccname": "use_cc_name",
            "mamba": "use_mamba_wide", "cnn": "use_cnn_consump"}[drop]
    def _b(weeks: int = 147, days: int = 7):
        return ZKMixV4(ZKMixV4Config(weeks=weeks, days=days, **_a34_cfg(**{flag: False})))
    return _b


ARM_BUILDERS = {
    "a32_generic":             build_a32_generic,  # generic cross-dataset arm
    "a32_ablate_na":           build_a32_generic,  # alias for A_no-miss ablation arm cited in Table 5
    "a33_generic":             build_a33_generic,  # generic + missingness-adaptive
    "a34_generic":             build_a34_generic,  # generic + periodicity (MSDM IIPC)
    "a34_no_diff":             build_a34_ablate("diff"),    # a34 drop-one ablations
    "a34_no_ip":               build_a34_ablate("ip"),
    "a34_no_mask":             build_a34_ablate("mask"),
    "a34_no_ccname":           build_a34_ablate("ccname"),
    "a34_no_mamba":            build_a34_ablate("mamba"),
    "a34_no_cnn":              build_a34_ablate("cnn"),
    # build-up (cumulative) ablation: S0 core → S1 +ccname → S2 +mask → S3=a33 (+diff) → S4=a34 (+periodicity)
    "a34_bu0_core":  lambda weeks=147, days=7: ZKMixV4(ZKMixV4Config(weeks=weeks, days=days, **_a34_cfg(
                        use_cc_name=False, use_mask_channel=False, use_diff_branch=False, use_interperiod_branch=False))),
    "a34_bu1_ccname": lambda weeks=147, days=7: ZKMixV4(ZKMixV4Config(weeks=weeks, days=days, **_a34_cfg(
                        use_mask_channel=False, use_diff_branch=False, use_interperiod_branch=False))),
    "a34_bu2_mask":  lambda weeks=147, days=7: ZKMixV4(ZKMixV4Config(weeks=weeks, days=days, **_a34_cfg(
                        use_diff_branch=False, use_interperiod_branch=False))),
    "a0_nanative":             _build_a0_nanative,
    "a1_cc_name_only":         build_a1,
    "a2_ms_only":              build_a2,
    "a3_cc_name_ms":           build_a3,
    "a4_cc_name_d1":           build_a4,
    "a5_cc_name_ms_nowide":    build_a5,
    "a6_branch_selective":     build_a6,
    "a7_mask_channel":         build_a7,
    "a8_temporal_branch":      build_a8,
    "a9_mask_temporal":        build_a9,
    "a10_focal":               build_a10,
    "a11_bilstm_wide":         build_a11,
    "a12_mamba_wide":          build_a12,   # Phase H — requires mamba_gpu env
    "a13_no_cnn_consump":      build_a13,   # Phase I ablation — requires mamba_gpu env
    "a14_no_mixer":            build_a14,   # Phase I ablation — requires mamba_gpu env
    "a14":                     build_a14,   # alias for A_21-block ablation arm cited in Table 5
    "a15_no_mask":             build_a15,   # Phase I ablation — requires mamba_gpu env
    "a16_no_temporal":         build_a16,   # Phase I ablation — requires mamba_gpu env
    "a9_lean":                 build_a9_lean,   # Phase J: A9 − Mixer (pruned)
    "a17_prce":                build_a17,        # Phase J: A9-lean + PRCE-CNN
    "a18_physsim_aug":         build_a18,        # Phase K: PhysSim augmentation
    "a19_ssl_identity":        build_a19,        # Phase K: SSL pre-training
    "a20_gnn_peer":            build_a20,        # Phase K: GNN peer comparison
    "a21_full_mitigation":     build_a21,        # Phase K: full stack
    "a22_sym_aug":             build_a22,        # Phase L: symmetric augmentation
    "a23_supcon":              build_a23,        # Phase L: SupCon pretraining
    "a31_diff":                build_a31,        # Phase N: multi-scale differencing branch
    "a24_eng_features":        build_a24,        # Phase L: engineered features branch
    "a25_signed_peer":         build_a25,        # Phase L: signed peer attention
    "a26_multitask":           build_a26,        # Phase L: multi-task head
    "a27_conf_learn":          build_a27,        # Phase L: confident learning weights
    "a28_sam_tta":             build_a28,        # Phase L: SAM + TTA
    "a29_tds_mtn_plus":        build_a29,        # Phase L: FULL stack
    "a30_eng_mamba":           build_a30,        # Phase L (L1): a14 + eng-features — requires mamba_gpu env
}

# b2 baselines (Zheng WDCNN, Zhu HORLN, Finardi HybridAttn, Che GRU-D, Wang MSDM)
# wired to the identical (B,W,D)->logit contract so run_n_clean evaluates them
# under the same honest protocol. Imported lazily-safe (no circular dependency).
from .baselines import BASELINE_BUILDERS  # noqa: E402
ARM_BUILDERS.update(BASELINE_BUILDERS)


def build_arm(name: str, weeks: int = 147, days: int = 7) -> nn.Module:
    if name == "a0_nanative":
        # Phase C model on NaN-native data — establishes the fair A0 baseline.
        from zkmix_lab.models import build_model as _build_phase_c  # type: ignore
        return _build_phase_c("zkmix_v3_name_nn", weeks=weeks, days=days)
    if name not in ARM_BUILDERS:
        raise ValueError(f"Unknown arm {name!r}. Valid: {sorted(ARM_BUILDERS)}")
    return ARM_BUILDERS[name](weeks=weeks, days=days)


__all__ = [
    "ZKMixV4Config",
    "ZKMixV4",
    "BiLSTMWide",
    "MambaWide",
    "build_a1", "build_a2", "build_a3", "build_a4", "build_a5", "build_a6",
    "build_a7", "build_a8", "build_a9", "build_a10", "build_a11", "build_a12",
    "build_a13", "build_a14", "build_a15", "build_a16",
    "build_a9_lean", "build_a17",
    "build_a18", "build_a19", "build_a20", "build_a21",
    "build_a22", "build_a23", "build_a24", "build_a25",
    "build_a26", "build_a27", "build_a28", "build_a29", "build_a30",
    "build_arm",
    "ARM_BUILDERS",
]
