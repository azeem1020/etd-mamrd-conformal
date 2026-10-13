"""Model zoo — Phase B + classical baselines.

All backbones are binary classifiers over weekly meter grids.

Common contract
---------------
* Input  : float tensor ``x`` of shape ``(N, W, D)`` — N customers,
           W weeks, D days/week.  D is always 7 in the SGCC pipeline
           but the modules do not hard-code it.
* Output : float tensor ``logit`` of shape ``(N,)``.  The training loop
           applies ``F.binary_cross_entropy_with_logits``; calibration
           is the caller's responsibility (sigmoid + APS lives in
           ``sct_etd.calibration`` and is unchanged).

The three backbones are deliberately written at *training-recipe
parity* — same Lion+cosine schedule, same binary-stratified sampler,
same binary head — so that a head-to-head bootstrap comparison
attributes the gap to the inductive bias rather than to optimisation
hyper-parameters.

ZKMix v2
~~~~~~~~
MLP-Mixer over the 2-D weekly grid.  The 2-D inductive bias maps onto
the data geometry: column-mixing aggregates *across-day-within-week*
(periodicity axis identified in Zheng §III), row-mixing aggregates
*across-week-at-fixed-day* (trend axis).  Kernel-stat injection is now
GPU-resident via ``kernels.AntiMeanKernel`` /
``kernels.LearnablePriorConv``.

TSMixer
~~~~~~~
Chen et al. ICML 2023, "TSMixer: An All-MLP Architecture for Time
Series".  Treats the input as a (L, C) sequence with L = W and C = D
(weekday channels).  Alternates time-mixing along L and feature-mixing
along C.

TimeMixer
~~~~~~~~~
Wang et al. ICLR 2024, "TimeMixer: Decomposable Multiscale Mixing".
We use the *past-decomposable mixing* encoder block (the forecasting
decoder is dropped — we are classifying, not forecasting), with
seasonal-trend decomposition at multiple scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import AntiMeanKernel, KernelSpec, LearnablePriorConv, weekly_kernel_stats


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Two-layer MLP with GELU + dropout, used in every Mixer-style block."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class BinaryHead(nn.Module):
    """Shared two-layer binary head — same wiring for all three backbones."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PartialConv2d(nn.Conv2d):
    """Partial Convolution layer (Liu et al. 2018, ECCV — NVIDIA).

    Reference: "Image Inpainting for Irregular Holes Using Partial
    Convolutions", arXiv:1804.07723, Eq. 1.

    For an input ``x`` of shape (N, C_in, H, W) and a *validity* mask of
    shape (N, 1, H, W) with values in {0, 1} (1 = valid observation,
    0 = missing), the partial convolution is:

        x'(i,j) = W^T (X(i,j) ⊙ M(i,j)) · ( sum(1) / sum(M(i,j)) ) + b
                  if sum(M(i,j)) > 0
                = 0   (effectively)
                  otherwise

    where:
      * ``M(i,j)`` is the mask window at output position (i,j);
      * ``sum(M(i,j))`` is the count of valid input cells in that window;
      * ``sum(1) = C_in · k_H · k_W`` is the constant window size;
      * the renormaliser ``sum(1)/sum(M)`` upweights the conv output when
        the receptive field has many missing cells so the magnitude of
        the activation is comparable to what it would have been on a
        fully-valid window.

    The propagated *output mask* is

        M'(i,j) = 1   if sum(M(i,j)) > 0
                = 0   otherwise

    and is returned alongside the convolution result so that downstream
    layers can chain the mask propagation if desired.

    Properties verified in tests (see tests/test_models.py):
      * all-ones mask  → bit-identical to ``nn.Conv2d`` (sanity).
      * all-zeros mask → all-zeros output, all-zeros propagated mask.
      * partial mask   → renormalisation matches the closed-form formula.
      * mask path has NO gradient (mask treated as a fixed indicator).
      * gradient on ``self.weight`` flows correctly through the masked
        convolution (so frozen weights stay frozen, learnable ones train).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, dilation=1,
                 bias: bool = True) -> None:
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias,
        )
        # The mask-sum kernel is all-ones over (1, in_channels, k_H, k_W) — a
        # 1-output-channel conv that sums valid input cells per output window
        # across all input channels.  Stored as a non-learnable buffer.
        self.register_buffer(
            "_mask_kernel",
            torch.ones(1, self.in_channels,
                       self.kernel_size[0], self.kernel_size[1]),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (N, C_in, H, W)
        mask: (N, 1, H, W)   — values in {0, 1}; broadcast across C_in.
                              May also be (N, C_in, H, W) if a per-channel mask
                              is provided; both shapes are accepted.
        Returns:
          out:      (N, C_out, H_out, W_out)
          new_mask: (N, 1,     H_out, W_out)
        """
        if x.dim() != 4:
            raise ValueError(f"PartialConv2d expects (N, C, H, W); got {tuple(x.shape)}.")
        if mask.dim() != 4 or mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"mask must be (N, 1|C_in, H, W) aligned with x; got {tuple(mask.shape)}."
            )
        if mask.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                f"mask channels must be 1 or C_in={x.shape[1]}; got C_mask={mask.shape[1]}."
            )
        # Treat the mask as a constant indicator: detach to prevent gradient
        # from flowing back to it through ``x * mask`` or the conv.  This
        # enforces the Liu et al. semantics that masks come from the data
        # (or a deterministic predicate of the input), never from a
        # learnable parameter.
        mask = mask.detach()
        # Standard conv on the mask-zeroed input.
        # The mask broadcasts along C_in if it has shape (N, 1, H, W).
        x_masked = x * mask
        out = self._conv_forward(x_masked, self.weight, self.bias)

        # Compute the window-wise valid-count, treating the mask as fixed
        # (no gradient through this path).
        #
        # Boundary handling: the convolution zero-pads its input.  We do
        # NOT want this zero-pad to be interpreted as "missing data" — that
        # would over-scale every boundary output even when all real cells
        # are valid.  We therefore use a *position-dependent* maximum
        # ``valid_max`` (the count of in-image cells each window covers)
        # as the numerator of the renormaliser.  On windows with no
        # missings, ``mask_sum == valid_max`` and the scale is exactly 1.0
        # — partial conv reduces to a standard conv bit-for-bit.
        with torch.no_grad():
            if mask.shape[1] == 1:
                mask_for_sum = mask.expand(-1, self.in_channels, -1, -1)
            else:
                mask_for_sum = mask
            mask_sum = nn.functional.conv2d(
                mask_for_sum, self._mask_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
            )  # (N, 1, H_out, W_out)
            valid_max = nn.functional.conv2d(
                torch.ones_like(mask_for_sum), self._mask_kernel, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
            )
            new_mask = (mask_sum > 0).to(x.dtype)
            # Clamp denominator to avoid div-by-zero at fully-invalid windows
            # (output zeroed by new_mask anyway, but scale must stay finite).
            scale = valid_max / torch.clamp(mask_sum, min=1.0)

        # Renormalise the conv output by the window-validity ratio.  Subtract
        # and add back bias so the renormaliser scales only the W·(X⊙M) term,
        # matching Liu et al. Eq. 1 exactly.
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (out - b) * scale + b
        else:
            out = out * scale
        # Zero positions where the window contained no valid cells.
        out = out * new_mask
        return out, new_mask


def _multi_pool(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Concatenate mean ⊕ max ⊕ std along ``dim``.

    Returns a tensor whose ``dim`` is collapsed and whose last-feature
    dimension is 3× the original feature dimension.  Used by all three
    backbones so they share the same pooling capacity.
    """
    return torch.cat(
        [x.mean(dim=dim), x.amax(dim=dim), x.std(dim=dim, unbiased=False)],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# ZKMix v2 — MLP-Mixer over (W, D) with anti-mean kernel injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZKMixConfig:
    weeks: int = 147
    days: int = 7
    hidden_dim: int = 96
    token_mlp_dim: int = 192          # MLP hidden for the across-week mix
    channel_mlp_dim: int = 192        # MLP hidden for the across-channel mix
    n_blocks: int = 4
    dropout: float = 0.10
    kernel: KernelSpec = field(default_factory=KernelSpec)
    stats: tuple[str, ...] = ("mean", "std", "max", "p99")
    inject_kernel_stats: bool = True  # False ⇒ mechanism-ablation backbone


class MixerBlock2D(nn.Module):
    """Standard MLP-Mixer block: token-mix (across W) then channel-mix."""

    def __init__(
        self, n_tokens: int, hidden: int, token_mlp: int, channel_mlp: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.token_mix = MLPBlock(n_tokens, token_mlp, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.channel_mix = MLPBlock(hidden, channel_mlp, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, W, hidden)
        y = self.norm1(x).transpose(1, 2)        # (N, hidden, W)
        y = self.token_mix(y).transpose(1, 2)    # (N, W, hidden)
        x = x + y
        y = self.norm2(x)
        return x + self.channel_mix(y)


class ZKMix(nn.Module):
    """ZKMix v2: kernel-prior + MLP-Mixer."""

    def __init__(self, cfg: ZKMixConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ZKMixConfig()
        self.cfg = cfg

        self.kernel = cfg.kernel.build()
        self._kernel_out_channels = cfg.kernel.out_channels

        # Two parallel embeddings of each week:
        #   raw daily values  → linear projection
        #   per-week kernel statistics → linear projection
        # Their sum is the input token to the Mixer stack.
        self.patch_proj = nn.Linear(cfg.days, cfg.hidden_dim)
        stat_dim = self._kernel_out_channels * len(cfg.stats)
        self.stat_proj = nn.Linear(stat_dim, cfg.hidden_dim)
        self.stat_norm = nn.LayerNorm(cfg.hidden_dim)

        self.blocks = nn.ModuleList(
            MixerBlock2D(
                n_tokens=cfg.weeks,
                hidden=cfg.hidden_dim,
                token_mlp=cfg.token_mlp_dim,
                channel_mlp=cfg.channel_mlp_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_blocks)
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        # Multi-stat pool → 3 × hidden_dim; head sees that.
        self.embed_dim = 3 * cfg.hidden_dim
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    # ---- helpers ----------------------------------------------------------

    def _kernel_response(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (N, 1, W, D) → response: (N, C, W, D)."""
        return self.kernel(grid)

    def _stat_embed(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (N, 1, W, D) → (N, W, hidden_dim) per-week stat features."""
        resp = self._kernel_response(grid)
        stats = weekly_kernel_stats(resp, stats=self.cfg.stats)  # (N, W, C*S)
        return self.stat_norm(self.stat_proj(stats))

    # ---- embedding (selective-head plug-in target) ------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(N, W, D) → (N, embed_dim) pre-head embedding."""
        if x.dim() != 3:
            raise ValueError(f"ZKMix expects (N, W, D); got tuple(x.shape)={tuple(x.shape)}.")
        n, w, d = x.shape
        if w != self.cfg.weeks or d != self.cfg.days:
            raise ValueError(
                f"ZKMix configured for (W={self.cfg.weeks}, D={self.cfg.days}); "
                f"got (W={w}, D={d})."
            )
        grid = x.unsqueeze(1)                              # (N, 1, W, D)
        tokens = self.patch_proj(x)                        # (N, W, hidden)
        if self.cfg.inject_kernel_stats:
            tokens = tokens + self._stat_embed(grid)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return _multi_pool(tokens, dim=1)                  # (N, 3 * hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ---------------------------------------------------------------------------
# TSMixer — Chen et al. 2023 (ICML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TSMixerConfig:
    weeks: int = 147
    days: int = 7
    n_blocks: int = 4
    time_mlp_dim: int = 192
    feature_mlp_dim: int = 64
    dropout: float = 0.10


class TSMixerBlock(nn.Module):
    """One TSMixer block: time-mix (across L) → feature-mix (across C)."""

    def __init__(self, length: int, channels: int, t_mlp: int, c_mlp: int, dropout: float) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(channels)
        self.time_mix = MLPBlock(length, t_mlp, dropout=dropout)
        self.norm_c = nn.LayerNorm(channels)
        self.channel_mix = MLPBlock(channels, c_mlp, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, L, C)
        y = self.norm_t(x).transpose(1, 2)        # (N, C, L)
        y = self.time_mix(y).transpose(1, 2)
        x = x + y
        y = self.norm_c(x)
        return x + self.channel_mix(y)


class TSMixer(nn.Module):
    """TSMixer for binary classification.

    The original paper targets multivariate forecasting; we keep the
    encoder stack verbatim and replace the linear forecasting head with
    a mean+max+std pool followed by ``BinaryHead``.
    """

    def __init__(self, cfg: TSMixerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TSMixerConfig()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            TSMixerBlock(
                length=cfg.weeks,
                channels=cfg.days,
                t_mlp=cfg.time_mlp_dim,
                c_mlp=cfg.feature_mlp_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_blocks)
        )
        self.norm = nn.LayerNorm(cfg.days)
        self.embed_dim = 3 * cfg.days
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TSMixer expects (N, W, D); got tuple(x.shape)={tuple(x.shape)}.")
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return _multi_pool(x, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ---------------------------------------------------------------------------
# TimeMixer — Wang et al. ICLR 2024
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeMixerConfig:
    weeks: int = 147
    days: int = 7
    n_scales: int = 3                     # number of downsampling levels
    moving_avg_kernel: int = 7            # seasonal-trend split kernel
    n_blocks: int = 3                     # past-decomposable mixing depth
    hidden_dim: int = 64
    season_mlp_dim: int = 128
    trend_mlp_dim: int = 128
    dropout: float = 0.10


class _MovingAvgDecomp(nn.Module):
    """Seasonal-trend decomposition by 1-D moving average.

    Matches the description in TimeMixer §3.1: trend = AvgPool1d(kernel),
    season = x - trend.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("moving_avg_kernel must be odd to preserve length.")
        self.kernel_size = kernel_size
        self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (N, L, C)  -> trend, season same shape
        trend = self.avgpool(x.transpose(1, 2)).transpose(1, 2)
        season = x - trend
        return season, trend


class _PDMBlock(nn.Module):
    """Past-decomposable mixing block: separate MLPs for season + trend."""

    def __init__(self, length: int, hidden: int, s_mlp: int, t_mlp: int, dropout: float) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(hidden)
        self.norm_t = nn.LayerNorm(hidden)
        self.season_mix = MLPBlock(length, s_mlp, dropout=dropout)
        self.trend_mix = MLPBlock(length, t_mlp, dropout=dropout)

    def forward(self, season: torch.Tensor, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.norm_s(season).transpose(1, 2)
        s_out = self.season_mix(s).transpose(1, 2)
        season = season + s_out
        t = self.norm_t(trend).transpose(1, 2)
        t_out = self.trend_mix(t).transpose(1, 2)
        trend = trend + t_out
        return season, trend


class TimeMixer(nn.Module):
    """TimeMixer encoder for binary classification.

    The model:
    1. flattens (W, D) into a 1-D length-``W*D`` sequence with a single
       channel,
    2. projects to ``hidden_dim`` channels (token embedding equivalent),
    3. builds ``n_scales`` downsampled views via stride-2 average pool,
    4. decomposes each scale into (season, trend) by a moving-average
       filter,
    5. runs ``n_blocks`` past-decomposable mixing blocks that mix
       *within each scale* (season and trend separately),
    6. up-samples coarser scales back to the finest length, sums, and
       pools (mean ⊕ max ⊕ std) before the binary head.
    """

    def __init__(self, cfg: TimeMixerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TimeMixerConfig()
        self.cfg = cfg
        self.length = cfg.weeks * cfg.days
        # ``embed_proj``, not ``embed``, to avoid colliding with the
        # backbone-level ``embed()`` method used by the selective wrapper.
        self.embed_proj = nn.Linear(1, cfg.hidden_dim)
        self.decomp = _MovingAvgDecomp(cfg.moving_avg_kernel)

        # Pre-compute the downsampled lengths so the blocks can size
        # their MLPs.  Each scale halves the length (ceil div).
        self.scale_lengths: list[int] = []
        L = self.length
        for _ in range(cfg.n_scales):
            self.scale_lengths.append(L)
            L = (L + 1) // 2
        self.scale_lengths.append(L)

        # One PDM stack per scale (depth = n_blocks).
        self.stacks = nn.ModuleList(
            nn.ModuleList(
                _PDMBlock(
                    length=L_scale,
                    hidden=cfg.hidden_dim,
                    s_mlp=cfg.season_mlp_dim,
                    t_mlp=cfg.trend_mlp_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_blocks)
            )
            for L_scale in self.scale_lengths
        )

        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.embed_dim = 3 * cfg.hidden_dim
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    @staticmethod
    def _downsample(x: torch.Tensor) -> torch.Tensor:
        # x : (N, L, C) → stride-2 average pool along L.
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).transpose(1, 2)

    def _upsample_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == target_len:
            return x
        return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TimeMixer expects (N, W, D); got tuple(x.shape)={tuple(x.shape)}.")
        n = x.shape[0]
        seq = x.reshape(n, self.length, 1)
        seq = self.embed_proj(seq)                         # (N, L, hidden)

        scales = [seq]
        for _ in range(self.cfg.n_scales):
            scales.append(self._downsample(scales[-1]))

        outputs: list[torch.Tensor] = []
        for stack, scaled in zip(self.stacks, scales):
            season, trend = self.decomp(scaled)
            for blk in stack:
                season, trend = blk(season, trend)
            outputs.append(self._upsample_to(season + trend, self.length))

        merged = torch.stack(outputs, dim=0).sum(dim=0)    # (N, L, hidden)
        merged = self.norm(merged)
        return _multi_pool(merged, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ---------------------------------------------------------------------------
# Classical baselines — vanilla 1-D CNN, BiLSTM, TCN, Transformer encoder
#
# These are deliberately *prior-free* — no anti-mean kernel, no 2-D weekly
# grid, no multi-scale decomposition.  They protect against the obvious
# reviewer objection: "does the ZKMix prior actually help, or does any
# modern architecture catch up?"  All four expose the standard
# ``embed(x)``/``embed_dim``/``forward(x)`` contract.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vanilla1DCNNConfig:
    weeks: int = 147
    days: int = 7
    channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 5
    dropout: float = 0.1


class Vanilla1DCNN(nn.Module):
    """Plain 1-D CNN — three blocks of (Conv1d → BN → GELU → MaxPool).

    Input  :  (N, W, D)  → flattened to (N, 1, W*D).
    Stack  :  Conv1d(k=5, padding=2) → BN → GELU → MaxPool(2)  × 3 blocks.
    Head   :  GlobalAvgPool over time → Linear → logit.

    The point of this backbone is "what does a textbook 1-D CNN, no
    domain prior, give you on this problem?"  It is intentionally
    plain.
    """

    def __init__(self, cfg: Vanilla1DCNNConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or Vanilla1DCNNConfig()
        self.cfg = cfg

        blocks: list[nn.Module] = []
        in_c = 1
        for out_c in cfg.channels:
            blocks.append(nn.Conv1d(in_c, out_c, kernel_size=cfg.kernel_size,
                                    padding=cfg.kernel_size // 2))
            blocks.append(nn.BatchNorm1d(out_c))
            blocks.append(nn.GELU())
            blocks.append(nn.MaxPool1d(2))
            blocks.append(nn.Dropout(cfg.dropout))
            in_c = out_c
        self.encoder = nn.Sequential(*blocks)
        self.embed_dim = cfg.channels[-1]
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Vanilla1DCNN expects (N, W, D); got {tuple(x.shape)}.")
        n = x.shape[0]
        seq = x.reshape(n, 1, -1)               # (N, 1, W*D)
        feat = self.encoder(seq)                # (N, C_last, T')
        return feat.mean(dim=-1)                # global-avg pool → (N, C_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


@dataclass(frozen=True)
class BiLSTMConfig:
    weeks: int = 147
    days: int = 7
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1


class BiLSTM(nn.Module):
    """Bidirectional LSTM over the daily sequence.

    The hidden state of each layer is bidirectional; we pool over time
    by ``mean ⊕ max ⊕ last`` to give the head a 3 × (2·hidden) embedding.
    Mirrors the receipe in ``sct_etd.models.vanilla_bigru`` but kept
    self-contained so this lab has no dependency on the upstream package.
    """

    def __init__(self, cfg: BiLSTMConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or BiLSTMConfig()
        self.cfg = cfg
        # Input is (N, L=W*D, 1) — daily values as a length-L sequence.
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        # mean ⊕ max ⊕ last  →  3 × (2·hidden).
        self.embed_dim = 3 * 2 * cfg.hidden_dim
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"BiLSTM expects (N, W, D); got {tuple(x.shape)}.")
        n = x.shape[0]
        seq = x.reshape(n, -1, 1)               # (N, L, 1)
        out, _ = self.rnn(seq)                  # (N, L, 2·hidden)
        return torch.cat([out.mean(1), out.amax(1), out[:, -1, :]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


@dataclass(frozen=True)
class TCNConfig:
    """TCN config — defaults chosen so the receptive field covers ~70 % of
    the SGCC sequence at the full regime (1029 days).

    Receptive field formula (Bai et al. 2018):
        RF = 2 · (k − 1) · Σ_{l=0}^{L-1} 2^l  +  1
           = 2 · (k − 1) · (2^L − 1)  +  1

    With ``k = 5`` and ``L = 6`` blocks: RF = 2·4·(2⁶ − 1) + 1 = **505**
    timesteps (~72 weeks).  The earlier (k=3, L=4) default gave RF ≈ 61
    timesteps (≈ 9 weeks), which underfits relative to the other
    backbones; bumped here for a fair head-to-head comparison.
    """

    weeks: int = 147
    days: int = 7
    channels: tuple[int, ...] = (32, 64, 64, 128, 128, 128)
    kernel_size: int = 5
    dropout: float = 0.1


class _TCNBlock(nn.Module):
    """One Bai et al. 2018 TCN residual block: two dilated causal convs."""

    def __init__(self, in_c: int, out_c: int, k: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (k - 1) * dilation  # causal padding to the left
        self.k = k
        self.pad = pad
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_c, out_c, kernel_size=k, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_c, out_c, kernel_size=k, dilation=dilation)
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # 1×1 conv on the residual when channels differ.
        self.residual = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def _causal_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Pad ``pad`` zeros on the left of the time dim, then convolve;
        # equivalent to causal masking but cheaper than the conv-mask trick.
        x = F.pad(x, (self.pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self._causal_conv(self.conv1, x)))
        out = self.drop(self.act(self._causal_conv(self.conv2, out)))
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network (Bai, Kolter, Koltun 2018).

    Stack of dilated causal convolutional blocks with residual
    connections; the dilation grows exponentially so the receptive
    field reaches the full sequence with O(log L) blocks.  We pool the
    final feature map by ``mean ⊕ last`` for a 2× (last channel) embedding.
    """

    def __init__(self, cfg: TCNConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TCNConfig()
        self.cfg = cfg
        blocks: list[nn.Module] = []
        in_c = 1
        for i, out_c in enumerate(cfg.channels):
            blocks.append(_TCNBlock(in_c, out_c, k=cfg.kernel_size,
                                     dilation=2 ** i, dropout=cfg.dropout))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self.embed_dim = 2 * cfg.channels[-1]
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TCN expects (N, W, D); got {tuple(x.shape)}.")
        n = x.shape[0]
        seq = x.reshape(n, 1, -1)               # (N, 1, L)
        feat = self.blocks(seq)                 # (N, C_last, L)
        return torch.cat([feat.mean(dim=-1), feat[:, :, -1]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


@dataclass(frozen=True)
class TransformerConfig:
    weeks: int = 147
    days: int = 7
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 192
    dropout: float = 0.1
    patch_size: int = 7  # one token per week (D=7)


class TransformerEncoder(nn.Module):
    """Vanilla transformer encoder over weekly patches.

    Each row of the (W, D) grid is a ``patch_size``-day patch projected
    linearly into ``d_model``.  We prepend a learnable [CLS] token and
    use sinusoidal positional encoding (kept fixed, not learned, so the
    model generalises to different ``weeks`` values under regime
    ablation).  The [CLS] token's final state is the embedding.
    """

    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or TransformerConfig()
        self.cfg = cfg
        if cfg.days % cfg.patch_size != 0:
            raise ValueError(
                f"days ({cfg.days}) must be a multiple of patch_size ({cfg.patch_size})."
            )
        # Each week's 7 daily values → one token.
        self.patch_proj = nn.Linear(cfg.patch_size, cfg.d_model)
        # Learnable [CLS] prepended to every sequence.
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.embed_dim = cfg.d_model
        self.head = BinaryHead(in_dim=self.embed_dim, dropout=cfg.dropout)

    @staticmethod
    def _sinusoidal_pos(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Standard Vaswani et al. 2017 sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model, device=device)
        pos = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TransformerEncoder expects (N, W, D); got {tuple(x.shape)}.")
        n, w, d = x.shape
        if d != self.cfg.days:
            raise ValueError(f"days mismatch: config={self.cfg.days}, input={d}.")
        tokens = self.patch_proj(x)                       # (N, W, d_model)
        cls = self.cls.expand(n, -1, -1)                  # (N, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)             # (N, W+1, d_model)
        pos = self._sinusoidal_pos(seq.shape[1], self.cfg.d_model, seq.device)
        seq = seq + pos
        out = self.encoder(seq)                           # (N, W+1, d_model)
        out = self.norm(out)
        return out[:, 0, :]                                # [CLS] embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ---------------------------------------------------------------------------
# Zheng baseline — wide-and-deep CNN (Zheng et al. IEEE TII 2018) — corrected
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZhengConfig:
    """Hyper-parameters from Zheng et al. 2018, Table III.

    α = 90 (wide branch Dense), β = 60 (deep branch Dense after pool/flatten),
    γ = 15 (Conv2D filters), R = 1 (number of conv layers).
    """

    weeks: int = 147
    days: int = 7
    alpha: int = 90
    beta: int = 60
    gamma: int = 15
    dropout: float = 0.1


def _eq7_nine_stencil_weights() -> torch.Tensor:
    """The nine 3×3 stencils that compute (g1 + g2)(V) per output position.

    Derivation
    ----------
    Eqs. 8 and 9 of Zheng et al. 2018 define g1 / g2 over a 3×3 window
    V[0..2, 0..2].  The *(g1+g2)* response at output position (i_out, j_out)
    expands to a single linear functional of the nine input cells.  For
    instance, at the centre (1, 1):

        g1[1,1] + g2[1,1]
          = (2·V[1,1] - V[0,1] - V[2,1])  +  (2·V[1,1] - V[1,0] - V[1,2])
          = 4·V[1,1] - V[0,1] - V[2,1] - V[1,0] - V[1,2]

    which is the discrete 4-neighbour Laplacian stencil.  Off-centre
    positions give different (asymmetric) stencils.  We pre-compute all
    nine stencils so the paper's "expand + 3×3 kernel + stride-3 conv"
    pipeline collapses to a single padded-conv with nine output channels
    (one channel per output position within a 3×3 window).

    Returns
    -------
    (9, 1, 3, 3) tensor — frozen weights for ``nn.Conv2d(1, 9, 3, padding=1)``.
    """
    # Row-anti-mean matrix M_g1 acts on rows: each output row is
    # 2·this − (sum of other two).
    M = torch.tensor(
        [[2.0, -1.0, -1.0],
         [-1.0, 2.0, -1.0],
         [-1.0, -1.0, 2.0]]
    )
    stencils = torch.zeros(9, 1, 3, 3)
    for i_out in range(3):
        for j_out in range(3):
            ch = i_out * 3 + j_out
            # g1[i_out, j_out] = sum_{r} M[i_out, r] · V[r, j_out]
            for r in range(3):
                stencils[ch, 0, r, j_out] += M[i_out, r]
            # g2[i_out, j_out] = sum_{c} M[j_out, c] · V[i_out, c]
            for c in range(3):
                stencils[ch, 0, i_out, c] += M[j_out, c]
    return stencils


class ZhengBaseline(nn.Module):
    """Paper-faithful PyTorch port of the Zheng wide-and-deep CNN (2018).

    Equivalence to the paper's "expand + Eq. 7-9 kernel + stride-3 Conv"
    --------------------------------------------------------------------
    The paper:
        (a) Expands V (W×D) → V' (3W × 3D) by replicating each 3-row /
            3-column sliding window.
        (b) Applies Eqs. 8-9 to every 3×3 patch of V'.
        (c) Convolves V' with a learnable 3×3 filter ŵ at stride 3 to
            produce γ feature channels at the original (W, D) resolution.

    Linearity makes (b) + (c) equivalent to "convolve V with the nine
    fixed (g1+g2) stencils → nine channels → 1×1 learnable conv to γ
    channels".  We pre-compute the nine stencils as a frozen Conv2d and
    use a learnable 1×1 conv on top.  This is *paper-faithful* — it
    reproduces what Zheng's pipeline computes, just without the
    expansion trick.

    Forward
    -------
    x : (N, W, D)
    out : (N,) binary logit.

    Architecture (Figure 5 of the paper):
        deep branch :  9-channel anti-mean stencil (frozen)
                       →  1×1 Conv(γ, ReLU)
                       →  MaxPool(2×2)
                       →  Flatten  →  Dense(β, ReLU)
        wide branch :  Flatten(x)  →  Dense(α, ReLU)
        fuse        :  concat  →  Dense(64, ReLU)  →  Dense(1)   (binary logit)
    """

    def __init__(self, cfg: ZhengConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ZhengConfig()
        self.cfg = cfg

        # Frozen nine-stencil anti-mean kernel.
        self.kernel = nn.Conv2d(
            in_channels=1, out_channels=9,
            kernel_size=3, padding=1, bias=False,
        )
        with torch.no_grad():
            self.kernel.weight.copy_(_eq7_nine_stencil_weights())
        self.kernel.weight.requires_grad_(False)

        # Learnable 1×1 conv = the Eq. 7 filter ŵ.
        self.conv = nn.Conv2d(9, cfg.gamma, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

        deep_pool_w = cfg.weeks // 2
        deep_pool_d = cfg.days // 2
        deep_flat = deep_pool_w * deep_pool_d * cfg.gamma
        self.deep_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(deep_flat, cfg.beta),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

        wide_in = cfg.weeks * cfg.days
        self.wide_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(wide_in, cfg.alpha),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

        self.embed_dim = cfg.alpha + cfg.beta
        self.fuse = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(64, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ZhengBaseline expects (N, W, D); got {tuple(x.shape)}.")
        # Deep branch: nine-stencil response → 1×1 conv (Eq. 7 ŵ) → pool.
        grid = x.unsqueeze(1)                              # (N, 1, W, D)
        resp = self.kernel(grid)                           # (N, 9, W, D)  (frozen)
        deep = self.pool(self.act(self.conv(resp)))        # (N, γ, W/2, D/2)
        deep_emb = self.deep_fc(deep)                      # (N, β)
        # Wide branch.
        wide_emb = self.wide_fc(x)                         # (N, α)
        return torch.cat([wide_emb, deep_emb], dim=-1)     # (N, α+β)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fuse(self.embed(x))).squeeze(-1)


# ---------------------------------------------------------------------------
# ZKMix v3 — Wide-and-Deep MIXER (2026-05-17 breakthrough candidate)
# ---------------------------------------------------------------------------
#
# Diagnosis from the 540-cell run (results in zkmix_lab_outputs/):
# Zheng wide-and-deep beats every modern architecture (incl. ZKMix v2) by
# 0.05–0.06 PR-AUC because it has three inductive biases ZKMix v2 lacks:
#   1. a WIDE branch — direct path raw → Dense classifier (Cheng et al. 2016)
#   2. a LOCAL CNN — preserves "consumption dropped weeks 80-95" patterns
#      (mixer's global token-mix destroys these in layer 1)
#   3. the kernel as THE deep input, not one of two streams pooled per-week
#
# ZKMix v3 contains the Zheng wide-and-deep CNN VERBATIM and adds a Mixer
# branch as a third parallel pathway.  The mixer is given the same
# kernel-stat injection as ZKMix v2 (which by itself buys +0.021 PR-AUC).
#
# Critical: zero out the mixer branch and we recover Zheng exactly.  So
# *architecturally*, ZKMix v3 cannot lose to Zheng — only tie or beat.


@dataclass(frozen=True)
class ZKMixV3Config:
    weeks: int = 147
    days: int = 7
    # Zheng wide-and-deep block (paper hyper-params verbatim).
    alpha: int = 90          # wide branch Dense
    beta: int = 60           # deep-CNN branch Dense
    gamma: int = 15          # deep-CNN 1×1 conv channels
    # Mixer branch (lightened from ZKMix v2: 2 blocks instead of 4).
    mixer_hidden: int = 96
    mixer_token_mlp: int = 192
    mixer_channel_mlp: int = 192
    mixer_n_blocks: int = 2
    mixer_out_dim: int = 64
    # Kernel-stat injection into the mixer branch (matches ZKMix v2).
    kernel: KernelSpec = field(default_factory=KernelSpec)
    stats: tuple[str, ...] = ("mean", "std", "max", "p99")
    inject_kernel_stats: bool = True
    # Fusion head.
    fuse_hidden: int = 96
    dropout: float = 0.10
    # Branch-level ablation switches.  At least ONE must be True; the
    # __init__ validates this so we don't accidentally build a no-op model.
    use_wide: bool = True
    use_deep_cnn: bool = True
    use_deep_mixer: bool = True
    # ============= Phase B.1 — Mask-channel extension =============
    # When ``use_mask_channel=True`` and the input was sentinel-preprocessed
    # (NaN → −1.0 marker), the model internally derives a second input
    # channel: a binary mask = (x == −1.0).float().  The deep-CNN branch's
    # first conv becomes Conv2d(2, 9, 3) with the frozen Eq.7 9-stencil on
    # channel 0 and a small *learnable* 3×3 kernel on channel 1 initialised
    # at zero so the model starts equivalent to the 1-channel variant.  The
    # wide branch and mixer patch_proj are also widened to consume both
    # channels.  Channel 0 of the consumption signal is restored to 0.0 at
    # missing positions so the kernel never sees the out-of-range −1.
    use_mask_channel: bool = False
    sentinel_value: float = -1.0
    # ============= Phase B.2 — Multi-scale dilated kernels =============
    # When ``use_multiscale=True``, a 4th parallel branch is added that
    # applies two learnable dilated 2-D convs to the raw consumption grid
    # at receptive fields of 9×9 (kernel 5×5, dilation 2) and 19×19
    # (kernel 7×7, dilation 3).  Each is followed by ReLU + MaxPool(2)
    # + Flatten + Linear(multiscale_dim).  Their outputs are concatenated
    # into the fusion embedding alongside (wide, deep-CNN, mixer).
    use_multiscale: bool = False
    multiscale_gamma: int = 8     # channels per dilated branch
    multiscale_dim: int = 32      # dim of the multiscale-branch FC output
    # ============= Native Missing-value Embedding (NaME) =============
    # Inspired by Chen & Guestrin 2016 (XGBoost) which learns a "default
    # direction" for missing values at each tree split.  Our adaptation
    # learns a single scalar substitution at the input layer for a CNN.
    # When ``use_name=True``, cells equal to ``sentinel_value`` are
    # replaced with the learnable scalar ``missing_token`` (initialised
    # at ``missing_token_init``) BEFORE all four branches.  This converts
    # the missing-value handling from a fixed preprocessing constant into
    # a trainable framework parameter that is co-optimised with the
    # classification objective.
    # Mutually exclusive with ``use_mask_channel`` — both are alternative
    # missing-value handlers and combining them is undefined behaviour.
    use_name: bool = False
    missing_token_init: float = -1.0
    missing_token_learnable: bool = True
    # When ``nan_native=True``, NaME locates missing cells via
    # ``torch.isnan(x)`` instead of the equality check against
    # ``sentinel_value``.  Used with the NaN-native preprocessor that
    # writes raw NaN cells to the CSV (no -1 substitution).  The model
    # then takes full responsibility for the substitution decision —
    # no preprocessing-time marker is involved.  Numerically equivalent
    # to the marker path when both are wired correctly; the difference
    # is purely architectural framing (clean NaME-only handling vs.
    # encode-then-substitute).
    nan_native: bool = False
    # ============= Partial Convolution at the input — NaME-v3 =============
    # When ``use_partial_conv=True``, the frozen Eq.7 9-stencil becomes a
    # PartialConv2d that explicitly excludes sentinel-marked cells from
    # the convolution and renormalises its output by the window-validity
    # ratio (Liu et al. 2018).  The 9-stencil weights remain frozen; only
    # the masking math is added.  Mutually exclusive with use_name and
    # use_mask_channel.
    use_partial_conv: bool = False


class ZKMixV3(nn.Module):
    """Wide-and-Deep MIXER.

    Forward graph
    -------------
    x : (N, W, D)

    ── wide branch ──────────────  Flatten(x) → Dense(α, ReLU+Dropout)         ──┐
    ── deep-CNN branch (Zheng) ──  9-stencil(frozen) → 1×1 Conv(γ, ReLU) →     ──┤
                                   MaxPool(2×2) → Dense(β, ReLU+Dropout)         │ concat
    ── deep-mixer branch ───────   patch_proj + kernel-stat injection →        ──┘
                                   N×mixer blocks → multi-pool →
                                   Dense(mixer_out_dim, ReLU+Dropout)
    ── fuse ────────────────────   Dense(fuse_hidden, ReLU+Dropout) → Dense(1)

    Equivalence guarantee
    ---------------------
    If the deep-mixer branch's outputs are zeroed, ZKMix v3 reduces *exactly*
    to ZhengBaseline (same wide branch, same 9-stencil deep CNN, same
    Dense fuse).  This is verified by ``tests/test_models.py``.
    """

    def __init__(self, cfg: ZKMixV3Config | None = None) -> None:
        super().__init__()
        cfg = cfg or ZKMixV3Config()
        self.cfg = cfg
        if not (cfg.use_wide or cfg.use_deep_cnn or cfg.use_deep_mixer):
            raise ValueError(
                "At least one of (use_wide, use_deep_cnn, use_deep_mixer) "
                "must be True — a model with zero branches has no input path."
            )
        # Exactly-one (or zero) of the three missing-value handlers may be on.
        active_handlers = (
            int(cfg.use_name) + int(cfg.use_mask_channel) + int(cfg.use_partial_conv)
        )
        if active_handlers > 1:
            raise ValueError(
                "At most one of (use_name, use_mask_channel, use_partial_conv) "
                "may be True — they are mutually exclusive alternative "
                "strategies for handling sentinel-marked missing cells."
            )

        # ============= Native Missing-value Embedding (NaME) =============
        # Learnable scalar parameter; replaces cells equal to sentinel_value
        # at the front of every branch.  At init = sentinel_value the model
        # is bit-identical to a non-NaME v3 on sentinel-encoded input.
        if cfg.use_name:
            self.missing_token = nn.Parameter(
                torch.tensor(cfg.missing_token_init, dtype=torch.float32),
                requires_grad=cfg.missing_token_learnable,
            )

        # ============= WIDE branch — canonical 2018 baseline verbatim =====
        # When use_mask_channel is on, the wide branch sees BOTH channels
        # flattened: consumption + mask.  This doubles the input dim.
        wide_factor = 2 if cfg.use_mask_channel else 1
        if cfg.use_wide:
            wide_in = cfg.weeks * cfg.days * wide_factor
            self.wide_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(wide_in, cfg.alpha),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # ============= DEEP-CNN branch =============
        # Default: Conv2d(1, 9, 3) with FROZEN Eq.7 9-stencil weights.
        # With use_mask_channel: Conv2d(2, 9, 3) (Phase B.1, dropped — kept
        #                       for the ablation table).
        # With use_partial_conv: PartialConv2d(1, 9, 3) with the SAME frozen
        #                       Eq.7 weights; only the masking math differs.
        if cfg.use_deep_cnn:
            in_ch = 2 if cfg.use_mask_channel else 1
            if cfg.use_partial_conv:
                # PartialConv2d takes the same constructor args as nn.Conv2d
                # and inherits weight/bias.  We initialise with the Eq.7
                # stencil and freeze, identical to the standard variant.
                self.zheng_kernel = PartialConv2d(
                    in_channels=1, out_channels=9, kernel_size=3, padding=1, bias=False,
                )
            else:
                self.zheng_kernel = nn.Conv2d(
                    in_channels=in_ch, out_channels=9, kernel_size=3, padding=1, bias=False,
                )
            with torch.no_grad():
                stencil = _eq7_nine_stencil_weights()  # (9, 1, 3, 3)
                if cfg.use_partial_conv or in_ch == 1:
                    self.zheng_kernel.weight.copy_(stencil)
                else:
                    # Channel 0: copy frozen stencil.  Channel 1: zero-init learnable.
                    w = torch.zeros(9, 2, 3, 3)
                    w[:, 0:1, :, :] = stencil
                    self.zheng_kernel.weight.copy_(w)
            if cfg.use_mask_channel:
                # Need a per-input-channel grad mask: freeze channel 0, train channel 1.
                # We do this via a forward hook that zeros the channel-0 grad
                # after backward.  Cleanest is to register the parameter as
                # requires_grad=True and zero its channel-0 slice every step.
                self.zheng_kernel.weight.requires_grad_(True)
                # Register a backward hook that zeroes out the channel-0 gradient.
                def _freeze_ch0_grad(grad: torch.Tensor) -> torch.Tensor:
                    g = grad.clone()
                    g[:, 0:1, :, :] = 0.0
                    return g
                self.zheng_kernel.weight.register_hook(_freeze_ch0_grad)
            else:
                # Both default and partial-conv variants keep the 9-stencil frozen.
                self.zheng_kernel.weight.requires_grad_(False)
            self.cnn_conv = nn.Conv2d(9, cfg.gamma, kernel_size=1)
            self.cnn_act = nn.ReLU(inplace=True)
            self.cnn_pool = nn.MaxPool2d(kernel_size=2)
            cnn_flat = (cfg.weeks // 2) * (cfg.days // 2) * cfg.gamma
            self.cnn_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cnn_flat, cfg.beta),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # ============= DEEP-MIXER branch — kernel-prior + MLP-Mixer =============
        if cfg.use_deep_mixer:
            self.kernel = cfg.kernel.build()
            self._kernel_out_channels = cfg.kernel.out_channels
            # With use_mask_channel, each week token concatenates the
            # consumption D values with the mask D values → 2D dim.
            patch_in_dim = cfg.days * (2 if cfg.use_mask_channel else 1)
            self.patch_proj = nn.Linear(patch_in_dim, cfg.mixer_hidden)
            stat_dim = self._kernel_out_channels * len(cfg.stats)
            self.stat_proj = nn.Linear(stat_dim, cfg.mixer_hidden)
            self.stat_norm = nn.LayerNorm(cfg.mixer_hidden)
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
            self.mixer_fc = nn.Sequential(
                nn.Linear(3 * cfg.mixer_hidden, cfg.mixer_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # ============= MULTI-SCALE DILATED branch — Phase B.2 =============
        # Two parallel learnable dilated 2-D convs on the raw consumption grid.
        # Receptive fields:
        #   ms_5x5_d2:  kernel 5, dilation 2 → RF 9×9   (~1 week × all days)
        #   ms_7x7_d3:  kernel 7, dilation 3 → RF 19×19 (~2.7 weeks × all days)
        # Each followed by MaxPool(2) and Flatten → Linear(multiscale_dim).
        if cfg.use_multiscale:
            self.ms_5x5_d2 = nn.Conv2d(
                in_channels=1, out_channels=cfg.multiscale_gamma,
                kernel_size=5, dilation=2, padding=4,
            )
            self.ms_7x7_d3 = nn.Conv2d(
                in_channels=1, out_channels=cfg.multiscale_gamma,
                kernel_size=7, dilation=3, padding=9,
            )
            self.ms_act = nn.ReLU(inplace=True)
            self.ms_pool = nn.MaxPool2d(kernel_size=2)
            ms_flat_one = (cfg.weeks // 2) * (cfg.days // 2) * cfg.multiscale_gamma
            self.ms_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2 * ms_flat_one, cfg.multiscale_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )

        # ============= FUSION =============
        self.embed_dim = (
            (cfg.alpha if cfg.use_wide else 0)
            + (cfg.beta if cfg.use_deep_cnn else 0)
            + (cfg.mixer_out_dim if cfg.use_deep_mixer else 0)
            + (cfg.multiscale_dim if cfg.use_multiscale else 0)
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.embed_dim, cfg.fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.fuse_hidden, 1)

    # ---- branch outputs (exposed for the Zheng-equivalence test) ---------

    def _derive_channels(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split sentinel-encoded input into (consumption, mask).

        Convention: cells equal to ``cfg.sentinel_value`` (default −1.0)
        are treated as missing.  ``consumption`` has those cells zeroed so
        the frozen 9-stencil never sees the out-of-range sentinel value
        (which would otherwise dominate gradients).  ``mask`` is 1.0 at
        missing positions, 0.0 elsewhere.
        """
        sent = self.cfg.sentinel_value
        mask = (x == sent).to(x.dtype)
        consumption = torch.where(mask > 0.5, torch.zeros_like(x), x)
        return consumption, mask

    def _wide(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_mask_channel:
            consumption, mask = self._derive_channels(x)
            # Concatenate along last axis (within the flatten):
            # shape (N, W, 2*D)  →  flatten →  (N, 2*W*D).
            x_w = torch.cat([consumption, mask], dim=-1)
        else:
            x_w = x
        return self.wide_fc(x_w)

    def _deep_cnn(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_partial_conv:
            # Validity mask: 1 where the cell is a real observation, 0 where
            # it carries the sentinel marker for "missing".  Built from the
            # raw input (NaME substitution has not been applied because
            # use_partial_conv is mutually exclusive with use_name).
            valid_mask = (x != self.cfg.sentinel_value).to(x.dtype).unsqueeze(1)
            # Zero out the sentinel cells in the consumption signal so the
            # convolution never multiplies its frozen weights against the
            # out-of-range −1 marker; the renormalisation inside the partial
            # conv compensates for the missing cells in the window.
            consumption = torch.where(
                x == self.cfg.sentinel_value, torch.zeros_like(x), x
            )
            grid = consumption.unsqueeze(1)                  # (N, 1, W, D)
            resp, _ = self.zheng_kernel(grid, valid_mask)    # PartialConv2d → (resp, new_mask)
        elif self.cfg.use_mask_channel:
            consumption, mask = self._derive_channels(x)
            grid = torch.stack([consumption, mask], dim=1)   # (N, 2, W, D)
            resp = self.zheng_kernel(grid)                   # (N, 9, W, D)
        else:
            grid = x.unsqueeze(1)                            # (N, 1, W, D)
            resp = self.zheng_kernel(grid)                   # (N, 9, W, D)
        h = self.cnn_pool(self.cnn_act(self.cnn_conv(resp))) # (N, γ, W/2, D/2)
        return self.cnn_fc(h)                                # (N, β)

    def _deep_mixer(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_mask_channel:
            consumption, mask = self._derive_channels(x)
            # Concatenate (consumption, mask) per week along the D axis.
            patch_in = torch.cat([consumption, mask], dim=-1)   # (N, W, 2D)
            kernel_in = consumption.unsqueeze(1)                # kernel sees clean signal
        else:
            patch_in = x
            kernel_in = x.unsqueeze(1)
        tokens = self.patch_proj(patch_in)                       # (N, W, hidden)
        if self.cfg.inject_kernel_stats:
            resp = self.kernel(kernel_in)
            stats = weekly_kernel_stats(resp, stats=self.cfg.stats)
            tokens = tokens + self.stat_norm(self.stat_proj(stats))
        for blk in self.mixer_blocks:
            tokens = blk(tokens)
        tokens = self.mixer_norm(tokens)
        pooled = _multi_pool(tokens, dim=1)                      # (N, 3 * hidden)
        return self.mixer_fc(pooled)                             # (N, mixer_out_dim)

    def _apply_name(self, x: torch.Tensor) -> torch.Tensor:
        """Native Missing-value Embedding (NaME) — Version 2.

        Replaces cells whose value equals ``cfg.sentinel_value`` with the
        learnable scalar parameter ``self.missing_token``.  The check
        ``x == sent`` is bit-exact because the sentinel is a constant float
        written by the preprocessing step; the data arrives at the model
        with these exact values.

        Inspired by Chen & Guestrin 2016 (XGBoost), which at each tree split
        learns a *default direction* for missing-value samples by trying both
        directions and picking the one that minimises training loss.  Our
        adaptation: at the input layer of a CNN, learn a *scalar embedding*
        that replaces sentinel-marked observations, co-trained end-to-end
        with the classifier objective.

        At initialisation with ``missing_token_init == sentinel_value``, the
        substitution is a no-op (replaces ``s`` with ``s``), so forward
        output is bit-identical to a model with ``use_name=False`` on the
        same sentinel-encoded input.  This is verified by
        ``test_zkmix_v3_name_init_matches_baseline`` in tests/.
        """
        # ``torch.where(cond, scalar0d, x)`` broadcasts the 0-d parameter
        # to ``x.shape`` while keeping the parameter on the autograd graph.
        if self.cfg.nan_native:
            # NaN-native path — model sees raw NaN cells (no preprocessing
            # marker), substitutes them with the learnable token.  NaN cells
            # MUST be substituted before any conv/linear; otherwise NaN
            # propagates and ruins the forward.
            return torch.where(torch.isnan(x), self.missing_token, x)
        sent = self.cfg.sentinel_value
        return torch.where(x == sent, self.missing_token, x)

    def _multiscale(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel learnable dilated convs on the consumption grid (Phase B.2).

        Returns ``(N, multiscale_dim)``.  When ``use_mask_channel`` is on the
        sentinel is masked to 0 first so the learnable kernels do not pick
        up the −1 spike as a dominant feature.
        """
        if self.cfg.use_mask_channel:
            consumption, _ = self._derive_channels(x)
        else:
            consumption = x
        grid = consumption.unsqueeze(1)                          # (N, 1, W, D)
        a = self.ms_pool(self.ms_act(self.ms_5x5_d2(grid)))      # (N, γ_ms, W/2, D/2)
        b = self.ms_pool(self.ms_act(self.ms_7x7_d3(grid)))      # (N, γ_ms, W/2, D/2)
        fused = torch.cat([a.flatten(1), b.flatten(1)], dim=1)   # (N, 2 * γ_ms * (W//2) * (D//2))
        return self.ms_fc(fused)                                 # (N, multiscale_dim)

    # ---- public API ------------------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ZKMixV3 expects (N, W, D); got {tuple(x.shape)}.")
        n, w, d = x.shape
        if w != self.cfg.weeks or d != self.cfg.days:
            raise ValueError(
                f"ZKMixV3 configured for (W={self.cfg.weeks}, D={self.cfg.days}); "
                f"got (W={w}, D={d})."
            )
        # NaME: substitute sentinel cells with the learnable scalar token
        # before all branches.  This MUST run before _derive_channels would
        # be called inside any branch — but use_name and use_mask_channel
        # are mutually exclusive (validated in __init__), so the mask path
        # is unreachable when NaME is active.
        if self.cfg.use_name:
            x = self._apply_name(x)
        parts: list[torch.Tensor] = []
        if self.cfg.use_wide:
            parts.append(self._wide(x))
        if self.cfg.use_deep_cnn:
            parts.append(self._deep_cnn(x))
        if self.cfg.use_deep_mixer:
            parts.append(self._deep_mixer(x))
        if self.cfg.use_multiscale:
            parts.append(self._multiscale(x))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fuse(self.embed(x))).squeeze(-1)


# ---------------------------------------------------------------------------
# Selective head — SelectiveNet-style for binary classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectiveConfig:
    """SelectiveNet composite-loss hyper-parameters.

    coverage_target  : target E[g(x)] for the selector.
    alpha_sel        : weight on selective BCE term in L_total.
    beta_aux         : weight on auxiliary (unselective) BCE term.
    lambda_cov       : coverage-hinge weight (Geifman & El-Yaniv 2019, Eq. 4).
    """

    coverage_target: float = 0.90
    alpha_sel: float = 0.5
    beta_aux: float = 0.5
    lambda_cov: float = 32.0
    head_hidden: int = 64
    dropout: float = 0.1


class SelectiveHead(nn.Module):
    """Binary SelectiveNet head: shared embedding → (logit, gate)."""

    def __init__(self, in_dim: int, cfg: SelectiveConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SelectiveConfig()
        self.cfg = cfg
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.selector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if emb.dim() != 2:
            raise ValueError(f"SelectiveHead expects (N, d); got {tuple(emb.shape)}.")
        logit = self.classifier(emb).squeeze(-1)
        gate = torch.sigmoid(self.selector(emb)).squeeze(-1)
        return logit, gate


class SelectiveModel(nn.Module):
    """Wraps any backbone exposing ``embed()`` and ``embed_dim``.

    Forward returns (logit, gate); the existing backbone head is discarded
    and a fresh ``SelectiveHead`` is trained jointly with the encoder.
    The combined module is what the selective-loss training mode optimises.
    """

    def __init__(self, backbone: nn.Module, cfg: SelectiveConfig | None = None) -> None:
        super().__init__()
        if not hasattr(backbone, "embed") or not hasattr(backbone, "embed_dim"):
            raise TypeError(
                "Backbone must expose ``embed(x)`` and ``embed_dim`` to be "
                "wrapped by SelectiveModel."
            )
        self.backbone = backbone
        self.head = SelectiveHead(in_dim=backbone.embed_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone.embed(x))


# ---------------------------------------------------------------------------
# Registry — single point of contact for the runner.
# ---------------------------------------------------------------------------


_BACKBONE_NAMES = Literal[
    "zheng_baseline",
    "zkmix_v2_frozen",
    "zkmix_v2_learnable",
    "zkmix_v2_no_kernel_stats",
    "zkmix_v3",
    "zkmix_v3_no_kernel_stats",
    "zkmix_v3_no_wide",
    "zkmix_v3_no_cnn",
    "zkmix_v3_no_mixer",
    "zkmix_v3_mixer_only",
    "zkmix_v3_mask",
    "zkmix_v3_ms",
    "zkmix_v3_name",
    "zkmix_v3_pconv",
    "tsmixer",
    "timemixer",
    "vanilla_cnn",
    "bilstm",
    "tcn",
    "transformer",
]


@dataclass(frozen=True)
class ModelSpec:
    name: _BACKBONE_NAMES
    weeks: int = 147
    days: int = 7

    def build(self) -> nn.Module:
        if self.name == "zheng_baseline":
            return ZhengBaseline(ZhengConfig(weeks=self.weeks, days=self.days))
        if self.name == "zkmix_v2_no_kernel_stats":
            return ZKMix(
                ZKMixConfig(
                    weeks=self.weeks, days=self.days,
                    kernel=KernelSpec(mode="frozen"),
                    inject_kernel_stats=False,
                )
            )
        if self.name == "zkmix_v2_frozen":
            return ZKMix(
                ZKMixConfig(weeks=self.weeks, days=self.days, kernel=KernelSpec(mode="frozen"))
            )
        if self.name == "zkmix_v2_learnable":
            return ZKMix(
                ZKMixConfig(
                    weeks=self.weeks,
                    days=self.days,
                    kernel=KernelSpec(mode="learnable_free"),
                )
            )
        if self.name == "zkmix_v3":
            return ZKMixV3(
                ZKMixV3Config(
                    weeks=self.weeks, days=self.days,
                    kernel=KernelSpec(mode="learnable_free"),
                    inject_kernel_stats=True,
                )
            )
        if self.name == "zkmix_v3_no_kernel_stats":
            return ZKMixV3(
                ZKMixV3Config(
                    weeks=self.weeks, days=self.days,
                    kernel=KernelSpec(mode="frozen"),
                    inject_kernel_stats=False,
                )
            )
        # ----- Ablation variants (2026-05-17) -----
        if self.name == "zkmix_v3_no_wide":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              use_wide=False)
            )
        if self.name == "zkmix_v3_no_cnn":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              use_deep_cnn=False)
            )
        if self.name == "zkmix_v3_no_mixer":
            # Should ≈ Zheng — wide + deep-CNN only, no mixer branch.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              use_deep_mixer=False)
            )
        if self.name == "zkmix_v3_mixer_only":
            # Should ≈ ZKMix v2 learnable — mixer branch only, no wide/cnn.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              use_wide=False, use_deep_cnn=False)
            )
        if self.name == "zkmix_v3_mask":
            # Phase B.1 — two-channel mask architecture.  Expects sentinel-
            # encoded input (NaN → −1.0); the model derives the binary mask
            # internally and feeds (consumption, mask) into a 2-channel
            # deep-CNN with frozen Eq.7 stencil on channel 0 + learnable
            # channel-1 weights initialised at 0.  Both wide and mixer
            # branches are widened to consume both channels.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_mask_channel=True)
            )
        if self.name == "zkmix_v3_ms":
            # Phase B.2 — multi-scale dilated kernels.  Adds a 4th parallel
            # branch with learnable 5×5/d=2 (RF 9×9) and 7×7/d=3 (RF 19×19)
            # convs operating on the raw consumption grid.  All other v3
            # components (wide, deep-CNN, mixer) unchanged.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_multiscale=True)
            )
        if self.name == "zkmix_v3_name":
            # NaME (Native Missing-value Embedding) — Version 2.  Replaces
            # the fixed sentinel substitution with a learnable scalar
            # parameter, co-trained with the rest of the network.  The
            # framework therefore handles missingness natively (inspired
            # by XGBoost's default-direction learning), turning the
            # preprocessing constant into a trainable framework component.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True)
            )
        # ----- NaME-v2 NaN-native variants (Phase C, 2026-05-18) -----
        # NaN-native NaME: the model reads raw NaN cells from the CSV and
        # substitutes them with the learnable token in the first forward op
        # (no preprocessing-side -1 marker).  This is the paper-canonical
        # framework formulation — handles missingness EXCLUSIVELY through
        # the trainable model parameter, eliminating any preprocessing-time
        # encoding step.  Functionally equivalent to the marker variant
        # at convergence; the difference is purity of the framing.
        if self.name == "zkmix_v3_name_nn":
            # The NaN-native canonical NaME-v2 — paper headline framework.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True,
                              nan_native=True)
            )
        if self.name == "zkmix_v3_name_nn_fixed":
            # NaME-OFF comparator: identical architecture and isnan-based
            # substitution mechanism as zkmix_v3_name_nn, but the token is
            # FROZEN at -1.0 (not trainable).  Paired Δ vs zkmix_v3_name_nn
            # isolates the LEARNABLE-VALUE contribution of NaME.  Differs
            # from v3_sent_bce only in mechanism (isnan vs sentinel marker),
            # which we expect to be numerically equivalent.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=False,
                              nan_native=True)
            )
        # ----- NaME-v2 component-removal ablations (Phase C, 2026-05-18) -----
        # Each variant keeps NaME-v2's learnable missing_token AND nan_native
        # handling (so missingness handling is identical to the headline);
        # only ONE structural branch is removed.  Paired Δ vs `zkmix_v3_name_nn`
        # yields the marginal contribution of that branch.
        if self.name == "zkmix_v3_name_no_wide":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True,
                              nan_native=True,
                              use_wide=False)
            )
        if self.name == "zkmix_v3_name_no_cnn":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True,
                              nan_native=True,
                              use_deep_cnn=False)
            )
        if self.name == "zkmix_v3_name_no_mixer":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True,
                              nan_native=True,
                              use_deep_mixer=False)
            )
        if self.name == "zkmix_v3_name_no_kernel_stats":
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=False,
                              use_name=True,
                              missing_token_init=-1.0,
                              missing_token_learnable=True,
                              nan_native=True)
            )
        if self.name == "zkmix_v3_pconv":
            # Partial Convolution at the input — NaME Version 3.  The frozen
            # Eq.7 9-stencil becomes a PartialConv2d (Liu et al. 2018) that
            # explicitly excludes sentinel-marked cells from the convolution
            # and renormalises by the window-validity ratio.  Strictly more
            # principled than the constant/scalar substitution variants.
            return ZKMixV3(
                ZKMixV3Config(weeks=self.weeks, days=self.days,
                              kernel=KernelSpec(mode="learnable_free"),
                              inject_kernel_stats=True,
                              use_partial_conv=True)
            )
        if self.name == "tsmixer":
            return TSMixer(TSMixerConfig(weeks=self.weeks, days=self.days))
        if self.name == "timemixer":
            return TimeMixer(TimeMixerConfig(weeks=self.weeks, days=self.days))
        if self.name == "vanilla_cnn":
            return Vanilla1DCNN(Vanilla1DCNNConfig(weeks=self.weeks, days=self.days))
        if self.name == "bilstm":
            return BiLSTM(BiLSTMConfig(weeks=self.weeks, days=self.days))
        if self.name == "tcn":
            return TCN(TCNConfig(weeks=self.weeks, days=self.days))
        if self.name == "transformer":
            return TransformerEncoder(TransformerConfig(weeks=self.weeks, days=self.days))
        raise ValueError(f"Unknown model name {self.name!r}.")  # pragma: no cover


BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "zheng_baseline": lambda: ModelSpec("zheng_baseline").build(),
    "zkmix_v2_frozen": lambda: ModelSpec("zkmix_v2_frozen").build(),
    "zkmix_v2_learnable": lambda: ModelSpec("zkmix_v2_learnable").build(),
    "zkmix_v2_no_kernel_stats": lambda: ModelSpec("zkmix_v2_no_kernel_stats").build(),
    "zkmix_v3": lambda: ModelSpec("zkmix_v3").build(),
    "zkmix_v3_no_kernel_stats": lambda: ModelSpec("zkmix_v3_no_kernel_stats").build(),
    "zkmix_v3_no_wide": lambda: ModelSpec("zkmix_v3_no_wide").build(),
    "zkmix_v3_no_cnn": lambda: ModelSpec("zkmix_v3_no_cnn").build(),
    "zkmix_v3_no_mixer": lambda: ModelSpec("zkmix_v3_no_mixer").build(),
    "zkmix_v3_mixer_only": lambda: ModelSpec("zkmix_v3_mixer_only").build(),
    "zkmix_v3_mask": lambda: ModelSpec("zkmix_v3_mask").build(),
    "zkmix_v3_ms": lambda: ModelSpec("zkmix_v3_ms").build(),
    "zkmix_v3_name": lambda: ModelSpec("zkmix_v3_name").build(),
    "zkmix_v3_name_nn": lambda: ModelSpec("zkmix_v3_name_nn").build(),
    "zkmix_v3_name_nn_fixed": lambda: ModelSpec("zkmix_v3_name_nn_fixed").build(),
    "zkmix_v3_name_no_wide": lambda: ModelSpec("zkmix_v3_name_no_wide").build(),
    "zkmix_v3_name_no_cnn": lambda: ModelSpec("zkmix_v3_name_no_cnn").build(),
    "zkmix_v3_name_no_mixer": lambda: ModelSpec("zkmix_v3_name_no_mixer").build(),
    "zkmix_v3_name_no_kernel_stats": lambda: ModelSpec("zkmix_v3_name_no_kernel_stats").build(),
    "zkmix_v3_pconv": lambda: ModelSpec("zkmix_v3_pconv").build(),
    "tsmixer": lambda: ModelSpec("tsmixer").build(),
    "timemixer": lambda: ModelSpec("timemixer").build(),
    "vanilla_cnn": lambda: ModelSpec("vanilla_cnn").build(),
    "bilstm": lambda: ModelSpec("bilstm").build(),
    "tcn": lambda: ModelSpec("tcn").build(),
    "transformer": lambda: ModelSpec("transformer").build(),
}


def available_models() -> list[str]:
    return sorted(BUILDERS.keys())


def build_model(name: str, weeks: int = 147, days: int = 7) -> nn.Module:
    return ModelSpec(name=name, weeks=weeks, days=days).build()  # type: ignore[arg-type]


def build_selective(name: str, weeks: int = 147, days: int = 7,
                    cfg: SelectiveConfig | None = None) -> SelectiveModel:
    """Build a backbone and wrap it in SelectiveModel.

    Convenience for the runner — separates the choice of backbone from
    the choice of head.  Both heads (BinaryHead and SelectiveHead) share
    the same encoder; the *active* head is the one returned by this
    wrapper, the BinaryHead is discarded.
    """
    return SelectiveModel(build_model(name, weeks=weeks, days=days), cfg=cfg)


__all__ = [
    "ZKMix",
    "ZKMixConfig",
    "ZKMixV3",
    "ZKMixV3Config",
    "TSMixer",
    "TSMixerConfig",
    "TimeMixer",
    "TimeMixerConfig",
    "ZhengBaseline",
    "ZhengConfig",
    "Vanilla1DCNN",
    "Vanilla1DCNNConfig",
    "BiLSTM",
    "BiLSTMConfig",
    "TCN",
    "TCNConfig",
    "TransformerEncoder",
    "TransformerConfig",
    "SelectiveHead",
    "SelectiveModel",
    "SelectiveConfig",
    "ModelSpec",
    "BinaryHead",
    "available_models",
    "build_model",
    "build_selective",
]
