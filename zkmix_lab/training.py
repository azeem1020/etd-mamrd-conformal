"""Training infrastructure — Lion + warmup-cosine + binary-stratified sampler.

The training recipe is held constant across all three backbones so the
bootstrap comparison attributes outcome to architecture, not to
optimiser tuning.

Conventions
-----------
* Loss     : ``BCEWithLogitsLoss`` with optional ``pos_weight``.
* Optimiser: Lion (Chen et al. NeurIPS 2023, "Symbolic Discovery of
             Optimisation Algorithms").  Sign-momentum update with
             decoupled weight decay.
* Schedule : linear warmup over ``warmup_steps`` then cosine decay to
             ``min_lr_ratio * lr`` over the remaining steps.
* Sampling : binary-stratified — every batch is composed of ``b_pos``
             positives and ``b_neg`` negatives, drawn without
             replacement within an epoch and reshuffled across epochs.

Determinism is enforced by ``seed_everything`` at the top of the loop.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set seeds for ``random``, ``numpy``, ``torch`` (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Lion optimiser (Chen et al. NeurIPS 2023)
# ---------------------------------------------------------------------------


class Lion(Optimizer):
    """Lion — EvoLved Sign Momentum.

    Update rule (per-parameter):

        c_t = β1 · m_{t-1} + (1 - β1) · g_t
        θ_t = θ_{t-1} − lr · ( sign(c_t)  +  λ · θ_{t-1} )
        m_t = β2 · m_{t-1} + (1 - β2) · g_t

    Notes
    -----
    The (β1, β2) defaults differ from Adam (0.9, 0.99 vs 0.9, 0.999) per
    Table 6 of the paper.  ``weight_decay`` is decoupled (AdamW-style).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid lr.")
        if not 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid betas.")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # Compute update direction (c_t) without mutating m.
                update = m.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)
                # Now update running momentum m_t.
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


# ---------------------------------------------------------------------------
# Warmup + cosine schedule
# ---------------------------------------------------------------------------


class WarmupCosineSchedule:
    """Lightweight scheduler — call ``.step()`` once per optimiser step.

    Linear warmup from 0 → base_lr for ``warmup_steps`` steps, then
    cosine decay from base_lr → ``min_lr_ratio * base_lr`` over the
    remaining steps.
    """

    def __init__(
        self,
        optimiser: Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        if warmup_steps > total_steps:
            raise ValueError("warmup_steps must be ≤ total_steps.")
        if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        self.optimiser = optimiser
        self.total_steps = max(1, total_steps)
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimiser.param_groups]
        self._step = 0

    def _scale(self) -> float:
        if self._step < self.warmup_steps:
            return self._step / max(1, self.warmup_steps)
        progress = (self._step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> float:
        scale = self._scale()
        for g, base in zip(self.optimiser.param_groups, self.base_lrs):
            g["lr"] = base * scale
        self._step += 1
        return scale


# ---------------------------------------------------------------------------
# Binary-stratified sampler
# ---------------------------------------------------------------------------


class BinaryStratifiedSampler(Sampler[int]):
    """Yield indices such that every length-``batch_size`` window is
    composed of ``n_pos`` positives + ``n_neg`` negatives.

    Within an epoch each positive is drawn at most once; negatives are
    drawn from the larger pool with replacement only if needed.  This
    matches the design used by ``sct_etd.experiments.grid_runner._train_zkmix``
    but is implemented as a standard ``torch.utils.data.Sampler`` so the
    runner stays clean.

    The sampler is *deterministic* per epoch given the ``seed`` and the
    epoch index supplied by ``set_epoch``.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        pos_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        if labels.ndim != 1:
            raise ValueError("labels must be 1-D.")
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1.")
        if not 0.0 < pos_fraction < 1.0:
            raise ValueError("pos_fraction must be in (0, 1).")
        self.pos_idx = np.where(labels > 0)[0]
        self.neg_idx = np.where(labels == 0)[0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("Need at least one positive and one negative example.")
        self.batch_size = batch_size
        self.n_pos = max(1, round(batch_size * pos_fraction))
        self.n_neg = batch_size - self.n_pos
        if self.n_neg < 1:
            raise ValueError("pos_fraction too high — batch has no negatives.")
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        # Number of full batches before either pool exhausts.  Positives
        # bound the epoch length (they are the scarcer class).
        return (len(self.pos_idx) // self.n_pos)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        pos_order = rng.permutation(self.pos_idx)
        neg_order = rng.permutation(self.neg_idx)
        n_batches = len(self)
        for b in range(n_batches):
            pos_slice = pos_order[b * self.n_pos : (b + 1) * self.n_pos]
            # For negatives, draw with replacement once we exhaust the pool
            # — keeps batch composition exact.
            start = (b * self.n_neg) % len(neg_order)
            end = start + self.n_neg
            if end <= len(neg_order):
                neg_slice = neg_order[start:end]
            else:
                neg_slice = rng.choice(self.neg_idx, size=self.n_neg, replace=False)
            batch = np.concatenate([pos_slice, neg_slice])
            rng.shuffle(batch)
            for i in batch:
                yield int(i)


# ---------------------------------------------------------------------------
# Tensor dataset
# ---------------------------------------------------------------------------


class WeeklyGridDataset(Dataset):
    """Lazy view over an (N, W, D) float tensor and an (N,) binary label."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 3:
            raise ValueError("features must be (N, W, D).")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must be (N,) aligned with features.")
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx]),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    eval_batch_size: int = 256
    pos_fraction: float = 0.5
    optimizer: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.01
    grad_clip_norm: float = 1.0
    patience: int = 8
    eval_every: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    num_workers: int = 0  # keep deterministic by default
    pin_memory: bool = True

    # Selective-training mode (used when the model is a ``SelectiveModel``).
    # When ``selective=True``, the loss is the SelectiveNet composite from
    # ``zkmix_lab.selective_loss``; otherwise plain BCEWithLogitsLoss.
    selective: bool = False
    coverage_target: float = 0.90
    alpha_sel: float = 0.5
    beta_aux: float = 0.5
    lambda_cov: float = 32.0

    # Loss type for the plain (non-selective) branch.
    #
    #   * ``"bce"``   — vanilla ``BCEWithLogitsLoss`` (default; the same
    #                   loss every cell in the original 540-cell run used).
    #   * ``"focal"`` — Lin et al. 2017 focal loss.  Down-weights easy
    #                   examples (high pt) so the gradient concentrates on
    #                   the decision boundary.  Designed for the kind of
    #                   class imbalance ETD presents (8.5 % positive prior).
    #                   Reduces to BCE when ``focal_gamma == 0``.
    #
    # Notes on ``focal_alpha``: our sampler is already 50/50 balanced, so
    # ``alpha`` does NOT need to compensate for class imbalance — it is
    # the per-class scalar weight applied AFTER the focusing factor.  The
    # Lin paper's defaults (γ=2, α=0.25) are tuned for foreground/background
    # detection without balanced sampling; we set α=0.5 (no class re-weight
    # on top of the sampler) and let γ do the focusing work.
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5

    # ----- Cheap-ablation knobs (2026-05-17 Phase 1) --------------------------
    # Applied on top of `loss_type`.  Defaults preserve the legacy 540-cell and
    # 30-sniff behaviour so existing manifests stay reproducible.
    #
    # ``label_smoothing``: ε in (0, 1).  Targets are mapped y → y(1-ε) + 0.5·ε
    # before BCE / ASL.  Defends against noisy SGCC inspection labels.  Ignored
    # by ``loss_type='focal'`` (no soft-target generalisation in the legacy fn).
    label_smoothing: float = 0.0
    # ``asl_gamma_pos`` / ``asl_gamma_neg`` / ``asl_margin``: Asymmetric Focal
    # Loss (Ridnik et al. 2021) hyperparameters.  Our defaults γ⁺=2, γ⁻=0, m=0
    # follow the theft-detection rationale: down-weight only confident-correct
    # positives (hard thieves matter); leave negatives at standard BCE.
    asl_gamma_pos: float = 2.0
    asl_gamma_neg: float = 0.0
    asl_margin: float = 0.0
    # ``cutmix_prob``: per-batch probability of applying CutMix on the (W, D)
    # grid (Yun et al. 2019, adapted to time-frequency tensors).  Augmentation
    # patch sampled from a Beta(α, α) distribution over the area fraction.
    cutmix_prob: float = 0.0
    cutmix_alpha: float = 1.0


@dataclass
class TrainResult:
    best_epoch: int
    best_val_pr_auc: float
    val_history: list[dict[str, float]] = field(default_factory=list)
    train_history: list[float] = field(default_factory=list)


def binary_focal_loss(
    logit: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017, "Focal Loss for Dense Object Detection").

    Definition
    ----------
    With ``p = σ(logit)``, ``pt = p`` if ``y == 1`` else ``1 - p``,
    ``α_t = α`` if ``y == 1`` else ``1 - α``:

        FL(logit, y) = − α_t · (1 − pt)^γ · log(pt)

    Properties verified in tests
    ----------------------------
    * γ = 0, α = 0.5 → FL == 0.5 · BCE (the standard reduction).
    * pt → 1 (model very confident, correct) → loss → 0.
    * pt → 0 (model very confident, wrong)  → loss → −α_t · log(pt).
    * Always non-negative; finite for any logit (we use ``logsigmoid``-style
      stable form to avoid log(0) blow-ups).
    """
    if logit.shape != y.shape:
        raise ValueError(
            f"logit and y must have the same shape; got {tuple(logit.shape)} vs {tuple(y.shape)}."
        )
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    # Numerically stable BCE-per-example using log-sigmoid identity.
    # log σ(x)   = −softplus(−x)
    # log(1−σ(x))= −softplus(x)
    log_p   = -torch.nn.functional.softplus(-logit)
    log_1mp = -torch.nn.functional.softplus(logit)
    # pt and log(pt) per example.
    log_pt = torch.where(y > 0.5, log_p, log_1mp)
    pt     = log_pt.exp().clamp(min=eps, max=1.0 - eps)
    alpha_t = torch.where(y > 0.5, torch.full_like(y, alpha), torch.full_like(y, 1.0 - alpha))
    loss = -alpha_t * (1.0 - pt).pow(gamma) * log_pt
    return loss.mean()


def asymmetric_focal_loss(
    logit: torch.Tensor,
    y: torch.Tensor,
    gamma_pos: float = 2.0,
    gamma_neg: float = 0.0,
    margin: float = 0.0,
    label_smoothing: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Asymmetric Focal Loss (Ridnik et al. 2021, ICCV).

    Definition (binary)
    -------------------
    With ``p = σ(logit)`` and shifted negative prob ``p_m = max(p − m, 0)``:

        L⁺(p)  = − (1 − p)^γ⁺  · log p              # positive samples
        L⁻(p)  = − p_m^γ⁻      · log(1 − p_m)       # negative samples
        L      = y · L⁺(p) + (1 − y) · L⁻(p)

    Soft-target generalisation
    --------------------------
    The y-weighted convex combination is well-defined for any y ∈ [0, 1]:
    hard targets {0, 1} recover the standard ASL exactly, while smoothed
    targets give a y-interpolation between the two branches.

    Properties (verified in tests)
    -----------------------------
    * γ⁺ = γ⁻ = γ, margin = 0, label_smoothing = 0 → equals
      ``binary_focal_loss`` with α = 0.5 up to floating-point tolerance.
    * label_smoothing = 0, hard y, γ⁺ = γ⁻ = 0, margin = 0 → 0.5 · BCE
      (the same α-induced halving as standard focal at γ=0).
    """
    if logit.shape != y.shape:
        raise ValueError(
            f"logit and y must have the same shape; got {tuple(logit.shape)} vs {tuple(y.shape)}."
        )
    if gamma_pos < 0.0 or gamma_neg < 0.0:
        raise ValueError("gammas must be non-negative.")
    if not (0.0 <= margin < 1.0):
        raise ValueError("margin must be in [0, 1).")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1).")
    if label_smoothing > 0.0:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # Numerically stable log σ via softplus identity.
    log_p   = -torch.nn.functional.softplus(-logit)
    log_1mp = -torch.nn.functional.softplus(logit)
    p = log_p.exp().clamp(min=eps, max=1.0 - eps)
    if margin > 0.0:
        p_m = (p - margin).clamp(min=0.0)
        # log(1 − p_m); clamp to avoid log(0) when p ≈ 1 + m.
        log_1mpm = torch.log((1.0 - p_m).clamp(min=eps))
    else:
        p_m = p
        log_1mpm = log_1mp
    loss_pos = -(1.0 - p).pow(gamma_pos) * log_p
    loss_neg = -p_m.pow(gamma_neg) * log_1mpm
    loss = y * loss_pos + (1.0 - y) * loss_neg
    # Halve to match focal/BCE scale (α = 0.5 convention).
    return 0.5 * loss.mean()


def _bce_with_label_smoothing(
    logit: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0,
) -> torch.Tensor:
    """BCE-with-logits, with optional symmetric label smoothing (Müller 2019).

    ``y → y(1−ε) + 0.5·ε`` before computing the loss.  The default ε = 0
    recovers vanilla BCE.
    """
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1).")
    if label_smoothing > 0.0:
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return nn.functional.binary_cross_entropy_with_logits(logit, y)


def apply_cutmix(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix on a (N, W, D) batch — paste a rectangular (W, D) patch from a
    shuffled twin into each example.

    Returns ``(x_mixed, y_a, y_b, lam)`` where ``y_a`` are the original
    labels, ``y_b`` are the labels of the shuffled donor batch, and
    ``lam = 1 − cut_area / total_area`` is the proportion of *original*
    pixels remaining after the paste.

    The loss is computed as ``lam · L(logit, y_a) + (1 − lam) · L(logit, y_b)``
    in the training loop, which matches the canonical CutMix recipe
    (Yun et al. 2019) generalised to time × frequency tensors.

    Notes
    -----
    * A single (W, D) bbox is sampled per batch (not per example), matching
      the original CutMix paper and keeping the loss interpretable.
    * Determinism: pass a seeded ``torch.Generator`` to make the batch
      augmentation reproducible.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (N, W, D); got shape {tuple(x.shape)}.")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be (N,) aligned with x.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    n, w, d = x.shape
    # λ ~ Beta(α, α); we want the surviving-area fraction, so clamp away
    # from 0 and 1 to avoid degenerate batches (empty or full paste).
    lam = float(np.random.beta(alpha, alpha))
    lam = float(np.clip(lam, 0.05, 0.95))
    cut_frac = 1.0 - lam
    cut_w = max(1, int(round(w * np.sqrt(cut_frac))))
    cut_d = max(1, int(round(d * np.sqrt(cut_frac))))
    if generator is None:
        cx = int(torch.randint(0, w, (1,)).item())
        cy = int(torch.randint(0, d, (1,)).item())
        perm = torch.randperm(n, device=x.device)
    else:
        cx = int(torch.randint(0, w, (1,), generator=generator).item())
        cy = int(torch.randint(0, d, (1,), generator=generator).item())
        perm = torch.randperm(n, generator=generator).to(x.device)
    x1 = max(0, cx - cut_w // 2); x2 = min(w, x1 + cut_w)
    y1 = max(0, cy - cut_d // 2); y2 = min(d, y1 + cut_d)
    x_mixed = x.clone()
    x_mixed[:, x1:x2, y1:y2] = x[perm, x1:x2, y1:y2]
    # Recompute lam from the actual integer-rounded patch area; otherwise
    # the loss mix is biased when the bbox is clipped at the boundary.
    lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (w * d)
    return x_mixed, y, y[perm], lam


def _plain_loss(logit: torch.Tensor, y: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Dispatch plain-mode loss based on ``cfg.loss_type``.

    Supports ``loss_type ∈ {'bce', 'focal', 'asl'}``.  ``label_smoothing``
    is respected by ``bce`` and ``asl``; ``focal`` ignores it (the legacy
    ``binary_focal_loss`` has no soft-target generalisation and we don't
    want to change its existing 540-cell / 30-sniff behaviour).
    """
    if cfg.loss_type == "bce":
        return _bce_with_label_smoothing(logit, y, label_smoothing=cfg.label_smoothing)
    if cfg.loss_type == "focal":
        return binary_focal_loss(logit, y, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
    if cfg.loss_type == "asl":
        return asymmetric_focal_loss(
            logit, y,
            gamma_pos=cfg.asl_gamma_pos, gamma_neg=cfg.asl_gamma_neg,
            margin=cfg.asl_margin, label_smoothing=cfg.label_smoothing,
        )
    raise ValueError(f"Unknown loss_type {cfg.loss_type!r}; use 'bce', 'focal', or 'asl'.")


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    optimiser: Lion,
    scheduler: WarmupCosineSchedule,
    device: torch.device,
    grad_clip_norm: float,
    *,
    cfg: TrainConfig,
) -> float:
    """One epoch of training; switches loss based on ``cfg.selective`` and ``cfg.loss_type``."""
    model.train()
    total = 0.0
    n = 0
    sel_fn = None
    if cfg.selective:
        from .selective_loss import selective_loss as sel_fn  # type: ignore

    use_cutmix = (cfg.cutmix_prob > 0.0) and not cfg.selective
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # CutMix on (N, W, D) — applied before the forward pass with probability
        # cfg.cutmix_prob.  Selective heads are exempt because the gate target
        # would also need mixing, which is out of scope for this phase.
        cutmix_active = use_cutmix and (random.random() < cfg.cutmix_prob)
        if cutmix_active:
            x, y_a, y_b, lam = apply_cutmix(x, y, alpha=cfg.cutmix_alpha)
        optimiser.zero_grad(set_to_none=True)
        out = model(x)
        if cfg.selective:
            if not (isinstance(out, tuple) and len(out) == 2):
                raise RuntimeError(
                    "selective=True but the model did not return (logit, gate); "
                    "wrap the backbone in zkmix_lab.models.SelectiveModel."
                )
            logit, gate = out
            loss, _ = sel_fn(  # type: ignore[misc]
                logit=logit,
                gate=gate,
                y=y,
                coverage_target=cfg.coverage_target,
                alpha_sel=cfg.alpha_sel,
                beta_aux=cfg.beta_aux,
                lambda_cov=cfg.lambda_cov,
            )
        else:
            if isinstance(out, tuple):
                raise RuntimeError(
                    "Model returned (logit, gate) but cfg.selective is False; "
                    "either set cfg.selective=True or use a non-selective backbone."
                )
            if cutmix_active:
                loss = lam * _plain_loss(out, y_a, cfg) + (1.0 - lam) * _plain_loss(out, y_b, cfg)
            else:
                loss = _plain_loss(out, y, cfg)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimiser.step()
        scheduler.step()
        total += float(loss.detach()) * x.shape[0]
        n += x.shape[0]
    return total / max(1, n)


@torch.no_grad()
def _eval_pr_auc(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(pr_auc, labels, sigmoid_scores)``.

    For selective models we score using the classifier head ONLY (ignoring
    the gate) — the gate is used at inference time for abstention, not as
    part of the ranking metric.  PR-AUC remains the early-stop signal.
    """
    from .metrics import pr_auc  # local import to avoid cycle

    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logit = out[0] if isinstance(out, tuple) else out
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    l = np.concatenate(labels)
    return pr_auc(l, s), l, s


def train(
    model: nn.Module,
    train_ds: WeeklyGridDataset,
    val_ds: WeeklyGridDataset,
    cfg: TrainConfig,
) -> tuple[TrainResult, nn.Module]:
    """Fit the model and return ``(result, best_model)``.

    ``best_model`` is the *parameter snapshot* with the highest val PR-AUC
    seen during training (early stopped after ``cfg.patience`` epochs of
    no improvement).  The returned object shares the same architecture
    object as ``model``; we restore the best state_dict before returning.
    """
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    sampler = BinaryStratifiedSampler(
        labels=train_ds.labels,
        batch_size=cfg.batch_size,
        pos_fraction=cfg.pos_fraction,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=getattr(cfg, "eval_batch_size", max(256, cfg.batch_size)),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    total_steps = max(1, len(sampler)) // 1 * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    if getattr(cfg, "optimizer", "AdamW").lower() == "lion":
        optimiser = Lion(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimiser = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
    scheduler = WarmupCosineSchedule(
        optimiser,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    result = TrainResult(best_epoch=-1, best_val_pr_auc=-1.0)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improve = 0

    for ep in range(cfg.epochs):
        sampler.set_epoch(ep)
        tr_loss = _epoch_loss(
            model, train_loader, optimiser, scheduler, device,
            cfg.grad_clip_norm, cfg=cfg,
        )
        eval_freq = getattr(cfg, "eval_every", 1)
        is_eval_epoch = ((ep + 1) % eval_freq == 0) or (ep == cfg.epochs - 1)
        if is_eval_epoch:
            val_pr, _, _ = _eval_pr_auc(model, val_loader, device)
            result.train_history.append(tr_loss)
            result.val_history.append({"epoch": ep, "pr_auc": val_pr, "loss": tr_loss})

            if val_pr > result.best_val_pr_auc:
                result.best_val_pr_auc = val_pr
                result.best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg.patience:
                    break
        else:
            result.train_history.append(tr_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return result, model


__all__ = [
    "Lion",
    "WarmupCosineSchedule",
    "BinaryStratifiedSampler",
    "WeeklyGridDataset",
    "TrainConfig",
    "TrainResult",
    "train",
    "seed_everything",
    "binary_focal_loss",
    "asymmetric_focal_loss",
    "apply_cutmix",
]
