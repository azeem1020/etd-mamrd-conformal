"""Phase L — Multi-task training (A26, A29).

Custom training loop that supports:
    * Two-headed model:   binary_logit + 6-way archetype_logit
    * Multi-task loss:    BCE + λ · CE(archetype)
    * Sample weights:     from Confident-Learning (A27); uniform fallback
    * SAM optimiser:      Sharpness-Aware Minimization (A28); opt-in
    * Class-balanced wt:  per-archetype inverse-frequency weighting in CE head

The standard ``zkmix_lab.training.train()`` is BCE-only and assumes
``model(x) → logit``; this module gives us the BCE+CE pathway needed by A26+.

Determinism mirrors zkmix_lab: seed_everything → deterministic sampler →
manual Lion or AdamW with cosine schedule.

NOTE
----
We DO NOT modify zkmix_lab. We re-use Lion + WarmupCosineSchedule +
BinaryStratifiedSampler + WeeklyGridDataset from zkmix_lab.training, and add
only what's specific to the multi-task formulation.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

# Re-use zkmix_lab building blocks (DO NOT MODIFY zkmix_lab files).
from zkmix_lab.training import (                                                # type: ignore
    Lion, WarmupCosineSchedule, BinaryStratifiedSampler,
    seed_everything, TrainConfig, TrainResult,
)


# ---------------------------------------------------------------------------
# Multi-task model wrapper
# ---------------------------------------------------------------------------

class MultiTaskZKMixV4(nn.Module):
    """Wrap a ZKMixV4 backbone and add a 6-way archetype head.

    Architecture
    ------------
    * Re-uses the entire backbone (CC-NaME, MS-Laplacian, etc.) AND its
      ``fuse`` projection (Linear → ReLU → Dropout). The binary head is the
      backbone's existing ``head``.
    * Adds a new linear ``archetype_head: fuse_hidden → n_archetypes``.

    Forward
    -------
    ``forward(x)`` returns ``(binary_logit, archetype_logit)``.

    Equivalence
    -----------
    With ``return_archetype=False`` at inference, the returned binary_logit is
    bit-identical to what the unwrapped backbone would return.
    """

    def __init__(
        self,
        backbone: "ZKMixV4",                                                    # type: ignore[name-defined]
        n_archetypes: int = 6,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        fuse_hidden = backbone.cfg.fuse_hidden
        self.archetype_head = nn.Linear(fuse_hidden, n_archetypes)
        # Zero-init last layer to keep init equivalence with binary-only model.
        nn.init.zeros_(self.archetype_head.weight)
        nn.init.zeros_(self.archetype_head.bias)
        self.n_archetypes = n_archetypes

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb        = self.backbone.embed(x)
        fused      = self.backbone.fuse(emb)
        bin_logit  = self.backbone.head(fused).squeeze(-1)
        arch_logit = self.archetype_head(fused)
        return bin_logit, arch_logit

    def predict_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time binary logit (drop the auxiliary head)."""
        emb       = self.backbone.embed(x)
        fused     = self.backbone.fuse(emb)
        return self.backbone.head(fused).squeeze(-1)

    @property
    def cfg(self):
        return self.backbone.cfg

    @property
    def embed_dim(self):
        return self.backbone.embed_dim


# ---------------------------------------------------------------------------
# Multi-task dataset — yields (x, y_binary, y_archetype, weight)
# ---------------------------------------------------------------------------

class PhaseLDataset(Dataset):
    """Unified Phase L dataset.

    Yields (features, binary_label, archetype_label, eng_feat, sample_weight).
    Any optional field can be None / uniform / zero; the training loop only
    consumes what its model needs.
    """

    def __init__(
        self,
        features: np.ndarray,
        binary_labels: np.ndarray,
        archetype_labels: np.ndarray | None = None,
        eng_features: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        if features.ndim != 3:
            raise ValueError("features must be (N, W, D)")
        N = features.shape[0]
        if binary_labels.shape != (N,):
            raise ValueError("binary_labels must be (N,)")
        if archetype_labels is None:
            archetype_labels = np.zeros(N, dtype=np.int64)
        if archetype_labels.shape != (N,):
            raise ValueError("archetype_labels must be (N,)")
        if eng_features is None:
            eng_features = np.zeros((N, 1), dtype=np.float32)
        if eng_features.ndim != 2 or eng_features.shape[0] != N:
            raise ValueError("eng_features must be (N, n_feat)")
        if weights is None:
            weights = np.ones(N, dtype=np.float32)
        if weights.shape != (N,):
            raise ValueError("weights must be (N,)")
        self.features         = features.astype(np.float32, copy=False)
        self.binary_labels    = binary_labels.astype(np.float32, copy=False)
        self.archetype_labels = archetype_labels.astype(np.int64, copy=False)
        self.eng_features     = eng_features.astype(np.float32, copy=False)
        self.weights          = weights.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.binary_labels[idx]),
            torch.tensor(self.archetype_labels[idx]),
            torch.from_numpy(self.eng_features[idx]),
            torch.tensor(self.weights[idx]),
        )


# Back-compat alias — old name still used elsewhere.
MultiTaskDataset = PhaseLDataset


# ---------------------------------------------------------------------------
# Class-balanced cross-entropy
# ---------------------------------------------------------------------------

def class_balanced_weights(
    archetype_labels: np.ndarray,
    beta: float = 0.9999,
    n_classes: int = 6,
) -> np.ndarray:
    """Cui et al. 2019 class-balanced weights — effective number of samples.

    Per-class weight ∝ (1 − β) / (1 − β ** n_c), where n_c is the count.
    Normalised so weights sum to ``n_classes``.

    Returns (n_classes,) float array suitable for ``F.cross_entropy(weight=)``.
    """
    counts = np.bincount(archetype_labels, minlength=n_classes).astype(np.float64)
    eff_n  = (1.0 - np.power(beta, np.maximum(counts, 1.0))) / (1.0 - beta)
    w      = (1.0 - beta) / np.maximum(eff_n, 1e-6)
    # Normalise so Σw = n_classes (keeps loss on familiar scale).
    w = w * n_classes / w.sum()
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multitask_loss(
    bin_logit: torch.Tensor,
    arch_logit: torch.Tensor,
    y_bin: torch.Tensor,
    y_arch: torch.Tensor,
    weights: torch.Tensor,
    *,
    aux_lambda: float = 0.30,
    cb_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task loss = BCE(theft) + λ · CE(archetype), sample-weighted.

    Returns (total_loss, dict with per-component scalars for logging).
    """
    # Primary: per-sample BCE × weight, then mean
    bce_per_sample = F.binary_cross_entropy_with_logits(
        bin_logit, y_bin, reduction="none"
    )                                                                            # (B,)
    primary = (bce_per_sample * weights).sum() / weights.sum().clamp(min=1.0)

    # Auxiliary: weighted cross-entropy on archetype
    if cb_weights is not None:
        ce_per_sample = F.cross_entropy(arch_logit, y_arch, weight=cb_weights, reduction="none")
    else:
        ce_per_sample = F.cross_entropy(arch_logit, y_arch, reduction="none")
    auxiliary = (ce_per_sample * weights).sum() / weights.sum().clamp(min=1.0)

    total = primary + aux_lambda * auxiliary
    return total, {
        "loss_primary":   float(primary.detach()),
        "loss_auxiliary": float(auxiliary.detach()),
        "loss_total":     float(total.detach()),
    }


# ---------------------------------------------------------------------------
# Multi-task training loop
# ---------------------------------------------------------------------------

@dataclass
class MTTrainConfig(TrainConfig):
    """Extends TrainConfig with multi-task knobs."""
    aux_lambda:        float = 0.30      # weight on archetype CE
    class_balanced:    bool  = True      # use Cui 2019 class-balanced weights
    cb_beta:           float = 0.9999
    use_sam:           bool  = False     # Sharpness-Aware Minimization (A28)
    sam_rho:           float = 0.05      # SAM neighbourhood radius
    early_stop_metric: str   = "pr_auc"  # which validation metric drives early stopping
    n_archetypes:      int   = 6


def _evaluate_mt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_eng_feat: bool,
    is_multitask: bool,
) -> dict[str, float]:
    """Compute PR-AUC + AUROC on the early-stop set using the binary head."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for batch in loader:
            x, y_bin, _arch, eng, _w = batch
            x   = x.to(device, non_blocking=True)
            eng = eng.to(device, non_blocking=True) if use_eng_feat else None
            if is_multitask and eng is not None:
                emb   = model.backbone.embed(x, eng_feat=eng)
                fused = model.backbone.fuse(emb)
                logit = model.backbone.head(fused).squeeze(-1)
            elif is_multitask:
                logit = model.predict_logit(x)
            elif eng is not None:
                logit = model(x, eng_feat=eng)
            else:
                logit = model(x)
            s = torch.sigmoid(logit).detach().cpu().numpy()
            all_y.append(y_bin.numpy())
            all_s.append(s)
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    if y.min() == y.max():
        return dict(pr_auc=float("nan"), auroc=float("nan"))
    return dict(
        pr_auc=float(average_precision_score(y, s)),
        auroc =float(roc_auc_score(y, s)),
    )


def train_phase_l(
    model: nn.Module,
    train_ds: PhaseLDataset,
    early_stop_ds: PhaseLDataset,
    cfg: MTTrainConfig,
    *,
    is_multitask: bool = False,
    use_eng_feat: bool = False,
) -> tuple[TrainResult, nn.Module]:
    """Unified Phase L training loop.

    Handles A24–A29 paths via flags:
      * ``is_multitask=True``  → model must return (binary, archetype) tuple;
                                   loss = BCE + aux_lambda · CE
      * ``is_multitask=False`` → model returns scalar logit; loss = BCE
      * ``use_eng_feat=True``  → pass batch's eng_feat tensor to model.forward
      * ``cfg.use_sam=True``    → SAM optimisation (two-step per batch)
      * sample weights from PhaseLDataset always applied

    Determinism + sampler + optimiser identical to zkmix_lab.training.train().
    """
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model  = model.to(device)

    # Stratified sampler over binary labels (same as Phase C convention).
    sampler = BinaryStratifiedSampler(
        labels=train_ds.binary_labels.astype(np.int64),
        batch_size=cfg.batch_size,
        pos_fraction=cfg.pos_fraction,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    es_loader = DataLoader(
        early_stop_ds,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    optim = Lion(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    # NB: use len(sampler), NOT len(train_loader). BinaryStratifiedSampler.__len__
    # returns the number of BATCHES (B), while __iter__ yields B*batch_size indices.
    # A DataLoader therefore computes len(train_loader)=len(sampler)//batch_size=0→1,
    # which would set total_steps=epochs and collapse the cosine schedule to the
    # min-LR floor after the first epoch (PR-AUC 0.55→0.30). Standard
    # zkmix_lab.training.train() uses len(sampler); we must match it exactly.
    steps_per_epoch = max(1, len(sampler))
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = max(1, int(total_steps * cfg.warmup_frac))
    schedule = WarmupCosineSchedule(
        optim, total_steps=total_steps, warmup_steps=warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    # Class-balanced weights (precomputed once on training archetype labels).
    cb_tensor: torch.Tensor | None = None
    if cfg.class_balanced:
        cb_np = class_balanced_weights(
            train_ds.archetype_labels, beta=cfg.cb_beta, n_classes=cfg.n_archetypes,
        )
        cb_tensor = torch.from_numpy(cb_np).to(device)

    # ── SAM helpers ──────────────────────────────────────────────────────────
    sam_state: dict = {}
    if cfg.use_sam:
        def _sam_first_step():
            with torch.no_grad():
                grad_norm = torch.sqrt(sum(
                    ((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None),
                    start=torch.tensor(0.0, device=device),
                )) + 1e-12
                scale = cfg.sam_rho / grad_norm
                sam_state["e_w"] = []
                for p in model.parameters():
                    if p.grad is None:
                        sam_state["e_w"].append(None); continue
                    e = p.grad * scale
                    p.add_(e)
                    sam_state["e_w"].append(e.clone())

        def _sam_second_step():
            with torch.no_grad():
                for p, e in zip(model.parameters(), sam_state["e_w"]):
                    if e is not None:
                        p.sub_(e)

    history_val:   list[dict[str, float]] = []
    history_train: list[float]            = []
    best_metric = -float("inf")
    best_epoch  = 0
    best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience    = cfg.patience
    since_best  = 0

    def _compute_loss(x, eng, y_bin, y_arch, w):
        if is_multitask:
            if use_eng_feat:
                # MultiTaskZKMixV4 forward through backbone manually to inject eng_feat
                emb        = model.backbone.embed(x, eng_feat=eng)
                fused      = model.backbone.fuse(emb)
                bin_logit  = model.backbone.head(fused).squeeze(-1)
                arch_logit = model.archetype_head(fused)
            else:
                bin_logit, arch_logit = model(x)
            return multitask_loss(
                bin_logit, arch_logit, y_bin, y_arch, w,
                aux_lambda=cfg.aux_lambda, cb_weights=cb_tensor,
            )
        else:
            if use_eng_feat:
                logit = model(x, eng_feat=eng)
            else:
                logit = model(x)
            # Plain BCE per-sample, weighted
            bce = F.binary_cross_entropy_with_logits(logit, y_bin, reduction="none")
            primary = (bce * w).sum() / w.sum().clamp(min=1.0)
            return primary, {"loss_total": float(primary.detach()),
                              "loss_primary": float(primary.detach()),
                              "loss_auxiliary": 0.0}

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_losses = []
        for batch in train_loader:
            x, y_bin, y_arch, eng, w = batch
            x      = x.to(device, non_blocking=True)
            y_bin  = y_bin.to(device, non_blocking=True)
            y_arch = y_arch.to(device, non_blocking=True)
            eng    = eng.to(device, non_blocking=True)
            w      = w.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            loss, parts = _compute_loss(x, eng, y_bin, y_arch, w)

            if not torch.isfinite(loss):
                continue

            if cfg.use_sam:
                loss.backward()
                _sam_first_step()
                optim.zero_grad(set_to_none=True)
                loss2, _ = _compute_loss(x, eng, y_bin, y_arch, w)
                loss2.backward()
                _sam_second_step()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()

            schedule.step()
            epoch_losses.append(parts["loss_total"])

        history_train.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

        val = _evaluate_mt(model, es_loader, device, use_eng_feat=use_eng_feat, is_multitask=is_multitask)
        history_val.append(val)
        cur = val.get(cfg.early_stop_metric, float("nan"))

        if np.isfinite(cur) and cur > best_metric:
            best_metric = cur
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best  = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    # Restore best
    model.load_state_dict(best_state)
    result = TrainResult(
        best_epoch=best_epoch,
        best_val_pr_auc=float(best_metric),
        val_history=history_val,
        train_history=history_train,
    )
    return result, model


__all__ = [
    "MultiTaskZKMixV4",
    "MultiTaskDataset",
    "PhaseLDataset",
    "MTTrainConfig",
    "multitask_loss",
    "class_balanced_weights",
    "train_phase_l",
]
