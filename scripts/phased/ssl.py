"""Phase K — Contrastive Self-Supervised Pre-training (A19).

SimCLR-style NT-Xent pre-training on the A9-lean backbone. All 42,372 SGCC
customers participate in the unsupervised objective; labels are not used.

Theft-detection mechanism
-------------------------
The pre-training objective forces the encoder f(·) to produce *stable*
representations under temporal cropping and small jitter: two views of the
same customer should map close, two views of different customers should map
far. Normal customers exhibit consumption identity that is stable across
windows — same seasonality, same appliance footprint. Light Droppers, in
contrast, *change identity* mid-window when they begin under-reporting.
Their two augmented views land in different regions of the representation
manifold, producing a *high intra-customer representation distance* — a
signal directly usable by the downstream classifier.

Design
------
* Encoder = A9-lean backbone, `embed(x) → (N, 212)`.
* Projector = 2-layer MLP `(212 → 128 → 64)`, used only during pre-training.
* Loss = NT-Xent (Chen et al., 2020), temperature τ = 0.5.
* Augmentations (each view independently sampled):
    - Temporal sub-crop: random 105-week window, reflection-padded back to 147.
    - Multiplicative jitter: `X * (1 + N(0, 0.03))`, applied at observed cells.
    - Day-of-week column drop: zero one random D-axis column with p=0.15.
* Optimiser: AdamW, lr=1e-3, weight_decay=0.01, 5 % warmup → cosine to 1e-5.
* Schedule: 30 epochs (smoke) / 30 epochs (full Kaggle), batch 512.

NaN safety
----------
Customer rows may contain NaN. Augmentations are applied AT THE NUMPY LEVEL
before any NaME / CC-NaME processing — the model sees a NaN-pattern-preserved
input, then imputes internally as usual. Jitter is applied multiplicatively
at observed cells (NaN cells remain NaN).

Fold safety
-----------
The SSL pre-training uses all rows of X without labels. Test-row features
participate but test LABELS are never accessed — this matches standard SimCLR
practice (Chen et al., 2020). No label leakage occurs.

Output
------
``ssl_pretrain_backbone(X, build_fn, ...) -> state_dict``

The returned state_dict is a snapshot of the entire model (the projector is
discarded). To use in a downstream arm:

    state = ssl_pretrain_backbone(X, build_a9_lean, n_epochs=30)
    model = build_a21()
    load_state_dict_compatible(model, state)   # loads matching params; new
                                                 # branches stay random.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Augmentations (numpy-level, NaN-preserving)
# ---------------------------------------------------------------------------

def _augment_view(
    X: np.ndarray, rng: np.random.Generator, weeks: int = 147
) -> np.ndarray:
    """Generate one augmented view of a batch of customers.

    X shape: (B, W, D). NaN positions are preserved.

    Pipeline:
      1. Temporal sub-crop (105 weeks at a random start) + reflection pad to W.
      2. Multiplicative jitter at observed cells.
      3. Day-of-week column drop with p=0.15.
    """
    B, W, D = X.shape

    # ── 1. Temporal sub-crop ──────────────────────────────────────────────
    crop_len = 105
    if W <= crop_len:
        out = X
    else:
        starts = rng.integers(0, W - crop_len + 1, size=B)
        out = np.empty_like(X)
        for i in range(B):
            s = starts[i]
            cropped = X[i, s:s + crop_len]                    # (crop_len, D)
            # Reflection-pad to W on both sides equally; truncate if over.
            pad_total = W - crop_len
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            padded = np.pad(cropped, ((pad_l, pad_r), (0, 0)), mode="reflect")
            out[i] = padded[:W]

    # ── 2. Multiplicative jitter at observed cells ────────────────────────
    sigma = 0.03
    jitter = 1.0 + sigma * rng.standard_normal(out.shape).astype(out.dtype)
    observed = ~np.isnan(out)
    out = np.where(observed, out * jitter, out)

    # ── 3. Day-of-week column drop with p=0.15 (one column per row) ──────
    drop_prob = 0.15
    drop_mask = rng.random(B) < drop_prob
    if drop_mask.any():
        drop_cols = rng.integers(0, D, size=B)
        for i in np.where(drop_mask)[0]:
            out[i, :, drop_cols[i]] = np.where(
                np.isnan(out[i, :, drop_cols[i]]),
                out[i, :, drop_cols[i]],
                0.0,
            )
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ContrastiveProjector(nn.Module):
    """2-layer MLP projector for SimCLR. Output is L2-normalised before NT-Xent."""

    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """SimCLR NT-Xent loss.

    z1, z2 : (B, D) — L2-normalised embeddings of paired views.
    Returns scalar loss (averaged over 2B anchors).
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                              # (2B, D)
    sim = z @ z.t() / temperature                                # (2B, 2B)

    # Mask self-similarity (diagonal) with very negative number.
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)

    # Positive indices: row i ∈ [0, B)  pairs with i + B; row i ∈ [B, 2B) pairs with i - B.
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    # Cross-entropy with positive index as target.
    return F.cross_entropy(sim, pos_idx)


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

def ssl_pretrain_backbone(
    X: np.ndarray,
    build_fn: Callable[[], nn.Module],
    *,
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    min_lr_ratio: float = 0.01,
    temperature: float = 0.5,
    proj_hidden: int = 128,
    proj_out: int = 64,
    device: str = "cuda",
    seed: int = 0,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Pre-train a backbone with SimCLR NT-Xent and return the model state_dict.

    Parameters
    ----------
    X        : (N, W, D) float32 — full unlabeled corpus (NaN-allowed).
    build_fn : zero-arg callable returning a model with ``.embed(x) → (N, d)``.

    Returns
    -------
    state_dict : the trained model's state_dict (projector NOT included).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be (N, W, D); got {X.shape}")

    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = build_fn().to(dev)
    model.train()

    # Determine embed dim from a tiny dummy pass.
    with torch.no_grad():
        dummy = torch.zeros(2, X.shape[1], X.shape[2], device=dev)
        emb_dim = model.embed(dummy).shape[1]

    projector = ContrastiveProjector(emb_dim, proj_hidden, proj_out).to(dev)
    projector.train()

    params = list(model.parameters()) + list(projector.parameters())
    optim  = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    N = X.shape[0]
    steps_per_epoch = math.ceil(N / batch_size)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)))

    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        ep_loss = 0.0
        ep_n    = 0
        for b_start in range(0, N, batch_size):
            idx = perm[b_start:b_start + batch_size]
            if len(idx) < 2:
                continue  # skip degenerate batch
            Xb = X[idx]                                          # (B, W, D)

            # Two independent augmented views (numpy-level).
            v1 = _augment_view(Xb, rng, weeks=Xb.shape[1])
            v2 = _augment_view(Xb, rng, weeks=Xb.shape[1])

            t1 = torch.from_numpy(v1).to(dev, non_blocking=True)
            t2 = torch.from_numpy(v2).to(dev, non_blocking=True)

            cur_lr = _lr_at(step)
            for g in optim.param_groups:
                g["lr"] = cur_lr

            optim.zero_grad(set_to_none=True)

            h1 = model.embed(t1)                                  # (B, d)
            h2 = model.embed(t2)
            z1 = projector(h1)
            z2 = projector(h2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            if not torch.isfinite(loss):
                # K-K3: NaN/Inf in SSL loss — bail out cleanly so caller can fall back.
                if verbose:
                    print(f"[ssl] NaN/Inf at epoch {epoch} step {step} — aborting SSL.", flush=True)
                raise RuntimeError("SSL loss became NaN/Inf")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optim.step()

            ep_loss += float(loss.detach())
            ep_n    += 1
            step    += 1

        if verbose:
            wall = time.time() - t0
            print(f"[ssl] epoch {epoch+1:02d}/{n_epochs}  "
                  f"loss={ep_loss/max(1, ep_n):.4f}  "
                  f"lr={cur_lr:.2e}  wall={wall:.0f}s",
                  flush=True)

    model.eval()
    # Return only the backbone state_dict; projector is discarded.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Compatible state-dict loading (skips shape-mismatched params)
# ---------------------------------------------------------------------------

def load_state_dict_compatible(
    model: nn.Module, state_dict: dict[str, torch.Tensor], verbose: bool = False
) -> tuple[list[str], list[str]]:
    """Load ``state_dict`` into ``model`` keeping only parameters with matching shape.

    Returns (loaded_keys, skipped_keys).
    """
    own = model.state_dict()
    compat: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            compat[k] = v
        else:
            skipped.append(k)
    missing_keys, unexpected_keys = model.load_state_dict(compat, strict=False)
    if verbose:
        print(f"[ssl-load] loaded={len(compat)}  skipped(shape mismatch)={len(skipped)}  "
              f"missing(in-model)={len(missing_keys)}", flush=True)
    return list(compat.keys()), skipped


# ---------------------------------------------------------------------------
# Phase L — Supervised Contrastive (SupCon) Loss (Khosla et al. 2020)
# ---------------------------------------------------------------------------

def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SupCon loss: pulls same-class embeddings together, pushes others apart.

    z      : (B, D) — L2-normalised embeddings.
    labels : (B,)   — int class labels (here: 6-way archetype label).

    Definition (Khosla et al. 2020 Eq 2)
    -------------------------------------
        L_i = − (1 / |P(i)|) · Σ_{p ∈ P(i)} log(
                  exp(z_i·z_p / τ) /
                  Σ_{a ∈ A(i)} exp(z_i·z_a / τ)
              )

    where A(i) = {1..B} \\ {i} and P(i) = {a ∈ A(i) : label_a = label_i}.
    Returns scalar mean over anchors that HAVE at least one positive in the batch.
    """
    B = z.size(0)
    sim = z @ z.t() / temperature                                 # (B, B)
    # Mask diagonal (self-similarity) with very-negative
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)

    # Numerical stability — subtract max per row before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = sim.exp()                                            # (B, B)
    denom = exp_sim.sum(dim=1, keepdim=True) + eps                 # (B, 1) over all non-self

    # Positive mask: same label, not self
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()                       # (B, B)
    pos_mask = pos_mask.masked_fill(self_mask, 0.0)

    n_pos_per_anchor = pos_mask.sum(dim=1)                          # (B,)
    valid = n_pos_per_anchor > 0
    if not valid.any():
        # No within-batch positives — degenerate case, return 0 (loss not defined).
        return torch.zeros((), device=z.device, requires_grad=True)

    log_prob_per_pair = sim - denom.log()                            # (B, B); only meaningful where pos_mask=1
    pos_log_prob = (pos_mask * log_prob_per_pair).sum(dim=1)
    per_anchor = -pos_log_prob[valid] / n_pos_per_anchor[valid].clamp(min=1.0)
    return per_anchor.mean()


def supcon_pretrain_backbone(
    X: np.ndarray,
    archetype_labels: np.ndarray,
    build_fn: Callable[[], nn.Module],
    *,
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    min_lr_ratio: float = 0.01,
    temperature: float = 0.07,
    proj_hidden: int = 128,
    proj_out: int = 64,
    device: str = "cuda",
    seed: int = 0,
    grad_clip: float = 1.0,
    verbose: bool = True,
    augment_views: bool = True,
) -> dict[str, torch.Tensor]:
    """Supervised contrastive pre-training of a backbone (A23).

    The same backbone signature/API as ``ssl_pretrain_backbone``, but uses
    ``archetype_labels`` to define positive/negative pairs.  When two views per
    sample are generated (default), the effective batch is 2B with paired
    same-label augmentations as automatic positives + all other same-label
    samples as additional positives.

    Returns the backbone state_dict (projector discarded).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be (N, W, D); got {X.shape}")
    if archetype_labels.shape[0] != X.shape[0]:
        raise ValueError("archetype_labels must align with X first dim")

    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = build_fn().to(dev)
    model.train()

    with torch.no_grad():
        dummy   = torch.zeros(2, X.shape[1], X.shape[2], device=dev)
        emb_dim = model.embed(dummy).shape[1]

    projector = ContrastiveProjector(emb_dim, proj_hidden, proj_out).to(dev)
    projector.train()

    params = list(model.parameters()) + list(projector.parameters())
    optim  = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    N = X.shape[0]
    steps_per_epoch = math.ceil(N / batch_size)
    total_steps  = steps_per_epoch * n_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) *
                     (1.0 + math.cos(math.pi * progress)))

    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        ep_loss = 0.0
        ep_n    = 0
        for b_start in range(0, N, batch_size):
            idx = perm[b_start:b_start + batch_size]
            if len(idx) < 2:
                continue
            Xb = X[idx]
            yb = archetype_labels[idx]

            cur_lr = _lr_at(step)
            for g in optim.param_groups:
                g["lr"] = cur_lr
            optim.zero_grad(set_to_none=True)

            if augment_views:
                v1 = _augment_view(Xb, rng, weeks=Xb.shape[1])
                v2 = _augment_view(Xb, rng, weeks=Xb.shape[1])
                t1 = torch.from_numpy(v1).to(dev, non_blocking=True)
                t2 = torch.from_numpy(v2).to(dev, non_blocking=True)
                h1 = model.embed(t1)
                h2 = model.embed(t2)
                z1 = projector(h1)
                z2 = projector(h2)
                z  = torch.cat([z1, z2], dim=0)
                # Concatenate labels — paired views share archetype label.
                y_t = torch.from_numpy(np.concatenate([yb, yb])).to(dev)
                loss = supcon_loss(z, y_t, temperature=temperature)
            else:
                t1 = torch.from_numpy(Xb).to(dev, non_blocking=True)
                h1 = model.embed(t1)
                z1 = projector(h1)
                y_t = torch.from_numpy(yb).to(dev)
                loss = supcon_loss(z1, y_t, temperature=temperature)

            if not torch.isfinite(loss):
                if verbose:
                    print(f"[supcon] NaN/Inf at epoch {epoch} step {step}; aborting", flush=True)
                raise RuntimeError("SupCon loss became NaN/Inf")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optim.step()

            ep_loss += float(loss.detach())
            ep_n    += 1
            step    += 1

        if verbose:
            wall = time.time() - t0
            print(f"[supcon] epoch {epoch+1:02d}/{n_epochs}  "
                  f"loss={ep_loss/max(1, ep_n):.4f}  "
                  f"lr={cur_lr:.2e}  wall={wall:.0f}s",
                  flush=True)

    model.eval()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


__all__ = [
    "ContrastiveProjector",
    "nt_xent_loss",
    "ssl_pretrain_backbone",
    "supcon_loss",
    "supcon_pretrain_backbone",
    "load_state_dict_compatible",
]
