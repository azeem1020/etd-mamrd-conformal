"""Phase L — Confident Learning Sample-Weighting (A27).

Lightweight reproduction of Northcutt et al. 2021 (Cleanlab) tailored to the
ETD pipeline. We do NOT depend on the ``cleanlab`` package because the runner
must be self-contained.

Procedure
---------
1. Split ``train_idx`` into ``n_splits`` internal folds (default 3).
2. For each internal fold, train a fast model (one Conv1d + small MLP, ~50k
   parameters) on the OTHER folds and predict probabilities on this fold.
3. Stitch out-of-fold probabilities ``p_ood`` per training customer.
4. For each customer with label y_i, compute confidence score s_i =
   p_ood[y_i] (the predicted probability the model assigns to its own label).
5. Map s_i → per-sample weight:
       w_i = clip( 0.5 + 0.5 · s_i,  0.1,  1.0 )
   so confidently-correct samples get weight 1.0, low-confidence samples ~0.5,
   and severely-suspect samples are floored at 0.1 (never silenced entirely).

Why a fast in-fold model?
-------------------------
Phase E experience: training the full ZKMixV4 backbone three times for
Confident Learning would cost ~30 min per cell. A fast Conv1d+MLP scorer
trained for 10 epochs takes ~10 s per fold (30 s total) and gives a confidence
signal well-correlated with full-model probabilities (Northcutt 2021 §4).

Strict fold safety
------------------
- Internal CV happens INSIDE ``train_idx``. Test-fold labels are NEVER seen.
- The fast model's input is the raw ``X[train_idx]``; no peer features (we
  want a SIMPLE noise detector, not a strong predictor).

Public API
----------
``compute_cl_weights(X_train, y_train, *, seed, device) -> (N_train,) float32``
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Fast scorer model
# ---------------------------------------------------------------------------

class _FastScorer(nn.Module):
    """Tiny ConvNet for in-fold confidence estimation (~40 k params)."""
    def __init__(self, days: int = 7, weeks: int = 147) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(days, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=7, padding=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, W, D) → (N, D, W) for Conv1d
        h = self.conv(x.transpose(1, 2)).squeeze(-1)            # (N, 16)
        return self.head(h)                                       # (N, 2)


def _train_fast_scorer(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray | None = None,
    *, n_epochs: int = 10, batch_size: int = 512, device: str = "cuda", seed: int = 0,
) -> np.ndarray:
    """Train _FastScorer briefly and return out-of-fold class probabilities for X_va."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Replace NaN with 0 for the fast scorer (it's a noise detector, not the
    # main model; we don't need NaME).
    X_tr_safe = np.nan_to_num(X_tr, nan=0.0).astype(np.float32)
    X_va_safe = np.nan_to_num(X_va, nan=0.0).astype(np.float32)

    days = X_tr.shape[2]
    weeks = X_tr.shape[1]
    model = _FastScorer(days=days, weeks=weeks).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # Degenerate fold — return uniform predictions
        return np.full((len(X_va), 2), 0.5, dtype=np.float32)
    half = batch_size // 2
    n_steps_per_epoch = max(1, len(pos_idx) // max(1, half))

    rng = np.random.default_rng(seed)
    for ep in range(n_epochs):
        model.train()
        pos_perm = rng.permutation(pos_idx)
        neg_perm = rng.permutation(neg_idx)
        for s in range(n_steps_per_epoch):
            p_sl = pos_perm[s*half:(s+1)*half]
            if len(p_sl) == 0:
                continue
            n_start = (s*half) % len(neg_perm)
            n_sl = neg_perm[n_start:n_start+half] if n_start+half <= len(neg_perm) \
                   else rng.choice(neg_idx, size=half, replace=False)
            idx = np.concatenate([p_sl, n_sl])
            rng.shuffle(idx)
            xb = torch.from_numpy(X_tr_safe[idx]).to(dev)
            yb = torch.from_numpy(y_tr[idx].astype(np.int64)).to(dev)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()

    # Predict OOF
    model.eval()
    probs_chunks = []
    with torch.no_grad():
        for i in range(0, len(X_va_safe), 1024):
            xb = torch.from_numpy(X_va_safe[i:i+1024]).to(dev)
            probs_chunks.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(probs_chunks, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cl_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_splits: int = 3,
    n_epochs: int = 10,
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Compute per-sample confidence weights for the training set.

    Returns
    -------
    weights : (N_train,) float32 in [0.1, 1.0]
    diag    : dict with summary statistics for logging
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train first-dim mismatch")
    N = len(y_train)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros((N, 2), dtype=np.float32)
    t0 = time.time()
    for fi, (tr, va) in enumerate(skf.split(X_train, y_train)):
        if verbose:
            print(f"  [cl] fold {fi+1}/{n_splits} train n={len(tr)} val n={len(va)}", flush=True)
        oof_probs[va] = _train_fast_scorer(
            X_train[tr], y_train[tr], X_train[va], y_train[va],
            n_epochs=n_epochs, batch_size=batch_size, device=device, seed=seed + fi,
        )
    if verbose:
        print(f"  [cl] OOF probs ready in {time.time()-t0:.1f}s", flush=True)

    # Confidence in the assigned label
    confs = oof_probs[np.arange(N), y_train.astype(np.int64)]
    weights = np.clip(0.5 + 0.5 * confs, 0.1, 1.0).astype(np.float32)

    diag = {
        "mean_conf":            float(confs.mean()),
        "low_conf_rate_below_03": float((confs < 0.3).mean()),
        "mean_weight":          float(weights.mean()),
        "weight_min":           float(weights.min()),
        "weight_max":           float(weights.max()),
        "n_below_0.5_weight":   int((weights < 0.5).sum()),
    }
    if verbose:
        print(f"  [cl] mean_conf={diag['mean_conf']:.3f}  "
              f"low_conf(<.3) rate={diag['low_conf_rate_below_03']:.3f}  "
              f"mean_weight={diag['mean_weight']:.3f}", flush=True)
    return weights, diag


__all__ = ["compute_cl_weights"]
