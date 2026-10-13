"""Phase L — Test-Time Augmentation (A28).

At inference, generate K augmented views per test customer, run each through
the model, then average sigmoid scores. This reduces variance and exploits
representational invariances learned during training.

Augmentations (matched to SimCLR/SupCon pre-training augmentations so the
model is on familiar ground):
    1. Identity (no augmentation) — anchor view
    2. Temporal sub-crop (105-week window, reflection-padded)
    3. Multiplicative jitter at σ=0.03
    4. Day-of-week column drop (1 random column, p=0.15)
    5. Temporal flip (reverse time)

Mean of sigmoid scores across views is returned.

Note on PRCE arms: TTA augments ONLY the own-consumption channels — the
peer-feature channels remain unchanged, since they are pre-computed from the
fold's normal peer pool and don't admit "views". This is intentional and
documented.
"""
from __future__ import annotations

import numpy as np
import torch


def _view_identity(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return X.copy()


def _view_subcrop(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N, W, D = X.shape
    crop = 105
    if W <= crop:
        return X.copy()
    out = np.empty_like(X)
    starts = rng.integers(0, W - crop + 1, size=N)
    for i in range(N):
        s = starts[i]
        c = X[i, s:s+crop]
        pad_total = W - crop
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        out[i] = np.pad(c, ((pad_l, pad_r), (0, 0)), mode="reflect")[:W]
    return out


def _view_jitter(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = 0.03
    j = 1.0 + sigma * rng.standard_normal(X.shape).astype(X.dtype)
    observed = ~np.isnan(X)
    return np.where(observed, X * j, X)


def _view_dow_drop(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = X.copy()
    N, W, D = out.shape
    for i in range(N):
        if rng.random() < 0.5:                       # 50 % of customers receive drop
            col = int(rng.integers(0, D))
            obs = ~np.isnan(out[i, :, col])
            out[i, :, col] = np.where(obs, 0.0, out[i, :, col])
    return out


def _view_flip(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return X[:, ::-1, :].copy()


_VIEWS = (
    _view_identity,
    _view_subcrop,
    _view_jitter,
    _view_dow_drop,
    _view_flip,
)


def tta_inference(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
    n_views: int = 5,
    seed: int = 0,
    peer_features: np.ndarray | None = None,
    eng_features: np.ndarray | None = None,
) -> np.ndarray:
    """Run TTA inference and return averaged sigmoid scores.

    Parameters
    ----------
    model        : the trained model. Must accept (N, W, D[+peer_D]) and return
                   either a single logit or a (binary_logit, archetype_logit)
                   tuple (MultiTaskZKMixV4 supported via ``predict_logit``).
    X            : (N, W, D) own consumption.
    peer_features: (N, W, P) optional pre-computed peer features.  These are
                   PASSED THROUGH every view unchanged (we only augment own
                   consumption, never peer features).
    n_views      : number of views to average.

    Returns
    -------
    scores : (N,) float32 — mean sigmoid score across views.
    """
    if n_views < 1 or n_views > len(_VIEWS):
        raise ValueError(f"n_views must be in [1, {len(_VIEWS)}]")

    rng = np.random.default_rng(seed)
    model.eval()

    n = X.shape[0]
    score_sum = np.zeros(n, dtype=np.float64)
    has_predict_logit = hasattr(model, "predict_logit")

    for v in range(n_views):
        Xv = _VIEWS[v](X, rng)
        # Replace NaN with 0 for the model input (the model's NaME path handles
        # NaN normally; here we feed already-imputed inputs from the augmented view).
        Xv = np.where(np.isnan(Xv), 0.0, Xv).astype(np.float32)
        if peer_features is not None:
            Xv = np.concatenate([Xv, peer_features], axis=-1)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.from_numpy(Xv[i:i+batch_size]).to(device)
                efb = torch.from_numpy(eng_features[i:i+batch_size]).to(device) if eng_features is not None else None

                if has_predict_logit and efb is not None:
                    # MultiTaskZKMixV4 with eng_feat: route through backbone manually
                    emb   = model.backbone.embed(xb, eng_feat=efb)
                    fused = model.backbone.fuse(emb)
                    logit = model.backbone.head(fused).squeeze(-1)
                elif has_predict_logit:
                    logit = model.predict_logit(xb)
                elif efb is not None:
                    logit = model(xb, eng_feat=efb)
                else:
                    logit = model(xb)
                    if isinstance(logit, tuple):
                        logit = logit[0]
                score_sum[i:i+batch_size] += torch.sigmoid(logit).detach().cpu().numpy()

    return (score_sum / n_views).astype(np.float32)


__all__ = ["tta_inference"]
