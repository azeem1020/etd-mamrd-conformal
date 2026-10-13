"""Training loop + metrics for MSDM reproduction (paper §IV-F, §V-B/§V-D).

Loss = class-weighted BCE (normal:abnormal = 1:5 → pos_weight=5, paper §V-D).
Optimizer Adam, lr 1e-3, 100 epochs. Metrics: F1, ROC-AUC, MAP (=average precision).
Results averaged over 3 seeds in run_experiments.py.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from .model import MSDM


def evaluate(model, Xte, yte, device, batch=512):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch):
            xb = torch.from_numpy(Xte[i:i+batch]).float().to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    s = np.concatenate(scores)
    # MAP variants — the paper's "MAP"=0.918 is inconsistent with its F1=0.606 under
    # the standard binary AP, so we report 3 definitions to identify which matches:
    #   map      = average_precision_score (standard binary AP = area under PR) [honest primary]
    #   map_macro= mean(AP_positive, AP_negative) — easy negative class inflates the mean
    #   map_at_100/200 = precision@K (top-K by score) — operational top-K variant
    ap_pos = average_precision_score(yte, s)
    ap_neg = average_precision_score(1 - yte, -s)
    order = np.argsort(-s)
    def _pk(k): k = min(k, len(yte)); return float(yte[order[:k]].sum() / k)
    out = {
        "auc": float(roc_auc_score(yte, s)),
        "map": float(ap_pos),                               # standard binary AP (primary)
        "map_macro": float(0.5 * (ap_pos + ap_neg)),
        "map_at_100": _pk(100), "map_at_200": _pk(200),
        "f1_0.5": float(f1_score(yte, (s >= 0.5).astype(int), zero_division=0)),
    }
    # best-F1 over thresholds (paper-comparable F1)
    ts = np.unique(s)
    if len(ts) > 200:
        ts = np.quantile(s, np.linspace(0, 1, 200))
    bf = max(f1_score(yte, (s >= t).astype(int), zero_division=0) for t in ts)
    out["f1"] = float(bf)
    return out, s


def train_msdm(Xtr, ytr, Xte, yte, *, scales, period, in_channels,
               filters=48, n_blocks=2, feat_dim=128, pos_weight=5.0,
               lr=1e-3, epochs=100, batch=256, device="cuda", seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    model = MSDM(scales=scales, period=period, in_channels=in_channels,
                 filters=filters, n_blocks=n_blocks, feat_dim=feat_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float())
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward(); opt.step()
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            m, _ = evaluate(model, Xte, yte, dev)
            print(f"  ep{ep:3d} loss={float(loss):.4f} AUC={m['auc']:.4f} MAP={m['map']:.4f} F1={m['f1']:.4f}")
    metrics, scores = evaluate(model, Xte, yte, dev)
    return model, metrics, scores


__all__ = ["train_msdm", "evaluate"]
