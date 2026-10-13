"""PhenoMoE — phenotype-routed mixture over a shared Wide-Deep-CNN backbone.

Phase N framework (2026-05-30). The a14 ZKMixV4 (Wide-Deep-CNN + Mamba-2) is the
shared "general leg": all customers train its representation (embed -> fuse). On
top of the shared fused representation we place:

  * a GENERALIST head (warm-started from the backbone's own head), and
  * K per-phenotype RESIDUAL experts (zero-initialised), selected by a LABEL-FREE
    router (Paper-4 switching-gap/magnitude phenotype, phased/phenotype.py).

  logit_i = generalist(fused_i) + expert_{pheno(i)}(fused_i)

Design rationale (honest):
  * Not new information — conditional computation. It lets each behavioral stratum
    own its decision boundary on the shared representation, where a monolith must
    compromise across strata with different optimal boundaries.
  * Zero-init residual => at init PhenoMoE == monolith a14 (G4-style invariance);
    data-poor strata (Light/Flat) keep residual ~0 and fall back to the generalist.
  * Router is consumption-derived and label-free => no leakage; phenotype labels
    never enter the loss. The router index is fit on TRAIN customers (tau/medians)
    and applied causally to every split (see phased/phenotype.py).

Detectability target (N1): lift the Heavy strata (~0.62-0.64 monolith) toward 0.70;
Light/Flat are structurally near-floor (Paper 4) and the generalist suffices there.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from zkmix_lab.training import (
    BinaryStratifiedSampler, Lion, WarmupCosineSchedule, seed_everything, TrainResult,
)
from phased.phenotype import PH_INDEX


class PhenoMoE(nn.Module):
    def __init__(self, backbone: nn.Module, n_pheno: int = 5,
                 expert_hidden: int = 64, dropout: float = 0.1,
                 active_experts: "set[int] | None" = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_pheno = n_pheno
        # Strata that get a specialised residual expert. Paper-4 says Light/Flat are
        # structurally undetectable, so an expert there only adds noise (verified:
        # Δ −0.01..−0.02). Default = Heavy_Dropper(0) + Heavy_Riser(2). All other
        # strata fall back to the shared generalist head.
        if active_experts is None:
            active_experts = {PH_INDEX["Heavy_Dropper"], PH_INDEX["Heavy_Riser"]}
        self.active_experts = set(active_experts)
        fuse_hidden = backbone.fuse[0].out_features

        self.generalist = nn.Linear(fuse_hidden, 1)
        # Warm-start generalist from the backbone's trained-style head init so that,
        # with zero-init experts, PhenoMoE == monolith at initialisation.
        with torch.no_grad():
            self.generalist.weight.copy_(backbone.head.weight)
            self.generalist.bias.copy_(backbone.head.bias)

        self.experts = nn.ModuleDict()
        for k in sorted(self.active_experts):
            e = nn.Sequential(
                nn.Linear(fuse_hidden, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            nn.init.zeros_(e[-1].weight)            # zero-init residual -> 0 at init
            nn.init.zeros_(e[-1].bias)
            self.experts[str(k)] = e

    def _fused(self, x: torch.Tensor, eng_feat: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone.fuse(self.backbone.embed(x, eng_feat=eng_feat))

    def forward(self, x: torch.Tensor, pheno_idx: torch.Tensor,
                eng_feat: torch.Tensor | None = None) -> torch.Tensor:
        fused = self._fused(x, eng_feat)
        out = self.generalist(fused).squeeze(-1)               # (N,) generalist
        for k in self.active_experts:                          # add residual only where signal exists
            m = pheno_idx == k
            if m.any():
                out = out.index_copy(
                    0, m.nonzero(as_tuple=True)[0],
                    out[m] + self.experts[str(k)](fused[m]).squeeze(-1),
                )
        return out


class PhenoDataset(Dataset):
    """(features, label, pheno_idx) — pheno_idx is the LABEL-FREE router index."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, pheno_idx: np.ndarray) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.pheno_idx = pheno_idx.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.features[i]),
                torch.tensor(self.labels[i]),
                torch.tensor(self.pheno_idx[i]))


@torch.no_grad()
def _eval_prauc(model, loader, device):
    from sklearn.metrics import average_precision_score
    model.eval(); ys, ss = [], []
    for x, y, p in loader:
        x = x.to(device); p = p.to(device)
        s = torch.sigmoid(model(x, p)).cpu().numpy()
        ys.append(y.numpy()); ss.append(s)
    y = np.concatenate(ys); s = np.concatenate(ss)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def train_pheno_moe(model, train_ds: PhenoDataset, val_ds: PhenoDataset, *,
                    epochs=30, patience=8, batch_size=256, pos_fraction=0.5,
                    lr=3e-4, weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01,
                    grad_clip_norm=1.0, device="cuda", seed=0):
    """Trainer faithful to the FIXED zkmix convention (total_steps = len(sampler)).
    Same sampler/optimiser/schedule/BCE as standard train(); see trainer-bug memo."""
    seed_everything(seed)
    dev = torch.device(device); model.to(dev)
    sampler = BinaryStratifiedSampler(labels=train_ds.labels.astype(np.int64),
                                      batch_size=batch_size, pos_fraction=pos_fraction, seed=seed)
    tl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    vl = DataLoader(val_ds, batch_size=512, shuffle=False)
    total_steps = max(1, len(sampler)) * epochs          # <-- len(sampler), the fix
    warmup = max(1, int(total_steps * warmup_frac))
    opt = Lion((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    sched = WarmupCosineSchedule(opt, total_steps=total_steps, warmup_steps=warmup, min_lr_ratio=min_lr_ratio)

    best = -1.0; best_ep = -1; best_state = None; since = 0
    for ep in range(epochs):
        sampler.set_epoch(ep); model.train()
        for x, y, p in tl:
            x = x.to(dev); y = y.to(dev); p = p.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x, p), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step(); sched.step()
        v = _eval_prauc(model, vl, dev)
        if np.isfinite(v) and v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_ep, best_val_pr_auc=best), model


__all__ = ["PhenoMoE", "PhenoDataset", "train_pheno_moe"]
