"""Imputation ablation (Phase L, NaN-wall test) — 2026-05-30.

Pre-registered question: is the ~0.56 PR-AUC wall set by the IMPUTATION
representation (CC-NaME fills missing with a learned scalar token tau~-1.0,
an out-of-[0,1]-manifold spike at 25.6% of positions) or is it STRUCTURAL
(Paper 1 regime confound + Paper 4 ceiling, max|r|=0.102, ~82% undetectable)?

Controlled design: hold a14 fixed; swap ONLY `model.name_module` (the fill used
by the wide + CNN value-branches). The mask channel (raw_mask from isnan, l.833)
and temporal branch (x_raw, l.863) read the ORIGINAL NaN tensor, so they are
identical across variants. Thus the only thing that changes is the value the
backbone sees at missing cells.

Variants:
  token  : baseline = unmodified a14 (learned CC-NaME tau ~ -1.0)  [OOD spike]
  zero   : missing -> 0.0                                          [control]
  mean   : missing -> per-customer observed mean (in [0,1])        [in-manifold]
  ffill  : missing -> last observed value (calendar order)         [continuity]

Pre-reg: paired Delta(variant - token) vs a14, material >= +0.01, BCa CI.
If any variant beats token by >=0.01 -> wall is imputation-bound (pursue).
If all tie/lose -> wall is structural (N1 pivot vindicated).
"""
from __future__ import annotations
import sys, os, csv, json, time, argparse
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent / "zkmix_lab")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import summarise
from phased.models import build_arm
from run_local_grid import load_and_preprocess, _stratified_three_way_split, _inference_batch
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

RESULTS = RESULTS_DIR


class FixedFillNaME(nn.Module):
    """Drop-in replacement for CC-NaME: deterministic, non-learned fill.

    Operates on (N, W, D) with NaN at missing cells; returns no-NaN tensor.
    The daily sequence order is row-major reshape (week*D + dow) = calendar order.
    """
    def __init__(self, mode: str) -> None:
        super().__init__()
        assert mode in {"zero", "mean", "ffill"}
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, w, d = x.shape
        flat = x.reshape(n, w * d)                       # (N, T) calendar order
        valid = ~torch.isnan(flat)
        zeros = torch.zeros_like(flat)
        xf = torch.where(valid, flat, zeros)
        cnt = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        cust_mean = xf.sum(dim=1, keepdim=True) / cnt    # (N,1) per-customer observed mean

        if self.mode == "zero":
            out = torch.where(valid, flat, zeros)
        elif self.mode == "mean":
            out = torch.where(valid, flat, cust_mean.expand_as(flat))
        else:  # ffill
            T = w * d
            idx = torch.arange(T, device=x.device).unsqueeze(0).expand(n, T)
            vidx = torch.where(valid, idx, torch.full_like(idx, -1))
            last = torch.cummax(vidx, dim=1).values      # last valid index, -1 if none yet
            gather_idx = last.clamp(min=0)
            filled = torch.gather(xf, 1, gather_idx)
            # leading positions with no prior valid (last==-1) -> per-customer mean
            no_prior = last < 0
            out = torch.where(valid, flat, filled)
            out = torch.where(no_prior, cust_mean.expand_as(flat), out)
        return out.reshape(n, w, d)


def run_cell(variant: str, X: np.ndarray, y: np.ndarray, seed: int, fold: int,
             kf: int = 0, epochs: int = 30, device: str = "cuda") -> dict:
    W, D = X.shape[1], X.shape[2]
    X_reg, rw = slice_regime(X, "full")
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(kf * 7 + fold))
    seed_everything(seed)
    model = build_arm("a14_no_mixer", weeks=rw, days=D)
    if variant != "token":
        model.name_module = FixedFillNaME(variant)   # swap fill; mask/temporal untouched
    cfg = TrainConfig(epochs=epochs, patience=8, batch_size=256, pos_fraction=0.5,
                      lr=3e-4, weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01,
                      grad_clip_norm=1.0, device=device)
    t0 = time.time()
    res, best = train(model, WeeklyGridDataset(X_reg[tr], y[tr]),
                      WeeklyGridDataset(X_reg[es], y[es]), cfg)
    wall = time.time() - t0
    dev = torch.device(device)
    ts = _inference_batch(best, X_reg[te], dev)
    m = summarise(y[te], ts)
    return dict(variant=variant, seed=seed, fold=fold, kf=kf,
                pr_auc=float(m["pr_auc"]), best_val_pr_auc=float(res.best_val_pr_auc),
                best_epoch=int(res.best_epoch), wall_s=round(wall, 1),
                val_labels_json=json.dumps(y[te].astype(int).tolist()),
                val_scores_json=json.dumps([round(float(s), 6) for s in ts]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["token", "zero", "mean", "ffill"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default=str(RESULTS / "imputation_ablation.csv"))
    args = ap.parse_args()

    print(f"[device] {args.device} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    X, y = load_and_preprocess()
    cells = [(v, s, f) for v in args.variants for s in args.seeds for f in args.folds]
    mpath = Path(args.manifest); mpath.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "seed", "fold", "kf", "pr_auc", "best_val_pr_auc",
            "best_epoch", "wall_s", "val_labels_json", "val_scores_json"]
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for i, (v, s, f) in enumerate(cells, 1):
            r = run_cell(v, X, y, s, f, epochs=args.epochs, device=args.device)
            w.writerow(r); fh.flush()
            print(f"[{i}/{len(cells)}] {v:6s} seed={s} fold={f} "
                  f"PR-AUC={r['pr_auc']:.4f} val={r['best_val_pr_auc']:.4f} "
                  f"ep={r['best_epoch']} wall={r['wall_s']}s", flush=True)
    print(f"[done] -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
