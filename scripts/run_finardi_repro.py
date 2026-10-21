"""Reproduction run #1 (GPU): HybridAttn (Finardi et al.) under ITS OWN
preprocessing vs the shared honest preprocessing.

Controlled to a single variable. Same architecture (phased.baselines.HybridAttn),
same fold split, same training recipe; ONLY the input normalisation changes:

  mode=ours    : Zheng outlier-clip + per-user min-max (the shared protocol).
                 Per-user scaling removes cross-consumer consumption level.
  mode=finardi : per-feature QuantileTransformer (10 quantiles, uniform output),
                 fit on TRAIN consumers only (leakage-safe). This is *global*
                 across consumers, so it PRESERVES consumption level -- the very
                 signal Sec. magnitude identifies as the discriminator.

If 'finardi' lifts ROC from ~0.72 toward the published ~0.93 while the
full-prevalence AP stays low, two things are shown at once: (a) the HybridAttn
re-implementation is faithful (it reaches the published regime under the
published preprocessing), and (b) the headline is a ranking/metric effect, not
an implementation gap. Single seed/fold -- a sanity check, not the n=9 headline.

Run under mamba_gpu. Heat-gated.
"""
from __future__ import annotations
import sys, json, time, argparse, subprocess
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from phased.runner_d import _stratified_three_way_split, _inference_batch
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

SGCC = SGCC_PATH


def gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"]).decode().splitlines()
        return int(out[0])
    except Exception:
        return -1


def cool(limit=70, maxwait=600):
    w = 0
    while gpu_temp() > limit and w < maxwait:
        time.sleep(20); w += 20


def raw_grid():
    df = pd.read_csv(SGCC)
    y = df["FLAG"].values.astype(np.int64)
    arr = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    nk = (arr.shape[1] // 7) * 7
    return arr[:, :nk], y, nk


def prep_ours(arr):
    """Zheng outlier-clip + per-user min-max (shared protocol)."""
    a = arr.copy()
    ceil = np.nanmean(a, 1, keepdims=True) + 2.0 * np.nanstd(a, 1, keepdims=True)
    a = np.where(a > ceil, ceil, a)
    lo = np.nanmin(a, 1, keepdims=True); hi = np.nanmax(a, 1, keepdims=True)
    den = hi - lo; den[den == 0] = 1.0
    return ((a - lo) / den).astype(np.float32)


def prep_finardi(arr, tr_idx):
    """Per-feature QuantileTransformer, uniform, 10 quantiles, fit on TRAIN only.
    sklearn QuantileTransformer ignores NaN in fit and passes it through."""
    qt = QuantileTransformer(n_quantiles=10, output_distribution="uniform",
                             subsample=10 ** 9, copy=True)
    qt.fit(arr[tr_idx])
    return qt.transform(arr).astype(np.float32)


def best_f1(y, s):
    P = int(y.sum())
    if P == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order]); k = np.arange(1, len(y) + 1)
    return float((2.0 * tp / (k + P)).max())


def balanced(y, s, R=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    ap = ro = f1 = 0.0
    for _ in range(R):
        sel = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        ap += average_precision_score(y[sel], s[sel])
        ro += roc_auc_score(y[sel], s[sel])
        f1 += best_f1(y[sel], s[sel])
    return ap / R, ro / R, f1 / R


def run(mode, seed=0, fold=0, kf=0, epochs=30, device="cuda"):
    arr, y, nk = raw_grid()
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(
        val, y, fractions=(0.33, 0.33, 0.34),
        rng=np.random.default_rng(kf * 7 + fold))

    if mode == "ours":
        A = prep_ours(arr)
    elif mode == "finardi":
        A = prep_finardi(arr, tr)
    else:
        raise ValueError(mode)
    X = A.reshape(len(y), nk // 7, 7)
    X_reg, rw = slice_regime(X, "full")

    cool()
    seed_everything(seed)
    model = build_arm("hybridattn", weeks=rw, days=7)
    cfg = TrainConfig(epochs=epochs, patience=8, batch_size=256, pos_fraction=0.5,
                      lr=3e-4, weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01,
                      grad_clip_norm=1.0, device=device, loss_type="bce")
    t0 = time.time()
    _, best = train(model, WeeklyGridDataset(X_reg[tr], y[tr]),
                    WeeklyGridDataset(X_reg[es], y[es]), cfg)
    wall = time.time() - t0
    dev = torch.device(device)
    cal_s = _inference_batch(best, X_reg[cal], dev)
    te_s = _inference_batch(best, X_reg[te], dev)
    yt = y[te]

    # native
    nat_ap = float(average_precision_score(yt, te_s))
    nat_roc = float(roc_auc_score(yt, te_s))
    nat_f1 = best_f1(yt, te_s)
    bal_ap, bal_roc, bal_f1 = balanced(yt, te_s)
    res = dict(mode=mode, seed=seed, fold=fold, n_test=int(len(te)),
               prev=float(yt.mean()), wall_s=round(wall, 1),
               nat_ap=nat_ap, nat_roc=nat_roc, nat_f1=nat_f1,
               bal_ap=bal_ap, bal_roc=bal_roc, bal_f1=bal_f1)
    print(f"[{mode}] n_test={len(te)} prev={yt.mean():.3f} wall={wall:.0f}s | "
          f"NATIVE AP={nat_ap:.3f} ROC={nat_roc:.3f} F1={nat_f1:.3f} | "
          f"BALANCED AP={bal_ap:.3f} ROC={bal_roc:.3f} F1={bal_f1:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["ours", "finardi"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=str(THIS.parent / "results/phase_n/repro_finardi.json"))
    args = ap.parse_args()
    print(f"[device] cuda avail={torch.cuda.is_available()} temp={gpu_temp()}C", flush=True)
    rows = []
    for m in args.modes:
        rows.append(run(m, seed=args.seed, fold=args.fold, epochs=args.epochs))
        torch.cuda.empty_cache(); cool(limit=55)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"[done] wrote {args.out}", flush=True)
    print("@@@ FINARDI REPRO DONE @@@", flush=True)


if __name__ == "__main__":
    main()
