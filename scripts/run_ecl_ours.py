"""Apply OUR approach (a14) to ECL (ElectricityLoadDiagrams2011-2014) with synthetic FDI theft.

ECL has NO native theft labels (370 industrial load profiles), so theft is injected
(MSDM convention, 6-attack FDI family). This is a SYNTHETIC-theft generalization check:
if a14 scores high here but ~0.55 AP on real SGCC, the SGCC ceiling is real-theft
subtlety, not a model weakness (supports C2).

Pipeline: 15-min → daily mean → 7-week (49-day) non-overlapping windows → FDI 10% →
our preprocessing (3σ clip + per-customer min-max) → a14 (weeks=7) → random k-fold →
shared metrics (PR-AUC/AP, ROC-AUC, F1, P@K).
"""
from __future__ import annotations
import sys, csv, time, argparse
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from run_local_grid import _stratified_three_way_split, _inference_batch
from msdm.fdi import ATTACKS
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ECL_TXT = ECL_PATH
WIN_DAYS = 49   # 7 weeks → a14 weeks=7 (temporal branch needs weeks≥7)


def load_ecl_daily():
    print("[ecl] parsing (710MB, European decimal)...", flush=True)
    df = pd.read_csv(ECL_TXT, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    daily = df.resample("D").mean()                 # 15-min → daily mean kW
    X = daily.values.T.astype(np.float64)           # (370 clients, n_days)
    print(f"[ecl] daily grid {X.shape}", flush=True)
    return X


def make_windows(X, win=WIN_DAYS):
    """Non-overlapping win-day windows per client → (N, win). Drop all-zero windows."""
    n, T = X.shape
    nw = T // win
    out = []
    for i in range(n):
        for w in range(nw):
            seg = X[i, w*win:(w+1)*win]
            if np.nansum(np.abs(seg)) > 1e-6:        # drop empty windows (no-signal)
                out.append(seg)
    return np.array(out, dtype=np.float64)           # (N, win)


def inject_fdi(W, rng, frac=0.10):
    N = len(W); y = np.zeros(N, np.int64); Wo = W.copy()
    idx = rng.choice(N, int(round(frac*N)), replace=False)
    for i in idx:
        Wo[i] = ATTACKS[rng.integers(6)](W[i], rng); y[i] = 1
    return Wo, y


def preprocess(W, norm="window"):
    """3σ clip + min-max → (N, weeks, 7). norm='window' (per-window, scale-removed)
    or 'global' (whole-dataset min/max, MSDM Eq.13, scale-PRESERVED → scaling attacks visible)."""
    m = np.nanmean(W, 1, keepdims=True); s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m + 3*s, m + 3*s, W)
    if norm == "global":
        lo = np.nanmin(W); hi = np.nanmax(W); d = (hi - lo) or 1.0
    else:
        lo = np.nanmin(W, 1, keepdims=True); hi = np.nanmax(W, 1, keepdims=True)
        d = hi - lo; d[d == 0] = 1.0
    W = (W - lo) / d
    return W.reshape(len(W), WIN_DAYS // 7, 7).astype(np.float32)


def run_cell(X, y, seed, fold, kf=0, epochs=30, device="cuda", backbone="a14_no_mixer"):
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, (0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(kf*7+fold))
    seed_everything(seed)
    model = build_arm(backbone, weeks=WIN_DAYS // 7, days=7)
    cfg = TrainConfig(epochs=epochs, patience=8, batch_size=256, pos_fraction=0.5,
                      lr=3e-4, weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01,
                      grad_clip_norm=1.0, device=device)
    _, best = train(model, WeeklyGridDataset(X[tr], y[tr]), WeeklyGridDataset(X[es], y[es]), cfg)
    s = _inference_batch(best, X[te], torch.device(device))
    yt = y[te]
    order = np.argsort(-s)
    return dict(seed=seed, fold=fold,
                pr_auc=float(average_precision_score(yt, s)),
                roc_auc=float(roc_auc_score(yt, s)),
                f1=float(max(f1_score(yt, (s >= t).astype(int), zero_division=0)
                             for t in np.quantile(s, np.linspace(0, 1, 100)))),
                p_at_100=float(yt[order[:min(100, len(yt))]].mean()),
                n_test=len(te), n_pos=int(yt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="a34_generic")
    ap.add_argument("--norm", default="window", choices=["window", "global"])
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "ecl_ours.csv"))
    a = ap.parse_args()

    Xd = load_ecl_daily()
    W = make_windows(Xd)
    print(f"[ecl] {len(W)} windows ({WIN_DAYS}d each)", flush=True)
    W, y = inject_fdi(W, np.random.default_rng(0), frac=a.frac)
    Xc = preprocess(W, norm=a.norm)
    print(f"[ecl] samples {Xc.shape} theft%={100*y.mean():.2f} (synthetic FDI)", flush=True)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "fold", "pr_auc", "roc_auc", "f1",
                                           "p_at_100", "n_test", "n_pos"])
        w.writeheader()
        for s in a.seeds:
            for f in a.folds:
                t0 = time.time(); r = run_cell(Xc, y, s, f, epochs=a.epochs, device=a.device, backbone=a.backbone)
                rows.append(r); w.writerow(r); fh.flush()
                print(f"[s{s} f{f}] AP={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
                      f"F1={r['f1']:.4f} P@100={r['p_at_100']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== ECL (synthetic FDI theft), a14, n={} ===".format(len(rows)))
    for c in ["pr_auc", "roc_auc", "f1", "p_at_100"]:
        v = [r[c] for r in rows]; print(f"  {c:9s} {np.mean(v):.4f} ± {np.std(v):.4f}")
    print("  (vs real SGCC a14 random: AP 0.553, ROC 0.877, F1 0.522)")


if __name__ == "__main__":
    main()
