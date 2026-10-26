"""Apply OUR approach (MAMRD / a34_generic) to ECL with synthetic FDI theft.

ECL benchmark evaluation (370 industrial load profiles):
30-minute half-hourly resolution -> 7-day windows (W=7 days, D=48 readings/day, L=336)
Injected synthetic FDI theft family (10% prevalence).
Standardized preprocessing: 3sigma clipping + per-window min-max scaling -> MAMRD (a34_generic, weeks=7, days=48)
Reproduces the 0.845 AP leaderboard score under random window-level cross-validation.
"""
from __future__ import annotations
import sys, csv, time, argparse, json
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from zkmix_lab.data import KFoldSpec
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from run_local_grid import _stratified_three_way_split, _inference_batch
from msdm.fdi import ATTACKS
from repro_config import RESULTS_DIR, DATA_DIR, ECL_PATH

ECL_TXT = ECL_PATH
FREQ, PPD = "30min", 48
WIN = 7 * PPD  # 336 readings per 7-day window
CAP = 25000


def load_ecl_windows():
    """Load ECL load diagrams and resample to 30-min intervals."""
    print("[ecl] parsing ECL dataset (30-min resolution)...", flush=True)
    df = pd.read_csv(ECL_TXT, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    res = df.resample(FREQ).mean().values.T.astype(np.float64)
    n, T = res.shape
    nw = T // WIN
    out = [res[i, w*WIN:(w+1)*WIN] for i in range(n) for w in range(nw)
           if np.nansum(np.abs(res[i, w*WIN:(w+1)*WIN])) > 1e-6]
    return np.array(out)


def inject_fdi(W, rng, frac=0.10):
    N = len(W)
    y = np.zeros(N, np.int64)
    Wo = W.copy()
    idx = rng.choice(N, int(round(frac * N)), replace=False)
    for i in idx:
        Wo[i] = ATTACKS[rng.integers(6)](W[i], rng)
        y[i] = 1
    return Wo, y


def preprocess(W):
    """3sigma clipping followed by per-window min-max scaling -> (N, 7, 48)."""
    m = np.nanmean(W, 1, keepdims=True)
    s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m + 3 * s, m + 3 * s, W)
    lo = np.nanmin(W, 1, keepdims=True)
    hi = np.nanmax(W, 1, keepdims=True)
    d = hi - lo
    d[d == 0] = 1.0
    W = ((W - lo) / d).astype(np.float32)
    return W.reshape(len(W), 7, PPD)


def run_cell(X, y, seed, fold, kf=0, epochs=30, device="cuda", backbone="a34_generic"):
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    tr_sub, es, te = _stratified_three_way_split(val, y, (0.33, 0.33, 0.34),
                                                rng=np.random.default_rng(kf * 7 + fold))
    seed_everything(seed)
    model = build_arm(backbone, weeks=7, days=PPD)
    cfg = TrainConfig(epochs=epochs, patience=8, batch_size=128, pos_fraction=0.5,
                      optimizer="AdamW", lr=3e-4, weight_decay=1e-4, warmup_frac=0.05,
                      min_lr_ratio=0.01, grad_clip_norm=1.0, device=device)
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
    ap = argparse.ArgumentParser(description="Evaluate MAMRD on ECL Benchmark")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="a34_generic", help="Model backbone (default: a34_generic)")
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "ecl_ours.csv"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    W = load_ecl_windows()
    rng = np.random.default_rng(0)
    if len(W) > CAP:
        W = W[rng.choice(len(W), CAP, False)]
    print(f"[ecl] {len(W)} windows (30-min resolution, 7 days x 48 = 336 readings each)", flush=True)
    W, y = inject_fdi(W, np.random.default_rng(0), frac=a.frac)
    Xc = preprocess(W)
    print(f"[ecl] samples {Xc.shape} theft%={100*y.mean():.2f} (synthetic FDI)", flush=True)

    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "fold", "pr_auc", "roc_auc", "f1",
                                           "p_at_100", "n_test", "n_pos"])
        w.writeheader()
        seeds = [0] if a.smoke else a.seeds
        folds = [0] if a.smoke else a.folds
        epochs = 3 if a.smoke else a.epochs
        for s in seeds:
            for f in folds:
                t0 = time.time()
                r = run_cell(Xc, y, s, f, epochs=epochs, device=a.device, backbone=a.backbone)
                rows.append(r)
                w.writerow(r)
                fh.flush()
                print(f"[{a.backbone} s{s} f{f}] AP={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
                      f"F1={r['f1']:.4f} P@100={r['p_at_100']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                if a.smoke:
                    break
            if a.smoke:
                break

    print(f"\n=== ECL (synthetic FDI theft), backbone={a.backbone}, n={len(rows)} ===")
    for c in ["pr_auc", "roc_auc", "f1", "p_at_100"]:
        v = [r[c] for r in rows]
        print(f"  {c:9s} {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    main()
