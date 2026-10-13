"""ECL at HOURLY resolution — test whether finer resolution lifts synthetic-theft AP
toward the field's 0.90+ (diagnosis: daily aggregation kills fdi6/shape attacks).

Hourly (24/day), 7-day windows = 168 points. FDI injected on the 168-pt series.
Runs MSDM (differencing, resolution-agnostic, hourly scales) and a31 (a14+diff, reshaped 7x24).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from zkmix_lab.data import KFoldSpec
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from run_local_grid import _stratified_three_way_split, _inference_batch
from msdm.fdi import ATTACKS
from msdm.train import train_msdm
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ECL = ECL_PATH
FREQ = "h"; PPD = 24; WIN = 7 * PPD; BACKBONE = "a31_diff"   # overridden in main()


def load():
    df = pd.read_csv(ECL, sep=";", decimal=",", index_col=0); df.index = pd.to_datetime(df.index)
    res = df.resample(FREQ).mean().values.T.astype(np.float64)   # (370, n_steps)
    n, T = res.shape; nw = T // WIN
    out = [res[i, w*WIN:(w+1)*WIN] for i in range(n) for w in range(nw)
           if np.nansum(np.abs(res[i, w*WIN:(w+1)*WIN])) > 1e-6]
    return np.array(out)


def fdi(W, rng, frac=0.10):
    y = np.zeros(len(W), int); Wo = W.copy()
    for i in rng.choice(len(W), int(frac*len(W)), False):
        Wo[i] = ATTACKS[rng.integers(6)](W[i], rng); y[i] = 1
    return Wo, y


def prep(W):  # per-window 3σ clip + min-max
    m = np.nanmean(W, 1, keepdims=True); s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m+3*s, m+3*s, W)
    lo = np.nanmin(W, 1, keepdims=True); hi = np.nanmax(W, 1, keepdims=True); d = hi-lo; d[d == 0] = 1
    return ((W-lo)/d).astype(np.float32)


def main():
    global FREQ, PPD, WIN, BACKBONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="30min")     # pandas resample freq
    ap.add_argument("--ppd", type=int, default=48)  # points per day (30min→48, 15min→96)
    ap.add_argument("--cap", type=int, default=25000)
    ap.add_argument("--backbone", default="a31_diff")
    ap.add_argument("--skip_msdm", action="store_true")
    a = ap.parse_args()
    FREQ = a.freq; PPD = a.ppd; WIN = 7 * PPD; BACKBONE = a.backbone
    print(f"[ecl] freq={FREQ} ppd={PPD} win={WIN}; loading...", flush=True)
    W = load(); print(f"[ecl] {W.shape} windows", flush=True)
    rng = np.random.default_rng(0)
    if len(W) > a.cap:
        W = W[rng.choice(len(W), a.cap, False)]
    W, y = fdi(W, np.random.default_rng(0), 0.10)
    Wp = prep(W)
    print(f"[ecl] n={len(Wp)} theft%={100*y.mean():.1f}", flush=True)
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
    # resolution-scaled diff scales (1step, halfday, day, 2day, 3day) capped < WIN
    sc = sorted({1, PPD // 2, PPD, 2 * PPD, 3 * PPD})
    sc = [x for x in sc if 0 < x < WIN]

    if not a.skip_msdm:
        print(f"\n=== MSDM @ {FREQ} (scales {sc}, period {PPD}) ===", flush=True)
        Xm = Wp[:, None, :]; aps = []
        for s in (0, 1, 2):
            tr, val = folds[0]; _, _, te = _stratified_three_way_split(val, y, (.33,.33,.34), rng=np.random.default_rng(0))
            _, m, _ = train_msdm(Xm[tr], y[tr], Xm[te], y[te], scales=sc, period=PPD,
                                 in_channels=1, epochs=30, batch=256, device="cuda", seed=s)
            aps.append(m['map']); print(f"  s{s}: AP={m['map']:.4f} ROC={m['auc']:.4f} F1={m['f1']:.4f}", flush=True)
        print(f"  MSDM {FREQ} AP={np.mean(aps):.4f}", flush=True)

    print(f"\n=== {BACKBONE} @ {FREQ}, grid (7,{PPD}) ===", flush=True)
    Xg = Wp.reshape(len(Wp), 7, PPD); aps2 = []
    for s in (0, 1, 2):
        tr, val = folds[0]; _, es, te = _stratified_three_way_split(val, y, (.33,.33,.34), rng=np.random.default_rng(0))
        seed_everything(s)
        model = build_arm(BACKBONE, weeks=7, days=PPD)
        cfg = TrainConfig(epochs=30, patience=8, batch_size=256, pos_fraction=0.5, lr=3e-4,
                          weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01, grad_clip_norm=1.0, device="cuda")
        _, best = train(model, WeeklyGridDataset(Xg[tr], y[tr]), WeeklyGridDataset(Xg[es], y[es]), cfg)
        scn = _inference_batch(best, Xg[te], torch.device("cuda"))
        aps2.append(average_precision_score(y[te], scn))
        print(f"  s{s}: AP={aps2[-1]:.4f} ROC={roc_auc_score(y[te],scn):.4f}", flush=True)
    print(f"  a31 {FREQ} AP={np.mean(aps2):.4f}", flush=True)
    print(f"\n(daily a31 0.729 | hourly: MSDM 0.873, a31 0.875)")


if __name__ == "__main__":
    main()
