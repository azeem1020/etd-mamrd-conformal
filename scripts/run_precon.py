"""Apply our approach to PRECON (Pakistan residential, LUMS) with synthetic FDI.

42 homes, 1-min Usage_kW. Tests the residential-scale hypothesis: unlike ECL
(industrial, 1000x spread → global norm collapses), residential PRECON has
moderate scale spread, so GLOBAL min-max is viable → preserves scale → the
scaling attack (fdi1) becomes detectable → synthetic AP should clear 0.90.
Compares norm=global vs window. Non-overlapping 7-day windows.
"""
from __future__ import annotations
import sys, argparse, glob
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from zkmix_lab.data import KFoldSpec
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from run_local_grid import _stratified_three_way_split, _inference_batch
from msdm.fdi import ATTACKS
from msdm.train import train_msdm
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

PRECON = (DATA_DIR / "precon")


def load(freq, ppd):
    win = 7 * ppd; out = []
    for f in sorted(glob.glob(str(PRECON / "House*.csv"))):
        s = pd.read_csv(f, usecols=["Date_Time", "Usage_kW"])
        s["Date_Time"] = pd.to_datetime(s["Date_Time"])
        v = s.set_index("Date_Time")["Usage_kW"].resample(freq).mean().values.astype(np.float64)
        nw = len(v) // win
        for w in range(nw):
            seg = v[w*win:(w+1)*win]
            if np.nansum(np.abs(seg)) > 1e-6:
                out.append(seg)
    return np.array(out)


def fdi(W, rng, frac=0.10):
    y = np.zeros(len(W), int); Wo = W.copy()
    for i in rng.choice(len(W), int(frac*len(W)), False):
        Wo[i] = ATTACKS[rng.integers(6)](W[i], rng); y[i] = 1
    return Wo, y


def prep(W, norm):
    m = np.nanmean(W, 1, keepdims=True); s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m+3*s, m+3*s, W)
    if norm == "global":
        lo = np.nanmin(W); hi = np.nanmax(W); d = (hi-lo) or 1.0
    else:
        lo = np.nanmin(W, 1, keepdims=True); hi = np.nanmax(W, 1, keepdims=True); d = hi-lo; d[d == 0] = 1
    return np.nan_to_num((W-lo)/d).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="30min"); ap.add_argument("--ppd", type=int, default=48)
    ap.add_argument("--norm", default="global", choices=["global", "window"])
    ap.add_argument("--backbone", default="a31_diff")
    a = ap.parse_args()
    print(f"[precon] freq={a.freq} ppd={a.ppd} norm={a.norm}; loading 42 homes...", flush=True)
    W = load(a.freq, a.ppd)
    W, y = fdi(W, np.random.default_rng(0), 0.10)
    Wp = prep(W, a.norm)
    print(f"[precon] n={len(Wp)} win={7*a.ppd} theft%={100*y.mean():.1f}", flush=True)
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
    sc = [x for x in sorted({1, a.ppd//2, a.ppd, 2*a.ppd, 3*a.ppd}) if 0 < x < 7*a.ppd]

    print(f"\n=== MSDM @ {a.freq} norm={a.norm} ===", flush=True)
    Xm = Wp[:, None, :]; aps = []
    for s in (0, 1, 2):
        tr, val = folds[0]; _, _, te = _stratified_three_way_split(val, y, (.33,.33,.34), rng=np.random.default_rng(0))
        _, m, _ = train_msdm(Xm[tr], y[tr], Xm[te], y[te], scales=sc, period=a.ppd,
                             in_channels=1, epochs=30, batch=256, device="cuda", seed=s)
        aps.append(m['map']); print(f"  s{s}: AP={m['map']:.4f} ROC={m['auc']:.4f} F1={m['f1']:.4f}", flush=True)
    print(f"  MSDM AP={np.mean(aps):.4f}", flush=True)

    print(f"\n=== a31 (a14+diff) @ {a.freq} norm={a.norm}, grid (7,{a.ppd}) ===", flush=True)
    Xg = Wp.reshape(len(Wp), 7, a.ppd); aps2 = []
    for s in (0, 1, 2):
        tr, val = folds[0]; _, es, te = _stratified_three_way_split(val, y, (.33,.33,.34), rng=np.random.default_rng(0))
        seed_everything(s)
        model = build_arm(a.backbone, weeks=7, days=a.ppd)
        cfg = TrainConfig(epochs=30, patience=8, batch_size=256, pos_fraction=0.5, lr=3e-4,
                          weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01, grad_clip_norm=1.0, device="cuda")
        _, best = train(model, WeeklyGridDataset(Xg[tr], y[tr]), WeeklyGridDataset(Xg[es], y[es]), cfg)
        scn = _inference_batch(best, Xg[te], torch.device("cuda"))
        aps2.append(average_precision_score(y[te], scn))
        print(f"  s{s}: AP={aps2[-1]:.4f} ROC={roc_auc_score(y[te],scn):.4f}", flush=True)
    print(f"  a31 AP={np.mean(aps2):.4f}", flush=True)
    print(f"\n(hypothesis: residential+global norm → fdi1 detectable → AP>0.90; ECL industrial capped ~0.88)")


if __name__ == "__main__":
    main()
