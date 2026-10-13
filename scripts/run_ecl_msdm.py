"""MSDM (reproduction) on the SAME ECL data as run_ecl_ours.py — head-to-head vs a14.

Identical windows, identical FDI injection (rng seed 0), identical random k-fold splits.
Only the model differs: MSDM (kMSD+AQ+MSCNet) vs our a14. Equal budget (30 epochs).
"""
from __future__ import annotations
import sys, csv, time, argparse
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
from zkmix_lab.data import KFoldSpec
from run_local_grid import _stratified_three_way_split
from run_ecl_ours import load_ecl_daily, make_windows, inject_fdi, WIN_DAYS
from msdm.train import train_msdm
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH


def preprocess_flat(W):
    """Same clip+min-max as run_ecl_ours, but keep flat (N,1,T) for MSDM's kMSD."""
    m = np.nanmean(W, 1, keepdims=True); s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m + 3*s, m + 3*s, W)
    lo = np.nanmin(W, 1, keepdims=True); hi = np.nanmax(W, 1, keepdims=True)
    d = hi - lo; d[d == 0] = 1.0
    return ((W - lo) / d)[:, None, :].astype(np.float32)   # (N,1,T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default=str(RESULTS_DIR / "ecl_msdm.csv"))
    a = ap.parse_args()

    Xd = load_ecl_daily(); W = make_windows(Xd)
    W, y = inject_fdi(W, np.random.default_rng(0), frac=0.10)   # SAME as run_ecl_ours
    Xc = preprocess_flat(W)
    print(f"[ecl] MSDM samples {Xc.shape} theft%={100*y.mean():.2f}", flush=True)
    scales = [1, 2, 7, 14, 28]    # window-appropriate (≤49d); period C=7
    rows = []
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "fold", "auc", "map", "map_macro", "f1"])
        w.writeheader()
        for s in a.seeds:
            for f in a.folds:
                folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
                tr, val = folds[f]
                es, cal, te = _stratified_three_way_split(val, y, (0.33, 0.33, 0.34),
                                                          rng=np.random.default_rng(0*7+f))
                t0 = time.time()
                _, m, _ = train_msdm(Xc[tr], y[tr], Xc[te], y[te], scales=scales, period=7,
                                     in_channels=1, epochs=a.epochs, batch=256,
                                     device=a.device, seed=s)
                r = dict(seed=s, fold=f, auc=m["auc"], map=m["map"],
                         map_macro=m["map_macro"], f1=m["f1"])
                rows.append(r); w.writerow(r); fh.flush()
                print(f"[s{s} f{f}] AUC={m['auc']:.4f} MAP(AP)={m['map']:.4f} "
                      f"F1={m['f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n=== MSDM on ECL (synthetic FDI), n={len(rows)} ===")
    for c in ["auc", "map", "f1"]:
        v = [r[c] for r in rows]; print(f"  {c:9s} {np.mean(v):.4f} ± {np.std(v):.4f}")
    print("  (our a14 on same ECL data: AP 0.712, ROC 0.856, F1 0.724)")


if __name__ == "__main__":
    main()
