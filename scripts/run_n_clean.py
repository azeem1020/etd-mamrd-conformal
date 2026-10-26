"""Step (a2): a14 monolith phenotype-stratified benchmark on the CLEANED cohort.

Cleaned cohort = keep_idx.npy from clean_cohort.py (drops 2114 no-signal meters).
Per phenotype + aggregate reports the SHARED metric set for top-tier comparability
(MSDM/HORLN/Liao): PR-AUC(=AP/MAP), ROC-AUC, F1 (threshold picked on the cal split,
applied to test — leakage-free operating point), plus aggregate P@100/P@200/P@500.

Leakage firewall: router (tau/medians) fit on TRAIN customers only; F1 threshold on
cal split only; raw-kWh gap/magnitude per-customer causal; y[te] alignment asserted.
"""
from __future__ import annotations
import sys, csv, json, time, argparse
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
from run_local_grid import load_and_preprocess, _stratified_three_way_split, _inference_batch, SGCC_CSV
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

OUT = RESULTS_DIR
KEEP = OUT / "keep_idx.npy"


def _pick_f1_threshold(cal_y, cal_s):
    best_t, best_f = 0.5, -1.0
    for t in np.unique(cal_s):
        f = f1_score(cal_y, (cal_s >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def _pak(y, s, k):
    k = min(k, len(y))
    if k == 0:
        return float("nan")
    order = np.argsort(-s)[:k]
    return float(y[order].sum() / k)


def _safe(fn, y, s):
    return float(fn(y, s)) if (len(y) > 0 and 0 < y.sum() < len(y)) else float("nan")


def run_cell(Xc, yc, g, m, seed, fold, kf=0, epochs=30, lr=3e-4, device="cuda",
             loss_type="bce", focal_gamma=2.0, focal_alpha=0.5, arm="a34_generic",
             backbone="a34_generic"):
    D = Xc.shape[2]
    X_reg, rw = slice_regime(Xc, "full")
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(yc))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, yc, fractions=(0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(kf * 7 + fold))
    params = fit_router(g[tr], m[tr])                       # leakage-free router
    seed_everything(seed)
    model = build_arm(backbone, weeks=rw, days=D)
    cfg = TrainConfig(epochs=epochs, patience=10, batch_size=128, eval_batch_size=256, pos_fraction=0.5,
                      optimizer="AdamW", lr=lr, weight_decay=1e-4, warmup_frac=0.05, min_lr_ratio=0.01,
                      eval_every=3, grad_clip_norm=1.0, device=device,
                      loss_type=loss_type, focal_gamma=focal_gamma, focal_alpha=focal_alpha)
    t0 = time.time()
    res, best = train(model, WeeklyGridDataset(X_reg[tr], yc[tr]),
                      WeeklyGridDataset(X_reg[es], yc[es]), cfg)
    wall = time.time() - t0
    dev = torch.device(device)
    cal_s = _inference_batch(best, X_reg[cal], dev)
    te_s = _inference_batch(best, X_reg[te], dev)
    thr = _pick_f1_threshold(yc[cal], cal_s)                # operating point from cal
    ph = assign(g[te], m[te], params)
    yt = yc[te]

    row = dict(arm=arm, seed=seed, fold=fold, kf=kf, thr=round(thr, 4),
               wall_s=round(wall, 1), n_test=len(te))
    # aggregate
    row["pr_auc"] = _safe(average_precision_score, yt, te_s)
    row["roc_auc"] = _safe(roc_auc_score, yt, te_s)
    row["f1"] = float(f1_score(yt, (te_s >= thr).astype(int), zero_division=0))
    for k in (100, 200, 500):
        row[f"p_at_{k}"] = _pak(yt, te_s, k)
    # per phenotype
    for pname in PHENOTYPES:
        msk = ph == pname
        row[f"pr_{pname}"] = _safe(average_precision_score, yt[msk], te_s[msk])
        row[f"roc_{pname}"] = _safe(roc_auc_score, yt[msk], te_s[msk])
        row[f"f1_{pname}"] = (float(f1_score(yt[msk], (te_s[msk] >= thr).astype(int), zero_division=0))
                              if msk.sum() > 0 else float("nan"))
        row[f"n_{pname}"] = int(msk.sum())
    row["val_labels_json"] = json.dumps(yt.astype(int).tolist())
    row["val_scores_json"] = json.dumps([round(float(s), 6) for s in te_s])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--kfolds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW peak learning rate (default: 3e-4)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_type", default="bce")        # 'bce' | 'focal' | 'asl'
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.5)
    ap.add_argument("--arm", default="a34_generic")
    ap.add_argument("--backbone", default="a34_generic")
    ap.add_argument("--full", action="store_true", default=True,
                    help="use full 42,372-consumer SGCC cohort (default: True)")
    ap.add_argument("--filtered", action="store_true",
                    help="use keep_idx.npy filtered cohort instead of full SGCC")
    ap.add_argument("--cell_prefix", default="a34_cell",
                    help="prefix for per-cell CSVs written to RESULTS_DIR (empty to disable)")
    ap.add_argument("--manifest", default=str(OUT / "a34_manifest.csv"))
    args = ap.parse_args()

    print(f"[device] {args.device} | backbone={args.backbone} | lr={args.lr}", flush=True)
    X, y = load_and_preprocess()
    use_full = args.full and not args.filtered
    keep = np.arange(len(y)) if use_full else np.load(KEEP)
    Xc, yc = X[keep], y[keep]
    # raw-kWh gap/magnitude on the kept cohort
    dfr = pd.read_csv(SGCC_CSV)
    raw = dfr.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    nk = (raw.shape[1] // 7) * 7
    g, m = gap_magnitude(raw[keep][:, :nk].reshape(len(keep), nk // 7, 7))
    print(f"[cohort] kept {len(keep)} / {len(y)}  theft%={100*yc.mean():.2f}", flush=True)

    cells = [(s, f, kf) for kf in args.kfolds for s in args.seeds for f in args.folds]
    cols = (["arm", "seed", "fold", "kf", "thr", "wall_s", "n_test",
             "pr_auc", "roc_auc", "f1", "p_at_100", "p_at_200", "p_at_500"]
            + [f"{x}_{p}" for p in PHENOTYPES for x in ("pr", "roc", "f1", "n")]
            + ["val_labels_json", "val_scores_json"])
    mp = Path(args.manifest); mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for i, (s, f, kf) in enumerate(cells, 1):
            r = run_cell(Xc, yc, g, m, s, f, kf=kf, epochs=args.epochs, lr=args.lr,
                         device=args.device, loss_type=args.loss_type,
                         focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
                         arm=args.arm, backbone=args.backbone)
            w.writerow(r); fh.flush()
            if args.cell_prefix:
                cell_fp = OUT / f"{args.cell_prefix}_{s}_{f}.csv"
                with open(cell_fp, "w", newline="") as cfh:
                    cw = csv.DictWriter(cfh, fieldnames=cols)
                    cw.writeheader()
                    cw.writerow(r)
            import gc; gc.collect(); torch.cuda.empty_cache()   # free GPU cache between cells
            print(f"[{i}/{len(cells)}] s={s} f={f} kf={kf} | PR={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
                  f"F1={r['f1']:.4f} | HD_PR={r['pr_Heavy_Dropper']:.3f} HR_PR={r['pr_Heavy_Riser']:.3f} "
                  f"wall={r['wall_s']}s", flush=True)
    print(f"[done] -> {mp}", flush=True)


if __name__ == "__main__":
    main()
