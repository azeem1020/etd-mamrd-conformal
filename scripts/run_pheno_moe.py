"""PhenoMoE runner (Phase N) — phenotype-routed MoE vs a14 monolith.

For each (seed, fold): build a14 backbone, wrap in PhenoMoE (label-free Paper-4
router, fit on TRAIN only), train, then report PR-AUC overall AND per phenotype on
the held-out test split. Paired against the existing a14 monolith cells (same
splits: seeds 0-4 x folds 0-2 x kf0 in phase_l_local/local_manifest.csv).

Headline question: does per-stratum specialisation lift the detectable Heavy
strata from ~0.62-0.64 toward 0.70, without leakage and without hurting the rest?
"""
from __future__ import annotations
import sys, csv, json, time, argparse
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import average_precision_score

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import seed_everything
from phased.models import build_arm
from run_local_grid import load_and_preprocess, _stratified_three_way_split, SGCC_CSV
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES, PH_INDEX
from phased.pheno_moe import PhenoMoE, PhenoDataset, train_pheno_moe
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

RESULTS = RESULTS_DIR


def _raw_grid():
    df = pd.read_csv(SGCC_CSV)
    raw = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    nk = (raw.shape[1] // 7) * 7
    return raw[:, :nk].reshape(len(df), nk // 7, 7)


@torch.no_grad()
def _infer(model, X, pidx, device, bs=1024):
    model.eval(); dev = torch.device(device); out = np.zeros(len(X), np.float32)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(dev)
        pb = torch.from_numpy(pidx[i:i+bs]).to(dev)
        out[i:i+bs] = torch.sigmoid(model(xb, pb)).cpu().numpy()
    return out


def run_cell(X, y, g, m, seed, fold, kf=0, epochs=30, device="cuda"):
    D = X.shape[2]
    X_reg, rw = slice_regime(X, "full")
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(kf * 7 + fold))
    # leakage-free router: fit tau/medians on TRAIN customers only
    params = fit_router(g[tr], m[tr])
    def pidx(idx):
        ph = assign(g[idx], m[idx], params)
        return np.array([PH_INDEX[p] for p in ph], dtype=np.int64)
    seed_everything(seed)
    backbone = build_arm("a14_no_mixer", weeks=rw, days=D)
    model = PhenoMoE(backbone, n_pheno=5)
    t0 = time.time()
    res, best = train_pheno_moe(
        model,
        PhenoDataset(X_reg[tr], y[tr], pidx(tr)),
        PhenoDataset(X_reg[es], y[es], pidx(es)),
        epochs=epochs, device=device, seed=seed,
    )
    wall = time.time() - t0
    p_te = pidx(te)
    s_te = _infer(best, X_reg[te], p_te, device)
    yt = y[te]
    row = dict(arm="pheno_moe", seed=seed, fold=fold, kf=kf,
               pr_auc=float(average_precision_score(yt, s_te)),
               best_val_pr_auc=float(res.best_val_pr_auc), best_epoch=int(res.best_epoch),
               wall_s=round(wall, 1))
    # per-phenotype PR-AUC
    for p in PHENOTYPES:
        msk = p_te == PH_INDEX[p]
        if msk.sum() > 0 and 0 < yt[msk].sum() < msk.sum():
            row[f"pr_{p}"] = float(average_precision_score(yt[msk], s_te[msk]))
        else:
            row[f"pr_{p}"] = float("nan")
    row["val_labels_json"] = json.dumps(yt.astype(int).tolist())
    row["val_scores_json"] = json.dumps([round(float(s), 6) for s in s_te])
    row["pheno_idx_json"] = json.dumps(p_te.astype(int).tolist())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default=str(RESULTS / "pheno_moe.csv"))
    args = ap.parse_args()

    print(f"[device] {args.device}", flush=True)
    X, y = load_and_preprocess()
    g, m = gap_magnitude(_raw_grid())
    cells = [(s, f) for s in args.seeds for f in args.folds]
    cols = (["arm", "seed", "fold", "kf", "pr_auc", "best_val_pr_auc", "best_epoch", "wall_s"]
            + [f"pr_{p}" for p in PHENOTYPES]
            + ["val_labels_json", "val_scores_json", "pheno_idx_json"])
    mp = Path(args.manifest); mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for i, (s, f) in enumerate(cells, 1):
            r = run_cell(X, y, g, m, s, f, epochs=args.epochs, device=args.device)
            w.writerow(r); fh.flush()
            hd, hr = r["pr_Heavy_Dropper"], r["pr_Heavy_Riser"]
            print(f"[{i}/{len(cells)}] seed={s} fold={f} all={r['pr_auc']:.4f} "
                  f"HD={hd:.3f} HR={hr:.3f} ep={r['best_epoch']} wall={r['wall_s']}s", flush=True)
    print(f"[done] -> {mp}", flush=True)


if __name__ == "__main__":
    main()
