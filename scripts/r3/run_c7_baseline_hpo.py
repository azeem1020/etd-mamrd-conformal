#!/usr/bin/env python3
"""Comment 7: Targeted Validation-Based Baseline HPO, WDCNN Native Interpolation, & Finardi 9-Cell Native Preprocessing.

1. Validation-Based HPO (D_es ONLY, zero test-set peeking) for HORLN and Self-Attention (hybridattn):
   - Candidate grid (6 configs): (lr, wd) in {(1e-4, 1e-4), (3e-4, 1e-4), (1e-3, 1e-4),
                                            (3e-4, 0.0), (3e-4, 1e-3), (1e-3, 1e-3)}.
   - Evaluates all 6 on Fold 0, Seed 0; selects argmax(best_val_pr_auc on D_es).
   - Evaluates the validation winner across all 9 matched cells (Folds 0-2 x Seeds 0-2)
     to test whether validation-tuned baselines close the gap with MAMRD (0.597 AP).
2. WDCNN with Zheng et al. (2018) linear neighbor interpolation (inside [0,1], 0 NaNs)
   across Folds 0-2 x Seeds 0-2 (9 cells).
3. Self-Attention (hybridattn) with Finardi et al. (2020) train-fit QuantileTransformer
   (n_quantiles=10, uniform, fit on D_train only) under the shared main protocol across
   all 9 matched cells (Folds 0-2 x Seeds 0-2).
"""
from __future__ import annotations
import sys, os, csv, gc, json, time, argparse, subprocess
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

csv.field_size_limit(10**7)

THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
_REPO_DIR = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(_REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import _ROOT, RESULTS_DIR, SGCC_PATH, scrub_paths

ORIG_REPO = _ROOT
ORIG_RES = RESULTS_DIR
R3_RES = _ROOT / "results" / "r3_revision"
OUT = R3_RES / "C7_baseline_hpo"
CELLS_DIR = OUT / "cells"
OUT.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import topk_tie_neutral
from phased.models import build_arm
from run_local_grid import load_and_preprocess, _stratified_three_way_split, _inference_batch

N_DIST = NormalDist()
HPO_GRID = [
    (1e-4, 1e-4),
    (3e-4, 1e-4),
    (1e-3, 1e-4),
    (3e-4, 0.0),
    (3e-4, 1e-3),
    (1e-3, 1e-3),
]


def bca_ci(data, B: int = 2000, alpha: float = 0.05, seed: int = 12345) -> tuple[float, float]:
    x = np.asarray([v for v in data if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        m = float(np.mean(x)) if n else float("nan")
        return m, m
    th = float(x.mean())
    rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    p_lt = min(max(float(np.mean(boot < th)), 1e-6), 1.0 - 1e-6)
    z0 = N_DIST.inv_cdf(p_lt)
    jk = np.array([np.delete(x, i).mean() for i in range(n)])
    jb = jk.mean()
    den = 6.0 * (np.sum((jb - jk) ** 2) ** 1.5)
    a = float(np.sum((jb - jk) ** 3) / den) if den else 0.0
    def q_adj(q):
        z = N_DIST.inv_cdf(q)
        return N_DIST.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
    return float(np.quantile(boot, q_adj(alpha / 2.0))), float(np.quantile(boot, q_adj(1.0 - alpha / 2.0)))


def gpu_temp() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


def cool_if_hot(limit: int = 82, target: int = 68) -> None:
    if gpu_temp() <= limit:
        return
    for _ in range(30):
        if gpu_temp() <= target:
            break
        time.sleep(10)


def _pick_f1_threshold(cal_y: np.ndarray, cal_s: np.ndarray) -> float:
    best_t, best_f = 0.5, -1.0
    for t in np.unique(cal_s):
        f = f1_score(cal_y, (cal_s >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def make_linear_interp_grid(X: np.ndarray) -> np.ndarray:
    n, w, d = X.shape
    flat = X.reshape(n, w * d).astype(np.float32).copy()
    t_idx = np.arange(w * d, dtype=np.float64)
    for i in range(n):
        row = flat[i]
        valid = ~np.isnan(row)
        nv = int(valid.sum())
        if nv == 0:
            flat[i] = 0.0
        elif nv == 1:
            flat[i] = float(row[valid][0])
        elif nv < len(row):
            flat[i] = np.interp(t_idx, t_idx[valid], row[valid]).astype(np.float32)
    return flat.reshape(n, w, d)


def prep_finardi_quantile_fold(raw_2d: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    arr = raw_2d[:, :1029].astype(np.float64)
    qt = QuantileTransformer(n_quantiles=10, output_distribution="uniform", subsample=10**9, random_state=0)
    qt.fit(arr[tr_idx])
    return qt.transform(arr).astype(np.float32).reshape(len(raw_2d), 147, 7)


def verify_math_invariants() -> bool:
    arr = np.array([
        [np.nan, 2.0, np.nan, 4.0] + [3.0] * 1025,
        [1.0, 3.0, 2.0, 5.0] + [2.5] * 1025,
        [2.0, 4.0, 3.0, 6.0] + [3.5] * 1025,
    ], dtype=np.float64)
    qt = QuantileTransformer(n_quantiles=3, output_distribution="uniform", random_state=0)
    qt.fit(arr)
    out = qt.transform(arr)
    assert np.isnan(out[0, 0]) and not np.isnan(out[0, 1])
    X_l = make_linear_interp_grid(arr[:, :1029].reshape(3, 147, 7))
    assert not np.isnan(X_l).any()
    return True


def run_single_cell(X_grid: np.ndarray, y: np.ndarray, folds: list, label: str, backbone: str,
                    seed: int, fold: int, lr: float, wd: float, epochs: int = 30, device: str = "cuda") -> dict:
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(fold))
    X_reg, rw = slice_regime(X_grid, "full")
    seed_everything(seed)
    model = build_arm(backbone, weeks=rw, days=X_grid.shape[2])
    cfg = TrainConfig(
        epochs=epochs, patience=10, batch_size=128, eval_batch_size=256,
        pos_fraction=0.5, optimizer="AdamW", lr=lr, weight_decay=wd,
        warmup_frac=0.05, min_lr_ratio=0.01, eval_every=3, grad_clip_norm=1.0,
        device=device, loss_type="bce",
    )
    t0 = time.time()
    res, best = train(model, WeeklyGridDataset(X_reg[tr], y[tr]), WeeklyGridDataset(X_reg[es], y[es]), cfg)
    wall = time.time() - t0
    dev = torch.device(device)
    cal_s = _inference_batch(best, X_reg[cal], dev)
    te_s = _inference_batch(best, X_reg[te], dev)
    thr = _pick_f1_threshold(y[cal], cal_s)
    yt = y[te]
    m100, p100 = topk_tie_neutral(yt, te_s, 100, n_perm=1000, seed=0)
    row = {
        "label": label, "backbone": backbone, "lr": lr, "wd": wd, "seed": seed, "fold": fold,
        "best_val_ap": float(res.best_val_pr_auc), "best_epoch": int(res.best_epoch),
        "wall_s": round(wall, 1), "pr_auc": float(average_precision_score(yt, te_s)),
        "roc_auc": float(roc_auc_score(yt, te_s)),
        "f1": float(f1_score(yt, (te_s >= thr).astype(int), zero_division=0)),
        "map_at_100": m100, "p_at_100": p100,
        "val_labels_json": json.dumps(yt.astype(int).tolist()),
        "val_scores_json": json.dumps([round(float(v), 6) for v in te_s]),
    }
    del model, best
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    if args.self_test:
        assert verify_math_invariants()
        print("[SELF-TEST PASS] run_c7_baseline_hpo invariants verified.", flush=True)
        return

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    sgcc_csv = SGCC_PATH
    df = pd.read_csv(sgcc_csv)
    raw_2d = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    X_base, y = load_and_preprocess(sgcc_csv)
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))

    if args.smoke:
        r = run_single_cell(X_base, y, folds, "smoke_horln", "horln", 0, 0, 3e-4, 1e-4, epochs=1, device=args.device)
        print(f"[SMOKE PASS] C7 HORLN ep=1 PR={r['pr_auc']:.4f} val={r['best_val_ap']:.4f}")
        return

    cols = ["label", "backbone", "lr", "wd", "seed", "fold", "best_val_ap", "best_epoch",
            "wall_s", "pr_auc", "roc_auc", "f1", "map_at_100", "p_at_100",
            "val_labels_json", "val_scores_json"]

    def get_or_run(fname, Xg, label, backbone, s, f, lr, wd):
        fp = CELLS_DIR / fname
        if args.resume and fp.exists() and fp.stat().st_size > 100:
            dfc = pd.read_csv(fp)
            if len(dfc) == 1 and np.isfinite(float(dfc["pr_auc"].iloc[0])):
                return dfc.iloc[0].to_dict(), True
        cool_if_hot()
        r = run_single_cell(Xg, y, folds, label, backbone, s, f, lr, wd, epochs=args.epochs, device=args.device)
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerow(r)
        return r, False

    # 1. Validation-Based HPO (Fold 0, Seed 0) for HORLN & Self-Attn (D_es selection ONLY)
    hpo_rows = []
    winners = {}
    for backbone in ("horln", "hybridattn"):
        log(f"\n=== HPO Grid on D_es (Fold 0, Seed 0) for {backbone} ===")
        cand_rows = []
        for lr, wd in HPO_GRID:
            tag = f"hpo_{backbone}_lr{lr}_wd{wd}_s0_f0"
            r, skipped = get_or_run(f"{tag}.csv", X_base, tag, backbone, 0, 0, lr, wd)
            hpo_rows.append(r); cand_rows.append(r)
            log(f"  [{'SKIP' if skipped else 'RUN'}] {backbone} lr={lr} wd={wd} | val_AP={r['best_val_ap']:.4f} test_AP={r['pr_auc']:.4f} ep={r['best_epoch']}")
        cand_rows_sorted = sorted(cand_rows, key=lambda d: float(d["best_val_ap"]), reverse=True)
        win = cand_rows_sorted[0]
        winners[backbone] = (float(win["lr"]), float(win["wd"]), float(win["best_val_ap"]))
        log(f"  -> Winner on D_es for {backbone}: lr={win['lr']}, wd={win['wd']} (val_AP={win['best_val_ap']:.4f})")

    pd.DataFrame(hpo_rows).to_csv(OUT / "c7_hpo_validation_grid.csv", index=False)

    # Evaluate validation winner across all 9 cells (or runner-up across 9 cells if default won, to verify!)
    tuned_rows, tuned_summary = [], []
    mamrd_ref = [float(next(csv.DictReader(open(ORIG_RES / f"a34_cell_{s}_{f}.csv")))["pr_auc"])
                 for s in range(3) for f in range(3)]
    for backbone in ("horln", "hybridattn"):
        w_lr, w_wd, _ = winners[backbone]
        lbl = f"tuned_{backbone}"
        v_cells = []
        for s in range(3):
            for f in range(3):
                r, skipped = get_or_run(f"{lbl}_{s}_{f}.csv", X_base, lbl, backbone, s, f, w_lr, w_wd)
                v_cells.append(r); tuned_rows.append(r)
                log(f"  [{'SKIP' if skipped else 'RUN'}] {lbl} s={s} f={f} (lr={w_lr}, wd={w_wd}) PR={r['pr_auc']:.4f} val={r['best_val_ap']:.4f}")
        dfv = pd.DataFrame(v_cells).sort_values(["seed", "fold"])
        aps = dfv["pr_auc"].astype(float).values
        alo, ahi = bca_ci(aps)
        diff = np.array(mamrd_ref) - aps
        dlo, dhi = bca_ci(diff)
        _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="exact")
        tuned_summary.append({
            "backbone": backbone, "lr_selected": w_lr, "wd_selected": w_wd, "n_cells": 9,
            "ap_mean": float(aps.mean()), "ap_ci_lo": alo, "ap_ci_hi": ahi,
            "roc_mean": float(dfv["roc_auc"].astype(float).mean()),
            "f1_mean": float(dfv["f1"].astype(float).mean()),
            "mamrd_minus_tuned_dAP": float(diff.mean()), "dAP_ci_lo": dlo, "dAP_ci_hi": dhi,
            "wilcoxon_p": float(p), "mamrd_wins_of_9": int(np.sum(diff > 0)),
        })

    # 2. WDCNN with Zheng et al. linear neighbor interpolation (9 cells)
    log("\n=== WDCNN with Zheng Linear Neighbor Interpolation (9 cells) ===")
    X_lin = make_linear_interp_grid(X_base)
    wd_cells = []
    for s in range(3):
        for f in range(3):
            r, skipped = get_or_run(f"wdcnn_lininterp_{s}_{f}.csv", X_lin, "wdcnn_linear_interp", "wdcnn", s, f, 3e-4, 1e-4)
            wd_cells.append(r)
            log(f"  [{'SKIP' if skipped else 'RUN'}] wdcnn_lininterp s={s} f={f} PR={r['pr_auc']:.4f}")
    df_wd = pd.DataFrame(wd_cells).sort_values(["seed", "fold"])
    wd_aps = df_wd["pr_auc"].astype(float).values
    alo, ahi = bca_ci(wd_aps)
    wd_diff = np.array(mamrd_ref) - wd_aps
    wd_dlo, wd_dhi = bca_ci(wd_diff)
    _, wd_p = wilcoxon(wd_diff, zero_method="wilcox", alternative="two-sided", method="exact")
    wd_sum = [{
        "variant": "wdcnn_linear_interp", "n_cells": 9,
        "ap_mean": float(wd_aps.mean()), "ap_ci_lo": alo, "ap_ci_hi": ahi,
        "roc_mean": float(df_wd["roc_auc"].astype(float).mean()), "f1_mean": float(df_wd["f1"].astype(float).mean()),
        "mamrd_minus_var_dAP": float(wd_diff.mean()), "dAP_ci_lo": wd_dlo, "dAP_ci_hi": wd_dhi,
        "wilcoxon_p": float(wd_p), "mamrd_wins_of_9": int(np.sum(wd_diff > 0)),
    }]
    pd.DataFrame(wd_sum).to_csv(OUT / "c7_wdcnn_interp_summary.csv", index=False)

    # 3. Finardi Self-Attn Native QuantileTransformer across ALL 9 matched cells (Folds 0-2 x Seeds 0-2)
    log("\n=== Self-Attn (hybridattn) with Finardi Native Train-Fit QuantileTransformer (9 cells) ===")
    qt_cache = {}
    fin_cells = []
    for s in range(3):
        for f in range(3):
            if f not in qt_cache:
                tr, _ = folds[f]
                qt_cache[f] = prep_finardi_quantile_fold(raw_2d, tr)
            r, skipped = get_or_run(f"hybridattn_finardi_qt_{s}_{f}.csv", qt_cache[f],
                                    "hybridattn_finardi_qt", "hybridattn", s, f, 3e-4, 1e-4)
            fin_cells.append(r)
            log(f"  [{'SKIP' if skipped else 'RUN'}] hybridattn_finardi_qt s={s} f={f} PR={r['pr_auc']:.4f}")
    df_fin = pd.DataFrame(fin_cells).sort_values(["seed", "fold"])
    fin_aps = df_fin["pr_auc"].astype(float).values
    alo, ahi = bca_ci(fin_aps)
    fin_diff = np.array(mamrd_ref) - fin_aps
    fin_dlo, fin_dhi = bca_ci(fin_diff)
    _, fin_p = wilcoxon(fin_diff, zero_method="wilcox", alternative="two-sided", method="exact")
    fin_sum = [{
        "variant": "hybridattn_finardi_native_qt10", "n_cells": 9,
        "ap_mean": float(fin_aps.mean()), "ap_ci_lo": alo, "ap_ci_hi": ahi,
        "roc_mean": float(df_fin["roc_auc"].astype(float).mean()), "f1_mean": float(df_fin["f1"].astype(float).mean()),
        "mamrd_minus_var_dAP": float(fin_diff.mean()), "dAP_ci_lo": fin_dlo, "dAP_ci_hi": fin_dhi,
        "wilcoxon_p": float(fin_p), "mamrd_wins_of_9": int(np.sum(fin_diff > 0)),
    }]
    pd.DataFrame(fin_sum).to_csv(OUT / "c7_finardi_native_9cell.csv", index=False)
    pd.DataFrame(tuned_summary).to_csv(OUT / "c7_tuned_baselines_summary.csv", index=False)

    log("\n--- C7 Tuned Baselines Summary ---")
    log(pd.DataFrame(tuned_summary).to_string(index=False))
    log("\n--- C7 WDCNN Linear Interp & Finardi Native QT Summary ---")
    log(pd.DataFrame(wd_sum + fin_sum).to_string(index=False))

    (OUT / "c7_summary.json").write_text(json.dumps({
        "hpo_winners": winners, "tuned_summary": tuned_summary,
        "wdcnn_linear_interp": wd_sum, "finardi_native_9cell": fin_sum,
    }, indent=2))
    log(f"\n[DONE] C7 completed in {(time.time() - t0)/60.0:.2f} min -> {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
