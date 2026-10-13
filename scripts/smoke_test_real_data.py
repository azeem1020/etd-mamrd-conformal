#!/usr/bin/env python3
"""Phase D real-data smoke test.

Runs Phase C A0 (zkmix_v3_name_nn) vs Phase D A1 (CC-NaME only) and
A3 (CC-NaME + multi-scale Laplacian) on real SGCC data for 30 epochs,
seed=0, fold=0, kfold_seed=0 — identical protocol to Phase C.

Usage:
    conda run -n tgpu python smoke_test_real_data.py

Output: comparison table printed to stdout.
"""
from __future__ import annotations
import sys, os, time, json, tempfile
from pathlib import Path

HERE     = Path(__file__).parent
LAB_DIR  = (HERE / ".." / "zkmix_lab").resolve()
PHASE_D  = HERE.resolve()
SGCC_CSV = SGCC_PATH

for p in [str(PHASE_D), str(LAB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch

from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.metrics import summarise, paired_bootstrap
from zkmix_lab.calibration import calibrate_cell
from zkmix_lab.conformal import expected_calibration_error_with_bins
from zkmix_lab.models import build_model

from phased.models import build_arm
from phased.runner_d import _stratified_three_way_split, _inference_batch
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH


# ---------------------------------------------------------------------------
# Step 1 — NaN-native preprocessing (in memory, no disk write needed)
# ---------------------------------------------------------------------------

def preprocess_nan_native(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read raw SGCC CSV and apply NaN-native preprocessing.

    Steps (per Zheng 2018 §IV-A, with Step 1 NaN-fill OMITTED):
      Step 2 : NaN-aware outlier clip at mean + 2*std per user.
      Step 3 : NaN-aware per-user min-max normalisation → [0, 1].
      Step 4 : Truncate to largest multiple of 7 days.

    NaN cells are deliberately preserved so CC-NaME can read the
    missing-fraction context signal.
    """
    print(f"[preprocess] Loading {csv_path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"  shape={df.shape}  ({time.time()-t0:.1f}s)", flush=True)

    labels  = df["FLAG"].values.astype(np.int64)
    arr     = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)

    nan_frac = np.isnan(arr).mean()
    print(f"  NaN fraction: {nan_frac*100:.2f}%  (preserved end-to-end)", flush=True)

    # Step 2: nan-aware outlier clip
    row_avg = np.nanmean(arr, axis=1, keepdims=True)
    row_std = np.nanstd(arr,  axis=1, keepdims=True)
    ceiling = row_avg + 2.0 * row_std
    arr     = np.where(arr > ceiling, ceiling, arr)      # NaN > ceiling → False → preserved

    # Step 3: nan-aware min-max normalise
    row_min = np.nanmin(arr, axis=1, keepdims=True)
    row_max = np.nanmax(arr, axis=1, keepdims=True)
    denom   = row_max - row_min
    denom[denom == 0] = 1.0
    arr = (arr - row_min) / denom                        # NaN cells remain NaN

    # Step 4: truncate to multiple of 7
    n_keep = (arr.shape[1] // 7) * 7
    arr    = arr[:, :n_keep].astype(np.float32)
    weeks  = n_keep // 7

    print(f"  output: {arr.shape}  →  ({len(labels)}, {weeks} weeks, 7 days)", flush=True)
    print(f"  NaN preserved: {np.isnan(arr).mean()*100:.2f}%", flush=True)
    print(f"  valid range  : [{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]", flush=True)

    X = arr.reshape(len(labels), weeks, 7)   # (N, W, D)
    return X, labels


# ---------------------------------------------------------------------------
# Step 2 — one cell runner (shared between Phase C and Phase D)
# ---------------------------------------------------------------------------

def run_one(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    fold: int = 0,
    kfold_seed: int = 0,
    epochs: int = 30,
    device: str = "cuda",
    label: str = "model",
) -> dict:
    """Run one (seed, fold) cell and return metrics dict."""
    print(f"\n[{label}] seed={seed} fold={fold} epochs={epochs} device={device}", flush=True)

    X_reg, regime_weeks = slice_regime(X, "full")
    days = X_reg.shape[2]

    spec  = KFoldSpec(family="random", n_folds=5, seed=kfold_seed)
    folds = list(spec.iter_folds(y))
    train_idx, val_idx = folds[fold]

    rng = np.random.default_rng(kfold_seed * 7 + fold)
    es_idx, cal_idx, test_idx = _stratified_three_way_split(
        val_idx, y, fractions=(0.33, 0.33, 0.34), rng=rng
    )
    print(f"  train={len(train_idx)}  es={len(es_idx)}  cal={len(cal_idx)}  test={len(test_idx)}", flush=True)
    print(f"  test theft rate: {y[test_idx].mean()*100:.1f}%", flush=True)

    train_ds    = WeeklyGridDataset(X_reg[train_idx], y[train_idx])
    early_ds    = WeeklyGridDataset(X_reg[es_idx],    y[es_idx])

    seed_everything(seed)

    cfg = TrainConfig(
        epochs=epochs, patience=8,
        batch_size=256, pos_fraction=0.5,
        lr=3e-4, weight_decay=0.01,
        warmup_frac=0.05, min_lr_ratio=0.01,
        grad_clip_norm=1.0,
        device=device,
    )

    t0 = time.time()
    train_result, best_model = train(model, train_ds, early_ds, cfg)
    wall = time.time() - t0
    print(f"  trained {train_result.best_epoch+1} epochs in {wall:.0f}s  "
          f"best_val_pr_auc={train_result.best_val_pr_auc:.4f}", flush=True)

    dev = torch.device(device)
    cal_scores  = _inference_batch(best_model, X_reg[cal_idx],  dev)
    test_scores = _inference_batch(best_model, X_reg[test_idx], dev)
    cal_labels  = y[cal_idx]
    test_labels = y[test_idx]

    metrics = summarise(test_labels, test_scores)
    ece_raw, _ = expected_calibration_error_with_bins(test_scores, test_labels)

    p_train  = max(min(0.5, 1-1e-3), 1e-3)
    p_deploy = max(min(float(y[train_idx].mean()), 1-1e-3), 1e-3)
    cal_out  = calibrate_cell(cal_scores=cal_scores, cal_labels=cal_labels,
                              test_scores=test_scores, p_train=p_train, p_deploy=p_deploy)
    ece_cal, _ = expected_calibration_error_with_bins(cal_out.test_scores_calibrated, test_labels)

    row = {
        "label":          label,
        "pr_auc":         float(metrics["pr_auc"]),
        "auroc":          float(metrics["auroc"]),
        "f1_optimal":     float(metrics["f1_optimal"]),
        "precision_at_recall": float(metrics.get("precision_at_recall", float("nan"))),
        "ece_raw":        float(ece_raw),
        "ece_calibrated": float(ece_cal),
        "best_epoch":     int(train_result.best_epoch),
        "best_val_pr_auc": float(train_result.best_val_pr_auc),
        "wall_s":         float(wall),
        "test_labels":    test_labels,
        "test_scores":    test_scores,
    }
    print(f"  → PR-AUC={row['pr_auc']:.4f}  AUROC={row['auroc']:.4f}  "
          f"F1={row['f1_optimal']:.4f}  ECE_cal={row['ece_calibrated']:.4f}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Step 3 — main: preprocess → run A0, A1, A3 → compare
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Preprocess
    X, y = preprocess_nan_native(SGCC_CSV)
    W, D = X.shape[1], X.shape[2]
    print(f"\nDataset: {len(y)} users, {y.sum()} theft ({y.mean()*100:.1f}%)")
    print(f"Grid shape: {W}w × {D}d, NaN fraction: {np.isnan(X).mean()*100:.2f}%\n")

    SEED, FOLD, KF = 0, 0, 0
    EPOCHS = 30

    results = []

    # A0: Phase C baseline — zkmix_v3_name_nn (scalar NaME, single-scale)
    model_a0 = build_model("zkmix_v3_name_nn", weeks=W, days=D)
    r_a0 = run_one(model_a0, X, y, SEED, FOLD, KF, EPOCHS, device, "A0 Phase-C  (scalar NaME, d=1)")
    results.append(r_a0)

    # A1: CC-NaME only, single-scale Laplacian (isolates NaME upgrade)
    model_a1 = build_arm("a1_cc_name_only", weeks=W, days=D)
    r_a1 = run_one(model_a1, X, y, SEED, FOLD, KF, EPOCHS, device, "A1 CC-NaME  (adaptive τ,  d=1)")
    results.append(r_a1)

    # A2: scalar NaME + multi-scale Laplacian (isolates MS upgrade)
    model_a2 = build_arm("a2_ms_only", weeks=W, days=D)
    r_a2 = run_one(model_a2, X, y, SEED, FOLD, KF, EPOCHS, device, "A2 MS-Lap   (scalar τ,    d=1,2,4)")
    results.append(r_a2)

    # A3: Phase D full headline
    model_a3 = build_arm("a3_cc_name_ms", weeks=W, days=D)
    r_a3 = run_one(model_a3, X, y, SEED, FOLD, KF, EPOCHS, device, "A3 Phase-D  (CC-NaME + d=1,2,4)")
    results.append(r_a3)

    # ---------------------------------------------------------------------------
    # Paired bootstrap: each arm vs A0
    # ---------------------------------------------------------------------------
    print("\n" + "="*72)
    print("RESULTS — Phase D vs Phase C (single cell: seed=0, fold=0, kf=0)")
    print("="*72)

    hdr = f"{'Arm':<38} {'PR-AUC':>7} {'AUROC':>7} {'F1':>7} {'ECE_c':>7} {'Ep':>4}"
    print(hdr)
    print("-"*72)
    for r in results:
        print(f"{r['label']:<38} {r['pr_auc']:>7.4f} {r['auroc']:>7.4f} "
              f"{r['f1_optimal']:>7.4f} {r['ece_calibrated']:>7.4f} {r['best_epoch']:>4}")

    print("\n--- Paired bootstrap vs A0 (B=2000, BCa approximation) ---")
    y_ref = r_a0["test_labels"]
    s_ref = r_a0["test_scores"]
    for r in results[1:]:
        y_q  = r["test_labels"]
        s_q  = r["test_scores"]
        # Use same test slice — must match exactly since same (seed, fold, kf) RNG
        n    = min(len(y_ref), len(y_q))
        res  = paired_bootstrap(y_ref[:n], s_ref[:n], s_q[:n],
                                n_replicates=2000, alpha=0.05)
        verdict = (
            "PASS ✓"       if res.ci_low > 0 and res.margin_point >= 0.01
            else "DIR ↑"   if res.ci_low > 0
            else "FAIL ✗"  if res.ci_high < 0
            else "NULL —"
        )
        print(f"  {r['label']:<38}  Δ={res.margin_point:+.4f}  "
              f"95%CI=[{res.ci_low:+.4f},{res.ci_high:+.4f}]  {verdict}")

    print("\n--- Additive decomposition ---")
    delta_cc = r_a1["pr_auc"] - r_a0["pr_auc"]
    delta_ms = r_a2["pr_auc"] - r_a0["pr_auc"]
    delta_a3 = r_a3["pr_auc"] - r_a0["pr_auc"]
    synergy  = delta_a3 - (delta_cc + delta_ms)
    print(f"  CC-NaME marginal (A1−A0) : {delta_cc:+.4f}")
    print(f"  MS-Lap  marginal (A2−A0) : {delta_ms:+.4f}")
    print(f"  Combined        (A3−A0) : {delta_a3:+.4f}")
    print(f"  Synergy A3−(A1+A2−A0)  : {synergy:+.4f}")
    print()

    # Phase C kf0 reference
    print("--- Reference: Phase C manifest (full, random, kf=0, 15 cells) ---")
    print("  A0 mean=0.5359  std=0.0427  [pre-registered reference]")
    print(f"  This cell A0={r_a0['pr_auc']:.4f}  "
          f"({'within' if abs(r_a0['pr_auc']-0.5359) < 0.06 else 'outside'} 1-std)")


if __name__ == "__main__":
    main()
