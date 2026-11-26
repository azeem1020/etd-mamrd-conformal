#!/usr/bin/env python3
"""Comment 5: Preprocessing, Laplacian Padding, & 2016-09-18 Reconstruction Sensitivity.

Evaluates 7 controlled architectural/preprocessing variants of MAMRD (a34_generic)
on SGCC Folds 0-2 x Seeds 0-2 = 9 cells each under identical TrainConfig (and exact
same train/es/cal/test indices as Table 3's 0.597 AP baseline):
  1. pad_replicate          : padding_mode="replicate" in Branches 2 & 3 Laplacian stencils
  2. pad_reflect            : padding_mode="reflect" in Branches 2 & 3 Laplacian stencils
  3. norm_zscore_user       : per-consumer 2-sigma clip + per-consumer z-score (clipped [-5, 5])
  4. norm_global_log_minmax : fold-wise TRAIN-only log1p global min-max (preserves absolute kWh scale)
  5. norm_global_quantile   : fold-wise TRAIN-only QuantileTransformer (preserves absolute kWh rank)
  6. sep18_as_nan           : keeps 1,035-day alignment intact, overwrites 2016-09-18 with NaN
  7. sep18_omitted_shifted  : drops column 2016-09-18 (1,034 -> 1,029 days, 1-day post-Sep-18 shift)
"""
from __future__ import annotations
import sys, os, csv, gc, json, time, argparse, subprocess
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
OUT = R3_RES / "C5_preprocessing_sensitivity"
CELLS_DIR = OUT / "cells"
OUT.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import topk_tie_neutral
from phased.models import build_arm
from run_local_grid import load_and_preprocess, _stratified_three_way_split, _inference_batch

N_DIST = NormalDist()


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


def set_laplacian_padding(model: nn.Module, padding_mode: str) -> None:
    """Replace padding_mode on Branches 2 & 3 Laplacian stencil Conv2d layers."""
    if padding_mode == "zeros":
        return
    for branch_attr in ("cnn_branch", "mask_cnn_branch"):
        branch = getattr(model, branch_attr, None)
        if branch is None:
            continue
        for scale_mod in branch.branches:
            conv = scale_mod.stencil
            assert isinstance(conv, nn.Conv2d)
            conv.padding_mode = padding_mode


def prep_per_user_minmax(raw_2d: np.ndarray) -> np.ndarray:
    arr = raw_2d.astype(np.float64, copy=True)
    ceil = np.nanmean(arr, axis=1, keepdims=True) + 2.0 * np.nanstd(arr, axis=1, keepdims=True)
    arr = np.where(arr > ceil, ceil, arr)
    lo, hi = np.nanmin(arr, axis=1, keepdims=True), np.nanmax(arr, axis=1, keepdims=True)
    den = hi - lo
    den[den == 0] = 1.0
    arr = ((arr - lo) / den)[:, :1029].astype(np.float32)
    return arr.reshape(len(raw_2d), 147, 7)


def prep_per_user_zscore(raw_2d: np.ndarray) -> np.ndarray:
    arr = raw_2d.astype(np.float64, copy=True)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    ceil = mu + 2.0 * sd
    arr = np.where(arr > ceil, ceil, arr)
    mu2 = np.nanmean(arr, axis=1, keepdims=True)
    sd2 = np.nanstd(arr, axis=1, keepdims=True)
    sd2_safe = np.where((sd2 > 1e-6) & np.isfinite(sd2), sd2, 1.0)
    z = np.clip((arr - mu2) / sd2_safe, -5.0, 5.0)[:, :1029].astype(np.float32)
    return z.reshape(len(raw_2d), 147, 7)


def prep_global_log_minmax_fold(raw_2d: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """Strictly train-fit global log1p min-max preserving absolute kWh magnitude across consumers."""
    arr = raw_2d[:, :1029].astype(np.float64, copy=True)
    tr_vals = arr[tr_idx]
    p99 = float(np.nanpercentile(tr_vals, 99.0))
    arr = np.where(arr > p99, p99, np.maximum(arr, 0.0))
    logv = np.log1p(arr)
    lo = float(np.nanmin(logv[tr_idx]))
    hi = float(np.nanmax(logv[tr_idx]))
    den = max(hi - lo, 1e-6)
    scaled = ((logv - lo) / den).astype(np.float32)
    return scaled.reshape(len(raw_2d), 147, 7)


def prep_global_quantile_fold(raw_2d: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    """Strictly train-fit QuantileTransformer preserving global cross-consumer rank/level."""
    arr = raw_2d[:, :1029].astype(np.float64)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", subsample=10**9, random_state=0)
    qt.fit(arr[tr_idx])
    out = qt.transform(arr).astype(np.float32)
    return out.reshape(len(raw_2d), 147, 7)


def verify_math_invariants() -> bool:
    """Verify reflect/replicate backward pass on Branches 2 & 3 and strict train-fit scalers."""
    for pm in ("replicate", "reflect"):
        m = build_arm("a34_generic", weeks=147, days=7)
        set_laplacian_padding(m, pm)
        x4d = torch.rand(4, 1, 147, 7, requires_grad=True)
        out_c = m.cnn_branch(x4d)
        out_m = m.mask_cnn_branch(x4d)
        loss = out_c.sum() + out_m.sum()
        loss.backward()
        assert torch.isfinite(out_c).all() and torch.isfinite(out_m).all(), f"Non-finite output for padding={pm}"
        assert torch.isfinite(x4d.grad).all(), f"Non-finite gradient for padding={pm}"

    # Verify global log-minmax preserves level ordering and NaNs
    rng = np.random.default_rng(0)
    raw = rng.uniform(1.0, 20.0, size=(20, 1029))
    raw[0, 10] = np.nan
    raw[1, :] = 100.0  # test consumer much larger than train -> remains larger up to p99
    X_log = prep_global_log_minmax_fold(raw, np.arange(15))
    assert np.isnan(X_log[0, 1, 3])
    assert X_log[1].mean() >= X_log[2].mean()
    X_z = prep_per_user_zscore(raw)
    assert np.isnan(X_z[0, 1, 3])
    assert abs(float(np.nanmean(X_z[2]))) < 1e-4
    return True


def run_cell_variant(X_all: np.ndarray | None, raw_2d: np.ndarray, y: np.ndarray, folds: list,
                     variant: str, pad_mode: str, fold_prep_fn, seed: int, fold: int,
                     epochs: int = 30, device: str = "cuda") -> dict:
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(fold))
    X_use = X_all if X_all is not None else fold_prep_fn(raw_2d, tr)
    X_reg, rw = slice_regime(X_use, "full")
    D = X_use.shape[2]

    seed_everything(seed)
    model = build_arm("a34_generic", weeks=rw, days=D)
    set_laplacian_padding(model, pad_mode)
    cfg = TrainConfig(
        epochs=epochs, patience=10, batch_size=128, eval_batch_size=256,
        pos_fraction=0.5, optimizer="AdamW", lr=3e-4, weight_decay=1e-4,
        warmup_frac=0.05, min_lr_ratio=0.01, eval_every=3, grad_clip_norm=1.0,
        device=device, loss_type="bce",
    )
    t0 = time.time()
    _, best = train(model, WeeklyGridDataset(X_reg[tr], y[tr]), WeeklyGridDataset(X_reg[es], y[es]), cfg)
    wall = time.time() - t0
    dev = torch.device(device)
    cal_s = _inference_batch(best, X_reg[cal], dev)
    te_s = _inference_batch(best, X_reg[te], dev)
    thr = _pick_f1_threshold(y[cal], cal_s)
    yt = y[te]
    m100, p100 = topk_tie_neutral(yt, te_s, 100, n_perm=1000, seed=0)
    row = {
        "variant": variant, "seed": seed, "fold": fold,
        "wall_s": round(wall, 1), "n_test": int(len(te)), "n_pos": int(yt.sum()),
        "pr_auc": float(average_precision_score(yt, te_s)),
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
        print("[SELF-TEST PASS] run_c5_preprocessing_sensitivity invariants verified.", flush=True)
        return

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    sgcc_csv = SGCC_PATH
    df = pd.read_csv(sgcc_csv)
    y = df["FLAG"].values.astype(np.int64)
    date_cols = list(df.columns[2:])
    assert "2016-09-18" in date_cols
    raw_2d = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))

    X_base, y_chk = load_and_preprocess(sgcc_csv)
    assert np.array_equal(y, y_chk)

    if args.smoke:
        r = run_cell_variant(X_base, raw_2d, y, folds, "smoke_reflect", "reflect", None, 0, 0, epochs=1, device=args.device)
        print(f"[SMOKE PASS] C5 reflect PR={r['pr_auc']:.4f} wall={r['wall_s']}s")
        return

    # Precompute static arrays for static variants
    X_zscore = prep_per_user_zscore(raw_2d)

    raw_sep18_nan = raw_2d.copy()
    sep18_pos = date_cols.index("2016-09-18")
    raw_sep18_nan[:, sep18_pos] = np.nan
    X_sep18_nan = prep_per_user_minmax(raw_sep18_nan)

    raw_sep18_omit = np.delete(raw_2d, sep18_pos, axis=1)  # 1,034 columns
    X_sep18_omit = prep_per_user_minmax(raw_sep18_omit)

    variants = [
        ("pad_replicate",          X_base,        "replicate", None),
        ("pad_reflect",            X_base,        "reflect",   None),
        ("norm_zscore_user",       X_zscore,      "zeros",     None),
        ("norm_global_log_minmax", None,          "zeros",     prep_global_log_minmax_fold),
        ("norm_global_quantile",   None,          "zeros",     prep_global_quantile_fold),
        ("sep18_as_nan",           X_sep18_nan,   "zeros",     None),
        ("sep18_omitted_shifted",  X_sep18_omit,  "zeros",     None),
    ]

    cols = ["variant", "seed", "fold", "wall_s", "n_test", "n_pos",
            "pr_auc", "roc_auc", "f1", "map_at_100", "p_at_100", "val_labels_json", "val_scores_json"]

    # Cache global-norm per fold so folds 0,1,2 aren't recomputed 3x across seeds
    fold_cache = {}
    all_dfs = {}

    for vname, X_stat, pad_mode, fn_prep in variants:
        log(f"\n=== Running C5 Variant: {vname} (pad={pad_mode}) ===")
        v_rows = []
        for s in range(3):
            for f in range(3):
                fp = CELLS_DIR / f"{vname}_{s}_{f}.csv"
                if args.resume and fp.exists() and fp.stat().st_size > 100:
                    df_c = pd.read_csv(fp)
                    if len(df_c) == 1 and np.isfinite(float(df_c["pr_auc"].iloc[0])):
                        v_rows.append(df_c.iloc[0].to_dict())
                        log(f"  [SKIP] {fp.name} PR={float(df_c['pr_auc'].iloc[0]):.4f}")
                        continue
                X_fold_ready = X_stat
                if X_stat is None:
                    if (vname, f) not in fold_cache:
                        tr, _ = folds[f]
                        fold_cache[(vname, f)] = fn_prep(raw_2d, tr)
                    X_fold_ready = fold_cache[(vname, f)]
                cool_if_hot()
                r = run_cell_variant(X_fold_ready, raw_2d, y, folds, vname, pad_mode, fn_prep, s, f,
                                     epochs=args.epochs, device=args.device)
                with open(fp, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerow(r)
                v_rows.append(r)
                log(f"  [{vname} s={s} f={f}] PR={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} F1={r['f1']:.4f} wall={r['wall_s']}s")
        all_dfs[vname] = pd.DataFrame(v_rows).sort_values(["seed", "fold"]).reset_index(drop=True)

    # Load Baseline MAMRD (9-cell) read-only for paired contrasts
    base_rows = []
    for s in range(3):
        for f in range(3):
            r = next(csv.DictReader(open(ORIG_RES / f"a34_cell_{s}_{f}.csv")))
            yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
            sc = np.array(json.loads(r["val_scores_json"]), dtype=float)
            m100, p100 = topk_tie_neutral(yt, sc, 100, n_perm=1000, seed=0)
            base_rows.append({
                "variant": "baseline_mamrd", "seed": s, "fold": f,
                "pr_auc": float(r["pr_auc"]), "roc_auc": float(r["roc_auc"]),
                "f1": float(r["f1"]), "map_at_100": m100, "p_at_100": p100,
            })
    df_base = pd.DataFrame(base_rows).sort_values(["seed", "fold"]).reset_index(drop=True)

    sum_rows, con_rows = [], []
    for vname, dfv in [("baseline_mamrd", df_base)] + list(all_dfs.items()):
        aps = dfv["pr_auc"].astype(float).values
        rocs = dfv["roc_auc"].astype(float).values
        f1s = dfv["f1"].astype(float).values
        alo, ahi = bca_ci(aps); rlo, rhi = bca_ci(rocs); flo, fhi = bca_ci(f1s)
        sum_rows.append({
            "variant": vname, "n_cells": len(dfv),
            "ap_mean": float(aps.mean()), "ap_ci_lo": alo, "ap_ci_hi": ahi,
            "roc_mean": float(rocs.mean()), "roc_ci_lo": rlo, "roc_ci_hi": rhi,
            "f1_mean": float(f1s.mean()), "f1_ci_lo": flo, "f1_ci_hi": fhi,
            "map100_mean": float(dfv["map_at_100"].astype(float).mean()),
            "p100_mean": float(dfv["p_at_100"].astype(float).mean()),
        })
        if vname != "baseline_mamrd":
            diff = aps - df_base["pr_auc"].astype(float).values
            dlo, dhi = bca_ci(diff)
            _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="exact")
            con_rows.append({
                "variant": vname, "ap_variant": float(aps.mean()), "ap_baseline": float(df_base["pr_auc"].mean()),
                "delta_ap": float(diff.mean()), "ci_lo": dlo, "ci_hi": dhi,
                "rel_pct": 100.0 * float(diff.mean() / df_base["pr_auc"].mean()),
                "wilcoxon_p": float(p), "wins_vs_base": int(np.sum(diff > 0)),
            })

    df_sum = pd.DataFrame(sum_rows)
    df_con = pd.DataFrame(con_rows)
    df_sum.to_csv(OUT / "c5_sensitivity_summary.csv", index=False)
    df_con.to_csv(OUT / "c5_paired_contrasts.csv", index=False)
    log("\n--- C5 Preprocessing, Padding & Sep-18 Sensitivity Summary ---")
    log(df_sum.to_string(index=False))
    log("\n--- C5 Paired Contrasts (Variant - Baseline MAMRD) ---")
    log(df_con.to_string(index=False))

    (OUT / "c5_summary.json").write_text(json.dumps({"summary": sum_rows, "contrasts": con_rows}, indent=2))
    log(f"\n[DONE] C5 completed in {(time.time() - t0)/60.0:.2f} min -> {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
