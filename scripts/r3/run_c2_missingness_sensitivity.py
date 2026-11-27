#!/usr/bin/env python3
"""Comment 2: Clean-Cohort Sensitivity & True 3-Way Neural Missingness Decomposition.

Part A (CPU): Evaluates all 7 detectors + A34_no-miss + 30-feature logistic mask-only
  baseline across 5 test-set subsets:
    1. full_cohort (all 42,372)
    2. excl_all_nan (-5)
    3. excl_all_zero (-2,109)
    4. excl_conflicting (-497 exact duplicates with conflicting labels)
    5. excl_all_nosignal (-2,114 -> clean 40,258 cohort)
  plus the isolated 2,114 no-signal subset alone.

Part B (GPU): Retrains MAMRD (a34_generic), A34_no_miss (a34_no_missingness), and
  HORLN on the clean 40,258-consumer cohort (keep_idx.npy; Folds 0-4 x Seeds 0-2 = 15 cells each; --folds).

Part C (GPU): True protocol-matched 3-way neural decomposition on full SGCC
  (Folds 0-4 x Seeds 0-2 = 15 cells each, identical TrainConfig; --folds):
    1. mask_only_neural: observed values set to constant 1.0, NaNs preserved -> a34_generic
    2. consumption_only_linear_interp: in-manifold [0,1] linear interpolation (0 NaNs) -> a34_no_missingness
    3. consumption_only_mean_fill: in-manifold [0,1] per-consumer observed mean fill (0 NaNs) -> a34_no_missingness
    4. combined_mamrd: loaded read-only (Folds 0-2 ORIG_RES, Folds 3-4 raw_folds34; see ref_cell_path)
"""
from __future__ import annotations
import sys, os, csv, gc, json, time, argparse, subprocess
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
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
RAW34 = R3_RES / "C1_5fold_and_stats" / "raw_folds34"
ABL34 = R3_RES / "C9_ablation_folds34"   # Folds 3-4 ablation arms (scripts/r3/run_r3_ablation_folds34.py)
FOLDS = [0, 1, 2, 3, 4]                  # overridden by --folds


def ref_cell_path(orig_pat: str, s: int, f: int) -> Path:
    """Cell file of a Revision 2 reference arm: Folds 0-2 from ORIG_RES, Folds 3-4 from the R3
    five-fold extensions (full model and baselines in raw_folds34, ablation arms in C9)."""
    if f < 3:
        return ORIG_RES / orig_pat.format(s=s, f=f)
    if orig_pat.startswith("bu/"):
        return ABL34 / orig_pat.format(s=s, f=f)
    return RAW34 / Path(orig_pat).name.format(s=s, f=f)
OUT = R3_RES / "C2_missingness_sensitivity"
CELLS_DIR = OUT / "cells"
OUT.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import topk_tie_neutral
from phased.models import build_arm
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
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


def _safe(fn, y: np.ndarray, s: np.ndarray) -> float:
    return float(fn(y, s)) if (len(y) > 0 and 0 < y.sum() < len(y)) else float("nan")


def make_mask_only_grid(X: np.ndarray) -> np.ndarray:
    """Observed positions set to constant 1.0; missing positions remain NaN."""
    return np.where(np.isnan(X), np.nan, 1.0).astype(np.float32)


def make_linear_interp_grid(X: np.ndarray) -> np.ndarray:
    """In-manifold [0,1] per-consumer linear interpolation along the 1,029-day sequence (0 NaNs remain)."""
    n, w, d = X.shape
    flat = X.reshape(n, w * d).astype(np.float32).copy()
    t_idx = np.arange(w * d, dtype=np.float64)
    for i in range(n):
        row = flat[i]
        valid = ~np.isnan(row)
        n_val = int(valid.sum())
        if n_val == 0:
            flat[i] = 0.0
        elif n_val == 1:
            flat[i] = float(row[valid][0])
        elif n_val < len(row):
            flat[i] = np.interp(t_idx, t_idx[valid], row[valid]).astype(np.float32)
    out = flat.reshape(n, w, d)
    assert not np.isnan(out).any(), "Linear interp grid still has NaNs"
    return out


def make_mean_fill_grid(X: np.ndarray) -> np.ndarray:
    """In-manifold [0,1] per-consumer observed mean fill (0 NaNs remain)."""
    n, w, d = X.shape
    flat = X.reshape(n, w * d).astype(np.float32).copy()
    valid = ~np.isnan(flat)
    cnt = valid.sum(axis=1, keepdims=True)
    sums = np.where(valid, flat, 0.0).sum(axis=1, keepdims=True)
    means = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
    out = np.where(valid, flat, np.broadcast_to(means, flat.shape)).astype(np.float32).reshape(n, w, d)
    assert not np.isnan(out).any(), "Mean-fill grid still has NaNs"
    return out


def verify_math_invariants() -> bool:
    """Verify 3-way tensor transformations leave zero leakage and exact shapes/NaN contracts."""
    rng = np.random.default_rng(42)
    X_toy = rng.uniform(0.0, 1.0, size=(6, 4, 7)).astype(np.float32)
    X_toy[0, :, :] = np.nan            # all-NaN row
    X_toy[1, :2, :] = np.nan           # partial NaN row
    X_toy[2, 1, 3] = np.nan            # isolated NaN
    X_mo = make_mask_only_grid(X_toy)
    assert np.array_equal(np.isnan(X_mo), np.isnan(X_toy))
    assert np.all(X_mo[~np.isnan(X_mo)] == 1.0)
    X_lin = make_linear_interp_grid(X_toy)
    assert not np.isnan(X_lin).any()
    assert X_lin.min() >= 0.0 and X_lin.max() <= 1.0
    assert np.allclose(X_lin[0], 0.0)
    assert np.allclose(X_lin[2, 1, 3], 0.5 * (X_toy[2, 1, 2] + X_toy[2, 1, 4]), atol=1e-5)
    X_mn = make_mean_fill_grid(X_toy)
    assert not np.isnan(X_mn).any()
    assert X_mn.min() >= 0.0 and X_mn.max() <= 1.0
    assert np.allclose(X_mn[0], 0.0)
    return True


def run_part_a_cached_exclusions(raw_arr: np.ndarray, y_all: np.ndarray, log) -> list[dict]:
    """Evaluate cached Folds 0-2 (and 0-4 if C1 present) across data-quality exclusions."""
    dropped_df = pd.read_csv(ORIG_RES / "dropped_ids.csv")
    conf_df = pd.read_csv(ORIG_RES / "conflicting_flag_ids.csv")
    keep_idx = np.load(ORIG_RES / "keep_idx.npy")

    idx_all_nan = set(dropped_df.loc[dropped_df["reason"] == "all_nan", "idx"].astype(int).tolist())
    idx_all_zero = set(dropped_df.loc[dropped_df["reason"] == "all_zero_observed", "idx"].astype(int).tolist())
    idx_nosig = set(dropped_df["idx"].astype(int).tolist())
    idx_conf = set(conf_df["idx"].astype(int).tolist())
    assert idx_conf.issubset(idx_nosig), "Expected conflicting duplicates to be subset of no-signal"
    assert len(keep_idx) + len(idx_nosig) == len(y_all)

    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_all))
    have_5fold = (RAW34 / "a34_cell_0_3.csv").exists() and (RAW34 / "horln_0_4.csv").exists() and (ABL34 / "bu" / "a34_no_missingness_2_4.csv").exists()
    f_sets = [("3_fold (0-2)", [0, 1, 2])] + ([("5_fold (0-4)", [0, 1, 2, 3, 4])] if have_5fold else [])

    # Precompute 30-feature logistic mask baseline
    X_raw_grid = raw_arr[:, :1029].reshape(len(y_all), 147, 7)
    miss_ind = np.isnan(X_raw_grid).astype(np.float32)
    f_overall = miss_ind.mean(axis=(1, 2))[:, None]
    f_dow = miss_ind.mean(axis=1)
    f_monthly = miss_ind.reshape(len(y_all), 21, 7, 7).mean(axis=(2, 3))
    n_trans = np.abs(np.diff(miss_ind.reshape(len(y_all), -1), axis=1)).mean(axis=1, keepdims=True)
    X_mask_feat = np.hstack([f_overall, f_dow, f_monthly, n_trans])

    det_specs = [
        ("MAMRD (matched)", "a34_cell_{s}_{f}.csv",             "a34_cell_{s}_{f}.csv",   range(3)),
        ("A34_no-miss",     "bu/a34_no_missingness_{s}_{f}.csv", "bu/a34_no_missingness_{s}_{f}.csv", range(3)),
        ("HORLN",           "baselines/horln_{s}_{f}.csv",      "horln_{s}_{f}.csv",      range(3)),
        ("Self-Attn",       "baselines/hybridattn_{s}_{f}.csv", "hybridattn_{s}_{f}.csv", range(3)),
        ("MSDM",            "baselines/msdm_net_{s}_{f}.csv",   "msdm_net_{s}_{f}.csv",   range(3)),
        ("WDCNN",           "baselines/wdcnn_{s}_{f}.csv",      "wdcnn_{s}_{f}.csv",      range(3)),
        ("DKGT-Net",        "baselines/dkgt_{s}_{f}.csv",       "dkgt_{s}_{f}.csv",       range(3)),
        ("GRU-D",           "baselines/grud_{s}_{f}.csv",       "grud_{s}_{f}.csv",       range(3)),
    ]

    subsets = [
        ("full_cohort (42,372)",       lambda idx: np.ones(len(idx), dtype=bool)),
        ("excl_all_nan (-5)",          lambda idx: np.array([i not in idx_all_nan for i in idx])),
        ("excl_all_zero (-2,109)",     lambda idx: np.array([i not in idx_all_zero for i in idx])),
        ("excl_conflicting (-497)",    lambda idx: np.array([i not in idx_conf for i in idx])),
        ("excl_all_nosignal (-2,114)", lambda idx: np.array([i not in idx_nosig for i in idx])),
        ("nosignal_only (2,114)",      lambda idx: np.array([i in idx_nosig for i in idx])),
    ]

    rows = []
    for fset_label, f_ids in f_sets:
        te_by_f = {}
        for f in f_ids:
            _, val = folds[f]
            _, _, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
            te_by_f[f] = te

        for det_name, orig_pat, raw34_pat, seeds in det_specs:
            if fset_label.startswith("5_fold") and raw34_pat is None:
                continue
            for sub_name, mask_fn in subsets:
                aps, rocs, ns, prevs = [], [], [], []
                for s in seeds:
                    for f in f_ids:
                        fp = ref_cell_path(orig_pat, s, f) if orig_pat.startswith("bu/") else ((ORIG_RES / orig_pat.format(s=s, f=f)) if f < 3 else (RAW34 / raw34_pat.format(s=s, f=f)))
                        r = next(csv.DictReader(open(fp)))
                        yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
                        sc = np.array(json.loads(r["val_scores_json"]), dtype=float)
                        te_idx = te_by_f[f]
                        assert np.array_equal(yt, y_all[te_idx])
                        m = mask_fn(te_idx)
                        ym, sm = yt[m], sc[m]
                        aps.append(_safe(average_precision_score, ym, sm))
                        rocs.append(_safe(roc_auc_score, ym, sm))
                        ns.append(int(m.sum()))
                        prevs.append(float(ym.mean()))
                lo, hi = bca_ci(aps)
                rows.append({
                    "fold_set": fset_label, "detector": det_name, "subset": sub_name,
                    "n_cells": len(aps), "n_test_mean": float(np.mean(ns)), "prev_pct": 100.0 * float(np.mean(prevs)),
                    "ap_mean": float(np.mean(aps)), "ap_ci_lo": lo, "ap_ci_hi": hi, "roc_mean": float(np.mean(rocs)),
                })

        # Logistic mask-only baseline on same subsets
        for sub_name, mask_fn in subsets:
            aps, rocs, ns, prevs = [], [], [], []
            for f in f_ids:
                tr, _ = folds[f]
                te_idx = te_by_f[f]
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(X_mask_feat[tr], y_all[tr])
                sc = clf.predict_proba(X_mask_feat[te_idx])[:, 1]
                m = mask_fn(te_idx)
                ym, sm = y_all[te_idx][m], sc[m]
                aps.append(_safe(average_precision_score, ym, sm))
                rocs.append(_safe(roc_auc_score, ym, sm))
                ns.append(int(m.sum()))
                prevs.append(float(ym.mean()))
            rows.append({
                "fold_set": fset_label, "detector": "Logistic-MaskOnly (30f)", "subset": sub_name,
                "n_cells": len(aps), "n_test_mean": float(np.mean(ns)), "prev_pct": 100.0 * float(np.mean(prevs)),
                "ap_mean": float(np.mean(aps)), "ap_ci_lo": float("nan"), "ap_ci_hi": float("nan"),
                "roc_mean": float(np.mean(rocs)),
            })

    df_a = pd.DataFrame(rows)
    df_a.to_csv(OUT / "c2a_cohort_exclusion_cached.csv", index=False)
    log("\n--- Part A: Cached Test-Set Cohort Exclusion Sensitivity (3-Fold) ---")
    log(df_a[df_a["fold_set"] == "3_fold (0-2)"][
        ["detector", "subset", "n_test_mean", "prev_pct", "ap_mean", "roc_mean"]
    ].to_string(index=False))
    return rows


def train_eval_cell(X_grid: np.ndarray, y: np.ndarray, g: np.ndarray, m: np.ndarray,
                    folds: list, variant_label: str, backbone: str, seed: int, fold: int,
                    epochs: int = 30, device: str = "cuda") -> dict:
    D = X_grid.shape[2]
    X_reg, rw = slice_regime(X_grid, "full")
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(fold))
    params = fit_router(g[tr], m[tr])
    seed_everything(seed)
    model = build_arm(backbone, weeks=rw, days=D)
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
        "variant": variant_label, "backbone": backbone, "seed": seed, "fold": fold,
        "wall_s": round(wall, 1), "n_test": int(len(te)), "n_pos": int(yt.sum()),
        "pr_auc": _safe(average_precision_score, yt, te_s),
        "roc_auc": _safe(roc_auc_score, yt, te_s),
        "f1": float(f1_score(yt, (te_s >= thr).astype(int), zero_division=0)),
        "map_at_100": m100, "p_at_100": p100,
        "val_labels_json": json.dumps(yt.astype(int).tolist()),
        "val_scores_json": json.dumps([round(float(v), 6) for v in te_s]),
    }
    del model, best
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_variant_9cells(X_grid: np.ndarray, y: np.ndarray, g: np.ndarray, m: np.ndarray,
                       folds: list, variant_label: str, backbone: str,
                       epochs: int, device: str, resume: bool, log) -> pd.DataFrame:
    cols = ["variant", "backbone", "seed", "fold", "wall_s", "n_test", "n_pos",
            "pr_auc", "roc_auc", "f1", "map_at_100", "p_at_100", "val_labels_json", "val_scores_json"]
    cell_rows = []
    for s in range(3):
        for f in FOLDS:
            fp = CELLS_DIR / f"{variant_label}_{s}_{f}.csv"
            if resume and fp.exists() and fp.stat().st_size > 100:
                df_c = pd.read_csv(fp)
                if len(df_c) == 1 and np.isfinite(float(df_c["pr_auc"].iloc[0])):
                    cell_rows.append(df_c.iloc[0].to_dict())
                    log(f"  [SKIP] {fp.name} PR={float(df_c['pr_auc'].iloc[0]):.4f}")
                    continue
            cool_if_hot()
            r = train_eval_cell(X_grid, y, g, m, folds, variant_label, backbone, s, f, epochs=epochs, device=device)
            with open(fp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerow(r)
            cell_rows.append(r)
            log(f"  [{variant_label} s={s} f={f}] PR={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} F1={r['f1']:.4f} wall={r['wall_s']}s")
    return pd.DataFrame(cell_rows).sort_values(["seed", "fold"]).reset_index(drop=True)


def summarise_df(df: pd.DataFrame, group_label: str) -> dict:
    aps = df["pr_auc"].astype(float).values
    rocs = df["roc_auc"].astype(float).values
    f1s = df["f1"].astype(float).values
    alo, ahi = bca_ci(aps)
    rlo, rhi = bca_ci(rocs)
    flo, fhi = bca_ci(f1s)
    return {
        "group": group_label, "variant": df["variant"].iloc[0], "n_cells": len(df),
        "ap_mean": float(np.mean(aps)), "ap_ci_lo": alo, "ap_ci_hi": ahi,
        "roc_mean": float(np.mean(rocs)), "roc_ci_lo": rlo, "roc_ci_hi": rhi,
        "f1_mean": float(np.mean(f1s)), "f1_ci_lo": flo, "f1_ci_hi": fhi,
        "map100_mean": float(df["map_at_100"].astype(float).mean()),
        "p100_mean": float(df["p_at_100"].astype(float).mean()),
    }


def paired_contrast(df_a: pd.DataFrame, df_b: pd.DataFrame, label: str) -> dict:
    a = df_a.sort_values(["seed", "fold"])["pr_auc"].astype(float).values
    b = df_b.sort_values(["seed", "fold"])["pr_auc"].astype(float).values
    diff = a - b
    lo, hi = bca_ci(diff)
    _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="exact")
    return {
        "contrast": label, "ap_a": float(a.mean()), "ap_b": float(b.mean()),
        "delta_ap": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
        "rel_pct": 100.0 * float(diff.mean() / b.mean()), "wilcoxon_p": float(p),
        "wins": int(np.sum(diff > 0)), "n_cells": len(diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="outer folds for Parts B-C (Revision 3 default: all five)")
    args = ap.parse_args()
    global FOLDS
    FOLDS = list(args.folds)

    if args.self_test:
        assert verify_math_invariants()
        print("[SELF-TEST PASS] run_c2_missingness_sensitivity invariants verified.", flush=True)
        return

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    sgcc_csv = SGCC_PATH
    df_raw = pd.read_csv(sgcc_csv)
    raw_arr = df_raw.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    X_full, y_full = load_and_preprocess(sgcc_csv)
    g_full, m_full = gap_magnitude(raw_arr[:, :1029].reshape(len(y_full), 147, 7))
    folds_full = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_full))

    if args.smoke:
        X_mo = make_mask_only_grid(X_full)
        r = train_eval_cell(X_mo, y_full, g_full, m_full, folds_full, "smoke_mo", "a34_generic", 0, 0, epochs=1, device=args.device)
        print(f"[SMOKE PASS] C2 smoke PR={r['pr_auc']:.4f} wall={r['wall_s']}s")
        return

    # PART A: Cached Cohort Exclusions (CPU)
    part_a_rows = run_part_a_cached_exclusions(raw_arr, y_full, log)

    # PART B: Clean-Cohort Retrain (keep_idx.npy = 40,258 consumers, --folds x Seeds 0-2)
    log("\n=== PART B: CLEAN-COHORT RETRAINING (N=40,258) ===")
    keep = np.load(ORIG_RES / "keep_idx.npy")
    X_clean, y_clean = X_full[keep], y_full[keep]
    g_clean, m_clean = g_full[keep], m_full[keep]
    folds_clean = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_clean))

    df_clean_mamrd = run_variant_9cells(X_clean, y_clean, g_clean, m_clean, folds_clean,
                                        "clean_mamrd", "a34_generic", args.epochs, args.device, args.resume, log)
    df_clean_nomiss = run_variant_9cells(X_clean, y_clean, g_clean, m_clean, folds_clean,
                                         "clean_a34_nomiss", "a34_no_missingness", args.epochs, args.device, args.resume, log)
    df_clean_horln = run_variant_9cells(X_clean, y_clean, g_clean, m_clean, folds_clean,
                                        "clean_horln", "horln", args.epochs, args.device, args.resume, log)

    part_b_sum = [
        summarise_df(df_clean_mamrd, "clean_cohort_40258"),
        summarise_df(df_clean_nomiss, "clean_cohort_40258"),
        summarise_df(df_clean_horln, "clean_cohort_40258"),
    ]
    pd.DataFrame(part_b_sum).to_csv(OUT / "c2b_clean_cohort_retrain.csv", index=False)

    # PART C: True Protocol-Matched 3-Way Neural Decomposition (Full 42,372 SGCC)
    log("\n=== PART C: TRUE 3-WAY NEURAL MISSINGNESS VS. CONSUMPTION DECOMPOSITION ===")
    X_mo = make_mask_only_grid(X_full)
    X_lin = make_linear_interp_grid(X_full)
    X_mean = make_mean_fill_grid(X_full)

    df_c_mask = run_variant_9cells(X_mo, y_full, g_full, m_full, folds_full,
                                   "mask_only_neural", "a34_generic", args.epochs, args.device, args.resume, log)
    df_c_lin = run_variant_9cells(X_lin, y_full, g_full, m_full, folds_full,
                                  "consumption_only_linear_interp", "a34_no_missingness", args.epochs, args.device, args.resume, log)
    df_c_mean = run_variant_9cells(X_mean, y_full, g_full, m_full, folds_full,
                                   "consumption_only_mean_fill", "a34_no_missingness", args.epochs, args.device, args.resume, log)

    # Load existing Combined MAMRD (0.597 AP) and A34_no_miss (0.570 AP) read-only for paired comparison
    def load_orig_9(pat, vname):
        r_list = []
        for s in range(3):
            for f in FOLDS:
                r = next(csv.DictReader(open(ref_cell_path(pat, s, f))))
                yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
                sc = np.array(json.loads(r["val_scores_json"]), dtype=float)
                m100, p100 = topk_tie_neutral(yt, sc, 100, n_perm=1000, seed=0)
                r_list.append({
                    "variant": vname, "seed": s, "fold": f,
                    "pr_auc": float(r["pr_auc"]), "roc_auc": float(r["roc_auc"]),
                    "f1": float(r["f1"]), "map_at_100": m100, "p_at_100": p100,
                })
        return pd.DataFrame(r_list)

    df_orig_mamrd = load_orig_9("a34_cell_{s}_{f}.csv", "combined_mamrd_ref")
    df_orig_nomiss = load_orig_9("bu/a34_no_missingness_{s}_{f}.csv", "a34_no_miss_token_ref")

    part_c_sum = [
        summarise_df(df_c_mask, "full_42372_3way"),
        summarise_df(df_c_lin, "full_42372_3way"),
        summarise_df(df_c_mean, "full_42372_3way"),
        summarise_df(df_orig_nomiss, "full_42372_3way"),
        summarise_df(df_orig_mamrd, "full_42372_3way"),
    ]
    pd.DataFrame(part_c_sum).to_csv(OUT / "c2c_three_way_neural.csv", index=False)

    contrasts = [
        paired_contrast(df_clean_mamrd, df_clean_nomiss, "CleanCohort: MAMRD - A34_no_miss"),
        paired_contrast(df_clean_mamrd, df_clean_horln, "CleanCohort: MAMRD - HORLN"),
        paired_contrast(df_orig_mamrd, df_c_lin, "3Way: Combined MAMRD - ConsumptionOnly (LinearInterp)"),
        paired_contrast(df_orig_mamrd, df_c_mean, "3Way: Combined MAMRD - ConsumptionOnly (MeanFill)"),
        paired_contrast(df_orig_mamrd, df_c_mask, "3Way: Combined MAMRD - MaskOnlyNeural"),
        paired_contrast(df_c_lin, df_c_mask, "3Way: ConsumptionOnly (LinearInterp) - MaskOnlyNeural"),
        paired_contrast(df_orig_nomiss, df_c_lin, "3Way: A34_no_miss (tau_base sentinel) - ConsumptionOnly (LinearInterp)"),
    ]
    df_con = pd.DataFrame(contrasts)
    df_con.to_csv(OUT / "c2_paired_contrasts.csv", index=False)

    log("\n--- Part B Summary ---")
    log(pd.DataFrame(part_b_sum).to_string(index=False))
    log("\n--- Part C Summary (True 3-Way Neural Comparison) ---")
    log(pd.DataFrame(part_c_sum).to_string(index=False))
    log("\n--- C2 Paired Contrasts ---")
    log(df_con.to_string(index=False))

    (OUT / "c2_summary.json").write_text(json.dumps({
        "clean_cohort_summary": part_b_sum,
        "three_way_neural_summary": part_c_sum,
        "paired_contrasts": contrasts,
    }, indent=2))
    log(f"\n[DONE] C2 completed in {(time.time() - t0)/60.0:.2f} min -> {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
