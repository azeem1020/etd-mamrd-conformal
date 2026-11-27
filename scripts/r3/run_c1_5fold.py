#!/usr/bin/env python3
"""Comment 1: Complete 5-Fold SGCC Evaluation (Folds 3 & 4) + Single-Model & Ensemble Consumer Bootstrap.

1. Trains ONLY missing Folds 3 & 4 on the RTX 5070 Ti GPU under the exact same protocol
   as run_n_clean.py (SGCC full 42,372 cohort, KFoldSpec(n_folds=5, seed=0),
   TrainConfig(epochs=30, lr=3e-4, wd=1e-4, AdamW, bs=128)):
     - MAMRD (a34_generic): seeds 0..4 x folds 3..4 (10 cells)
     - 6 baselines (horln, hybridattn, msdm_net, wdcnn, dkgt, grud): seeds 0..2 x folds 3..4 (36 cells)
   Loads Folds 0..2 strictly read-only from results/phase_n_retrained/.
2. Verifies 100% customer disjointness across all 5 test folds (0..4).
3. Computes full 5-fold Table 3 metrics, paired tests, BOTH seed-ensemble and single-model
   consumer-level bootstraps (3-fold vs 5-fold), 5-fold Mondrian conformal audit, and
   5-fold constrained positive calibration.
"""
from __future__ import annotations
import sys, os, csv, gc, json, time, argparse, subprocess
from pathlib import Path
from collections import defaultdict
from statistics import NormalDist
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import friedmanchisquare, wilcoxon
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
OUT = R3_RES / "C1_5fold_and_stats"
RAW34 = OUT / "raw_folds34"
OUT.mkdir(parents=True, exist_ok=True)
RAW34.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec, slice_regime
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import topk_tie_neutral
from phased.models import build_arm
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from run_local_grid import load_and_preprocess, _stratified_three_way_split, _inference_batch
from conformal_common import (
    NREP, resplit_rng, stratified_half, cc_recall_threshold, cell_summary, cell_bca,
)

N_DIST = NormalDist()
TRAIN_F_IDS = [3, 4]
ALL_F_IDS = [0, 1, 2, 3, 4]
DETECTORS = [
    ("GRU-D",           "grud",       "baselines/grud_{s}_{f}.csv",       range(3)),
    ("Self-Attn",       "hybridattn", "baselines/hybridattn_{s}_{f}.csv", range(3)),
    ("DKGT-Net",        "dkgt",       "baselines/dkgt_{s}_{f}.csv",       range(3)),
    ("WDCNN",           "wdcnn",      "baselines/wdcnn_{s}_{f}.csv",      range(3)),
    ("MSDM",            "msdm_net",   "baselines/msdm_net_{s}_{f}.csv",   range(3)),
    ("HORLN",           "horln",      "baselines/horln_{s}_{f}.csv",      range(3)),
    ("MAMRD (matched)", "a34_cell",   "a34_cell_{s}_{f}.csv",             range(3)),
    ("MAMRD (5-seed)",  "a34_cell",   "a34_cell_{s}_{f}.csv",             range(5)),
]


def gpu_temp() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


def cool_if_hot(limit: int = 82, target: int = 68) -> None:
    t = gpu_temp()
    if t <= limit:
        return
    print(f"  [heat] GPU at {t}C > {limit}C; cooling to {target}C...", flush=True)
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


def _pak(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = min(k, len(y))
    if k == 0:
        return float("nan")
    order = np.argsort(-s)[:k]
    return float(y[order].sum() / k)


def _safe(fn, y: np.ndarray, s: np.ndarray) -> float:
    return float(fn(y, s)) if (len(y) > 0 and 0 < y.sum() < len(y)) else float("nan")


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


def get_cell_path(prefix: str, s: int, f: int, orig_pat: str) -> Path:
    if f in (0, 1, 2):
        return ORIG_RES / orig_pat.format(s=s, f=f)
    return RAW34 / f"{prefix}_{s}_{f}.csv"


def run_cell(X_reg: np.ndarray, rw: int, D: int, y: np.ndarray, g: np.ndarray, m: np.ndarray,
             folds: list, backbone: str, seed: int, fold: int, epochs: int = 30, device: str = "cuda") -> dict:
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(fold))
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
    ph = assign(g[te], m[te], params)
    yt = y[te]

    row = dict(arm=backbone, seed=seed, fold=fold, kf=0, thr=round(thr, 4),
               wall_s=round(wall, 1), n_test=len(te),
               pr_auc=_safe(average_precision_score, yt, te_s),
               roc_auc=_safe(roc_auc_score, yt, te_s),
               f1=float(f1_score(yt, (te_s >= thr).astype(int), zero_division=0)))
    for k in (100, 200, 500):
        row[f"p_at_{k}"] = _pak(yt, te_s, k)
    for pname in PHENOTYPES:
        msk = ph == pname
        row[f"pr_{pname}"] = _safe(average_precision_score, yt[msk], te_s[msk])
        row[f"roc_{pname}"] = _safe(roc_auc_score, yt[msk], te_s[msk])
        row[f"f1_{pname}"] = float(f1_score(yt[msk], (te_s[msk] >= thr).astype(int), zero_division=0)) if msk.sum() > 0 else float("nan")
        row[f"n_{pname}"] = int(msk.sum())
    row["val_labels_json"] = json.dumps(yt.astype(int).tolist())
    row["val_scores_json"] = json.dumps([round(float(v), 6) for v in te_s])
    del model, best
    gc.collect()
    torch.cuda.empty_cache()
    return row


def load_cell_csv(fp: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    r = next(csv.DictReader(open(fp)))
    return r, np.array(json.loads(r["val_labels_json"]), dtype=int), np.array(json.loads(r["val_scores_json"]), dtype=float)


def collapse_grp(ph: np.ndarray) -> np.ndarray:
    out = np.empty(len(ph), dtype=object)
    for i, p in enumerate(ph):
        if p in ("Heavy_Riser", "Heavy_Dropper"):
            out[i] = "Heavy"
        elif p in ("Light_Riser", "Light_Dropper"):
            out[i] = "Light"
        else:
            out[i] = "Flat"
    return out


def compute_all_stats(y_all: np.ndarray, g_all: np.ndarray, m_all: np.ndarray, folds: list, log) -> None:
    cols = (["arm", "seed", "fold", "kf", "thr", "wall_s", "n_test",
             "pr_auc", "roc_auc", "f1", "p_at_100", "p_at_200", "p_at_500"]
            + [f"{x}_{p}" for p in PHENOTYPES for x in ("pr", "roc", "f1", "n")]
            + ["val_labels_json", "val_scores_json"])

    # Verify 5-fold disjointness of te across Folds 0..4
    te_sets = []
    fold_grps = {}
    for f in ALL_F_IDS:
        tr, val = folds[f]
        _, _, te = _stratified_three_way_split(val, y_all, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        te_sets.append(set(te.tolist()))
        fold_grps[f] = collapse_grp(assign(g_all[te], m_all[te], fit_router(g_all[tr], m_all[tr])))
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(te_sets[i] & te_sets[j]) == 0, f"Test fold overlap between {i} and {j}"
    total_unique_te = sum(len(s) for s in te_sets)
    total_unique_pos = sum(int(y_all[list(te_sets[f])].sum()) for f in ALL_F_IDS)
    log(f"[DISJOINT PASS] All 5 test folds pairwise disjoint: {total_unique_te} unique test consumers, "
        f"{total_unique_pos} unique test thefts ({100*total_unique_pos/total_unique_te:.2f}%)")

    # Load all predictions for Folds 0..4
    preds = {}
    meta_rows = {}
    for det_label, prefix, orig_pat, seeds in DETECTORS:
        for s in seeds:
            for f in ALL_F_IDS:
                fp = get_cell_path(prefix, s, f, orig_pat)
                r, yt, sc = load_cell_csv(fp)
                preds[(det_label, s, f)] = (yt, sc)
                meta_rows[(det_label, s, f)] = r

    # 1. Table 3 5-Fold Main Results
    t3_rows = []
    matched_ap_by_det = {}
    for det_label, prefix, orig_pat, seeds in DETECTORS:
        aps, rocs, f1s, maps, p100s = [], [], [], [], []
        for s in seeds:
            for f in ALL_F_IDS:
                r = meta_rows[(det_label, s, f)]
                yt, sc = preds[(det_label, s, f)]
                ap_v = float(r["pr_auc"])
                roc_v = float(r["roc_auc"])
                f1_v = float(r["f1"])
                m100, p100 = topk_tie_neutral(yt, sc, 100, n_perm=1000, seed=0)
                aps.append(ap_v); rocs.append(roc_v); f1s.append(f1_v); maps.append(m100); p100s.append(p100)
        if det_label != "MAMRD (5-seed)":
            matched_ap_by_det[det_label] = aps
        ap_lo, ap_hi = bca_ci(aps)
        roc_lo, roc_hi = bca_ci(rocs)
        f1_lo, f1_hi = bca_ci(f1s)
        t3_rows.append({
            "detector": det_label, "n_cells": len(aps), "n_folds": 5, "n_seeds": len(list(seeds)),
            "ap_mean": float(np.mean(aps)), "ap_ci_lo": ap_lo, "ap_ci_hi": ap_hi,
            "roc_mean": float(np.mean(rocs)), "roc_ci_lo": roc_lo, "roc_ci_hi": roc_hi,
            "f1_mean": float(np.mean(f1s)), "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "map100_mean": float(np.mean(maps)), "p100_mean": float(np.mean(p100s)),
        })
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(OUT / "table3_5fold_main_results.csv", index=False)
    log("\n--- 5-Fold SGCC Main Results (Folds 0-4) ---")
    log(df_t3.to_string(index=False))

    # Paired tests across 15 matched cells (MAMRD vs HORLN + Friedman)
    d_ap = np.array(matched_ap_by_det["MAMRD (matched)"]) - np.array(matched_ap_by_det["HORLN"])
    d_lo, d_hi = bca_ci(d_ap)
    w_stat, w_p = wilcoxon(d_ap, zero_method="wilcox", alternative="two-sided", method="exact")
    fr_stat, fr_p = friedmanchisquare(
        matched_ap_by_det["GRU-D"], matched_ap_by_det["Self-Attn"], matched_ap_by_det["DKGT-Net"],
        matched_ap_by_det["WDCNN"], matched_ap_by_det["MSDM"], matched_ap_by_det["HORLN"],
        matched_ap_by_det["MAMRD (matched)"],
    )
    n_wins = int(np.sum(d_ap > 0))
    log(f"\n[15-Cell Paired MAMRD - HORLN] dAP = {d_ap.mean():+.4f} [{d_lo:+.4f}, {d_hi:+.4f}], "
        f"Wilcoxon p = {w_p:.6e}, MAMRD>HORLN in {n_wins}/15 cells | Friedman chi2={fr_stat:.2f}, p={fr_p:.3e}")

    # 2. Consumer-level Bootstraps (B = 1,000, seed 2026) on 3-Fold vs 5-Fold:
    #    BOTH Seed-Ensemble AND Single-Model (per-seed + seed-averaged paired dAP)
    cb_rows = []
    B_CONS = 1000
    BOOT_SEED = 2026
    matched_names = [d[0] for d in DETECTORS if d[0] != "MAMRD (5-seed)"]

    for label_win, f_ids in [("3_fold (Folds 0-2)", [0, 1, 2]), ("5_fold (Folds 0-4)", [0, 1, 2, 3, 4])]:
        nf = len(f_ids)
        # (a) Seed-ensemble consumer bootstrap (matches compute_consumer_bootstrap.py exactly)
        rng = np.random.default_rng(BOOT_SEED)
        ens_scores = {}
        for f in f_ids:
            yv = preds[("HORLN", 0, f)][0]
            for d in matched_names:
                ens_scores[(d, f)] = (yv, np.mean([preds[(d, s, f)][1] for s in range(3)], axis=0))

        ens_boot_ap = {d: np.zeros(B_CONS) for d in matched_names}
        ens_boot_roc = {d: np.zeros(B_CONS) for d in matched_names}
        for f in f_ids:
            yv = ens_scores[("HORLN", f)][0]
            n_te = len(yv)
            for b in range(B_CONS):
                idx = rng.integers(0, n_te, n_te)
                yb = yv[idx]
                for d in matched_names:
                    sb = ens_scores[(d, f)][1][idx]
                    ens_boot_ap[d][b] += average_precision_score(yb, sb) / nf
                    ens_boot_roc[d][b] += roc_auc_score(yb, sb) / nf

        for d in matched_names:
            pt_ap = float(np.mean([average_precision_score(*ens_scores[(d, f)]) for f in f_ids]))
            lo, hi = np.percentile(ens_boot_ap[d], [2.5, 97.5])
            pt_roc = float(np.mean([roc_auc_score(*ens_scores[(d, f)]) for f in f_ids]))
            rlo, rhi = np.percentile(ens_boot_roc[d], [2.5, 97.5])
            cb_rows.append({
                "fold_set": label_win, "mode": "seed_ensemble", "detector_or_contrast": d,
                "point_ap": pt_ap, "ci_lo": float(lo), "ci_hi": float(hi),
                "point_roc": pt_roc, "roc_ci_lo": float(rlo), "roc_ci_hi": float(rhi),
                "n_rep_le_0": float("nan"), "p_one_sided_boot": float("nan"),
            })

        diff_ens = ens_boot_ap["MAMRD (matched)"] - ens_boot_ap["HORLN"]
        pt_diff_ens = float(np.mean([
            average_precision_score(*ens_scores[("MAMRD (matched)", f)]) - average_precision_score(*ens_scores[("HORLN", f)])
            for f in f_ids
        ]))
        lo, hi = np.percentile(diff_ens, [2.5, 97.5])
        n_le0 = int(np.sum(diff_ens <= 0.0))
        cb_rows.append({
            "fold_set": label_win, "mode": "seed_ensemble", "detector_or_contrast": "PAIRED_MAMRD_minus_HORLN",
            "point_ap": pt_diff_ens, "ci_lo": float(lo), "ci_hi": float(hi),
            "point_roc": float("nan"), "roc_ci_lo": float("nan"), "roc_ci_hi": float("nan"),
            "n_rep_le_0": n_le0, "p_one_sided_boot": n_le0 / B_CONS,
        })

        # (b) Single-model consumer-level bootstrap (keeping single seeds un-ensembled!)
        rng_s = np.random.default_rng(BOOT_SEED)
        single_boot_mamrd = np.zeros((3, B_CONS))
        single_boot_horln = np.zeros((3, B_CONS))
        for f in f_ids:
            yv = preds[("HORLN", 0, f)][0]
            n_te = len(yv)
            for b in range(B_CONS):
                idx = rng_s.integers(0, n_te, n_te)
                yb = yv[idx]
                for s in range(3):
                    single_boot_mamrd[s, b] += average_precision_score(yb, preds[("MAMRD (matched)", s, f)][1][idx]) / nf
                    single_boot_horln[s, b] += average_precision_score(yb, preds[("HORLN", s, f)][1][idx]) / nf

        for s in range(3):
            diff_s = single_boot_mamrd[s] - single_boot_horln[s]
            pt_s = float(np.mean([
                average_precision_score(*preds[("MAMRD (matched)", s, f)]) - average_precision_score(*preds[("HORLN", s, f)])
                for f in f_ids
            ]))
            lo, hi = np.percentile(diff_s, [2.5, 97.5])
            n0 = int(np.sum(diff_s <= 0.0))
            cb_rows.append({
                "fold_set": label_win, "mode": f"single_model_seed_{s}", "detector_or_contrast": "PAIRED_MAMRD_minus_HORLN",
                "point_ap": pt_s, "ci_lo": float(lo), "ci_hi": float(hi),
                "point_roc": float("nan"), "roc_ci_lo": float("nan"), "roc_ci_hi": float("nan"),
                "n_rep_le_0": n0, "p_one_sided_boot": n0 / B_CONS,
            })

        # Average across the 3 un-ensembled seeds inside each consumer bootstrap replicate
        diff_single_avg = np.mean(single_boot_mamrd - single_boot_horln, axis=0)
        pt_single_avg = float(np.mean([
            average_precision_score(*preds[("MAMRD (matched)", s, f)]) - average_precision_score(*preds[("HORLN", s, f)])
            for s in range(3) for f in f_ids
        ]))
        lo, hi = np.percentile(diff_single_avg, [2.5, 97.5])
        n0 = int(np.sum(diff_single_avg <= 0.0))
        cb_rows.append({
            "fold_set": label_win, "mode": "single_model_3seed_mean", "detector_or_contrast": "PAIRED_MAMRD_minus_HORLN",
            "point_ap": pt_single_avg, "ci_lo": float(lo), "ci_hi": float(hi),
            "point_roc": float("nan"), "roc_ci_lo": float("nan"), "roc_ci_hi": float("nan"),
            "n_rep_le_0": n0, "p_one_sided_boot": n0 / B_CONS,
        })

    df_cb = pd.DataFrame(cb_rows)
    df_cb.to_csv(OUT / "consumer_bootstrap_3fold_vs_5fold.csv", index=False)
    log("\n--- Consumer-Level Bootstrap: 3-Fold vs. 5-Fold (Seed-Ensemble & Single-Model) ---")
    log(df_cb[df_cb["detector_or_contrast"] == "PAIRED_MAMRD_minus_HORLN"][
        ["fold_set", "mode", "point_ap", "ci_lo", "ci_hi", "n_rep_le_0", "p_one_sided_boot"]
    ].to_string(index=False))

    # 3. 5-Fold Conformal Audit (alpha = 0.20)
    base_rates = {}
    for gn in ("Heavy", "Light", "Flat"):
        pos = sum(int(((fold_grps[f] == gn) & (preds[("HORLN", 0, f)][0] == 1)).sum()) for f in ALL_F_IDS)
        tot = sum(int((fold_grps[f] == gn).sum()) for f in ALL_F_IDS)
        base_rates[gn] = pos / tot

    c6_arms = [
        ("MAMRD (5-seed)", "a34_cell",   "a34_cell_{s}_{f}.csv",             range(5)),
        ("HORLN",          "horln",      "baselines/horln_{s}_{f}.csv",      range(3)),
        ("MSDM",           "msdm_net",   "baselines/msdm_net_{s}_{f}.csv",   range(3)),
        ("WDCNN",          "wdcnn",      "baselines/wdcnn_{s}_{f}.csv",      range(3)),
        ("GRU-D",          "grud",       "baselines/grud_{s}_{f}.csv",       range(3)),
        ("Self-Attn",      "hybridattn", "baselines/hybridattn_{s}_{f}.csv", range(3)),
        ("DKGT-Net",       "dkgt",       "baselines/dkgt_{s}_{f}.csv",       range(3)),
    ]
    conf_rows = []
    for arm_idx, (det_label, prefix, orig_pat, seeds) in enumerate(c6_arms):
        cond_c, sing_c, red_c, hh_c, fh_c, fr_c, fv_c = [], [], [], [], [], [], []
        for s in seeds:
            for f in ALL_F_IDS:
                yt, sc = preds[(det_label if det_label != "MAMRD (5-seed)" else "MAMRD (5-seed)", s, f)]
                grp = fold_grps[f]
                ca, sa, ra, hha, fha, fra, fva = [], [], [], [], [], [], []
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(arm_idx, s, f, rep))
                    yc, sc_c, gc = yt[ci], sc[ci], grp[ci]
                    ye, sc_e, ge = yt[ti], sc[ti], grp[ti]
                    thrs = {gn: cc_recall_threshold(sc_c[gc == gn], yc[gc == gn], 0.20)[0] for gn in ("Heavy", "Light", "Flat")}
                    sthr = min(thrs.values())
                    cf = np.zeros(len(ye), dtype=bool)
                    for gn in ("Heavy", "Light", "Flat"):
                        me = ge == gn
                        fg = sc_e[me] >= thrs[gn]
                        cf[me] = fg
                        if gn == "Heavy" and fg.any():
                            hha.append(float(ye[me][fg].mean()))
                        if gn == "Flat":
                            if fg.any():
                                fha.append(float(ye[me][fg].mean()))
                            if (ye[me] == 1).any():
                                rflat = float(fg[ye[me] == 1].mean())
                                fra.append(rflat)
                                fva.append(1.0 if rflat < 0.80 else 0.0)
                    sf = sc_e >= sthr
                    c_f, s_f = float(cf.mean()), float(sf.mean())
                    ca.append(c_f); sa.append(s_f)
                    ra.append(1.0 - (c_f / s_f if s_f > 0 else 1.0))
                cond_c.append(ca); sing_c.append(sa); red_c.append(ra)
                hh_c.append(hha); fh_c.append(fha); fr_c.append(fra); fv_c.append(fva)
        c_m, _, _, _ = cell_summary(cond_c)
        s_m, _, _, _ = cell_summary(sing_c)
        r_m, _, _, _ = cell_summary(red_c)
        r_lo, r_hi = cell_bca(red_c, B=2000, seed=0)
        hh_m, _, _, _ = cell_summary(hh_c)
        fh_m, _, _, _ = cell_summary(fh_c)
        fr_m, _, _, _ = cell_summary(fr_c)
        fv_m, _, _, _ = cell_summary(fv_c)
        conf_rows.append({
            "detector": det_label, "n_cells": len(list(seeds)) * 5,
            "heavy_hit_pct": 100.0 * hh_m, "heavy_lift": hh_m / base_rates["Heavy"],
            "flat_hit_pct": 100.0 * fh_m, "flat_lift": fh_m / base_rates["Flat"],
            "cond_insp_pct": 100.0 * c_m, "single_insp_pct": 100.0 * s_m,
            "workload_red_pct": 100.0 * r_m, "workload_red_ci_lo": 100.0 * r_lo, "workload_red_ci_hi": 100.0 * r_hi,
            "flat_recall_pct": 100.0 * fr_m, "flat_viol_pct": 100.0 * fv_m,
        })
    df_conf = pd.DataFrame(conf_rows)
    df_conf.to_csv(OUT / "table6_5fold_conformal.csv", index=False)
    log("\n--- 5-Fold SGCC Mondrian Conformal Audit (@ 80% target) ---")
    log(df_conf.to_string(index=False))

    # 4. 5-Fold Constrained Positive Calibration
    def ece_bin(p, y):
        edges = np.linspace(0.0, 1.0, 11)
        e = 0.0
        for i in range(10):
            m_ = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
            if m_.sum() > 0:
                e += (m_.sum() / len(p)) * abs(p[m_].mean() - y[m_].mean())
        return float(e)

    cal_rows = []
    for det_label, prefix, orig_pat, seeds in DETECTORS:
        er_l, ea_l, br_l, ba_l, w_l, b_l = [], [], [], [], [], []
        for s in seeds:
            for f in ALL_F_IDS:
                y_v, sc = preds[(det_label, s, f)]
                sc = np.clip(sc, 1e-6, 1.0 - 1e-6)
                ci, ti = np.arange(0, len(y_v), 2), np.arange(1, len(y_v), 2)
                zc = np.log(sc[ci] / (1.0 - sc[ci])); ze = np.log(sc[ti] / (1.0 - sc[ti]))
                yc, ye = y_v[ci], y_v[ti]
                n_cal = float(len(yc))
                def fn(th):
                    w = np.exp(np.clip(th[0], -20.0, 20.0)); lin = w * zc + th[1]
                    nll = (np.sum(np.logaddexp(0.0, lin) - yc * lin) + 0.5 * (w ** 2) / 1e4) / n_cal
                    r_ = expit(lin) - yc
                    return float(nll), np.array([((float(np.sum(r_ * zc)) + w / 1e4) * w) / n_cal, float(np.sum(r_)) / n_cal])
                res = minimize(fn, np.array([0.0, 0.0]), method="BFGS", jac=True, options={"gtol": 1e-9, "maxiter": 500})
                w_hat, b_hat = float(np.exp(res.x[0])), float(res.x[1])
                if not res.success:
                    def fn_box(wb):
                        w, b = float(wb[0]), float(wb[1]); lin = w * zc + b
                        nll = (np.sum(np.logaddexp(0.0, lin) - yc * lin) + 0.5 * (w ** 2) / 1e4) / n_cal
                        r_ = expit(lin) - yc
                        return float(nll), np.array([(float(np.sum(r_ * zc)) + w / 1e4) / n_cal, float(np.sum(r_)) / n_cal])
                    res_lb = minimize(fn_box, np.array([1.0, 0.0]), method="L-BFGS-B", jac=True, bounds=[(1e-8, None), (None, None)], options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500})
                    if res_lb.success:
                        w_hat, b_hat = float(res_lb.x[0]), float(res_lb.x[1])
                pa = expit(w_hat * ze + b_hat)
                er_l.append(ece_bin(sc[ti], ye)); ea_l.append(ece_bin(pa, ye))
                br_l.append(float(np.mean((sc[ti] - ye) ** 2))); ba_l.append(float(np.mean((pa - ye) ** 2)))
                w_l.append(w_hat); b_l.append(b_hat)
        cal_rows.append({
            "detector": det_label, "n_cells": len(er_l),
            "ece_raw": float(np.mean(er_l)), "ece_cal": float(np.mean(ea_l)),
            "brier_raw": float(np.mean(br_l)), "brier_cal": float(np.mean(ba_l)),
            "w_mean": float(np.mean(w_l)), "w_min": float(np.min(w_l)), "b_mean": float(np.mean(b_l)),
        })
    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(OUT / "c8_5fold_calibration.csv", index=False)

    summary = {
        "total_unique_test_consumers": total_unique_te,
        "total_unique_test_positives": total_unique_pos,
        "paired_15cell_mamrd_minus_horln": {
            "dAP": float(d_ap.mean()), "ci_lo": d_lo, "ci_hi": d_hi,
            "wilcoxon_p": float(w_p), "wins": n_wins, "friedman_chi2": float(fr_stat), "friedman_p": float(fr_p),
        },
        "table3_5fold": t3_rows,
        "consumer_bootstrap_contrasts": [r for r in cb_rows if r["detector_or_contrast"] == "PAIRED_MAMRD_minus_HORLN"],
        "table6_5fold_conformal": conf_rows,
    }
    (OUT / "c1_summary.json").write_text(json.dumps(summary, indent=2))


def verify_math_invariants() -> bool:
    """Verify fold disjointness, router cutoff leakage firewall, and BCa CI helper."""
    rng = np.random.default_rng(0)
    y_dummy = np.array([0] * 90 + [1] * 10)
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_dummy))
    te_sets = []
    for f in range(5):
        tr, val = folds[f]
        es, cal, te = _stratified_three_way_split(val, y_dummy, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        assert len(set(tr) & set(te)) == 0 and len(set(es) & set(te)) == 0 and len(set(cal) & set(te)) == 0
        te_sets.append(set(te.tolist()))
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(te_sets[i] & te_sets[j]) == 0
    lo, hi = bca_ci([0.5, 0.55, 0.6, 0.58, 0.62], B=200, seed=0)
    assert lo < hi
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    if args.self_test:
        assert verify_math_invariants()
        print("[SELF-TEST PASS] run_c1_5fold invariants verified.", flush=True)
        return

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t_start = time.time()
    X, y = load_and_preprocess(SGCC_PATH)
    X_reg, rw = slice_regime(X, "full")
    D = X.shape[2]
    raw = pd.read_csv(SGCC_PATH).drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    nk = (raw.shape[1] // 7) * 7
    g, m = gap_magnitude(raw[:, :nk].reshape(len(y), nk // 7, 7))
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))

    if args.smoke:
        r = run_cell(X_reg, rw, D, y, g, m, folds, "a34_generic", seed=0, fold=3, epochs=1, device=args.device)
        log(f"[SMOKE PASS] a34_generic s=0 f=3 ep=1 PR={r['pr_auc']:.4f} wall={r['wall_s']}s")
        return

    cols = (["arm", "seed", "fold", "kf", "thr", "wall_s", "n_test",
             "pr_auc", "roc_auc", "f1", "p_at_100", "p_at_200", "p_at_500"]
            + [f"{x}_{p}" for p in PHENOTYPES for x in ("pr", "roc", "f1", "n")]
            + ["val_labels_json", "val_scores_json"])

    jobs = [
        ("a34_generic", "a34_cell",   range(5), TRAIN_F_IDS),
        ("horln",       "horln",      range(3), TRAIN_F_IDS),
        ("wdcnn",       "wdcnn",      range(3), TRAIN_F_IDS),
        ("msdm_net",    "msdm_net",   range(3), TRAIN_F_IDS),
        ("dkgt",        "dkgt",       range(3), TRAIN_F_IDS),
        ("hybridattn",  "hybridattn", range(3), TRAIN_F_IDS),
        ("grud",        "grud",       range(3), TRAIN_F_IDS),
    ]
    total_cells = sum(len(list(s)) * len(f) for _, _, s, f in jobs)
    done_idx = 0
    for backbone, prefix, seeds, fold_ids in jobs:
        for s in seeds:
            for f in fold_ids:
                done_idx += 1
                fp = RAW34 / f"{prefix}_{s}_{f}.csv"
                if args.resume and fp.exists() and fp.stat().st_size > 100:
                    df_chk = pd.read_csv(fp)
                    if len(df_chk) == 1 and np.isfinite(float(df_chk["pr_auc"].iloc[0])):
                        log(f"[{done_idx}/{total_cells}] SKIP existing {fp.name} PR={float(df_chk['pr_auc'].iloc[0]):.4f}")
                        continue
                cool_if_hot()
                row = run_cell(X_reg, rw, D, y, g, m, folds, backbone, s, f, epochs=args.epochs, device=args.device)
                with open(fp, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerow(row)
                log(f"[{done_idx}/{total_cells}] {prefix} s={s} f={f} | PR={row['pr_auc']:.4f} ROC={row['roc_auc']:.4f} "
                    f"F1={row['f1']:.4f} wall={row['wall_s']}s T={gpu_temp()}C")

    compute_all_stats(y, g, m, folds, log)
    log(f"\n[DONE] C1 completed in {(time.time() - t_start)/60.0:.2f} min -> {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
