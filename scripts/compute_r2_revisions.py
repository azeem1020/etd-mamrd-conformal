"""Single source of Tables 2-5 of the manuscript and of the statistics quoted with them.

Reads the cached prediction records in RESULTS_DIR (no retraining) plus the raw SGCC/ECL
files (MAMRD_SGCC / MAMRD_ECL) and writes unrounded values to RESULTS_DIR/r2_audit/:
  sgcc_/ecl_partition_manifest.csv, sgcc_/ecl_fold_counts.csv  partition manifests and per-fold counts
  router_params.csv             train-fitted consumption-group cutoffs per fold (Methods)
  table2_partition_ledger.csv   Table 2 (means over Folds 0-2; conformal halves on the Table 6 /
                                Supp. Table S6 resplits)
  table3_main_results.csv       Table 3 (AP, ROC-AUC, F1 with cell BCa intervals; MAP@100, P@100; ECL AP and ROC-AUC with intervals)
  table3_tests.csv              paired MAMRD-HORLN tests, Friedman test, consumer-level bootstrap
  table4_fpr_budgets.csv        Table 4 rows (recall at 1 %/5 % FPR by ROC interpolation; inspected %)
  table4_prospective.csv        Table 4 note (Workflow B threshold transfer on the Table 6 resplits)
  table5_ablation.csv           Table 5 (build-up and subtractive contrasts, BCa, exact Wilcoxon)
  mask_only_baseline.csv        missingness-only classifier and missing-rate statistics (Introduction)
  conformal_eligibility_ledger.csv  share of resplits with a finite conformal threshold per group
All intervals use bca_ci below (B = 2,000, seed 12345). Manuscript values are these numbers
rounded once.
"""
from __future__ import annotations
import os, sys, json, csv, glob
from pathlib import Path
from collections import defaultdict
from statistics import NormalDist
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

csv.field_size_limit(10**7)
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import RESULTS_DIR, SGCC_PATH, ECL_PATH, DATA_DIR
# Data paths come only from repro_config (MAMRD_SGCC / MAMRD_ECL or data/); no silent fallback.
from zkmix_lab.data import KFoldSpec
from zkmix_lab.metrics import topk_tie_neutral, recall_at_fpr as recall_at_fpr_roc
from sklearn.metrics import roc_curve
from run_local_grid import load_and_preprocess, _stratified_three_way_split
from phased.phenotype import gap_magnitude, fit_router, assign
from msdm.fdi import ATTACKS
from conformal_common import NREP, resplit_rng, stratified_half, n_min as conformal_n_min
# Arm indices of the conformal resplits (Table 6 / Supp. Table S6), reused for Workflow B below
from run_decision_value import _ARM_ORDER as SGCC_ARM_ORDER
from run_ecl_trust_analysis import ARMS as ECL_ARMS, _ARM_ORDER as ECL_ARM_ORDER

# Conformal resplits: exactly those of run_decision_value.py (Table 6) -- arm index and
# seed set per detector, same RNG (conformal_common.resplit_rng). Only labels are needed.
CONF_ARMS = [(0, range(5))] + [(i, range(3)) for i in range(1, 7)]
CONF_RESPLITS = []   # one dict per (arm, seed, fold, rep): calibration positives per group

RES = RESULTS_DIR
OUT_R2 = RES / "r2_audit"
OUT_R2.mkdir(parents=True, exist_ok=True)
N_DIST = NormalDist()


def bca_ci(data, B=2000, alpha=0.05, seed=12345):
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
    lo = float(np.quantile(boot, q_adj(alpha / 2.0)))
    hi = float(np.quantile(boot, q_adj(1.0 - alpha / 2.0)))
    return lo, hi


def three_group(ph: np.ndarray) -> np.ndarray:
    out = np.empty(len(ph), dtype=object)
    for i, v in enumerate(ph):
        if v in ("Heavy_Dropper", "Heavy_Riser"):
            out[i] = "Heavy"
        elif v in ("Light_Dropper", "Light_Riser"):
            out[i] = "Light"
        else:
            out[i] = "Flat"
    return out


def recall_at_fpr(y, s, b):
    """(recall at FPR = b by ROC interpolation [zkmix_lab.metrics.recall_at_fpr], inspected fraction),
    inspected = (TP + FP) / N = pi * recall + (1 - pi) * b at that operating point."""
    r = recall_at_fpr_roc(y, s, b)
    pi = float(np.mean(y))
    return r, pi * r + (1.0 - pi) * b


def load_cell(fp):
    r = pd.read_csv(fp).iloc[0]
    return r, np.array(json.loads(r["val_labels_json"]), dtype=int), np.array(json.loads(r["val_scores_json"]), dtype=float)


print("=====================================================================")
print("1. LOADING SGCC RAW & NORMALIZED DATA + CALENDAR DAY-OF-WEEK CHECK")
print("=====================================================================")
if SGCC_PATH.exists():
    raw_df = pd.read_csv(SGCC_PATH)
    id_col, flag_col = raw_df.columns[0], raw_df.columns[1]
    date_cols = list(raw_df.columns[2:])
    parsed_dates = pd.to_datetime(date_cols, format="mixed")
    print(f"SGCC total rows={len(raw_df)}, total date columns={len(date_cols)}")
    print(f"First date={parsed_dates[0].strftime('%Y-%m-%d (%A)')}, "
          f"1029th date={parsed_dates[1028].strftime('%Y-%m-%d (%A)')}, "
          f"Last date={parsed_dates[-1].strftime('%Y-%m-%d (%A)')}")
    dow_names = [parsed_dates[d].strftime("%A") for d in range(7)]
    print("Weekly grid (147, 7) column index -> Calendar Day-of-Week:")
    for c_idx, dname in enumerate(dow_names):
        print(f"  Col {c_idx}: {dname}")

    X_norm, y_all = load_and_preprocess(SGCC_PATH)
    val_arr = raw_df[date_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X_raw = val_arr[:, :1029].reshape(len(raw_df), 147, 7)
    g_all, m_all = gap_magnitude(X_raw)
    cust_ids = raw_df[id_col].astype(str).to_numpy()

    print("\n=====================================================================")
    print("2. POINT 3: SGCC PARTITION MANIFESTS & EXACT COHORT/GROUP COUNTS")
    print("=====================================================================")
    folds_sgcc = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_all))
    sgcc_manifest_rows = []
    fold_stats = []
    router_rows = []   # train-fitted consumption-group router parameters per fold (Methods)

    for f in (0, 1, 2):
        tr, val = folds_sgcc[f]
        es, cal, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        # Verify exact disjointness
        assert len(set(tr) & set(cal)) == 0
        assert len(set(tr) & set(es)) == 0
        assert len(set(tr) & set(te)) == 0
        assert len(set(cal) & set(es)) == 0
        assert len(set(cal) & set(te)) == 0
        assert len(set(es) & set(te)) == 0
        assert len(tr) + len(cal) + len(es) + len(te) == len(y_all)

        params = fit_router(g_all[tr], m_all[tr])
        router_rows.append({"fold": f, "n_train": len(tr), "tau_f": params["tau"],
                            "mu_plus": params["med_drop"], "mu_minus": params["med_rise"]})
        ph_all = assign(g_all, m_all, params)
        grp_all = three_group(ph_all)

        for split_name, idx_arr in [("train", tr), ("early_stop", es), ("threshold_cal", cal), ("test", te)]:
            for idx in idx_arr:
                sgcc_manifest_rows.append({
                    "fold": f,
                    "split": split_name,
                    "row_index": int(idx),
                    "customer_id": cust_ids[idx],
                    "label": int(y_all[idx]),
                    "group_3": grp_all[idx],
                    "phenotype_5": ph_all[idx],
                })

        # Conformal 50/50 resplits on te: the 1,380 resplits used for Table 6 / Fig. 5
        # (7 detectors x seed-fold cells x 20), regenerated from labels via conformal_common.
        ci_counts = defaultdict(list)
        ti_counts = defaultdict(list)
        yt = y_all[te]
        gt = grp_all[te]
        for arm_idx, seeds in CONF_ARMS:
            for s in seeds:
                for rep in range(NREP):
                    ci, ti = stratified_half(yt, resplit_rng(arm_idx, s, f, rep))
                    ci_counts["total"].append(len(ci))
                    ci_counts["pos"].append(int(yt[ci].sum()))
                    ti_counts["total"].append(len(ti))
                    ti_counts["pos"].append(int(yt[ti].sum()))
                    row = dict(arm_index=arm_idx, seed=s, fold=f, rep=rep)
                    for gname in ("Heavy", "Light", "Flat"):
                        ci_counts[f"{gname}_tot"].append(int((gt[ci] == gname).sum()))
                        ci_counts[f"{gname}_pos"].append(int(((gt[ci] == gname) & (yt[ci] == 1)).sum()))
                        ti_counts[f"{gname}_tot"].append(int((gt[ti] == gname).sum()))
                        ti_counts[f"{gname}_pos"].append(int(((gt[ti] == gname) & (yt[ti] == 1)).sum()))
                        row[f"{gname}_pos"] = ci_counts[f"{gname}_pos"][-1]
                    CONF_RESPLITS.append(row)

        st_row = {
            "fold": f,
            "train_tot": len(tr), "train_pos": int(y_all[tr].sum()),
            "es_tot": len(es), "es_pos": int(y_all[es].sum()),
            "cal_tot": len(cal), "cal_pos": int(y_all[cal].sum()),
            "te_tot": len(te), "te_pos": int(y_all[te].sum()),
            "te_Heavy_tot": int((gt == "Heavy").sum()), "te_Heavy_pos": int(((gt == "Heavy") & (yt == 1)).sum()),
            "te_Light_tot": int((gt == "Light").sum()), "te_Light_pos": int(((gt == "Light") & (yt == 1)).sum()),
            "te_Flat_tot": int((gt == "Flat").sum()), "te_Flat_pos": int(((gt == "Flat") & (yt == 1)).sum()),
            "conf_cal_tot_mean": np.mean(ci_counts["total"]), "conf_cal_pos_mean": np.mean(ci_counts["pos"]),
            "conf_cal_Heavy_pos_mean": np.mean(ci_counts["Heavy_pos"]),
            "conf_cal_Light_pos_mean": np.mean(ci_counts["Light_pos"]),
            "conf_cal_Flat_pos_mean": np.mean(ci_counts["Flat_pos"]),
            "conf_cal_Flat_pos_min": np.min(ci_counts["Flat_pos"]),
            "conf_cal_Flat_pos_max": np.max(ci_counts["Flat_pos"]),
            "conf_eval_tot_mean": np.mean(ti_counts["total"]), "conf_eval_pos_mean": np.mean(ti_counts["pos"]),
            "conf_eval_Heavy_pos_mean": np.mean(ti_counts["Heavy_pos"]),
            "conf_eval_Light_pos_mean": np.mean(ti_counts["Light_pos"]),
            "conf_eval_Flat_pos_mean": np.mean(ti_counts["Flat_pos"]),
        }
        fold_stats.append(st_row)

    pd.DataFrame(sgcc_manifest_rows).to_csv(OUT_R2 / "sgcc_partition_manifest.csv", index=False)
    df_fstats = pd.DataFrame(fold_stats)
    df_fstats.to_csv(OUT_R2 / "sgcc_fold_counts.csv", index=False)
    pd.DataFrame(router_rows).to_csv(OUT_R2 / "router_params.csv", index=False)
    print(df_fstats.to_string(index=False))
else:
    print(f"[INFO] SGCC raw CSV not found at {SGCC_PATH}. Loading precomputed verified partition manifests.")
    manifest_fstats = OUT_R2 / "sgcc_fold_counts.csv"
    if not manifest_fstats.exists():
        manifest_fstats = THIS.parent / "partition_manifests" / "sgcc_fold_counts.csv"
    if manifest_fstats.exists():
        df_fstats = pd.read_csv(manifest_fstats)
        print(df_fstats.to_string(index=False))
    else:
        print("[WARNING] No cached sgcc_fold_counts.csv found.")

print("\n=====================================================================")
print("3. POINT 3 & 4: ECL WINDOW/CLIENT MANIFEST & CLIENT OVERLAP AUDIT")
print("=====================================================================")
if ECL_PATH.exists():
    ecl_df = pd.read_csv(ECL_PATH, sep=";", decimal=",", index_col=0)
    ecl_df.index = pd.to_datetime(ecl_df.index)
    ecl_res = ecl_df.resample("30min").mean().values.T.astype(np.float64)
    n_clients, T_ecl = ecl_res.shape
    WIN = 7 * 48  # 336
    nw = T_ecl // WIN
    win_records = []
    W_list = []
    for cid in range(n_clients):
        cname = str(ecl_df.columns[cid])
        for w in range(nw):
            seg = ecl_res[cid, w * WIN : (w + 1) * WIN]
            if np.nansum(np.abs(seg)) > 1e-6:
                W_list.append(seg)
                win_records.append((cid, cname, w))

    W_arr = np.array(W_list)
    rng_ecl = np.random.default_rng(0)
    if len(W_arr) > 25000:
        sel_idx = rng_ecl.choice(len(W_arr), 25000, replace=False)
        W_arr = W_arr[sel_idx]
        win_records = [win_records[i] for i in sel_idx]

    frng = np.random.default_rng(0)
    y_ecl = np.zeros(len(W_arr), dtype=int)
    atk_ecl = -np.ones(len(W_arr), dtype=int)
    W_obs = W_arr.copy()   # observed (post-FDI) windows, as in run_ecl_trust.build_dataset
    for i in frng.choice(len(W_arr), int(0.10 * len(W_arr)), replace=False):
        t = int(frng.integers(6))
        atk_ecl[i] = t
        W_obs[i] = ATTACKS[t](W_arr[i], frng)
        y_ecl[i] = 1
    # Consumption magnitude as in run_ecl_trust.py (observed, post-FDI, pre-normalisation): this is
    # the High/Low split used by the ECL conformal analysis (Supp. Table S6).
    mag_all = np.nanmean(W_obs, axis=1)
    mag_all = np.where(np.isfinite(mag_all), mag_all, 0.0)

    ecl_folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_ecl))
    ecl_manifest_rows = []
    ecl_fold_summary = []

    for f in (0, 1, 2):
        tr, val = ecl_folds[f]
        # Held-out thirds as in run_ecl_trust.split_idx: (reserved, unused; early-stop; test)
        cal, es, te = _stratified_three_way_split(val, y_ecl, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        assert len(set(tr) & set(te)) == 0
        assert len(set(cal) & set(te)) == 0
        assert len(set(es) & set(te)) == 0

        tr_clients = {win_records[i][0] for i in tr}
        te_clients = {win_records[i][0] for i in te}
        shared_clients = tr_clients & te_clients

        # Magnitude median on train (matching run_ecl_trust.py)
        med_tr = float(np.median(mag_all[tr]))
        heavy_te = mag_all[te] > med_tr
        yt_ecl = y_ecl[te]

        ecl_fold_summary.append({
            "fold": f,
            "train_win": len(tr), "train_pos": int(y_ecl[tr].sum()),
            "es_win": len(es), "es_pos": int(y_ecl[es].sum()),
            "cal_win": len(cal), "cal_pos": int(y_ecl[cal].sum()),
            "te_win": len(te), "te_pos": int(y_ecl[te].sum()),
            "te_Heavy_win": int(heavy_te.sum()), "te_Heavy_pos": int((heavy_te & (yt_ecl == 1)).sum()),
            "te_Light_win": int((~heavy_te).sum()), "te_Light_pos": int(((~heavy_te) & (yt_ecl == 1)).sum()),
            "unique_tr_clients": len(tr_clients),
            "unique_te_clients": len(te_clients),
            "shared_tr_te_clients": len(shared_clients),
            "pct_te_clients_in_tr": 100.0 * len(shared_clients) / len(te_clients),
        })

        for split_name, idx_arr in [("train", tr), ("early_stop", es), ("reserved_unused", cal), ("test", te)]:
            for idx in idx_arr:
                cid, cname, wid = win_records[idx]
                ecl_manifest_rows.append({
                    "fold": f,
                    "split": split_name,
                    "window_index": int(idx),
                    "client_id": cid,
                    "client_name": cname,
                    "client_week_idx": wid,
                    "label": int(y_ecl[idx]),
                    "attack_type": int(atk_ecl[idx]),
                })

    pd.DataFrame(ecl_manifest_rows).to_csv(OUT_R2 / "ecl_partition_manifest.csv", index=False)
    df_ecl_sum = pd.DataFrame(ecl_fold_summary)
    df_ecl_sum.to_csv(OUT_R2 / "ecl_fold_counts.csv", index=False)
    print(df_ecl_sum.to_string(index=False))
else:
    print(f"[INFO] ECL raw dataset not found at {ECL_PATH}. Loading precomputed verified partition manifests.")
    manifest_ecl = OUT_R2 / "ecl_fold_counts.csv"
    if not manifest_ecl.exists():
        manifest_ecl = THIS.parent / "partition_manifests" / "ecl_fold_counts.csv"
    if manifest_ecl.exists():
        df_ecl_sum = pd.read_csv(manifest_ecl)
        print(df_ecl_sum.to_string(index=False))
    else:
        print("[WARNING] No cached ecl_fold_counts.csv found.")

print("\n=====================================================================")
print("4. POINT 5: 9-CELL VS 15-CELL MAMRD & CONSUMER-BOOTSTRAP VS CELL BCa")
print("=====================================================================")
DET_SGCC = [
    ("GRU-D", "baselines/grud_{s}_{f}.csv", range(3), range(3)),
    ("Self-Attn", "baselines/hybridattn_{s}_{f}.csv", range(3), range(3)),
    ("DKGT-Net", "baselines/dkgt_{s}_{f}.csv", range(3), range(3)),
    ("WDCNN", "baselines/wdcnn_{s}_{f}.csv", range(3), range(3)),
    ("MSDM", "baselines/msdm_net_{s}_{f}.csv", range(3), range(3)),
    ("HORLN", "baselines/horln_{s}_{f}.csv", range(3), range(3)),
    ("MAMRD (9-cell)", "a34_cell_{s}_{f}.csv", range(3), range(3)),
    ("MAMRD (15-cell)", "a34_cell_{s}_{f}.csv", range(5), range(3)),
]

cell_records = []
raw_preds = {}  # (name, s, f) -> (y, scores, thr)
for name, pat, seeds, folds in DET_SGCC:
    for s in seeds:
        for f in folds:
            fp = RES / pat.format(s=s, f=f)
            r = pd.read_csv(fp).iloc[0]
            yv = np.array(json.loads(r["val_labels_json"]), dtype=int)
            sv = np.array(json.loads(r["val_scores_json"]), dtype=float)
            raw_preds[(name, s, f)] = (yv, sv, float(r["thr"]))
            cell_records.append({
                "detector": name, "seed": s, "fold": f,
                "ap": float(r["pr_auc"]), "roc": float(r["roc_auc"]),
                "f1": float(r["f1"]), "p100": float(r["p_at_100"]),
                "p200": float(r["p_at_200"]),
                # MAP@100 (Eq. map100) and P@100 with tied scores in random order (expected value):
                # the SGCC file stores thefts first, so index-order tie-breaking would favour positives
                **dict(zip(("map100", "p100"), topk_tie_neutral(yv, sv, 100))),
                "p100_stored": float(r["p_at_100"]),
            })

df_cells = pd.DataFrame(cell_records)
print("\n--- Summary across cells (9-cell vs 15-cell) ---")
for name, _, _, _ in DET_SGCC:
    sub = df_cells[df_cells.detector == name]
    ap_m = sub.ap.mean(); ap_lo, ap_hi = bca_ci(sub.ap.values)
    roc_m = sub.roc.mean(); roc_lo, roc_hi = bca_ci(sub.roc.values)
    f1_m = sub.f1.mean(); f1_lo, f1_hi = bca_ci(sub.f1.values)
    p100_m = sub.p100.mean(); p100_lo, p100_hi = bca_ci(sub.p100.values)
    p200_m = sub.p200.mean(); p200_lo, p200_hi = bca_ci(sub.p200.values)
    print(f"{name:16s} (n={len(sub):2d}) | AP={ap_m:.3f} [{ap_lo:.3f},{ap_hi:.3f}] | "
          f"ROC={roc_m:.3f} [{roc_lo:.3f},{roc_hi:.3f}] | F1={f1_m:.3f} [{f1_lo:.3f},{f1_hi:.3f}] | "
          f"P@100={p100_m:.3f} [{p100_lo:.3f},{p100_hi:.3f}] | P@200={p200_m:.3f} [{p200_lo:.3f},{p200_hi:.3f}]")

# ---- Table 3 (tab:main_results): unrounded means + cell BCa intervals -> r2_audit/table3_main_results.csv
ECL_ARM_OF = {"GRU-D": "grud", "Self-Attn": "hybridattn", "WDCNN": "wdcnn", "MSDM": "msdm",
              "HORLN": "horln", "MAMRD (9-cell)": "a34_generic"}   # DKGT-Net and the 15-cell row: SGCC only
table3_rows = []
for name, _, _, _ in DET_SGCC:
    sub = df_cells[df_cells.detector == name]
    row = {"detector": name, "n_cells": len(sub)}
    for col in ("ap", "roc", "f1"):
        lo, hi = bca_ci(sub[col].values)
        row.update({col: sub[col].mean(), f"{col}_lo": lo, f"{col}_hi": hi})
    row.update({"map100": sub.map100.mean(), "p100": sub.p100.mean()})
    if name in ECL_ARM_OF:
        ecl_rows = [pd.read_csv(RES / "ecl_trust" / f"{ECL_ARM_OF[name]}_{s}_{f}.csv").iloc[0]
                    for s in range(3) for f in range(3)]
        for col in ("ap", "roc"):
            vals = [float(r[col]) for r in ecl_rows]
            lo, hi = bca_ci(vals)
            row.update({f"ecl_{col}": float(np.mean(vals)), f"ecl_{col}_lo": lo, f"ecl_{col}_hi": hi})
        row["ecl_n_cells"] = len(ecl_rows)
    table3_rows.append(row)
pd.DataFrame(table3_rows).to_csv(OUT_R2 / "table3_main_results.csv", index=False)
print(f"Exported {OUT_R2 / 'table3_main_results.csv'}")

# ---- Statistical tests quoted with Table 3 -> r2_audit/table3_tests.csv
test_rows = []
# Paired comparison MAMRD vs HORLN on exact 9 matched cells (seeds 0..2 x folds 0..2)
m9 = df_cells[df_cells.detector == "MAMRD (9-cell)"].set_index(["seed", "fold"])
h9 = df_cells[df_cells.detector == "HORLN"].set_index(["seed", "fold"])
for col in ("ap", "roc", "f1", "p100", "p200"):
    diff9 = (m9[col] - h9[col]).values
    lo, hi = bca_ci(diff9)
    stat_w, p_w = wilcoxon(diff9)
    print(f"Paired 9-cell MAMRD - HORLN {col:4s}: mean={diff9.mean():+.4f} BCa=[{lo:+.4f},{hi:+.4f}] Wilcoxon p={p_w:.4f}")
    test_rows.append({"test": "paired_cell_MAMRD_minus_HORLN", "metric": col, "n": len(diff9),
                      "estimate": diff9.mean(), "ci_lo": lo, "ci_hi": hi, "p_value": p_w,
                      "rel_pct": 100.0 * diff9.mean() / h9[col].mean(), "n_positive": int((diff9 > 0).sum())})

# Friedman test across the 7 detectors on the 9 matched cells (seeds 0..2 x folds 0..2)
det7_names = ["GRU-D", "Self-Attn", "DKGT-Net", "WDCNN", "MSDM", "HORLN", "MAMRD (9-cell)"]
mat_9x7 = np.column_stack([
    df_cells[df_cells.detector == d].sort_values(["seed", "fold"]).ap.values
    for d in det7_names
])
chi2_9, p_f9 = friedmanchisquare(*[mat_9x7[:, j] for j in range(7)])
print(f"Friedman test (7 detectors, 9 matched seed-fold cells): chi2={chi2_9:.2f}, p={p_f9:.2e}")
test_rows.append({"test": "friedman_7_detectors_9_cells", "metric": "ap", "n": 9,
                  "estimate": chi2_9, "p_value": p_f9, "df": 6})
# MAMRD best in every matched cell against every baseline?
test_rows.append({"test": "MAMRD_best_in_all_9_cells", "metric": "ap", "n": 9,
                  "estimate": float(np.all(mat_9x7[:, -1] > mat_9x7[:, :-1].max(axis=1)))})

# Fold-averaged consumer-level bootstrap (averaging 3 seeds per consumer within each fold,
# then bootstrapping held-out consumers B=1000 times per fold)
print("\n--- Consumer-level bootstrap (seed-averaged scores per consumer within each fold, B=1000) ---")
B_CONS = 1000
rng_cb = np.random.default_rng(2026)
cons_boot_ap = {d: np.zeros(B_CONS) for d in det7_names}
cons_boot_roc = {d: np.zeros(B_CONS) for d in det7_names}
fold_seed_avg_scores = {}

for f in (0, 1, 2):
    yv = raw_preds[("HORLN", 0, f)][0]
    n_te = len(yv)
    for d in det7_names:
        s_avg = np.mean([raw_preds[(d, s, f)][1] for s in (0, 1, 2)], axis=0)
        fold_seed_avg_scores[(d, f)] = (yv, s_avg)
    for b in range(B_CONS):
        b_idx = rng_cb.integers(0, n_te, n_te)
        yb = yv[b_idx]
        for d in det7_names:
            sb = fold_seed_avg_scores[(d, f)][1][b_idx]
            cons_boot_ap[d][b] += average_precision_score(yb, sb) / 3.0
            cons_boot_roc[d][b] += roc_auc_score(yb, sb) / 3.0

for d in det7_names:
    pt_ap = np.mean([average_precision_score(*fold_seed_avg_scores[(d, f)]) for f in (0, 1, 2)])
    lo_ap, hi_ap = np.percentile(cons_boot_ap[d], [2.5, 97.5])
    pt_roc = np.mean([roc_auc_score(*fold_seed_avg_scores[(d, f)]) for f in (0, 1, 2)])
    lo_roc, hi_roc = np.percentile(cons_boot_roc[d], [2.5, 97.5])
    print(f"  {d:16s} | Seed-ensemble AP={pt_ap:.3f} Cons-Boot95%=[{lo_ap:.3f},{hi_ap:.3f}] | "
          f"ROC={pt_roc:.3f} [{lo_roc:.3f},{hi_roc:.3f}]")

diff_cb_ap = cons_boot_ap["MAMRD (9-cell)"] - cons_boot_ap["HORLN"]
pt_diff_cb = np.mean([
    average_precision_score(*fold_seed_avg_scores[("MAMRD (9-cell)", f)]) -
    average_precision_score(*fold_seed_avg_scores[("HORLN", f)])
    for f in (0, 1, 2)
])
print(f"  Paired Consumer-Bootstrap MAMRD - HORLN AP: {pt_diff_cb:+.4f} "
      f"95% CI=[{np.percentile(diff_cb_ap, 2.5):+.4f}, {np.percentile(diff_cb_ap, 97.5):+.4f}] "
      f"(replicates <= 0: {int((diff_cb_ap <= 0).sum())} of {B_CONS})")
for d in ("MAMRD (9-cell)", "HORLN"):
    test_rows.append({"test": "consumer_bootstrap_seed_ensemble", "metric": f"ap:{d}", "n": B_CONS,
                      "estimate": np.mean([average_precision_score(*fold_seed_avg_scores[(d, f)]) for f in (0, 1, 2)]),
                      "ci_lo": np.percentile(cons_boot_ap[d], 2.5), "ci_hi": np.percentile(cons_boot_ap[d], 97.5)})
test_rows.append({"test": "consumer_bootstrap_paired_MAMRD_minus_HORLN", "metric": "ap", "n": B_CONS,
                  "estimate": pt_diff_cb, "ci_lo": np.percentile(diff_cb_ap, 2.5),
                  "ci_hi": np.percentile(diff_cb_ap, 97.5), "n_replicates_le_0": int((diff_cb_ap <= 0).sum())})
pd.DataFrame(test_rows).to_csv(OUT_R2 / "table3_tests.csv", index=False)
print(f"Exported {OUT_R2 / 'table3_tests.csv'}")

print("\n=====================================================================")
print("5. POINT 7: ABLATION PROVENANCE & PAIRED CIs (BUILD-UP & SUBTRACTIVE)")
print("=====================================================================")
bu_stages = [
    ("S0: Core (BiMamba2 + 2D Laplacian CNN, scalar fill)", "bu/a34_bu0_core_{s}_{f}.csv"),
    ("S1: + Context-conditioned fill token (CC-NaME)", "bu/a34_bu1_ccname_{s}_{f}.csv"),
    ("S2: + 2D missing-mask Laplacian CNN", "bu/a34_bu2_mask_{s}_{f}.csv"),
    ("S3: + Multi-scale lagged differencing (a33 / no-IIPC)", "bu/a33_generic_{s}_{f}.csv"),
    ("S4: + Inter/intra-period conv (full MAMRD, 9-cell)", "a34_cell_{s}_{f}.csv"),
]

bu_data = {}
for label, pat in bu_stages:
    rows = []
    for s in (0, 1, 2):
        for f in (0, 1, 2):
            r = pd.read_csv(RES / pat.format(s=s, f=f)).iloc[0]
            rows.append({"seed": s, "fold": f, "ap": float(r["pr_auc"]),
                         "roc": float(r["roc_auc"]), "f1": float(r["f1"]), "p100": float(r["p_at_100"])})
    bu_data[label] = pd.DataFrame(rows).set_index(["seed", "fold"])

# Also load a32_sgcc.csv (9 cells: S3 minus both CC-NaME and Mask CNN)
a32_fp = RES / "a32_sgcc.csv"
if a32_fp.exists():
    a32_df = pd.read_csv(a32_fp).set_index(["seed", "fold"])
    bu_data["A32: Generic core + differencing (no CC-NaME, no Mask CNN)"] = a32_df.rename(
        columns={"pr_auc": "ap", "roc_auc": "roc", "p_at_100": "p100"}
    )[["ap", "roc", "f1", "p100"]]

# Also load controlled S4 missingness ablation (Reviewer 2, Comment 4) if present
a34_no_miss_key = "A34: Controlled S4 with missingness pathways disabled"
no_miss_rows = []
for s in (0, 1, 2):
    for f in (0, 1, 2):
        for candidate in [
            RES / f"bu/a34_no_missingness_{s}_{f}.csv",
            RES / f"a34_no_missingness_cell_{s}_{f}.csv",
            RES / f"bu/a34_no_missingness_cell_{s}_{f}.csv",
        ]:
            if candidate.exists():
                r = pd.read_csv(candidate).iloc[0]
                no_miss_rows.append({"seed": s, "fold": f, "ap": float(r["pr_auc"]),
                                     "roc": float(r["roc_auc"]), "f1": float(r["f1"]), "p100": float(r["p_at_100"])})
                break
if len(no_miss_rows) == 9:
    bu_data[a34_no_miss_key] = pd.DataFrame(no_miss_rows).set_index(["seed", "fold"])

prev_key = None
s0_key = bu_stages[0][0]
s4_key = bu_stages[-1][0]
for label, _ in bu_stages:
    df_s = bu_data[label]
    ap_m = df_s.ap.mean(); ap_lo, ap_hi = bca_ci(df_s.ap.values)
    roc_m = df_s.roc.mean(); f1_m = df_s.f1.mean(); p100_m = df_s.p100.mean()
    if prev_key is None:
        inc_str = "(baseline)"
        tot_str = "(baseline)"
    else:
        d_inc = (df_s.ap - bu_data[prev_key].ap).values
        ilo, ihi = bca_ci(d_inc)
        inc_str = f"{d_inc.mean():+.4f} [{ilo:+.4f},{ihi:+.4f}]"
        d_tot = (df_s.ap - bu_data[s0_key].ap).values
        tlo, thi = bca_ci(d_tot)
        tot_str = f"{d_tot.mean():+.4f} [{tlo:+.4f},{thi:+.4f}]"
    print(f"{label:55s} | AP={ap_m:.4f}[{ap_lo:.4f},{ap_hi:.4f}] ROC={roc_m:.4f} F1={f1_m:.4f} P@100={p100_m:.4f} | "
          f"dPrev={inc_str} | dS0={tot_str}")
    prev_key = label

print("\n--- Subtractive / Cross-Arm Paired Contrasts (n=9 matched cells) ---")
s3_key = bu_stages[3][0]
s2_key = bu_stages[2][0]
s1_key = bu_stages[1][0]
a32_key = "A32: Generic core + differencing (no CC-NaME, no Mask CNN)"
contrasts = [
    ("Full MAMRD (S4) minus No-IIPC (S3 = a33_generic)", bu_data[s4_key].ap - bu_data[s3_key].ap),
    ("No-IIPC (S3) minus No-Diff-No-IIPC (S2 = a34_bu2_mask)", bu_data[s3_key].ap - bu_data[s2_key].ap),
    ("Full MAMRD (S4) minus No-Diff-No-IIPC (S2 = a34_bu2_mask)", bu_data[s4_key].ap - bu_data[s2_key].ap),
    ("No-IIPC (S3) minus No-Missingness (A32: no CC-NaME, no Mask)", bu_data[s3_key].ap - bu_data[a32_key].ap),
    ("Full MAMRD (S4) minus No-Missingness (A32: no CC-NaME, no Mask)", bu_data[s4_key].ap - bu_data[a32_key].ap),
    ("S2 (+Mask CNN) minus S1 (CC-NaME only, no Mask CNN)", bu_data[s2_key].ap - bu_data[s1_key].ap),
    ("S1 (+CC-NaME) minus S0 (Core scalar fill only)", bu_data[s1_key].ap - bu_data[s0_key].ap),
]
if a34_no_miss_key in bu_data:
    contrasts.insert(5, ("Full MAMRD (S4) minus Controlled No-Missingness (A34: no CC-NaME, no Mask)",
                         bu_data[s4_key].ap - bu_data[a34_no_miss_key].ap))
for cname, cser in contrasts:
    vals = cser.values
    lo, hi = bca_ci(vals)
    print(f"  {cname:64s}: dAP={vals.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

# ---- Table 5 (tab:ablation_results) -> r2_audit/table5_ablation.csv (unrounded)
# Panel A: build-up S0..S4 on the 9 matched cells; step = S_k - S_{k-1}, cumulative = S_k - S0
# (paired over cells, BCa CI, exact two-sided Wilcoxon p). Panel B: subtractive contrasts.
table5_rows = []
stage_vids = ["S0", "S1", "S2", "S3", "S4"]
prev_k = None
for i, (label, _) in enumerate(bu_stages):
    df_s = bu_data[label]
    ap_lo, ap_hi = bca_ci(df_s.ap.values)
    row = {"panel": "A", "variant": stage_vids[i], "n_cells": len(df_s),
           "ap": df_s.ap.mean(), "ap_lo": ap_lo, "ap_hi": ap_hi}
    if prev_k is not None:
        d_inc = (df_s.ap - bu_data[prev_k].ap).values
        d_tot = (df_s.ap - bu_data[s0_key].ap).values
        ilo, ihi = bca_ci(d_inc); tlo, thi = bca_ci(d_tot)
        row.update({"step_delta": d_inc.mean(), "step_lo": ilo, "step_hi": ihi,
                    "step_p": wilcoxon(df_s.ap.values, bu_data[prev_k].ap.values)[1],
                    "cum_delta": d_tot.mean(), "cum_lo": tlo, "cum_hi": thi,
                    "cum_p": wilcoxon(df_s.ap.values, bu_data[s0_key].ap.values)[1]})
    table5_rows.append(row)
    prev_k = label
# 15-cell value, reported only where 15 runs exist (S4 = MAMRD; build-up arms have 9 cells)
table5_rows[-1]["ap_15cell"] = df_cells[df_cells.detector == "MAMRD (15-cell)"].ap.mean()

b_contrasts = [("S3 - S4", s3_key), ("S2 - S4", s2_key), ("S1 - S2", s1_key)]
if a34_no_miss_key in bu_data:
    b_contrasts.append(("A34_no-miss - S4", a34_no_miss_key))
b_contrasts.append(("A32 - S4", a32_key))
for cid, key in b_contrasts:
    ref = s2_key if cid == "S1 - S2" else s4_key
    a1, a2 = bu_data[key].ap, bu_data[ref].ap
    vals = (a1 - a2).values
    lo, hi = bca_ci(vals); alo, ahi = bca_ci(a1.values)
    table5_rows.append({"panel": "B", "variant": cid, "n_cells": len(vals),
                        "ap": a1.mean(), "ap_lo": alo, "ap_hi": ahi, "ref_ap": a2.mean(),
                        "step_delta": vals.mean(), "step_lo": lo, "step_hi": hi,
                        "step_p": wilcoxon(a1.values, a2.values)[1],
                        "rel_pct": 100.0 * vals.mean() / a2.mean()})
pd.DataFrame(table5_rows).to_csv(OUT_R2 / "table5_ablation.csv", index=False)
print(f"Exported {OUT_R2 / 'table5_ablation.csv'}")

print("\n=====================================================================")
print("6. POINT 8: MISSINGNESS-ONLY BASELINE ON SGCC (3 FOLDS)")
print("=====================================================================")
if SGCC_PATH.exists():
    # Mask-only summary features per customer (30 in total): (1) overall missingness fraction,
    # (2) 7 day-of-week missingness fractions, (3) 21 four-week-block missingness fractions,
    # (4) rate of observed/missing transitions along the day sequence
    mask_obs = (~np.isnan(X_raw)).astype(np.float32)  # (N, 147, 7), 1=observed, 0=missing
    miss_ind = 1.0 - mask_obs
    f_overall = miss_ind.mean(axis=(1, 2), keepdims=False)[:, None]  # (N, 1)
    f_dow = miss_ind.mean(axis=1)  # (N, 7)
    f_monthly = miss_ind.reshape(len(X_raw), 21, 7, 7).mean(axis=(2, 3))  # (N, 21)
    miss_flat = miss_ind.reshape(len(X_raw), -1)  # (N, 1029)
    # boundary transitions
    n_transitions = np.abs(np.diff(miss_flat, axis=1)).mean(axis=1, keepdims=True)
    X_mask_feat = np.hstack([f_overall, f_dow, f_monthly, n_transitions])

    mask_aps, mask_rocs = [], []
    for f in (0, 1, 2):
        tr, val = folds_sgcc[f]
        es, cal, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_mask_feat[tr], y_all[tr])
        prob_te = clf.predict_proba(X_mask_feat[te])[:, 1]
        mask_aps.append(average_precision_score(y_all[te], prob_te))
        mask_rocs.append(roc_auc_score(y_all[te], prob_te))
    n_feat = X_mask_feat.shape[1]
    print(f"Mask-only classifier ({n_feat} missingness features, 3 folds): "
          f"AP = {np.mean(mask_aps):.3f} (folds: {[round(x,3) for x in mask_aps]}), "
          f"ROC-AUC = {np.mean(mask_rocs):.3f} (folds: {[round(x,3) for x in mask_rocs]})")
    # Consumer-level missing-reading rate by label (raw 1,035-day matrix and 1,029-day grid)
    miss_rate_raw = np.isnan(val_arr).mean(axis=1)
    miss_rate_grid = miss_ind.reshape(len(X_raw), -1).mean(axis=1)
    pd.DataFrame([{
        "n_features": n_feat, "ap_mean": np.mean(mask_aps), "roc_mean": np.mean(mask_rocs),
        **{f"ap_fold{f}": v for f, v in enumerate(mask_aps)}, **{f"roc_fold{f}": v for f, v in enumerate(mask_rocs)},
        "base_rate_all": float(y_all.mean()), "ap_over_base_rate": np.mean(mask_aps) / float(y_all.mean()),
        "cell_missing_pct_raw": 100.0 * float(np.isnan(val_arr).mean()),
        "cell_missing_pct_grid": 100.0 * float(np.isnan(X_raw).mean()),
        "median_consumer_missing_pct_theft_raw": 100.0 * float(np.median(miss_rate_raw[y_all == 1])),
        "median_consumer_missing_pct_honest_raw": 100.0 * float(np.median(miss_rate_raw[y_all == 0])),
        "median_consumer_missing_pct_theft_grid": 100.0 * float(np.median(miss_rate_grid[y_all == 1])),
        "median_consumer_missing_pct_honest_grid": 100.0 * float(np.median(miss_rate_grid[y_all == 0])),
    }]).to_csv(OUT_R2 / "mask_only_baseline.csv", index=False)
    print(f"Exported {OUT_R2 / 'mask_only_baseline.csv'}")
else:
    print("[SKIP] Mask-only baseline requires the raw SGCC CSV (set MAMRD_SGCC); no values reported.")

print("\n=====================================================================")
print("7. TABLE 4: RECALL & INSPECTED % AT 1 % / 5 % FPR (ROC INTERPOLATION) + WORKFLOW B PROSPECTIVE")
print("=====================================================================")
# Rows of Table 4: per cell, recall at FPR = b by ROC interpolation (recall_at_fpr) and the
# inspected share at that operating point, (TP + FP) / N = pi * recall + (1 - pi) * b; averaged
# over cells. Note of Table 4 (Workflow B): threshold = (1 - b) quantile of the calibration-half
# negatives, applied to the evaluation half, on the same 20 resplits per cell as Table 6
# (SGCC, conformal_common.resplit_rng with the run_decision_value arm index) and Supp. Table S6 (ECL).
BUDGETS = (0.01, 0.05)
SGCC_ARM_KEY = {"GRU-D": "GRU-D", "Self-Attn": "HybridAttn", "DKGT-Net": "DKGT", "WDCNN": "WDCNN",
                "MSDM": "MSDM", "HORLN": "HORLN", "MAMRD (9-cell)": "a34_NaME", "MAMRD (15-cell)": "a34_NaME"}
ECL_DISPLAY = {"grud": "GRU-D", "hybridattn": "Self-Attn", "wdcnn": "WDCNN", "msdm": "MSDM",
               "horln": "HORLN", "a34_generic": "MAMRD (9-cell)"}


def operating_points(cells, arm_index):
    """cells: list of (seed, fold, y, s) -> (row-level dict per budget, prospective dict per budget)."""
    rows, prosp = {}, {}
    for b in BUDGETS:
        rec, ins, pis = [], [], []
        pr, pf, pi_ = [], [], []
        for s, f, y, sc in cells:
            r, i = recall_at_fpr(y, sc, b)
            rec.append(r); ins.append(100.0 * i); pis.append(100.0 * float(np.mean(y)))
            for rep in range(NREP):
                ci, ti = stratified_half(y, resplit_rng(arm_index, s, f, rep))
                t = np.quantile(sc[ci][y[ci] == 0], 1.0 - b)
                ye, se = y[ti], sc[ti]
                pr.append(float((se[ye == 1] > t).mean()))
                pf.append(100.0 * float((se[ye == 0] > t).mean()))
                pi_.append(100.0 * float((se > t).mean()))
        rows[b] = {"recall": np.mean(rec), "insp_pct": np.mean(ins), "pi_pct": np.mean(pis), "n_cells": len(cells)}
        prosp[b] = {"recall": np.mean(pr), "fpr_pct": np.mean(pf), "insp_pct": np.mean(pi_), "n_resplits": len(pr)}
    return rows, prosp


t4_rows, t4_prosp = [], []
for name, pat, seeds, folds in DET_SGCC:
    cells = [(s, f, *raw_preds[(name, s, f)][:2]) for s in seeds for f in folds]
    rows, prosp = operating_points(cells, SGCC_ARM_ORDER[SGCC_ARM_KEY[name]])
    for b in BUDGETS:
        t4_rows.append({"dataset": "SGCC", "detector": name, "budget_fpr": b, **rows[b]})
        t4_prosp.append({"dataset": "SGCC", "detector": name, "budget_fpr": b, **prosp[b]})
    print(f"{name:16s} | SGCC " + " | ".join(
        f"@{int(b*100)}%FPR: Rec={rows[b]['recall']:.3f} Insp={rows[b]['insp_pct']:.2f}% "
        f"(Prosp: Rec={prosp[b]['recall']:.3f} FPR={prosp[b]['fpr_pct']:.2f}% Insp={prosp[b]['insp_pct']:.2f}%)"
        for b in BUDGETS))
for arm in ECL_ARMS:
    cells = []
    for s in (0, 1, 2):
        for f in (0, 1, 2):
            _, y, sc = load_cell(RES / "ecl_trust" / f"{arm}_{s}_{f}.csv")
            cells.append((s, f, y, sc))
    rows, prosp = operating_points(cells, ECL_ARM_ORDER[arm])
    for b in BUDGETS:
        t4_rows.append({"dataset": "ECL", "detector": ECL_DISPLAY[arm], "budget_fpr": b, **rows[b]})
        t4_prosp.append({"dataset": "ECL", "detector": ECL_DISPLAY[arm], "budget_fpr": b, **prosp[b]})
    print(f"{ECL_DISPLAY[arm]:16s} | ECL  " + " | ".join(
        f"@{int(b*100)}%FPR: Rec={rows[b]['recall']:.3f} Insp={rows[b]['insp_pct']:.2f}% "
        f"(Prosp: Rec={prosp[b]['recall']:.3f} FPR={prosp[b]['fpr_pct']:.2f}% Insp={prosp[b]['insp_pct']:.2f}%)"
        for b in BUDGETS))
pd.DataFrame(t4_rows).to_csv(OUT_R2 / "table4_fpr_budgets.csv", index=False)
pd.DataFrame(t4_prosp).to_csv(OUT_R2 / "table4_prospective.csv", index=False)
print(f"Exported {OUT_R2 / 'table4_fpr_budgets.csv'} and table4_prospective.csv")

print("\n=====================================================================")
print("8. CONFORMAL FINITE-SAMPLE ELIGIBILITY LEDGER (per resplit)")
print("=====================================================================")
# Reviewer 2 Comment 6: share of calibration/evaluation resplits in which each group has
# n_cal >= ceil((1-alpha)/alpha), i.e. a finite order statistic k = ceil((n+1)(1-alpha)) <= n.
# Computed on the 1,380 resplits used for Table 6. No resplit is excluded: when n_cal is
# below n_min, the whole group is inspected (q_hat = 0; conformal_common.cc_recall_threshold).
df_conf = pd.DataFrame(CONF_RESPLITS)
if df_conf.empty:   # raw SGCC absent: use the released per-resplit records of Table 6
    _rs = pd.read_csv(RES / "trust" / "decision_value_sgcc_resplits.csv")
    _rs = _rs[np.isclose(_rs["guarantee"], 0.80)]
    df_conf = pd.DataFrame({"Heavy_pos": _rs["ncal_high"].values, "Light_pos": _rs["ncal_low"].values,
                            "Flat_pos": _rs["ncal_flat"].values})
eligibility_rows = []
for alpha in (0.20, 0.10, 0.05):
    nmin = conformal_n_min(alpha)
    for gname in ("Heavy", "Light", "Flat"):
        v = df_conf[f"{gname}_pos"]
        share = float((v >= nmin).mean()) * 100.0
        eligibility_rows.append({
            "target_coverage": round(1.0 - alpha, 2),
            "alpha": alpha,
            "n_min_required": nmin,
            "group": gname,
            "n_resplits": int(len(v)),
            "n_pos_mean": float(v.mean()),
            "n_pos_min": int(v.min()),
            "n_pos_max": int(v.max()),
            "pct_resplits_finite_threshold": share,
            "rule_when_below_n_min": "-" if share == 100.0 else "whole group inspected (q_hat = 0)",
        })
df_elig = pd.DataFrame(eligibility_rows)
df_elig.to_csv(OUT_R2 / "conformal_eligibility_ledger.csv", index=False)
print(df_elig.to_string(index=False))


print("\n=====================================================================")
print("9. TABLE 2: PARTITION LEDGER (means over Folds 0-2)")
print("=====================================================================")
# SGCC rows from sgcc_fold_counts.csv (section 2). Conformal halves: means over the 1,380 Table 6
# resplits (equal number per fold, so the mean of fold means equals the pooled mean).
# ECL rows from ecl_fold_counts.csv (section 3); ECL conformal halves from the cached ECL test cells on
# the 1,080 resplits of Supp. Table S6 (run_ecl_trust_analysis: 6 detectors x 9 cells x 20).
t2 = []
fs = df_fstats
sg_total_n = float((fs.train_tot + fs.es_tot + fs.cal_tot + fs.te_tot).mean())
sg_total_pos = float((fs.train_pos + fs.es_pos + fs.cal_pos + fs.te_pos).mean())
for part, n_col, p_col, extra in [
        ("train", "train_tot", "train_pos", {}),
        ("early_stop", "es_tot", "es_pos", {}),
        ("threshold_cal", "cal_tot", "cal_pos", {}),
        ("test", "te_tot", "te_pos", {"grp1_pos": fs.te_Heavy_pos.mean(), "grp2_pos": fs.te_Light_pos.mean(),
                                       "grp3_pos": fs.te_Flat_pos.mean()}),
        ("conf_cal", "conf_cal_tot_mean", "conf_cal_pos_mean",
         {"grp1_pos": fs.conf_cal_Heavy_pos_mean.mean(), "grp2_pos": fs.conf_cal_Light_pos_mean.mean(),
          "grp3_pos": fs.conf_cal_Flat_pos_mean.mean()}),
        ("conf_eval", "conf_eval_tot_mean", "conf_eval_pos_mean",
         {"grp1_pos": fs.conf_eval_Heavy_pos_mean.mean(), "grp2_pos": fs.conf_eval_Light_pos_mean.mean(),
          "grp3_pos": fs.conf_eval_Flat_pos_mean.mean()})]:
    n, pos = float(fs[n_col].mean()), float(fs[p_col].mean())
    t2.append({"dataset": "SGCC", "partition": part, "n": n, "pos": pos, "pos_pct": 100.0 * pos / n,
               "share_of_total_pct": 100.0 * n / sg_total_n, "groups": "Heavy/Light/Flat", **extra})
t2.append({"dataset": "SGCC", "partition": "total", "n": sg_total_n, "pos": sg_total_pos,
           "pos_pct": 100.0 * sg_total_pos / sg_total_n, "share_of_total_pct": 100.0})

es_ = df_ecl_sum
ecl_cc = defaultdict(lambda: defaultdict(list))   # fold -> key -> list over resplits
for arm in ECL_ARMS:
    for s in (0, 1, 2):
        for f in (0, 1, 2):
            r, y, _ = load_cell(RES / "ecl_trust" / f"{arm}_{s}_{f}.csv")
            heavy = np.array(json.loads(r["val_mag_json"]), float) > float(r["mag_median_tr"])
            row_f = es_[es_.fold == f].iloc[0]
            assert len(y) == row_f.te_win and int(y.sum()) == row_f.te_pos, "ECL cache/manifests disagree"
            assert int((heavy & (y == 1)).sum()) == row_f.te_Heavy_pos, "ECL heavy split disagrees"
            for rep in range(NREP):
                ci, ti = stratified_half(y, resplit_rng(ECL_ARM_ORDER[arm], s, f, rep))
                for half, idx in (("cal", ci), ("eval", ti)):
                    ecl_cc[f][f"{half}_n"].append(len(idx)); ecl_cc[f][f"{half}_pos"].append(int(y[idx].sum()))
                    ecl_cc[f][f"{half}_hi"].append(int((heavy[idx] & (y[idx] == 1)).sum()))
                    ecl_cc[f][f"{half}_lo"].append(int((~heavy[idx] & (y[idx] == 1)).sum()))
fm = lambda k: float(np.mean([np.mean(ecl_cc[f][k]) for f in (0, 1, 2)]))
ecl_total_n = float((es_.train_win + es_.es_win + es_.cal_win + es_.te_win).mean())
ecl_total_pos = float((es_.train_pos + es_.es_pos + es_.cal_pos + es_.te_pos).mean())
for part, n, pos, extra in [
        ("train", es_.train_win.mean(), es_.train_pos.mean(), {}),
        ("early_stop", es_.es_win.mean(), es_.es_pos.mean(), {}),
        ("reserved_unused", es_.cal_win.mean(), es_.cal_pos.mean(), {}),   # not used by run_ecl_trust.py
        ("test", es_.te_win.mean(), es_.te_pos.mean(),
         {"grp1_pos": es_.te_Heavy_pos.mean(), "grp2_pos": es_.te_Light_pos.mean()}),
        ("conf_cal", fm("cal_n"), fm("cal_pos"), {"grp1_pos": fm("cal_hi"), "grp2_pos": fm("cal_lo")}),
        ("conf_eval", fm("eval_n"), fm("eval_pos"), {"grp1_pos": fm("eval_hi"), "grp2_pos": fm("eval_lo")})]:
    n, pos = float(n), float(pos)
    t2.append({"dataset": "ECL", "partition": part, "n": n, "pos": pos, "pos_pct": 100.0 * pos / n,
               "share_of_total_pct": 100.0 * n / ecl_total_n, "groups": "High/Low consumption", **extra})
t2.append({"dataset": "ECL", "partition": "total", "n": ecl_total_n, "pos": ecl_total_pos,
           "pos_pct": 100.0 * ecl_total_pos / ecl_total_n, "share_of_total_pct": 100.0})
df_t2 = pd.DataFrame(t2)
df_t2.to_csv(OUT_R2 / "table2_partition_ledger.csv", index=False)
print(df_t2.to_string(index=False))
print(f"Exported {OUT_R2 / 'table2_partition_ledger.csv'}")
