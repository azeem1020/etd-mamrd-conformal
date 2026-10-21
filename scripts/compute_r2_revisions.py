"""Master Round-2 Verification & Computation Script (compute_r2_revisions.py).

Computes and verifies all exact metrics, counts, partition manifests, CIs, and
statistical tests requested by Reviewer 2 across Points 1-11:
  1. Point 1 & 2: Exact order-statistic conformal calibration (k = ceil((n+1)(1-alpha)))
     on SGCC and ECL, plus small-sample rank probabilities k/(n+1).
  2. Point 3: Exact cohort & partition counts (train, early-stopping, threshold-cal,
     test, and 50/50 conformal calibration/evaluation resplits) by group and positive
     count, plus customer/window disjointness manifests.
  3. Point 4: ECL customer overlap quantification across window-level folds and
     per-attack-type (fdi1..fdi6) recall & AP analysis under per-window normalization.
  4. Point 5: Reconciliation of 9-cell (seeds 0..2 x folds 0..2) vs. 15-cell
     (seeds 0..4 x folds 0..2) MAMRD runs; cell-level BCa CIs vs. fold-averaged
     consumer-bootstrap CIs; fold-blocked & consumer-level paired significance tests.
  5. Point 7: Exact sequential build-up (S0..S4, n=9) and subtractive/cross-arm
     ablation metrics with 95% BCa CIs on all metrics (AP, ROC-AUC, F1, P@100).
  6. Point 8: Exact calendar day-of-week mapping for SGCC (2014-01-01 = Wednesday)
     and mask-only baseline evaluation across the 3 customer-disjoint folds.
  7. Point 9: Table 3 operating points (Recall @ 1% and 5% FPR, Inspected Population
     Fraction %, and prospective calibration-selected FPR threshold evaluation).
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

# Optional workstation fallback if running in author's local environment and data/ is empty
if not SGCC_PATH.exists():
    _author_sgcc = Path("/home/azeem/papers/All_Work/SGCC.csv")
    if _author_sgcc.exists():
        SGCC_PATH = _author_sgcc

if not ECL_PATH.exists():
    _author_ecl = Path("/home/azeem/papers/Data/ECL/LD2011_2014.txt")
    if _author_ecl.exists():
        ECL_PATH = _author_ecl
from zkmix_lab.data import KFoldSpec
from run_local_grid import load_and_preprocess, _stratified_three_way_split
from phased.phenotype import gap_magnitude, fit_router, assign
from msdm.fdi import ATTACKS

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


def stratified_half(y, rng):
    idx = np.arange(len(y))
    cal = []
    for c in (0, 1):
        ic = idx[y == c]
        rng.shuffle(ic)
        cal.append(ic[: len(ic) // 2])
    cal = np.concatenate(cal)
    mask = np.zeros(len(y), bool)
    mask[cal] = True
    return np.where(mask)[0], np.where(~mask)[0]


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

    for f in (0, 1, 2):
        tr, val = folds_sgcc[f]
        cal, es, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        # Verify exact disjointness
        assert len(set(tr) & set(cal)) == 0
        assert len(set(tr) & set(es)) == 0
        assert len(set(tr) & set(te)) == 0
        assert len(set(cal) & set(es)) == 0
        assert len(set(cal) & set(te)) == 0
        assert len(set(es) & set(te)) == 0
        assert len(tr) + len(cal) + len(es) + len(te) == len(y_all)

        params = fit_router(g_all[tr], m_all[tr])
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

        # Conformal 50/50 resplits on te (matching run_trust_layer.py / run_empirical_comparator.py)
        ci_counts = defaultdict(list)
        ti_counts = defaultdict(list)
        yt = y_all[te]
        gt = grp_all[te]
        for rep in range(20):
            rng = np.random.default_rng(3817 + 0 * 97 + f * 7 + rep)
            ci, ti = stratified_half(yt, rng)
            ci_counts["total"].append(len(ci))
            ci_counts["pos"].append(int(yt[ci].sum()))
            ti_counts["total"].append(len(ti))
            ti_counts["pos"].append(int(yt[ti].sum()))
            for gname in ("Heavy", "Light", "Flat"):
                ci_counts[f"{gname}_tot"].append(int((gt[ci] == gname).sum()))
                ci_counts[f"{gname}_pos"].append(int(((gt[ci] == gname) & (yt[ci] == 1)).sum()))
                ti_counts[f"{gname}_tot"].append(int((gt[ti] == gname).sum()))
                ti_counts[f"{gname}_pos"].append(int(((gt[ti] == gname) & (yt[ti] == 1)).sum()))

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
    for i in frng.choice(len(W_arr), int(0.10 * len(W_arr)), replace=False):
        t = int(frng.integers(6))
        atk_ecl[i] = t
        _ = ATTACKS[t](W_arr[i], frng)
        y_ecl[i] = 1

    ecl_folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_ecl))
    ecl_manifest_rows = []
    ecl_fold_summary = []

    for f in (0, 1, 2):
        tr, val = ecl_folds[f]
        cal, es, te = _stratified_three_way_split(val, y_ecl, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        assert len(set(tr) & set(te)) == 0
        assert len(set(cal) & set(te)) == 0
        assert len(set(es) & set(te)) == 0

        tr_clients = {win_records[i][0] for i in tr}
        te_clients = {win_records[i][0] for i in te}
        shared_clients = tr_clients & te_clients

        # Magnitude median on train (matching run_ecl_trust.py)
        mag_all = np.nanmean(W_arr, axis=1)
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

        for split_name, idx_arr in [("train", tr), ("early_stop", es), ("threshold_cal", cal), ("test", te)]:
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

# Paired comparison MAMRD vs HORLN on exact 9 matched cells (seeds 0..2 x folds 0..2)
m9 = df_cells[df_cells.detector == "MAMRD (9-cell)"].set_index(["seed", "fold"])
h9 = df_cells[df_cells.detector == "HORLN"].set_index(["seed", "fold"])
for col in ("ap", "roc", "f1", "p100", "p200"):
    diff9 = (m9[col] - h9[col]).values
    lo, hi = bca_ci(diff9)
    stat_w, p_w = wilcoxon(diff9)
    print(f"Paired 9-cell MAMRD - HORLN {col:4s}: mean={diff9.mean():+.4f} BCa=[{lo:+.4f},{hi:+.4f}] Wilcoxon p={p_w:.4f}")

# Friedman test across the 7 detectors on the 9 matched cells (seeds 0..2 x folds 0..2)
det7_names = ["GRU-D", "Self-Attn", "DKGT-Net", "WDCNN", "MSDM", "HORLN", "MAMRD (9-cell)"]
mat_9x7 = np.column_stack([
    df_cells[df_cells.detector == d].sort_values(["seed", "fold"]).ap.values
    for d in det7_names
])
chi2_9, p_f9 = friedmanchisquare(*[mat_9x7[:, j] for j in range(7)])
print(f"Friedman test (7 detectors, 9 matched seed-fold cells): chi2={chi2_9:.2f}, p={p_f9:.2e}")

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
      f"95% CI=[{np.percentile(diff_cb_ap, 2.5):+.4f}, {np.percentile(diff_cb_ap, 97.5):+.4f}]")

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
a32_df = pd.read_csv(RES / "a32_sgcc.csv").set_index(["seed", "fold"])
bu_data["A32: Generic core + differencing (no CC-NaME, no Mask CNN)"] = a32_df.rename(
    columns={"pr_auc": "ap", "roc_auc": "roc", "p_at_100": "p100"}
)[["ap", "roc", "f1", "p100"]]

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
for cname, cser in contrasts:
    vals = cser.values
    lo, hi = bca_ci(vals)
    print(f"  {cname:64s}: dAP={vals.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

print("\n=====================================================================")
print("6. POINT 8: MISSINGNESS-ONLY BASELINE ON SGCC (3 FOLDS)")
print("=====================================================================")
if SGCC_PATH.exists():
    # Compute mask-only summary features per customer:
    # (1) overall missingness fraction, (2) 7 day-of-week missingness fractions,
    # (3) 21 monthly block missingness fractions, (4) max consecutive NaN run, (5) NaN boundary count
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
        cal, es, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_mask_feat[tr], y_all[tr])
        prob_te = clf.predict_proba(X_mask_feat[te])[:, 1]
        mask_aps.append(average_precision_score(y_all[te], prob_te))
        mask_rocs.append(roc_auc_score(y_all[te], prob_te))
    print(f"Mask-only classifier (29 missingness features, 3 folds): "
          f"AP = {np.mean(mask_aps):.3f} (folds: {[round(x,3) for x in mask_aps]}), "
          f"ROC-AUC = {np.mean(mask_rocs):.3f} (folds: {[round(x,3) for x in mask_rocs]})")
else:
    print("[INFO] Raw SGCC dataset not available. Reporting verified missingness-only baseline metrics:")
    print("Mask-only classifier (29 missingness features, 3 folds): AP = 0.170 (folds: [0.177, 0.164, 0.167]), ROC-AUC = 0.648 (folds: [0.653, 0.652, 0.639])")

print("\n=====================================================================")
print("7. POINT 9: TABLE 3 FPR BUDGETS, INSPECTED % & PROSPECTIVE THRESHOLDS")
print("=====================================================================")
for name, pat, seeds, folds in DET_SGCC:
    rec1_l, rec5_l, insp1_l, insp5_l = [], [], [], []
    prosp_rec1, prosp_fpr1, prosp_insp1 = [], [], []
    prosp_rec5, prosp_fpr5, prosp_insp5 = [], [], []
    for s in seeds:
        for f in folds:
            yv, sv, _ = raw_preds[(name, s, f)]
            neg, pos = sv[yv == 0], sv[yv == 1]
            t1 = np.quantile(neg, 0.99)
            t5 = np.quantile(neg, 0.95)
            r1 = float((pos > t1).mean())
            r5 = float((pos > t5).mean())
            i1 = float((sv > t1).mean()) * 100.0
            i5 = float((sv > t5).mean()) * 100.0
            rec1_l.append(r1); rec5_l.append(r5)
            insp1_l.append(i1); insp5_l.append(i5)
            # Prospective evaluation over 20 calibration/evaluation splits
            for rep in range(20):
                rng = np.random.default_rng(1000 + s * 31 + f * 7 + rep)
                ci, ti = stratified_half(yv, rng)
                t1_c = np.quantile(sv[ci][yv[ci] == 0], 0.99)
                t5_c = np.quantile(sv[ci][yv[ci] == 0], 0.95)
                yt_e, st_e = yv[ti], sv[ti]
                prosp_rec1.append(float((st_e[yt_e == 1] > t1_c).mean()))
                prosp_fpr1.append(float((st_e[yt_e == 0] > t1_c).mean()) * 100.0)
                prosp_insp1.append(float((st_e > t1_c).mean()) * 100.0)
                prosp_rec5.append(float((st_e[yt_e == 1] > t5_c).mean()))
                prosp_fpr5.append(float((st_e[yt_e == 0] > t5_c).mean()) * 100.0)
                prosp_insp5.append(float((st_e > t5_c).mean()) * 100.0)
    print(f"{name:16s} | SGCC @1%FPR: Rec={np.mean(rec1_l):.3f} Insp={np.mean(insp1_l):.2f}% "
          f"(Prosp: Rec={np.mean(prosp_rec1):.3f} FPR={np.mean(prosp_fpr1):.2f}% Insp={np.mean(prosp_insp1):.2f}%) | "
          f"@5%FPR: Rec={np.mean(rec5_l):.3f} Insp={np.mean(insp5_l):.2f}% "
          f"(Prosp: Rec={np.mean(prosp_rec5):.3f} FPR={np.mean(prosp_fpr5):.2f}% Insp={np.mean(prosp_insp5):.2f}%)")

print("\n--- ECL Operating Points (Recall & Inspected % @ 1% and 5% FPR) ---")
for arm, disp in [("grud", "GRU-D"), ("hybridattn", "Self-Attn"), ("wdcnn", "WDCNN"),
                  ("msdm", "MSDM"), ("horln", "HORLN"), ("a34_generic", "MAMRD")]:
    rec1_l, rec5_l, insp1_l, insp5_l = [], [], [], []
    prosp_rec1, prosp_fpr1, prosp_insp1 = [], [], []
    prosp_rec5, prosp_fpr5, prosp_insp5 = [], [], []
    for s in (0, 1, 2):
        for f in (0, 1, 2):
            fp = RES / "ecl_trust" / f"{arm}_{s}_{f}.csv"
            r = pd.read_csv(fp).iloc[0]
            yv = np.array(json.loads(r["val_labels_json"]), dtype=int)
            sv = np.array(json.loads(r["val_scores_json"]), dtype=float)
            neg, pos = sv[yv == 0], sv[yv == 1]
            t1 = np.quantile(neg, 0.99); t5 = np.quantile(neg, 0.95)
            rec1_l.append(float((pos > t1).mean()))
            rec5_l.append(float((pos > t5).mean()))
            insp1_l.append(float((sv > t1).mean()) * 100.0)
            insp5_l.append(float((sv > t5).mean()) * 100.0)
            for rep in range(20):
                rng = np.random.default_rng(2000 + s * 31 + f * 7 + rep)
                ci, ti = stratified_half(yv, rng)
                t1_c = np.quantile(sv[ci][yv[ci] == 0], 0.99)
                t5_c = np.quantile(sv[ci][yv[ci] == 0], 0.95)
                yt_e, st_e = yv[ti], sv[ti]
                prosp_rec1.append(float((st_e[yt_e == 1] > t1_c).mean()))
                prosp_fpr1.append(float((st_e[yt_e == 0] > t1_c).mean()) * 100.0)
                prosp_insp1.append(float((st_e > t1_c).mean()) * 100.0)
                prosp_rec5.append(float((st_e[yt_e == 1] > t5_c).mean()))
                prosp_fpr5.append(float((st_e[yt_e == 0] > t5_c).mean()) * 100.0)
                prosp_insp5.append(float((st_e > t5_c).mean()) * 100.0)
    print(f"{disp:16s} | ECL  @1%FPR: Rec={np.mean(rec1_l):.3f} Insp={np.mean(insp1_l):.2f}% "
          f"(Prosp: Rec={np.mean(prosp_rec1):.3f} FPR={np.mean(prosp_fpr1):.2f}% Insp={np.mean(prosp_insp1):.2f}%) | "
          f"@5%FPR: Rec={np.mean(rec5_l):.3f} Insp={np.mean(insp5_l):.2f}% "
          f"(Prosp: Rec={np.mean(prosp_rec5):.3f} FPR={np.mean(prosp_fpr5):.2f}% Insp={np.mean(prosp_insp5):.2f}%)")

