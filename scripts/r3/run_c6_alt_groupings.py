#!/usr/bin/env python3
"""Comment 6: Robustness of Mondrian Conformal Auditing Across 4 Grouping Definitions.

Note on Folds 0-2 diagnostic vs. authoritative 5-fold manuscript results:
  This standalone script evaluates the Folds 0-2 (range(3)) diagnostic grid into
  results/r3_revision/_3fold_diagnostics/C6_alt_groupings/.
  The authoritative 5-fold Revision 3 manuscript results (35.5% / 33.4% / 24.2% / 45.3%
  mean per-split workload reductions across Folds 0-4) are computed by
  scripts/r3/build_r3_5fold_tables_and_figures.py into
  results/r3_revision/r3_manuscript_verified_ledger.json and results/r3_revision/tables/.

Evaluates the post-hoc Mondrian split-conformal audit on the exact cached test-fold
predictions (SGCC Folds 0-2, all 7 detectors, identical conformal_common.py resplits
with zero eligibility gate, alpha in {0.20, 0.10, 0.05}) across four strictly
leakage-free, label-free grouping schemes fitted exclusively on D_train:
  1. original_block_contrast   : C_first5 (Wed-Sun) vs C_last2 (Mon-Tue) -> Heavy / Light / Flat
  2. calendar_weekday_weekend  : True calendar Mon-Fri vs Sat-Sun        -> Heavy / Light / Flat
  3. magnitude_tertiles        : Training-fold kWh/day tertiles          -> Low_kWh / Mid_kWh / High_kWh
  4. magnitude_x_variability   : Training-fold medians of (mean, SD)     -> 2x2 quadrants (4 groups)
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
import pandas as pd

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
OUT = R3_RES / "_3fold_diagnostics" / "C6_alt_groupings"
OUT.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec
from run_local_grid import _stratified_three_way_split
from phased.phenotype import gap_magnitude, fit_router, assign
from conformal_common import (
    NREP, n_min, resplit_rng, stratified_half, cc_recall_threshold,
    cell_summary, cell_bca,
)

ALPHAS = [0.20, 0.10, 0.05]
ARMS = [
    ("a34_NaME",   str(ORIG_RES / "a34_cell_{s}_{f}.csv"),             range(5), range(3)),
    ("HORLN",      str(ORIG_RES / "baselines/horln_{s}_{f}.csv"),      range(3), range(3)),
    ("MSDM",       str(ORIG_RES / "baselines/msdm_net_{s}_{f}.csv"),   range(3), range(3)),
    ("WDCNN",      str(ORIG_RES / "baselines/wdcnn_{s}_{f}.csv"),      range(3), range(3)),
    ("GRU-D",      str(ORIG_RES / "baselines/grud_{s}_{f}.csv"),       range(3), range(3)),
    ("HybridAttn", str(ORIG_RES / "baselines/hybridattn_{s}_{f}.csv"), range(3), range(3)),
    ("DKGT",       str(ORIG_RES / "baselines/dkgt_{s}_{f}.csv"),       range(3), range(3)),
]


def collapse_router_groups(ph: np.ndarray) -> np.ndarray:
    out = np.empty(len(ph), dtype=object)
    for i, p in enumerate(ph):
        if p in ("Heavy_Riser", "Heavy_Dropper"):
            out[i] = "Heavy"
        elif p in ("Light_Riser", "Light_Dropper"):
            out[i] = "Light"
        else:
            out[i] = "Flat"
    return out


def nanmean_zero(arr2d: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(arr2d), axis=1)
    s = np.nansum(arr2d, axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def nanstd_zero(arr2d: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(arr2d), axis=1)
    m = nanmean_zero(arr2d)[:, None]
    valid = ~np.isnan(arr2d)
    sq = np.where(valid, (arr2d - m) ** 2, 0.0).sum(axis=1)
    return np.where(cnt > 1, np.sqrt(sq / np.maximum(cnt, 1)), 0.0)


def main() -> None:
    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    sgcc_csv = SGCC_PATH
    log(f"=== C6: ALTERNATIVE MONDRIAN GROUPINGS (data={scrub_paths(str(sgcc_csv))}) ===")
    df = pd.read_csv(sgcc_csv)
    date_cols = list(df.columns[2:1031])  # first 1,029 days matching W=147 * D=7
    parsed = pd.to_datetime(date_cols, format="%Y-%m-%d")
    dow = parsed.dayofweek.to_numpy()
    wkday_mask = dow < 5   # Mon-Fri (735 days)
    wkend_mask = dow >= 5  # Sat-Sun (294 days)
    log(f"1,029 modeled days: {wkday_mask.sum()} Mon-Fri days, {wkend_mask.sum()} Sat-Sun days")

    y_all = df["FLAG"].values.astype(np.int64)
    raw = df.drop(columns=["CONS_NO", "FLAG"]).values[:, :1029].astype(np.float64)
    X_block = raw.reshape(len(df), 147, 7)

    # Precompute raw per-consumer features (each consumer's own observed series)
    g_orig, m_all = gap_magnitude(X_block)
    mu_wkday = nanmean_zero(raw[:, wkday_mask])
    mu_wkend = nanmean_zero(raw[:, wkend_mask])
    g_cal = (mu_wkday - mu_wkend).astype(np.float64)
    sd_all = nanstd_zero(raw).astype(np.float64)

    # Reconstruct Folds 0-2 identically to run_decision_value.py
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_all))
    scheme_names = [
        "original_block_contrast",
        "calendar_weekday_weekend",
        "magnitude_tertiles",
        "magnitude_x_variability",
    ]
    scheme_groups = {
        "original_block_contrast": ("Heavy", "Light", "Flat"),
        "calendar_weekday_weekend": ("Heavy", "Light", "Flat"),
        "magnitude_tertiles": ("High_kWh", "Mid_kWh", "Low_kWh"),
        "magnitude_x_variability": ("HighMag_HighVar", "HighMag_LowVar", "LowMag_HighVar", "LowMag_LowVar"),
    }

    fold_grp_te = {sc: {} for sc in scheme_names}
    te_idx_by_fold = {}
    router_cutoffs_log = []

    for f in range(3):
        tr, val = folds[f]
        _, _, te = _stratified_three_way_split(val, y_all, fractions=(0.33, 0.33, 0.34),
                                               rng=np.random.default_rng(f))
        te_idx_by_fold[f] = te

        # Assert exact label match against cached MAMRD file
        r0 = next(csv.DictReader(open(ORIG_RES / f"a34_cell_0_{f}.csv")))
        y_cached = np.array(json.loads(r0["val_labels_json"]), dtype=int)
        assert np.array_equal(y_all[te], y_cached), f"Label mismatch on fold {f}"

        # Scheme 1: original block contrast
        p1 = fit_router(g_orig[tr], m_all[tr], flat_frac=0.165)
        fold_grp_te["original_block_contrast"][f] = collapse_router_groups(assign(g_orig[te], m_all[te], p1))

        # Scheme 2: calendar Mon-Fri vs Sat-Sun
        p2 = fit_router(g_cal[tr], m_all[tr], flat_frac=0.165)
        fold_grp_te["calendar_weekday_weekend"][f] = collapse_router_groups(assign(g_cal[te], m_all[te], p2))
        router_cutoffs_log.append({
            "fold": f, "orig_tau": p1["tau"], "orig_med_d": p1["med_drop"], "orig_med_r": p1["med_rise"],
            "cal_tau": p2["tau"], "cal_med_d": p2["med_drop"], "cal_med_r": p2["med_rise"],
        })

        # Scheme 3: pure magnitude tertiles fitted strictly on tr
        q33, q67 = np.quantile(m_all[tr], [1.0 / 3.0, 2.0 / 3.0])
        g3 = np.where(m_all[te] > q67, "High_kWh", np.where(m_all[te] > q33, "Mid_kWh", "Low_kWh"))
        fold_grp_te["magnitude_tertiles"][f] = g3

        # Scheme 4: magnitude x variability 2x2 fitted strictly on tr medians
        m_med = float(np.median(m_all[tr]))
        sd_med = float(np.median(sd_all[tr]))
        hi_m = m_all[te] > m_med
        hi_s = sd_all[te] > sd_med
        g4 = np.where(hi_m & hi_s, "HighMag_HighVar",
             np.where(hi_m & (~hi_s), "HighMag_LowVar",
             np.where((~hi_m) & hi_s, "LowMag_HighVar", "LowMag_LowVar")))
        fold_grp_te["magnitude_x_variability"][f] = g4

    log("Calendar router cutoffs (Folds 0-2): " + json.dumps(router_cutoffs_log, indent=2))

    # Load all cached scores once
    preds = {}
    for arm_name, pat, seeds, fold_ids in ARMS:
        for s in seeds:
            for f in fold_ids:
                fp = Path(pat.format(s=s, f=f))
                r = next(csv.DictReader(open(fp)))
                yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
                sc = np.array(json.loads(r["val_scores_json"]), dtype=float)
                preds[(arm_name, s, f)] = (yt, sc)

    summary_rows = []
    group_detail_rows = []

    for sc_name in scheme_names:
        grps = scheme_groups[sc_name]
        # Base prevalence per group across Folds 0-2
        base_rates = {}
        for gn in grps:
            pos = sum(int(((fold_grp_te[sc_name][f] == gn) & (y_all[te_idx_by_fold[f]] == 1)).sum()) for f in range(3))
            tot = sum(int((fold_grp_te[sc_name][f] == gn).sum()) for f in range(3))
            base_rates[gn] = pos / max(tot, 1)

        for alpha in ALPHAS:
            req_n = n_min(alpha)
            target = 1.0 - alpha
            for arm_idx, (arm_name, _, seeds, fold_ids) in enumerate(ARMS):
                cond_cells, sing_cells, red_cells = [], [], []
                rec_all_cells = []
                grp_stats = {gn: {"ncal": [], "rec": [], "viol": [], "prec": [], "triv": []} for gn in grps}

                for s in seeds:
                    for f in fold_ids:
                        yt, scores = preds[(arm_name, s, f)]
                        grp = fold_grp_te[sc_name][f]
                        c_arr, s_arr, r_arr, ra_arr = [], [], [], []
                        g_cell = {gn: {"ncal": [], "rec": [], "viol": [], "prec": [], "triv": []} for gn in grps}

                        for rep in range(NREP):
                            rng = resplit_rng(arm_idx, s, f, rep)
                            ci, ti = stratified_half(yt, rng)
                            yc, sc_c, gc = yt[ci], scores[ci], grp[ci]
                            ye, sc_e, ge = yt[ti], scores[ti], grp[ti]

                            thrs = {}
                            for gn in grps:
                                mcal = gc == gn
                                npos = int(yc[mcal].sum())
                                thr_g, triv_g = cc_recall_threshold(sc_c[mcal], yc[mcal], alpha)
                                thrs[gn] = thr_g
                                g_cell[gn]["ncal"].append(npos)
                                g_cell[gn]["triv"].append(1.0 if triv_g else 0.0)

                            single_thr = min(thrs.values())
                            cond_flag = np.zeros(len(ye), dtype=bool)
                            for gn in grps:
                                me = ge == gn
                                f_g = sc_e[me] >= thrs[gn]
                                cond_flag[me] = f_g
                                pos_e = ye[me] == 1
                                if pos_e.any():
                                    rec_g = float(f_g[pos_e].mean())
                                    g_cell[gn]["rec"].append(rec_g)
                                    g_cell[gn]["viol"].append(1.0 if rec_g < target else 0.0)
                                if f_g.any():
                                    g_cell[gn]["prec"].append(float(ye[me][f_g].mean()))

                            sing_flag = sc_e >= single_thr
                            c_frac = float(cond_flag.mean())
                            s_frac = float(sing_flag.mean())
                            red = 1.0 - (c_frac / s_frac if s_frac > 0 else 1.0)
                            c_arr.append(c_frac); s_arr.append(s_frac); r_arr.append(red)
                            ra_arr.append(float(cond_flag[ye == 1].mean()))

                        cond_cells.append(c_arr); sing_cells.append(s_arr); red_cells.append(r_arr)
                        rec_all_cells.append(ra_arr)
                        for gn in grps:
                            for k_m in ("ncal", "rec", "viol", "prec", "triv"):
                                grp_stats[gn][k_m].append(g_cell[gn][k_m])

                c_mean, c_pool, n_cells, n_vals = cell_summary(cond_cells)
                s_mean, _, _, _ = cell_summary(sing_cells)
                r_mean, r_pool, _, _ = cell_summary(red_cells)
                ra_mean, _, _, _ = cell_summary(rec_all_cells)
                assert np.isclose(r_mean, r_pool, atol=1e-9)
                r_lo, r_hi = cell_bca(red_cells, B=2000, seed=0)

                summary_rows.append({
                    "scheme": sc_name,
                    "guarantee": round(target, 2),
                    "alpha": alpha,
                    "arm": arm_name,
                    "n_cells": n_cells,
                    "n_resplits": n_vals,
                    "cond_insp_pct": 100.0 * c_mean,
                    "single_insp_pct": 100.0 * s_mean,
                    "workload_red_pct": 100.0 * r_mean,
                    "workload_red_ci_lo_pct": 100.0 * r_lo,
                    "workload_red_ci_hi_pct": 100.0 * r_hi,
                    "overall_realized_recall_pct": 100.0 * ra_mean,
                })

                for gn in grps:
                    nc_vals = np.concatenate(grp_stats[gn]["ncal"])
                    rec_m, _, _, _ = cell_summary(grp_stats[gn]["rec"])
                    viol_m, _, _, _ = cell_summary(grp_stats[gn]["viol"])
                    prec_m, _, _, _ = cell_summary(grp_stats[gn]["prec"])
                    triv_m, _, _, _ = cell_summary(grp_stats[gn]["triv"])
                    group_detail_rows.append({
                        "scheme": sc_name,
                        "guarantee": round(target, 2),
                        "arm": arm_name,
                        "group": gn,
                        "base_rate_pct": 100.0 * base_rates[gn],
                        "ncal_mean": float(nc_vals.mean()),
                        "ncal_min": int(nc_vals.min()),
                        "ncal_max": int(nc_vals.max()),
                        "n_min_req": int(req_n),
                        "finite_quantile_pct": 100.0 * (1.0 - triv_m),
                        "trivial_fallback_pct": 100.0 * triv_m,
                        "realized_recall_pct": 100.0 * rec_m,
                        "target_violation_pct": 100.0 * viol_m,
                        "hit_rate_pct": 100.0 * prec_m,
                        "lift": float(prec_m / base_rates[gn]) if base_rates[gn] > 0 else float("nan"),
                    })

    df_sum = pd.DataFrame(summary_rows)
    df_det = pd.DataFrame(group_detail_rows)
    df_sum.to_csv(OUT / "alt_groupings_summary.csv", index=False)
    df_det.to_csv(OUT / "alt_groupings_group_details.csv", index=False)

    df_mamrd = df_sum[(df_sum["arm"] == "a34_NaME")].copy()
    df_mamrd.to_csv(OUT / "alt_groupings_mamrd_comparison.csv", index=False)

    # Sanity check against Table 6 MAMRD: 28.6% / 49.1%, 35.5% [30.2%, 40.4%]
    chk = df_mamrd[(df_mamrd["scheme"] == "original_block_contrast") & (df_mamrd["guarantee"] == 0.8)].iloc[0]
    assert abs(chk["cond_insp_pct"] - 28.6) < 0.2 and abs(chk["workload_red_pct"] - 35.5) < 0.2, chk.to_dict()
    log(f"\n[SANITY PASS] original_block_contrast @ 0.80 MAMRD matches Table 6: "
        f"Cond={chk['cond_insp_pct']:.1f}% Single={chk['single_insp_pct']:.1f}% Red={chk['workload_red_pct']:.1f}%")

    log("\n--- MAMRD Across 4 Grouping Schemes (@ 0.80, 0.90, 0.95) ---")
    log(df_mamrd[["scheme", "guarantee", "cond_insp_pct", "single_insp_pct",
                  "workload_red_pct", "workload_red_ci_lo_pct", "workload_red_ci_hi_pct",
                  "overall_realized_recall_pct"]].to_string(index=False))

    log("\n--- MAMRD Per-Group Eligibility & Recall Details (@ 0.80) ---")
    log(df_det[(df_det["arm"] == "a34_NaME") & (df_det["guarantee"] == 0.8)][
        ["scheme", "group", "base_rate_pct", "ncal_mean", "ncal_min", "finite_quantile_pct",
         "realized_recall_pct", "target_violation_pct", "hit_rate_pct", "lift"]
    ].to_string(index=False))

    (OUT / "c6_summary.json").write_text(json.dumps({
        "router_cutoffs_folds0_2": router_cutoffs_log,
        "mamrd_across_schemes": df_mamrd.to_dict(orient="records"),
    }, indent=2))
    log(f"\n[DONE] Saved C6 artifacts to {OUT}")
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
