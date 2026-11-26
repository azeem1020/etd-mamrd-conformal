#!/usr/bin/env python3
"""Comment 4: Conformal Calibration Sample-Size, Selective-Inspection IPW, & Distribution-Shift Sensitivity.

Note on Folds 0-2 diagnostic vs. authoritative 5-fold manuscript results:
  This standalone script evaluates the Folds 0-2 (range(3)) diagnostic grid into
  results/r3_revision/_3fold_diagnostics/C4_conformal_sensitivity/.
  The authoritative 5-fold Revision 3 manuscript results (115 fits / 500 splits,
  76 unique Flat thefts across 5 folds) are computed by
  scripts/r3/build_r3_5fold_tables_and_figures.py into
  results/r3_revision/r3_manuscript_verified_ledger.json and results/r3_revision/tables/.

Runs 100% on CPU from cached MAMRD 15-cell SGCC predictions (Folds 0-2):
  Part A1: Within-fold calibration split fraction sweep f_cal in {0.20, 0.35, 0.50, 0.65, 0.80}
           at target recall 0.80 and 0.90 across 15 cells x 20 repeats = 300 splits.
  Part A2: Pooled-3-fold larger calibration sweep per seed (f_cal in {0.15, 0.25, 0.40, 0.50, 0.65})
           yielding larger n_flat up to ~31 thefts at target recall {0.80, 0.90, 0.95}.
  Part B:  Selective-inspection bias & Tibshirani et al. (2019) Corollary 1 IPW recovery:
           - Bernoulli selective inspection with propensity e(x) = max(p_min, s^gamma),
             p_min=0.05, gamma in {0.0, 0.5, 1.0, 1.5, 2.0}, comparing Naive unweighted vs.
             Exact IPW weighted conformal quantile.
           - Plus positivity-floor violation test (deterministic top-50% truncation).
  Part C:  Distribution shift stress tests (exchangeability violated):
           - Cross-fold transfer (calibrate Fold f1 -> evaluate Fold f2 != f1)
           - Missingness-regime shift (low-missingness cal -> high-missingness eval, & reverse)
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
OUT = R3_RES / "_3fold_diagnostics" / "C4_conformal_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

from zkmix_lab.data import KFoldSpec
from run_local_grid import _stratified_three_way_split
from phased.phenotype import gap_magnitude, fit_router, assign
from conformal_common import NREP, resplit_rng, cc_recall_threshold, cell_summary, cell_bca

GROUPS = ("Heavy", "Light", "Flat")


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


def stratified_frac(y: np.ndarray, f_cal: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    cal = []
    for c in (0, 1):
        ic = idx[y == c].copy()
        rng.shuffle(ic)
        k = max(1, int(round(f_cal * len(ic))))
        k = min(k, len(ic) - 1)
        cal.append(ic[:k])
    cal_idx = np.concatenate(cal)
    mask = np.zeros(len(y), dtype=bool)
    mask[cal_idx] = True
    return np.where(mask)[0], np.where(~mask)[0]


def weighted_conformal_flags(s_cal_pos: np.ndarray, w_cal_pos: np.ndarray,
                             s_eval: np.ndarray, w_eval: np.ndarray, alpha: float) -> np.ndarray:
    """Exact Tibshirani et al. (2019) weighted split-conformal quantile for nonconformity R = 1 - s.

    For each test unit j with weight w_j, form normalized masses:
      p_i(j) = w_i / (sum_k w_k + w_j),  p_test(j) = w_j / (sum_k w_k + w_j).
    Flag j iff R_j <= WeightedQuantile_{1-alpha}({(R_i, p_i(j))}_i union {(inf, p_test(j))}),
    equivalently flag iff total weight of calibration thefts with R_i < R_j (strictly smaller
    nonconformity = strictly larger score s_i > s_j) is < (1 - alpha) * (sum_k w_k + w_j)?
    Careful with ties: flag iff R_j <= q_hat where q_hat = inf{r: sum_{R_i <= r} w_i >= (1-alpha)*(W + w_j)}.
    """
    n = len(s_cal_pos)
    if n == 0:
        return np.ones(len(s_eval), dtype=bool)
    r_cal = 1.0 - np.asarray(s_cal_pos, dtype=float)
    w_cal = np.asarray(w_cal_pos, dtype=float)
    order = np.argsort(r_cal, kind="mergesort")
    r_sorted = r_cal[order]
    w_cum = np.cumsum(w_cal[order])
    W_tot = float(w_cum[-1])

    r_eval = 1.0 - np.asarray(s_eval, dtype=float)
    needed = (1.0 - alpha) * (W_tot + np.asarray(w_eval, dtype=float))
    # First index idx where w_cum[idx] >= needed
    pos_idx = np.searchsorted(w_cum, needed, side="left")
    flags = np.empty(len(s_eval), dtype=bool)
    inf_mask = pos_idx >= n
    flags[inf_mask] = True
    fin_mask = ~inf_mask
    if fin_mask.any():
        flags[fin_mask] = r_eval[fin_mask] <= r_sorted[pos_idx[fin_mask]]
    return flags


def main() -> None:
    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    sgcc_csv = SGCC_PATH
    log(f"=== C4: CONFORMAL SAMPLE-SIZE, IPW SELECTIVE-INSPECTION & SHIFT SENSITIVITY ===")
    df = pd.read_csv(sgcc_csv)
    y_all = df["FLAG"].values.astype(np.int64)
    raw = df.drop(columns=["CONS_NO", "FLAG"]).values[:, :1029].astype(np.float64)
    miss_rate_all = np.isnan(raw).mean(axis=1)
    g_all, m_all = gap_magnitude(raw.reshape(len(df), 147, 7))

    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_all))
    fold_info = {}
    for f in range(3):
        tr, val = folds[f]
        _, _, te = _stratified_three_way_split(val, y_all, (0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        ph = assign(g_all[te], m_all[te], fit_router(g_all[tr], m_all[tr]))
        fold_info[f] = (te, collapse_grp(ph), miss_rate_all[te])

    preds = {}
    for s in range(5):
        for f in range(3):
            r = next(csv.DictReader(open(ORIG_RES / f"a34_cell_{s}_{f}.csv")))
            yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
            sc = np.array(json.loads(r["val_scores_json"]), dtype=float)
            assert np.array_equal(yt, y_all[fold_info[f][0]])
            preds[(s, f)] = (yt, sc)

    # ------------------------------------------------------------------
    # PART A1: Within-fold calibration fraction sweep
    # ------------------------------------------------------------------
    a1_rows = []
    for alpha in (0.20, 0.10):
        target = 1.0 - alpha
        for f_cal in (0.20, 0.35, 0.50, 0.65, 0.80):
            cond_cells, sing_cells, red_cells = [], [], []
            g_stats = {gn: {"ncal": [], "rec": [], "viol": [], "triv": []} for gn in GROUPS}
            for s in range(5):
                for f in range(3):
                    yt, sc = preds[(s, f)]
                    grp = fold_info[f][1]
                    c_arr, s_arr, r_arr = [], [], []
                    gc_cell = {gn: {"ncal": [], "rec": [], "viol": [], "triv": []} for gn in GROUPS}
                    for rep in range(NREP):
                        rng = resplit_rng(100 + int(round(100 * f_cal)), s, f, rep)
                        ci, ti = stratified_frac(yt, f_cal, rng)
                        yc, sc_c, gc = yt[ci], sc[ci], grp[ci]
                        ye, sc_e, ge = yt[ti], sc[ti], grp[ti]
                        thrs = {}
                        for gn in GROUPS:
                            mc = gc == gn
                            thr_g, triv_g = cc_recall_threshold(sc_c[mc], yc[mc], alpha)
                            thrs[gn] = thr_g
                            gc_cell[gn]["ncal"].append(int(yc[mc].sum()))
                            gc_cell[gn]["triv"].append(1.0 if triv_g else 0.0)
                        s_thr = min(thrs.values())
                        cf = np.zeros(len(ye), dtype=bool)
                        for gn in GROUPS:
                            me = ge == gn
                            fg = sc_e[me] >= thrs[gn]
                            cf[me] = fg
                            pos_e = ye[me] == 1
                            if pos_e.any():
                                rec_g = float(fg[pos_e].mean())
                                gc_cell[gn]["rec"].append(rec_g)
                                gc_cell[gn]["viol"].append(1.0 if rec_g < target else 0.0)
                        sf = sc_e >= s_thr
                        cf_f, sf_f = float(cf.mean()), float(sf.mean())
                        c_arr.append(cf_f); s_arr.append(sf_f)
                        r_arr.append(1.0 - (cf_f / sf_f if sf_f > 0 else 1.0))
                    cond_cells.append(c_arr); sing_cells.append(s_arr); red_cells.append(r_arr)
                    for gn in GROUPS:
                        for km in ("ncal", "rec", "viol", "triv"):
                            g_stats[gn][km].append(gc_cell[gn][km])
            c_m, _, _, _ = cell_summary(cond_cells)
            s_m, _, _, _ = cell_summary(sing_cells)
            r_m, _, _, _ = cell_summary(red_cells)
            for gn in GROUPS:
                nc_flat = np.concatenate(g_stats[gn]["ncal"])
                rec_flat = np.concatenate(g_stats[gn]["rec"])
                rec_m, _, _, _ = cell_summary(g_stats[gn]["rec"])
                viol_m, _, _, _ = cell_summary(g_stats[gn]["viol"])
                triv_m, _, _, _ = cell_summary(g_stats[gn]["triv"])
                a1_rows.append({
                    "target_recall": round(target, 2), "f_cal": f_cal, "group": gn,
                    "ncal_mean": float(nc_flat.mean()), "ncal_min": int(nc_flat.min()), "ncal_max": int(nc_flat.max()),
                    "finite_pct": 100.0 * (1.0 - triv_m),
                    "realized_recall_pct": 100.0 * rec_m,
                    "recall_sd_pct": 100.0 * float(rec_flat.std(ddof=1)),
                    "violation_pct": 100.0 * viol_m,
                    "cond_insp_pct": 100.0 * c_m, "single_insp_pct": 100.0 * s_m, "workload_red_pct": 100.0 * r_m,
                })
    df_a1 = pd.DataFrame(a1_rows)
    df_a1.to_csv(OUT / "c4a_sample_size_sweep.csv", index=False)
    log("\n--- Part A1: Within-Fold Calibration Size Sweep (Flat group) ---")
    log(df_a1[df_a1["group"] == "Flat"].to_string(index=False))

    # ------------------------------------------------------------------
    # PART A2: Pooled 3-fold per seed (larger Flat n_cal up to ~31)
    # ------------------------------------------------------------------
    a2_rows = []
    for alpha in (0.20, 0.10, 0.05):
        target = 1.0 - alpha
        for f_cal in (0.15, 0.25, 0.40, 0.50, 0.65):
            cond_cells, sing_cells, red_cells = [], [], []
            g_stats = {gn: {"ncal": [], "rec": [], "viol": [], "triv": []} for gn in GROUPS}
            for s in range(5):
                yt = np.concatenate([preds[(s, f)][0] for f in range(3)])
                sc = np.concatenate([preds[(s, f)][1] for f in range(3)])
                grp = np.concatenate([fold_info[f][1] for f in range(3)])
                c_arr, s_arr, r_arr = [], [], []
                gc_cell = {gn: {"ncal": [], "rec": [], "viol": [], "triv": []} for gn in GROUPS}
                for rep in range(NREP):
                    rng = resplit_rng(200 + int(round(100 * f_cal)), s, 0, rep)
                    ci, ti = stratified_frac(yt, f_cal, rng)
                    yc, sc_c, gc = yt[ci], sc[ci], grp[ci]
                    ye, sc_e, ge = yt[ti], sc[ti], grp[ti]
                    thrs = {}
                    for gn in GROUPS:
                        mc = gc == gn
                        thr_g, triv_g = cc_recall_threshold(sc_c[mc], yc[mc], alpha)
                        thrs[gn] = thr_g
                        gc_cell[gn]["ncal"].append(int(yc[mc].sum()))
                        gc_cell[gn]["triv"].append(1.0 if triv_g else 0.0)
                    s_thr = min(thrs.values())
                    cf = np.zeros(len(ye), dtype=bool)
                    for gn in GROUPS:
                        me = ge == gn
                        fg = sc_e[me] >= thrs[gn]
                        cf[me] = fg
                        pos_e = ye[me] == 1
                        if pos_e.any():
                            rec_g = float(fg[pos_e].mean())
                            gc_cell[gn]["rec"].append(rec_g)
                            gc_cell[gn]["viol"].append(1.0 if rec_g < target else 0.0)
                    sf = sc_e >= s_thr
                    cf_f, sf_f = float(cf.mean()), float(sf.mean())
                    c_arr.append(cf_f); s_arr.append(sf_f)
                    r_arr.append(1.0 - (cf_f / sf_f if sf_f > 0 else 1.0))
                cond_cells.append(c_arr); sing_cells.append(s_arr); red_cells.append(r_arr)
                for gn in GROUPS:
                    for km in ("ncal", "rec", "viol", "triv"):
                        g_stats[gn][km].append(gc_cell[gn][km])
            c_m, _, _, _ = cell_summary(cond_cells)
            s_m, _, _, _ = cell_summary(sing_cells)
            r_m, _, _, _ = cell_summary(red_cells)
            for gn in GROUPS:
                nc_flat = np.concatenate(g_stats[gn]["ncal"])
                rec_flat = np.concatenate(g_stats[gn]["rec"])
                rec_m, _, _, _ = cell_summary(g_stats[gn]["rec"])
                viol_m, _, _, _ = cell_summary(g_stats[gn]["viol"])
                triv_m, _, _, _ = cell_summary(g_stats[gn]["triv"])
                a2_rows.append({
                    "target_recall": round(target, 2), "f_cal": f_cal, "group": gn,
                    "ncal_mean": float(nc_flat.mean()), "ncal_min": int(nc_flat.min()), "ncal_max": int(nc_flat.max()),
                    "finite_pct": 100.0 * (1.0 - triv_m),
                    "realized_recall_pct": 100.0 * rec_m,
                    "recall_sd_pct": 100.0 * float(rec_flat.std(ddof=1)),
                    "violation_pct": 100.0 * viol_m,
                    "cond_insp_pct": 100.0 * c_m, "single_insp_pct": 100.0 * s_m, "workload_red_pct": 100.0 * r_m,
                })
    df_a2 = pd.DataFrame(a2_rows)
    df_a2.to_csv(OUT / "c4a_pooled_size_sweep.csv", index=False)
    log("\n--- Part A2: Pooled 3-Fold Calibration Size Sweep (Flat group) ---")
    log(df_a2[df_a2["group"] == "Flat"].to_string(index=False))

    # ------------------------------------------------------------------
    # PART B: Selective-Inspection Bias & Tibshirani IPW Weighted Conformal
    # ------------------------------------------------------------------
    alpha = 0.20
    target = 0.80
    b_rows = []
    configs = [
        ("uniform_e1", 0.0, False),
        ("selective_gamma0.5", 0.5, False),
        ("selective_gamma1.0", 1.0, False),
        ("selective_gamma1.5", 1.5, False),
        ("selective_gamma2.0", 2.0, False),
        ("truncated_top50_positivity_violated", 0.0, True),
    ]
    for cfg_name, gamma, truncate in configs:
        for method in ("naive_unweighted", "tibshirani_ipw"):
            rec_overall, insp_overall = [], []
            g_rec = {gn: [] for gn in GROUPS}
            g_viol = {gn: [] for gn in GROUPS}
            g_ncal = {gn: [] for gn in GROUPS}

            for s in range(5):
                for f in range(3):
                    yt, sc = preds[(s, f)]
                    grp = fold_info[f][1]
                    ro_c, io_c = [], []
                    gr_c = {gn: [] for gn in GROUPS}
                    gv_c = {gn: [] for gn in GROUPS}
                    gn_c = {gn: [] for gn in GROUPS}

                    for rep in range(NREP):
                        rng = resplit_rng(0, s, f, rep)
                        ci, ti = stratified_frac(yt, 0.50, rng)
                        yc, sc_c, gc = yt[ci], sc[ci], grp[ci]
                        ye, sc_e, ge = yt[ti], sc[ti], grp[ti]

                        # Simulation of historical inspection policy on calibration half:
                        if truncate:
                            med_sc = np.median(sc_c)
                            prop_c = np.where(sc_c >= med_sc, 1.0, 0.0)
                            prop_e = np.where(sc_e >= med_sc, 1.0, 0.0)
                            # IPW clipped to 1/0.05 on test when zero, demonstrating breakdown
                            w_e = 1.0 / np.maximum(prop_e, 0.05)
                            inspected = prop_c > 0.5
                        else:
                            prop_c = np.maximum(0.05, sc_c ** gamma)
                            prop_e = np.maximum(0.05, sc_e ** gamma)
                            w_e = 1.0 / prop_e
                            aud_rng = resplit_rng(500 + int(round(10 * gamma)), s, f, rep)
                            inspected = aud_rng.random(len(yc)) < prop_c

                        cf = np.zeros(len(ye), dtype=bool)
                        for gn in GROUPS:
                            # Audited positives in calibration
                            mc_pos = (gc == gn) & (yc == 1) & inspected
                            me = ge == gn
                            sc_cal_pos = sc_c[mc_pos]
                            gn_c[gn].append(int(mc_pos.sum()))

                            if method == "naive_unweighted":
                                thr_g, _ = cc_recall_threshold(sc_cal_pos, np.ones(len(sc_cal_pos), dtype=int), alpha)
                                fg = sc_e[me] >= thr_g
                            else:
                                w_cal_pos = 1.0 / np.maximum(prop_c[mc_pos], 0.05)
                                fg = weighted_conformal_flags(sc_cal_pos, w_cal_pos, sc_e[me], w_e[me], alpha)

                            cf[me] = fg
                            pos_e = ye[me] == 1
                            if pos_e.any():
                                rec_g = float(fg[pos_e].mean())
                                gr_c[gn].append(rec_g)
                                gv_c[gn].append(1.0 if rec_g < target else 0.0)

                        ro_c.append(float(cf[ye == 1].mean()))
                        io_c.append(float(cf.mean()))

                    rec_overall.append(ro_c); insp_overall.append(io_c)
                    for gn in GROUPS:
                        g_rec[gn].append(gr_c[gn])
                        g_viol[gn].append(gv_c[gn])
                        g_ncal[gn].append(gn_c[gn])

            ro_m, _, _, _ = cell_summary(rec_overall)
            io_m, _, _, _ = cell_summary(insp_overall)
            row = {
                "scenario": cfg_name, "gamma": gamma, "positivity_violated": truncate,
                "method": method, "overall_recall_pct": 100.0 * ro_m, "overall_insp_pct": 100.0 * io_m,
            }
            for gn in GROUPS:
                rm, _, _, _ = cell_summary(g_rec[gn])
                vm, _, _, _ = cell_summary(g_viol[gn])
                ncm, _, _, _ = cell_summary(g_ncal[gn])
                row[f"{gn}_ncal_mean"] = ncm
                row[f"{gn}_recall_pct"] = 100.0 * rm
                row[f"{gn}_viol_pct"] = 100.0 * vm
            b_rows.append(row)

    df_b = pd.DataFrame(b_rows)
    df_b.to_csv(OUT / "c4b_selective_inspection_ipw.csv", index=False)
    log("\n--- Part B: Selective Inspection Bias & Tibshirani IPW Recovery (@ 80% target) ---")
    log(df_b[["scenario", "method", "overall_recall_pct", "Heavy_recall_pct", "Light_recall_pct",
              "Flat_recall_pct", "overall_insp_pct"]].to_string(index=False))

    # ------------------------------------------------------------------
    # PART C: Covariate / Missingness & Cross-Fold Distribution Shift
    # ------------------------------------------------------------------
    c_rows = []
    # 1. Cross-fold shift: calibrate on Fold f1, test on Fold f2 != f1 (seed-matched)
    cf_rec, cf_insp = [], []
    cf_g_rec = {gn: [] for gn in GROUPS}
    for s in range(5):
        for f1 in range(3):
            for f2 in range(3):
                if f1 == f2:
                    continue
                yc, sc_c = preds[(s, f1)]; gc = fold_info[f1][1]
                ye, sc_e = preds[(s, f2)]; ge = fold_info[f2][1]
                cf = np.zeros(len(ye), dtype=bool)
                for gn in GROUPS:
                    mc = gc == gn
                    thr_g, _ = cc_recall_threshold(sc_c[mc], yc[mc], alpha)
                    me = ge == gn
                    fg = sc_e[me] >= thr_g
                    cf[me] = fg
                    cf_g_rec[gn].append([float(fg[ye[me] == 1].mean())])
                cf_rec.append([float(cf[ye == 1].mean())])
                cf_insp.append([float(cf.mean())])
    c_rows.append({
        "shift_regime": "cross_fold_transfer (f1 != f2)",
        "n_evals": len(cf_rec),
        "overall_recall_pct": 100.0 * float(np.mean(cf_rec)),
        "Heavy_recall_pct": 100.0 * float(np.mean(cf_g_rec["Heavy"])),
        "Light_recall_pct": 100.0 * float(np.mean(cf_g_rec["Light"])),
        "Flat_recall_pct": 100.0 * float(np.mean(cf_g_rec["Flat"])),
        "overall_insp_pct": 100.0 * float(np.mean(cf_insp)),
    })

    # 2. Missingness-regime shift: low-missingness cal -> high-missingness eval, & reverse
    for dir_name, cal_low in [("low_miss_to_high_miss", True), ("high_miss_to_low_miss", False)]:
        m_rec, m_insp = [], []
        mg_rec = {gn: [] for gn in GROUPS}
        for s in range(5):
            for f in range(3):
                yt, sc = preds[(s, f)]
                grp, mr = fold_info[f][1], fold_info[f][2]
                med_mr = float(np.median(mr))
                low_mask = mr <= med_mr
                c_mask = low_mask if cal_low else ~low_mask
                e_mask = ~c_mask
                yc, sc_c, gc = yt[c_mask], sc[c_mask], grp[c_mask]
                ye, sc_e, ge = yt[e_mask], sc[e_mask], grp[e_mask]
                cf = np.zeros(len(ye), dtype=bool)
                for gn in GROUPS:
                    mc = gc == gn
                    thr_g, _ = cc_recall_threshold(sc_c[mc], yc[mc], alpha)
                    me = ge == gn
                    fg = sc_e[me] >= thr_g
                    cf[me] = fg
                    if (ye[me] == 1).any():
                        mg_rec[gn].append(float(fg[ye[me] == 1].mean()))
                m_rec.append(float(cf[ye == 1].mean()))
                m_insp.append(float(cf.mean()))
        c_rows.append({
            "shift_regime": dir_name,
            "n_evals": len(m_rec),
            "overall_recall_pct": 100.0 * float(np.mean(m_rec)),
            "Heavy_recall_pct": 100.0 * float(np.mean(mg_rec["Heavy"])),
            "Light_recall_pct": 100.0 * float(np.mean(mg_rec["Light"])),
            "Flat_recall_pct": 100.0 * float(np.mean(mg_rec["Flat"])),
            "overall_insp_pct": 100.0 * float(np.mean(m_insp)),
        })

    df_c = pd.DataFrame(c_rows)
    df_c.to_csv(OUT / "c4c_distribution_shift.csv", index=False)
    log("\n--- Part C: Covariate / Cross-Fold & Missingness-Regime Shift (@ 80% target) ---")
    log(df_c.to_string(index=False))

    (OUT / "c4_summary.json").write_text(json.dumps({
        "a1_within_fold_flat": df_a1[df_a1["group"] == "Flat"].to_dict(orient="records"),
        "a2_pooled_flat": df_a2[df_a2["group"] == "Flat"].to_dict(orient="records"),
        "b_selective_ipw": b_rows,
        "c_distribution_shift": c_rows,
    }, indent=2))
    (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))
    log(f"\n[DONE] Saved C4 artifacts to {OUT}")


if __name__ == "__main__":
    main()
