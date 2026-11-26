#!/usr/bin/env python3
"""Compute verified 5-fold client-disjoint ECL Table 2, Table S6 (lift/gap), and Table S7 (board)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

csv.field_size_limit(10**7)
THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
_REPO_DIR = SCRIPTS_DIR.parent
for p in (str(THIS), str(SCRIPTS_DIR), str(_REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import _ROOT, ECL_PATH
from run_c3_ecl_client_disjoint import build_ecl_dataset_with_clients, get_client_disjoint_splits, bca_ci
from conformal_common import NREP, resplit_rng, stratified_half, cc_recall_threshold

R3_RES = _ROOT / "results" / "r3_revision"
CELLS_DIR = R3_RES / "C3_ecl_client_disjoint" / "cells"
OUT_JSON = R3_RES / "C3_ecl_client_disjoint" / "c3_tables_verified.json"

ARM_ORDER = [
    ("GRU-D", "grud", 0),
    ("Self-Attn", "hybridattn", 2),
    ("WDCNN", "wdcnn", 1),
    ("MSDM", "msdm", 3),
    ("HORLN", "horln", 4),
    ("MAMRD", "a34_generic", 5),
]


def topk_tie_neutral(y: np.ndarray, scores: np.ndarray, K: int = 100, draws: int = 200) -> float:
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(draws):
        idx = np.lexsort((rng.random(len(scores)), -scores))[:K]
        ry = y[idx]
        s = float(ry.sum())
        if s == 0:
            vals.append(0.0)
        else:
            p_at = np.cumsum(ry) / (np.arange(len(ry)) + 1.0)
            vals.append(float((ry * p_at).sum() / s))
    return float(np.mean(vals))


def best_oracle_f1(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    pos = float(y.sum())
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(pos, 1.0)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(np.max(f1))


def main() -> None:
    if not ECL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ECL raw dataset at {ECL_PATH}. Obtain LD2011_2014.txt per data/README.md "
            "or set MAMRD_ECL."
        )
    Wp, y, atk_type, cids, mag = build_ecl_dataset_with_clients()
    splits = get_client_disjoint_splits(y, cids, n_folds=5)

    # Fold partition High/Low stats
    fold_part = []
    for f, (tr, es, te) in enumerate(splits):
        med_tr = float(np.median(mag[tr]))
        hi = mag[te] > med_tr
        lo = ~hi
        # Check avg over 20 conformal stratified halves on te
        cal_hi_l, cal_lo_l, eval_hi_l, eval_lo_l = [], [], [], []
        for rep in range(NREP):
            ci, ei = stratified_half(y[te], resplit_rng(5, 0, f, rep))
            cal_hi_l.append(int((y[te][ci][hi[ci]] == 1).sum()))
            cal_lo_l.append(int((y[te][ci][lo[ci]] == 1).sum()))
            eval_hi_l.append(int((y[te][ei][hi[ei]] == 1).sum()))
            eval_lo_l.append(int((y[te][ei][lo[ei]] == 1).sum()))
        fold_part.append({
            "fold": f,
            "te_win": int(len(te)), "te_pos": int(y[te].sum()),
            "te_hi_win": int(hi.sum()), "te_hi_pos": int((y[te][hi] == 1).sum()),
            "te_lo_win": int(lo.sum()), "te_lo_pos": int((y[te][lo] == 1).sum()),
            "cal_hi_pos_mean": float(np.mean(cal_hi_l)), "cal_lo_pos_mean": float(np.mean(cal_lo_l)),
            "eval_hi_pos_mean": float(np.mean(eval_hi_l)), "eval_lo_pos_mean": float(np.mean(eval_lo_l)),
        })

    board_rows = []
    liftgap_rows = []

    for dname, arm, arm_idx in ARM_ORDER:
        aps, rocs, f1s, m100s = [], [], [], []
        cell_lh, cell_ll, cell_gap = [], [], []
        cell_rech, cell_recl = [], []
        cell_cond_insp, cell_single_insp, cell_red = [], [], []
        for s in (0, 1, 2):
            for f in (0, 1, 2, 3, 4):
                tr, es, te = splits[f]
                med_tr = float(np.median(mag[tr]))
                hi_mask = (mag[te] > med_tr)
                lo_mask = ~hi_mask

                row = next(csv.DictReader(open(CELLS_DIR / f"{arm}_{s}_{f}.csv")))
                yv = np.array(json.loads(row["val_labels_json"]), dtype=int)
                sv = np.array(json.loads(row["val_scores_json"]), dtype=float)
                assert len(yv) == len(te) and np.array_equal(yv, y[te]), f"Label mismatch {arm} {s} {f}"

                # Use stored unrounded training-time cell AP and ROC to avoid 6-decimal score
                # rounding ties (e.g., saturated MSDM scores), while computing F1/MAP@100 and
                # conformal slices directly from val_scores_json.
                aps.append(float(row["ap"]))
                rocs.append(float(row["roc"]))
                f1s.append(best_oracle_f1(yv, sv))
                m100s.append(topk_tie_neutral(yv, sv, K=100))

                pi_h = float(yv[hi_mask].mean())
                pi_l = float(yv[lo_mask].mean())

                lhs, lls, gaps, rhs, rls, cis, sis, reds = [], [], [], [], [], [], [], []
                for rep in range(NREP):
                    ci, ei = stratified_half(yv, resplit_rng(arm_idx, s, f, rep))
                    qh, _ = cc_recall_threshold(sv[ci][hi_mask[ci]], yv[ci][hi_mask[ci]], 0.20)
                    ql, _ = cc_recall_threshold(sv[ci][lo_mask[ci]], yv[ci][lo_mask[ci]], 0.20)
                    qg = min(qh, ql)

                    fl_h = sv[ei][hi_mask[ei]] >= qh
                    fl_l = sv[ei][lo_mask[ei]] >= ql
                    ye_h = yv[ei][hi_mask[ei]]
                    ye_l = yv[ei][lo_mask[ei]]

                    hh = float(ye_h[fl_h].mean()) if fl_h.sum() > 0 else float("nan")
                    hl = float(ye_l[fl_l].mean()) if fl_l.sum() > 0 else float("nan")
                    lh = hh / pi_h if (np.isfinite(hh) and pi_h > 0) else float("nan")
                    ll = hl / pi_l if (np.isfinite(hl) and pi_l > 0) else float("nan")
                    if np.isfinite(lh) and np.isfinite(ll):
                        lhs.append(lh); lls.append(ll); gaps.append(lh - ll)
                    rhs.append(float((fl_h & (ye_h == 1)).sum() / max(int((ye_h == 1).sum()), 1)))
                    rls.append(float((fl_l & (ye_l == 1)).sum() / max(int((ye_l == 1).sum()), 1)))
                    ci_f = float(np.where(hi_mask[ei], sv[ei] >= qh, sv[ei] >= ql).mean())
                    si_f = float((sv[ei] >= qg).mean())
                    cis.append(ci_f); sis.append(si_f)
                    if si_f > 0:
                        reds.append(1.0 - ci_f / si_f)

                cell_lh.append(float(np.mean(lhs)))
                cell_ll.append(float(np.mean(lls)))
                cell_gap.append(float(np.mean(gaps)))
                cell_rech.append(float(np.mean(rhs)))
                cell_recl.append(float(np.mean(rls)))
                cell_cond_insp.append(float(np.mean(cis)))
                cell_single_insp.append(float(np.mean(sis)))
                cell_red.append(float(np.mean(reds)))

        glo, ghi = bca_ci(cell_gap)
        board_rows.append({
            "detector": dname,
            "n_cells": 15,
            "ap_mean": float(np.mean(aps)),
            "roc_mean": float(np.mean(rocs)),
            "f1_oracle_mean": float(np.mean(f1s)),
            "map100_mean": float(np.mean(m100s)),
        })
        liftgap_rows.append({
            "detector": dname,
            "n_cells": 15,
            "lift_high": float(np.mean(cell_lh)),
            "lift_low": float(np.mean(cell_ll)),
            "gap_mean": float(np.mean(cell_gap)),
            "gap_ci_lo": glo,
            "gap_ci_hi": ghi,
            "rec_high_pct": 100.0 * float(np.mean(cell_rech)),
            "rec_low_pct": 100.0 * float(np.mean(cell_recl)),
            "cond_insp_pct": 100.0 * float(np.mean(cell_cond_insp)),
            "single_insp_pct": 100.0 * float(np.mean(cell_single_insp)),
            "workload_red_pct": 100.0 * float(np.mean(cell_red)),
        })

    # Mean prevalence in High / Low across 5 test folds
    hi_prev = float(np.mean([r["te_hi_pos"] / r["te_hi_win"] for r in fold_part])) * 100.0
    lo_prev = float(np.mean([r["te_lo_pos"] / r["te_lo_win"] for r in fold_part])) * 100.0
    out = {
        "fold_part_summary": {
            "te_win_mean": float(np.mean([r["te_win"] for r in fold_part])),
            "te_pos_mean": float(np.mean([r["te_pos"] for r in fold_part])),
            "te_hi_win_mean": float(np.mean([r["te_hi_win"] for r in fold_part])),
            "te_hi_pos_mean": float(np.mean([r["te_hi_pos"] for r in fold_part])),
            "te_lo_win_mean": float(np.mean([r["te_lo_win"] for r in fold_part])),
            "te_lo_pos_mean": float(np.mean([r["te_lo_pos"] for r in fold_part])),
            "cal_hi_pos_mean": float(np.mean([r["cal_hi_pos_mean"] for r in fold_part])),
            "cal_lo_pos_mean": float(np.mean([r["cal_lo_pos_mean"] for r in fold_part])),
            "eval_hi_pos_mean": float(np.mean([r["eval_hi_pos_mean"] for r in fold_part])),
            "eval_lo_pos_mean": float(np.mean([r["eval_lo_pos_mean"] for r in fold_part])),
            "high_prev_pct": hi_prev,
            "low_prev_pct": lo_prev,
        },
        "board_rows": board_rows,
        "liftgap_rows": liftgap_rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
