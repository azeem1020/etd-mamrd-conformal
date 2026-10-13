"""Build the ECL (synthetic-attack) leaderboard at honest n=9 from LOCKED stored
predictions in results/phase_n/ecl_trust/<arm>_<seed>_<fold>.csv.

No GPU re-run: every metric is recomputed from the cached per-fold test
predictions (val_labels_json / val_scores_json), so the numbers are guaranteed
consistent with the AP already reported in the paper (a34_generic = 0.845).

Metric definitions match run_n_clean.py (the SGCC source) exactly:
  PR-AUC   = average_precision_score(y, s)          [= AP, threshold-free]
  ROC-AUC  = roc_auc_score(y, s)                    [threshold-free]
  MAP@100  = _mapk(y, s, 100)  (average precision @ top-100, threshold-free)
  F1       = best F1 over unique test-fold thresholds.
             NOTE: the secondary corpus did not cache a calibration split, so
             unlike SGCC (cal-thresholded) F1 is reported at its maximizing
             test threshold. This is disclosed in the manuscript caption.
"""
from __future__ import annotations
from pathlib import Path
import csv, json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

E = (RESULTS_DIR / "ecl_trust")
ARMS = [("grud", "GRU-D"), ("hybridattn", "Self-attention"), ("wdcnn", "Wide-and-deep CNN"),
        ("msdm", "MSDM"), ("horln", "HORLN"), ("a34_generic", "MAMRD")]


def _mapk(y, s, K):
    o = np.argsort(-s)[:K]; yk = y[o]
    if yk.sum() == 0:
        return 0.0
    return float((np.cumsum(yk) / (np.arange(len(yk)) + 1) * yk).sum() / yk.sum())


def _best_f1(y, s):
    best = 0.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best:
            best = f
    return float(best)


def cell_metrics(path):
    row = next(csv.DictReader(open(path)))
    y = np.array(json.loads(row["val_labels_json"]))
    s = np.array(json.loads(row["val_scores_json"]))
    return {
        "pr": average_precision_score(y, s),   # recomputed; equals cached `ap`
        "roc": roc_auc_score(y, s),
        "f1": _best_f1(y, s),
        "map100": _mapk(y, s, 100),
        "ap_cached": float(row["ap"]),
        "roc_cached": float(row["roc"]),
    }


def main():
    out_rows = []
    print(f"{'arm':<18}{'PR-AUC':>9}{'ROC-AUC':>9}{'F1':>9}{'MAP@100':>9}   (n=9, mean)")
    print("-" * 62)
    for key, label in ARMS:
        cells = sorted(E.glob(f"{key}_*_*.csv"))
        assert len(cells) == 9, f"{key}: expected 9 cells, found {len(cells)}"
        m = [cell_metrics(p) for p in cells]
        agg = {k: float(np.mean([c[k] for c in m])) for k in ("pr", "roc", "f1", "map100")}
        # consistency check: recomputed PR-AUC vs cached AP must agree to 3 dp
        d = abs(np.mean([c["pr"] for c in m]) - np.mean([c["ap_cached"] for c in m]))
        flag = "" if d < 1e-3 else f"  !! AP drift {d:.4f}"
        print(f"{label:<18}{agg['pr']:>9.3f}{agg['roc']:>9.3f}{agg['f1']:>9.3f}{agg['map100']:>9.3f}{flag}")
        out_rows.append({"arm": key, "label": label, **{k: round(v, 4) for k, v in agg.items()}})

    out = E.parent / "ecl_board.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "label", "pr", "roc", "f1", "map100"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
