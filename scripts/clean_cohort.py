"""clean_cohort.py — Step (a1): cohort cleaning.

RULE:
  DROP customer iff  nansum(|x|) < NO_SIGNAL_EPS   (= 1e-6 kWh total).
  This single criterion captures BOTH:
    R1 all-NaN customers (no observed reading at all), and
    R2 all-zero-observed meters (every observed reading is 0)
  i.e. customers with ZERO recoverable consumption signal — uninformative for any
  detector and for phenotype assignment (gap/magnitude both 0). Removing them is
  removal of NO-SIGNAL rows, not cherry-picking: it cannot inflate true detection
  (these carry no separable information; their theft labels are unrecoverable).

NOT removed (would be cheating / unjustified):
  * cleanlab "suspect" positives  -> circular (deletes hard true positives).
  * conflicting-label exact-duplicate clusters -> FLAGGED for transparency (R3),
    kept in cohort; they are an irreducible label-noise floor, not our choice to delete.

Outputs (results/phase_n_retrained/):
  clean_cohort.json        summary + counts + breakdowns
  dropped_ids.csv          CONS_NO, FLAG, reason for every dropped customer
  keep_idx.npy             int64 row indices (into the 42372-row SGCC order) to KEEP
  conflicting_flag_ids.csv CONS_NO, FLAG, cluster_exact, idx  (R3 flag, not dropped;
                           computed from the SGCC readings, no external input)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from run_local_grid import SGCC_CSV
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

NO_SIGNAL_EPS = 1e-6
OUT = RESULTS_DIR


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SGCC_CSV)
    cons = df["CONS_NO"].values
    y = df["FLAG"].values.astype(np.int64)
    vals = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    N = len(y)

    n_obs = np.sum(~np.isnan(vals), axis=1)                 # observed readings
    n_nonzero = np.nansum(vals > 0, axis=1).astype(int)     # observed nonzero readings
    total = np.nansum(np.abs(vals), axis=1)                 # total consumption

    no_signal = total < NO_SIGNAL_EPS                        # the drop rule
    all_nan = n_obs == 0
    all_zero_observed = (n_obs > 0) & (n_nonzero == 0)
    keep_mask = ~no_signal
    keep_idx = np.where(keep_mask)[0].astype(np.int64)

    # phenotype of dropped (descriptive; router fit on full pop for this report only)
    g, m = gap_magnitude(vals[:, :(vals.shape[1] // 7) * 7].reshape(N, -1, 7))
    ph = assign(g, m, fit_router(g, m))

    drop_idx = np.where(no_signal)[0]
    reason = np.where(all_nan[drop_idx], "all_nan", "all_zero_observed")
    dropped = pd.DataFrame({"idx": drop_idx, "CONS_NO": cons[drop_idx],
                            "FLAG": y[drop_idx], "reason": reason,
                            "phenotype": ph[drop_idx], "n_obs": n_obs[drop_idx]})
    dropped.to_csv(OUT / "dropped_ids.csv", index=False)
    np.save(OUT / "keep_idx.npy", keep_idx)

    # R3 conflicting-label exact-duplicate clusters (flag, do NOT drop).
    # An exact-duplicate cluster is a set of >= 2 customers whose raw daily readings
    # are identical in every column (a missing reading matches only a missing reading).
    # A cluster is conflicting when it contains both labels. Cluster ids follow the
    # first appearance of each cluster in the SGCC row order.
    row_key = pd.util.hash_pandas_object(pd.DataFrame(vals), index=False).values
    key_count = pd.Series(row_key).map(pd.Series(row_key).value_counts()).values
    dup_idx = np.where(key_count > 1)[0]
    ex = pd.DataFrame({"idx": dup_idx, "CONS_NO": cons[dup_idx],
                       "FLAG": y[dup_idx], "key": row_key[dup_idx]})
    first_row = ex.groupby("key").idx.transform("min")
    ex["cluster_exact"] = first_row.rank(method="dense").astype(np.int64) - 1
    mixed = ex.groupby("cluster_exact").FLAG.transform("nunique") > 1
    exm = ex.loc[mixed, ["CONS_NO", "FLAG", "cluster_exact", "idx"]]
    exm.to_csv(OUT / "conflicting_flag_ids.csv", index=False)
    conf = {"n_clusters": int(exm.cluster_exact.nunique()),
            "n_customers": int(len(exm)),
            "theft": int((exm.FLAG == 1).sum()),
            "honest": int((exm.FLAG == 0).sum()),
            "all_no_signal": bool(no_signal[exm.idx.values].all())}

    summary = {
        "pre_registered_rule": "drop iff nansum(|x|) < 1e-6 (no consumption signal)",
        "NO_SIGNAL_EPS": NO_SIGNAL_EPS,
        "N_total": int(N),
        "N_kept": int(keep_mask.sum()),
        "N_dropped": int(no_signal.sum()),
        "dropped_all_nan": int(all_nan.sum()),
        "dropped_all_zero_observed": int(all_zero_observed.sum()),
        "dropped_theft": int(y[no_signal].sum()),
        "dropped_honest": int((y[no_signal] == 0).sum()),
        "dropped_phenotype_breakdown": {p: int((ph[no_signal] == p).sum()) for p in PHENOTYPES},
        "theft_prevalence_before": float(y.mean()),
        "theft_prevalence_after": float(y[keep_mask].mean()),
        "conflicting_label_exact_dup_FLAGGED_not_dropped": conf,
    }
    (OUT / "clean_cohort.json").write_text(json.dumps(summary, indent=2))

    print("=== clean_cohort (Step a1) ===")
    print(f"rule: drop iff nansum(|x|) < {NO_SIGNAL_EPS}")
    print(f"N total        : {N}")
    print(f"N kept         : {keep_mask.sum()}  ({100*keep_mask.mean():.2f}%)")
    print(f"N dropped      : {int(no_signal.sum())}  (all_nan={int(all_nan.sum())}, "
          f"all_zero_observed={int(all_zero_observed.sum())})")
    print(f"  dropped theft={int(y[no_signal].sum())}  honest={int((y[no_signal]==0).sum())}")
    print(f"  dropped phenotypes: " +
          ", ".join(f"{p}={int((ph[no_signal]==p).sum())}" for p in PHENOTYPES))
    print(f"theft prevalence: {y.mean():.5f} -> {y[keep_mask].mean():.5f}")
    print(f"R3 conflicting-label exact-dup (flagged, kept): "
          f"{conf['n_clusters']} clusters, {conf['n_customers']} cust "
          f"({conf['theft']} theft / {conf['honest']} honest)")
    print(f"[artifacts] -> {OUT}")


if __name__ == "__main__":
    main()
