#!/usr/bin/env python3
"""Build Revision 3 5-fold SGCC and 5-fold client-disjoint ECL partition manifests.

Generates:
  - partition_manifests/sgcc_partition_manifest.csv  (211,860 rows: Folds 0..4 x 42,372)
  - partition_manifests/sgcc_fold_counts.csv         (per-fold SGCC counts, Folds 0..4)
  - partition_manifests/ecl_partition_manifest.csv   (125,000 rows: Folds 0..4 x 25,000, client-disjoint)
  - partition_manifests/ecl_fold_counts.csv          (per-fold ECL client-disjoint counts, Folds 0..4)

Performs zero-drift assertions against:
  - partition_manifests/archive_r2/sgcc_partition_manifest.csv (exact match on Folds 0..2)
  - results/r3_revision/C3_ecl_client_disjoint/c3_ecl_client_disjoint_partitions.csv (if present)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
ROOT_DIR = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import SGCC_PATH, ECL_PATH  # noqa: E402
from zkmix_lab.data import KFoldSpec  # noqa: E402
from phased.phenotype import gap_magnitude, fit_router, assign  # noqa: E402
from run_local_grid import _stratified_three_way_split  # noqa: E402
from msdm.fdi import ATTACKS  # noqa: E402


MANIFEST_DIR = ROOT_DIR / "partition_manifests"
R3_C3_PARTITIONS = (
    ROOT_DIR / "results" / "r3_revision" / "C3_ecl_client_disjoint" / "c3_ecl_client_disjoint_partitions.csv"
)


def collapse_group3(ph: np.ndarray) -> np.ndarray:
    """Collapse 5 fine router labels into Heavy / Light / Flat."""
    out = np.empty(len(ph), dtype=object)
    for i, label in enumerate(ph):
        if label in ("Heavy_Riser", "Heavy_Dropper"):
            out[i] = "Heavy"
        elif label in ("Light_Riser", "Light_Dropper"):
            out[i] = "Light"
        else:
            out[i] = "Flat"
    return out


def build_sgcc_manifests() -> None:
    """Generate 5-fold SGCC partition manifest and per-fold counts."""
    print(f"[1/2] Loading SGCC from {SGCC_PATH.name}...", flush=True)
    df = pd.read_csv(SGCC_PATH)
    cids = df["CONS_NO"].astype(str).to_numpy()
    y = df["FLAG"].values.astype(int)
    raw = df.drop(columns=["CONS_NO", "FLAG"]).values[:, :1029].astype(np.float64)
    g, m = gap_magnitude(raw.reshape(len(df), 147, 7))

    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))

    rows = []
    fold_counts = []
    test_sets = []

    for fold in range(5):
        tr, val = folds[fold]
        es, cal, te = _stratified_three_way_split(
            val, y, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(fold)
        )
        params = fit_router(g[tr], m[tr])
        ph = assign(g, m, params)
        grp3 = collapse_group3(ph)

        splits = [
            ("train", np.sort(tr)),
            ("early_stop", np.sort(es)),
            ("threshold_cal", np.sort(cal)),
            ("test", np.sort(te)),
        ]
        # Pairwise within-fold disjointness
        s_sets = {name: set(arr.tolist()) for name, arr in splits}
        names = [name for name, _ in splits]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert len(s_sets[names[i]] & s_sets[names[j]]) == 0
        test_sets.append(s_sets["test"])

        for sp_name, idx_arr in splits:
            for idx in idx_arr:
                rows.append((
                    fold,
                    sp_name,
                    int(idx),
                    cids[idx],
                    int(y[idx]),
                    str(grp3[idx]),
                    str(ph[idx]),
                ))

        te_g = grp3[te]
        te_y = y[te]
        fold_counts.append({
            "fold": fold,
            "train_tot": int(len(tr)),
            "train_pos": int(y[tr].sum()),
            "es_tot": int(len(es)),
            "es_pos": int(y[es].sum()),
            "cal_tot": int(len(cal)),
            "cal_pos": int(y[cal].sum()),
            "te_tot": int(len(te)),
            "te_pos": int(te_y.sum()),
            "te_Heavy_tot": int((te_g == "Heavy").sum()),
            "te_Heavy_pos": int(((te_g == "Heavy") & (te_y == 1)).sum()),
            "te_Light_tot": int((te_g == "Light").sum()),
            "te_Light_pos": int(((te_g == "Light") & (te_y == 1)).sum()),
            "te_Flat_tot": int((te_g == "Flat").sum()),
            "te_Flat_pos": int(((te_g == "Flat") & (te_y == 1)).sum()),
        })

    # Assert 0 cross-fold test overlap & exact manuscript sums
    for i in range(5):
        for j in range(i + 1, 5):
            assert len(test_sets[i] & test_sets[j]) == 0, f"SGCC test overlap F{i}/F{j}"

    df_counts = pd.DataFrame(fold_counts)
    assert len(rows) == 211860, f"Expected 211,860 SGCC manifest rows, got {len(rows)}"
    assert int(df_counts["te_tot"].sum()) == 14420
    assert int(df_counts["te_pos"].sum()) == 1235
    assert int(df_counts["te_Flat_pos"].sum()) == 76

    # Zero-drift check on Folds 0..2 against archived R2 manifest
    r2_sgcc = MANIFEST_DIR / "archive_r2" / "sgcc_partition_manifest.csv"
    if r2_sgcc.exists():
        expected = {}
        with open(r2_sgcc, "r", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                expected[(int(r["fold"]), int(r["row_index"]))] = (
                    r["split"], r["customer_id"], int(r["label"]), r["group_3"], r["phenotype_5"]
                )
        actual_02 = {
            (f, idx): (sp, cid, lbl, g3, p5)
            for (f, sp, idx, cid, lbl, g3, p5) in rows if f in (0, 1, 2)
        }
        assert actual_02 == expected, "SGCC Folds 0..2 diverged from archived R2 manifest!"
        print("  [ZERO-DRIFT OK] SGCC Folds 0..2 match archive_r2/sgcc_partition_manifest.csv byte-for-row.")

    out_manifest = MANIFEST_DIR / "sgcc_partition_manifest.csv"
    with open(out_manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "split", "row_index", "customer_id", "label", "group_3", "phenotype_5"])
        w.writerows(rows)
    df_counts.to_csv(MANIFEST_DIR / "sgcc_fold_counts.csv", index=False)
    print(f"  Wrote {out_manifest.name} ({len(rows)} rows) and sgcc_fold_counts.csv")


def build_ecl_manifests() -> None:
    """Generate 5-fold client-disjoint GroupKFold ECL partition manifest and counts."""
    print(f"[2/2] Loading ECL from {ECL_PATH.name}...", flush=True)
    df = pd.read_csv(ECL_PATH, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    res_df = df.resample("30min").mean()
    client_names = list(res_df.columns)
    res = res_df.values.T.astype(np.float64)

    win_len = 7 * 48
    cap = 25000
    n_clients, total_steps = res.shape
    nw = total_steps // win_len

    wins, cids, cnames, cweeks = [], [], [], []
    for i in range(n_clients):
        cname = str(client_names[i])
        for w in range(nw):
            seg = res[i, w * win_len : (w + 1) * win_len]
            if np.nansum(np.abs(seg)) > 1e-6:
                wins.append(seg)
                cids.append(i)
                cnames.append(cname)
                cweeks.append(w)

    W = np.array(wins, dtype=np.float64)
    cids_arr = np.array(cids, dtype=np.int64)
    cnames_arr = np.array(cnames, dtype=object)
    cweeks_arr = np.array(cweeks, dtype=np.int64)

    rng = np.random.default_rng(0)
    if len(W) > cap:
        sel = rng.choice(len(W), cap, False)
        W = W[sel]
        cids_arr = cids_arr[sel]
        cnames_arr = cnames_arr[sel]
        cweeks_arr = cweeks_arr[sel]

    frng = np.random.default_rng(0)
    y = np.zeros(len(W), dtype=int)
    atk_type = np.full(len(W), -1, dtype=int)
    Wo = W.copy()
    for i in frng.choice(len(W), int(0.10 * len(W)), False):
        k = int(frng.integers(6))
        Wo[i] = ATTACKS[k](W[i], frng)
        y[i] = 1
        atk_type[i] = k + 1

    mag = np.nanmean(Wo, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)

    gkf = GroupKFold(n_splits=5)
    rows = []
    fold_counts = []

    for f, (tr_idx, held_idx) in enumerate(gkf.split(y, y, groups=cids_arr)):
        held_clients = np.unique(cids_arr[held_idx])
        perm = np.random.default_rng(100 + f).permutation(held_clients)
        n_es_c = max(1, int(round(0.35 * len(perm))))
        es_clients = set(perm[:n_es_c].tolist())
        te_clients = set(perm[n_es_c:].tolist())
        es_idx = held_idx[np.isin(cids_arr[held_idx], list(es_clients))]
        te_idx = held_idx[np.isin(cids_arr[held_idx], list(te_clients))]

        c_tr = set(cids_arr[tr_idx].tolist())
        c_es = set(cids_arr[es_idx].tolist())
        c_te = set(cids_arr[te_idx].tolist())
        sh_tr_es = len(c_tr & c_es)
        sh_tr_te = len(c_tr & c_te)
        sh_es_te = len(c_es & c_te)
        assert sh_tr_es == 0 and sh_tr_te == 0 and sh_es_te == 0, f"ECL client leakage Fold {f}"

        med_tr = float(np.median(mag[tr_idx]))
        hi_te = mag[te_idx] > med_tr
        lo_te = ~hi_te

        for sp_name, idx_arr in (
            ("train", np.sort(tr_idx)),
            ("early_stop", np.sort(es_idx)),
            ("test", np.sort(te_idx)),
        ):
            for idx in idx_arr:
                rows.append((
                    f,
                    sp_name,
                    int(idx),
                    int(cids_arr[idx]),
                    str(cnames_arr[idx]),
                    int(cweeks_arr[idx]),
                    int(y[idx]),
                    int(atk_type[idx]),
                ))

        fold_counts.append({
            "fold": f,
            "train_win": int(len(tr_idx)),
            "train_pos": int(y[tr_idx].sum()),
            "train_clients": len(c_tr),
            "es_win": int(len(es_idx)),
            "es_pos": int(y[es_idx].sum()),
            "es_clients": len(c_es),
            "te_win": int(len(te_idx)),
            "te_pos": int(y[te_idx].sum()),
            "te_clients": len(c_te),
            "te_High_win": int(hi_te.sum()),
            "te_High_pos": int((y[te_idx][hi_te] == 1).sum()),
            "te_Low_win": int(lo_te.sum()),
            "te_Low_pos": int((y[te_idx][lo_te] == 1).sum()),
            "shared_tr_es_clients": sh_tr_es,
            "shared_tr_te_clients": sh_tr_te,
            "shared_es_te_clients": sh_es_te,
        })

    assert len(rows) == 125000, f"Expected 125,000 ECL manifest rows, got {len(rows)}"
    df_counts = pd.DataFrame(fold_counts)

    # Zero-drift check against C3 partitions CSV (either in repo or check after sync)
    if R3_C3_PARTITIONS.exists():
        df_ref = pd.read_csv(R3_C3_PARTITIONS)
        cols = ["fold", "train_win", "train_pos", "train_clients",
                "es_win", "es_pos", "es_clients", "te_win", "te_pos", "te_clients"]
        assert df_counts[cols].equals(df_ref[cols]), "ECL counts differ from C3 reference partitions!"
        print("  [ZERO-DRIFT OK] ECL counts match c3_ecl_client_disjoint_partitions.csv identically.")

    out_manifest = MANIFEST_DIR / "ecl_partition_manifest.csv"
    with open(out_manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "split", "window_index", "client_id", "client_name", "client_week_idx", "label", "attack_type"])
        w.writerows(rows)
    df_counts.to_csv(MANIFEST_DIR / "ecl_fold_counts.csv", index=False)
    print(f"  Wrote {out_manifest.name} ({len(rows)} rows) and ecl_fold_counts.csv")


def main() -> None:
    build_sgcc_manifests()
    build_ecl_manifests()
    print("[PASS] Revision 3 5-fold SGCC & client-disjoint ECL manifests generated.")


if __name__ == "__main__":
    main()
