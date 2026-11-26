#!/usr/bin/env python3
"""Deterministic 5-Fold Partition Disjointness & Ledger Verification Script.

Verifies:
  1. 100% customer-disjoint partitioning on SGCC across Folds 0--4 (211,860 rows;
     0 within-fold overlap and 0 cross-fold test overlap across 14,420 unique test consumers).
  2. 100% window-disjoint AND 100% client-disjoint GroupKFold partitioning on ECL across
     Folds 0--4 (125,000 rows; 296/26/48 clients per fold; 0 client overlap across splits).

Usage:
    python partition_manifests/verify_disjointness.py
    python scripts/verify_partition_manifests.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def find_manifest_dir() -> Path:
    """Locate the active Revision 3 partition_manifests directory."""
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "partition_manifests",
        Path(__file__).resolve().parent.parent / "partition_manifests",
    ]
    for c in candidates:
        if (c / "sgcc_partition_manifest.csv").exists() and (c / "ecl_partition_manifest.csv").exists():
            return c
    raise FileNotFoundError("Could not locate 5-fold partition manifests in partition_manifests/.")


def audit_sgcc(manifest_dir: Path) -> None:
    """Audit 5-fold SGCC customer disjointness within and across folds."""
    p = manifest_dir / "sgcc_partition_manifest.csv"
    print(f"\n[1/2] Auditing SGCC Partition Manifest: {p.name}")
    print("=" * 72)

    splits_by_fold = defaultdict(lambda: defaultdict(set))
    pos_by_fold = defaultdict(lambda: defaultdict(int))
    grp_pos_by_fold = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with open(p, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_rows = 0
        for r in reader:
            total_rows += 1
            fold = int(r["fold"])
            sp = r["split"]
            cid = r["customer_id"]
            lbl = int(r["label"])
            grp = r.get("group_3", "")

            splits_by_fold[fold][sp].add(cid)
            if lbl == 1:
                pos_by_fold[fold][sp] += 1
                grp_pos_by_fold[fold][sp][grp] += 1

    print(f"Total manifest rows read: {total_rows} (Expected 211,860 across 5 folds)")
    assert total_rows == 211860, f"Row count mismatch: got {total_rows}"
    folds = sorted(splits_by_fold.keys())
    assert folds == [0, 1, 2, 3, 4], f"Expected folds [0..4], got {folds}"

    for fold in folds:
        sp_dict = splits_by_fold[fold]
        assert set(sp_dict.keys()) == {"train", "early_stop", "threshold_cal", "test"}
        tr, es, cal, te = sp_dict["train"], sp_dict["early_stop"], sp_dict["threshold_cal"], sp_dict["test"]

        tot = len(tr) + len(es) + len(cal) + len(te)
        print(f"\nFold {fold}: Total customers = {tot} (Train={len(tr)}, ES={len(es)}, Cal={len(cal)}, Test={len(te)})")
        assert tot == 42372, f"Fold {fold} total customers != 42,372"

        assert len(tr & es) == 0, f"Fold {fold}: Overlap between train and early_stop!"
        assert len(tr & cal) == 0, f"Fold {fold}: Overlap between train and threshold_cal!"
        assert len(tr & te) == 0, f"Fold {fold}: Overlap between train and test!"
        assert len(es & cal) == 0, f"Fold {fold}: Overlap between early_stop and threshold_cal!"
        assert len(es & te) == 0, f"Fold {fold}: Overlap between early_stop and test!"
        assert len(cal & te) == 0, f"Fold {fold}: Overlap between threshold_cal and test!"
        print("  [OK] Within-fold pairwise customer overlap: 0 across all 6 split pairs")
        print(
            f"  Test Positives: {pos_by_fold[fold]['test']} "
            f"(Heavy={grp_pos_by_fold[fold]['test']['Heavy']}, "
            f"Light={grp_pos_by_fold[fold]['test']['Light']}, "
            f"Flat={grp_pos_by_fold[fold]['test']['Flat']})"
        )

    for i in range(5):
        for j in range(i + 1, 5):
            overlap = len(splits_by_fold[i]["test"] & splits_by_fold[j]["test"])
            assert overlap == 0, f"Cross-fold test overlap between Fold {i} and Fold {j}: {overlap}"

    tot_te = sum(len(splits_by_fold[f]["test"]) for f in folds)
    tot_te_pos = sum(pos_by_fold[f]["test"] for f in folds)
    tot_flat_pos = sum(grp_pos_by_fold[f]["test"]["Flat"] for f in folds)
    assert tot_te == 14420, f"Expected 14,420 unique test consumers, got {tot_te}"
    assert tot_te_pos == 1235, f"Expected 1,235 unique test thefts, got {tot_te_pos}"
    assert tot_flat_pos == 76, f"Expected 76 unique Flat test thefts, got {tot_flat_pos}"
    print(f"\n  [OK] Cross-fold test customer overlap: 0 across all 10 fold pairs ({tot_te} unique test consumers, {tot_te_pos} thefts, {tot_flat_pos} Flat thefts)")

    mean_tr = sum(len(splits_by_fold[f]["train"]) for f in folds) / 5.0
    mean_tr_pos = sum(pos_by_fold[f]["train"] for f in folds) / 5.0
    mean_es = sum(len(splits_by_fold[f]["early_stop"]) for f in folds) / 5.0
    mean_es_pos = sum(pos_by_fold[f]["early_stop"] for f in folds) / 5.0
    mean_cal = sum(len(splits_by_fold[f]["threshold_cal"]) for f in folds) / 5.0
    mean_cal_pos = sum(pos_by_fold[f]["threshold_cal"] for f in folds) / 5.0
    mean_te = sum(len(splits_by_fold[f]["test"]) for f in folds) / 5.0
    mean_te_pos = sum(pos_by_fold[f]["test"] for f in folds) / 5.0
    mean_h_pos = sum(grp_pos_by_fold[f]["test"]["Heavy"] for f in folds) / 5.0
    mean_l_pos = sum(grp_pos_by_fold[f]["test"]["Light"] for f in folds) / 5.0
    mean_f_pos = sum(grp_pos_by_fold[f]["test"]["Flat"] for f in folds) / 5.0

    print("\nSGCC Mean Counts Across Folds 0--4 (Reconciling Table 2):")
    print(f"  Training (D_train):     {mean_tr:.1f} total | {mean_tr_pos:.1f} pos ({100*mean_tr_pos/mean_tr:.2f}%)")
    print(f"  Early-Stopping (D_es):  {mean_es:.1f} total | {mean_es_pos:.1f} pos ({100*mean_es_pos/mean_es:.2f}%)")
    print(f"  Threshold-Cal (D_cal):  {mean_cal:.1f} total | {mean_cal_pos:.1f} pos ({100*mean_cal_pos/mean_cal:.2f}%)")
    print(f"  Held-Out Test (D_test): {mean_te:.1f} total | {mean_te_pos:.1f} pos ({100*mean_te_pos/mean_te:.2f}%)")
    print(f"    -> Test Positives Breakdown: Heavy={mean_h_pos:.1f}, Light={mean_l_pos:.1f}, Flat={mean_f_pos:.1f}")


def audit_ecl(manifest_dir: Path) -> None:
    """Audit 5-fold ECL window and client-level GroupKFold disjointness."""
    p = manifest_dir / "ecl_partition_manifest.csv"
    print(f"\n[2/2] Auditing ECL Partition Manifest: {p.name}")
    print("=" * 72)

    windows_by_fold = defaultdict(lambda: defaultdict(set))
    clients_by_fold = defaultdict(lambda: defaultdict(set))
    pos_by_fold = defaultdict(lambda: defaultdict(int))

    with open(p, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total_rows = 0
        for r in reader:
            total_rows += 1
            fold = int(r["fold"])
            sp = r["split"]
            wid = int(r["window_index"])
            cid = int(r["client_id"])
            lbl = int(r["label"])

            windows_by_fold[fold][sp].add(wid)
            clients_by_fold[fold][sp].add(cid)
            if lbl == 1:
                pos_by_fold[fold][sp] += 1

    print(f"Total manifest rows read: {total_rows} (Expected 125,000 across 5 folds)")
    assert total_rows == 125000, f"Row count mismatch: got {total_rows}"
    folds = sorted(windows_by_fold.keys())
    assert folds == [0, 1, 2, 3, 4], f"Expected folds [0..4], got {folds}"

    heldout_client_sets = []
    for fold in folds:
        sp_w = windows_by_fold[fold]
        sp_c = clients_by_fold[fold]
        assert set(sp_w.keys()) == {"train", "early_stop", "test"}, f"Unexpected splits: {set(sp_w.keys())}"

        tr_w, es_w, te_w = sp_w["train"], sp_w["early_stop"], sp_w["test"]
        tr_c, es_c, te_c = sp_c["train"], sp_c["early_stop"], sp_c["test"]

        tot = len(tr_w) + len(es_w) + len(te_w)
        print(f"\nFold {fold}: Total windows = {tot} (Train={len(tr_w)}, ES={len(es_w)}, Test={len(te_w)})")
        assert tot == 25000, f"Fold {fold} total windows != 25,000"

        assert len(tr_w & es_w) == 0, f"Fold {fold}: Window overlap between train and early_stop!"
        assert len(tr_w & te_w) == 0, f"Fold {fold}: Window overlap between train and test!"
        assert len(es_w & te_w) == 0, f"Fold {fold}: Window overlap between early_stop and test!"

        assert len(tr_c) == 296 and len(es_c) == 26 and len(te_c) == 48, (
            f"Fold {fold} client counts unexpected: {len(tr_c)}/{len(es_c)}/{len(te_c)}"
        )
        assert len(tr_c & es_c) == 0, f"Fold {fold}: Client overlap between train and early_stop!"
        assert len(tr_c & te_c) == 0, f"Fold {fold}: Client overlap between train and test!"
        assert len(es_c & te_c) == 0, f"Fold {fold}: Client overlap between early_stop and test!"
        heldout_client_sets.append(es_c | te_c)

        print("  [OK] Window overlap: 0 across all 3 pairs; Client overlap: 0 across all 3 pairs "
              f"(Train={len(tr_c)}, ES={len(es_c)}, Test={len(te_c)} clients)")

    for i in range(5):
        for j in range(i + 1, 5):
            assert len(heldout_client_sets[i] & heldout_client_sets[j]) == 0, (
                f"Held-out client overlap between Fold {i} and Fold {j}"
            )
    assert len(set.union(*heldout_client_sets)) == 370, "Held-out client sets do not cover all 370 clients"
    print("\n  [OK] Cross-fold held-out client overlap: 0 across all 10 fold pairs (all 370 clients partitioned)")

    mean_tr = sum(len(windows_by_fold[f]["train"]) for f in folds) / 5.0
    mean_tr_pos = sum(pos_by_fold[f]["train"] for f in folds) / 5.0
    mean_es = sum(len(windows_by_fold[f]["early_stop"]) for f in folds) / 5.0
    mean_es_pos = sum(pos_by_fold[f]["early_stop"] for f in folds) / 5.0
    mean_te = sum(len(windows_by_fold[f]["test"]) for f in folds) / 5.0
    mean_te_pos = sum(pos_by_fold[f]["test"] for f in folds) / 5.0

    print("\nECL Mean Counts Across Folds 0--4 (Client-Disjoint GroupKFold, Reconciling Table 2):")
    print(f"  Training (D_train):     {mean_tr:.1f} total | {mean_tr_pos:.1f} pos ({100*mean_tr_pos/mean_tr:.2f}%) | 296 clients")
    print(f"  Early-Stopping (D_es):  {mean_es:.1f} total | {mean_es_pos:.1f} pos ({100*mean_es_pos/mean_es:.2f}%) | 26 clients")
    print(f"  Held-Out Test (D_test): {mean_te:.1f} total | {mean_te_pos:.1f} pos ({100*mean_te_pos/mean_te:.2f}%) | 48 clients")


def main() -> None:
    print("=" * 72)
    print("MAMRD REVISION 3 (5-FOLD) PARTITION DISJOINTNESS & TABLE 2 AUDIT")
    print("=" * 72)
    manifest_dir = find_manifest_dir()
    print(f"Using manifest directory: {manifest_dir}")
    audit_sgcc(manifest_dir)
    audit_ecl(manifest_dir)
    print("\n" + "=" * 72)
    print("[PASS] ALL PARTITIONS 100% DISJOINT ACROSS ALL 5 FOLDS (EXIT 0)")
    print("=" * 72)


if __name__ == "__main__":
    main()
