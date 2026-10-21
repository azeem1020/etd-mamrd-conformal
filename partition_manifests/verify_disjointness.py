#!/usr/bin/env python3
"""Deterministic Partition Disjointness & Ledger Verification Script.

Verifies 100% customer-disjoint partitioning on SGCC and window-disjoint
partitioning on ECL across Folds 0--2, reproducing the cohort counts in Table 2.

Usage:
    python verify_disjointness.py
    python scripts/verify_partition_manifests.py
"""
import sys, os, csv
from pathlib import Path
from collections import defaultdict

def find_manifest_dir():
    # Try current directory first, then standard repo paths
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "partition_manifests",
        Path(__file__).resolve().parent.parent / "partition_manifests",
        Path(__file__).resolve().parent.parent / "results" / "phase_n" / "r2_audit",
    ]
    for c in candidates:
        if (c / "sgcc_partition_manifest.csv").exists() and (c / "ecl_partition_manifest.csv").exists():
            return c
    raise FileNotFoundError("Could not locate partition manifests in standard search paths.")

def audit_sgcc(manifest_dir):
    p = manifest_dir / "sgcc_partition_manifest.csv"
    print(f"\n[1/2] Auditing SGCC Partition Manifest: {p.name}")
    print("=" * 70)
    
    # fold -> split -> set of customer_ids
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

    print(f"Total manifest rows read: {total_rows} (Expected 127,116 across 3 folds)")
    assert total_rows == 127116, f"Row count mismatch: got {total_rows}"
    
    for fold in sorted(splits_by_fold.keys()):
        sp_dict = splits_by_fold[fold]
        tr = sp_dict["train"]
        es = sp_dict["early_stop"]
        cal = sp_dict["threshold_cal"]
        te = sp_dict["test"]
        
        tot = len(tr) + len(es) + len(cal) + len(te)
        print(f"\nFold {fold}: Total customers = {tot} (Train={len(tr)}, ES={len(es)}, Cal={len(cal)}, Test={len(te)})")
        assert tot == 42372, f"Fold {fold} total customers != 42,372"
        
        # Deterministic disjointness assertions
        assert len(tr & es) == 0, f"Fold {fold}: Overlap detected between train and early_stop!"
        assert len(tr & cal) == 0, f"Fold {fold}: Overlap detected between train and threshold_cal!"
        assert len(tr & te) == 0, f"Fold {fold}: Overlap detected between train and test!"
        assert len(es & cal) == 0, f"Fold {fold}: Overlap detected between early_stop and threshold_cal!"
        assert len(es & te) == 0, f"Fold {fold}: Overlap detected between early_stop and test!"
        assert len(cal & te) == 0, f"Fold {fold}: Overlap detected between threshold_cal and test!"
        print(f"  [OK] Pairwise customer overlap: 0 across all 6 pairs (100% customer-disjoint)")
        print(f"  Test Positives: {pos_by_fold[fold]['test']} "
              f"(Heavy={grp_pos_by_fold[fold]['test']['Heavy']}, "
              f"Light={grp_pos_by_fold[fold]['test']['Light']}, "
              f"Flat={grp_pos_by_fold[fold]['test']['Flat']})")
              
    # Means across Folds 0--2 matching Table 2
    mean_tr = sum(len(splits_by_fold[f]["train"]) for f in (0, 1, 2)) / 3.0
    mean_tr_pos = sum(pos_by_fold[f]["train"] for f in (0, 1, 2)) / 3.0
    mean_es = sum(len(splits_by_fold[f]["early_stop"]) for f in (0, 1, 2)) / 3.0
    mean_es_pos = sum(pos_by_fold[f]["early_stop"] for f in (0, 1, 2)) / 3.0
    mean_cal = sum(len(splits_by_fold[f]["threshold_cal"]) for f in (0, 1, 2)) / 3.0
    mean_cal_pos = sum(pos_by_fold[f]["threshold_cal"] for f in (0, 1, 2)) / 3.0
    mean_te = sum(len(splits_by_fold[f]["test"]) for f in (0, 1, 2)) / 3.0
    mean_te_pos = sum(pos_by_fold[f]["test"] for f in (0, 1, 2)) / 3.0
    mean_h_pos = sum(grp_pos_by_fold[f]["test"]["Heavy"] for f in (0, 1, 2)) / 3.0
    mean_l_pos = sum(grp_pos_by_fold[f]["test"]["Light"] for f in (0, 1, 2)) / 3.0
    mean_f_pos = sum(grp_pos_by_fold[f]["test"]["Flat"] for f in (0, 1, 2)) / 3.0
    
    print("\nSGCC Mean Counts Across Folds 0--2 (Reconciling Table 2):")
    print(f"  Training (D_train):     {mean_tr:.1f} total | {mean_tr_pos:.1f} pos ({100*mean_tr_pos/mean_tr:.2f}%)")
    print(f"  Early-Stopping (D_es):  {mean_es:.1f} total | {mean_es_pos:.1f} pos ({100*mean_es_pos/mean_es:.2f}%)")
    print(f"  Threshold-Cal (D_cal):  {mean_cal:.1f} total | {mean_cal_pos:.1f} pos ({100*mean_cal_pos/mean_cal:.2f}%)")
    print(f"  Held-Out Test (D_test): {mean_te:.1f} total | {mean_te_pos:.1f} pos ({100*mean_te_pos/mean_te:.2f}%)")
    print(f"    -> Test Positives Breakdown: Heavy={mean_h_pos:.1f}, Light={mean_l_pos:.1f}, Flat={mean_f_pos:.1f}")

def audit_ecl(manifest_dir):
    p = manifest_dir / "ecl_partition_manifest.csv"
    print(f"\n[2/2] Auditing ECL Partition Manifest: {p.name}")
    print("=" * 70)
    
    # fold -> split -> set of window_indices
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

    print(f"Total manifest rows read: {total_rows} (Expected 75,000 across 3 folds)")
    assert total_rows == 75000, f"Row count mismatch: got {total_rows}"
    
    for fold in sorted(windows_by_fold.keys()):
        sp_dict = windows_by_fold[fold]
        tr_w = sp_dict["train"]
        es_w = sp_dict["early_stop"]
        cal_w = sp_dict["threshold_cal"]
        te_w = sp_dict["test"]
        
        tot = len(tr_w) + len(es_w) + len(cal_w) + len(te_w)
        print(f"\nFold {fold}: Total windows = {tot} (Train={len(tr_w)}, ES={len(es_w)}, Cal={len(cal_w)}, Test={len(te_w)})")
        assert tot == 25000, f"Fold {fold} total windows != 25,000"
        
        # Window-level disjointness
        assert len(tr_w & es_w) == 0, f"Fold {fold}: Window overlap between train and early_stop!"
        assert len(tr_w & cal_w) == 0, f"Fold {fold}: Window overlap between train and threshold_cal!"
        assert len(tr_w & te_w) == 0, f"Fold {fold}: Window overlap between train and test!"
        assert len(es_w & cal_w) == 0, f"Fold {fold}: Window overlap between early_stop and threshold_cal!"
        assert len(es_w & te_w) == 0, f"Fold {fold}: Window overlap between early_stop and test!"
        assert len(cal_w & te_w) == 0, f"Fold {fold}: Window overlap between threshold_cal and test!"
        print(f"  [OK] Pairwise window overlap: 0 across all 6 pairs (100% window-disjoint)")
        
        # Client sharing check (Point 4 demonstration)
        tr_c = clients_by_fold[fold]["train"]
        te_c = clients_by_fold[fold]["test"]
        shared_c = tr_c & te_c
        pct_shared = 100.0 * len(shared_c) / len(te_c)
        print(f"  Client Overlap: {len(tr_c)} train clients, {len(te_c)} test clients -> "
              f"{len(shared_c)} shared ({pct_shared:.1f}% of test clients in train; "
              f"demonstrating window-level evaluation)")

    mean_tr = sum(len(windows_by_fold[f]["train"]) for f in (0, 1, 2)) / 3.0
    mean_tr_pos = sum(pos_by_fold[f]["train"] for f in (0, 1, 2)) / 3.0
    mean_es = sum(len(windows_by_fold[f]["early_stop"]) for f in (0, 1, 2)) / 3.0
    mean_es_pos = sum(pos_by_fold[f]["early_stop"] for f in (0, 1, 2)) / 3.0
    mean_cal = sum(len(windows_by_fold[f]["threshold_cal"]) for f in (0, 1, 2)) / 3.0
    mean_cal_pos = sum(pos_by_fold[f]["threshold_cal"] for f in (0, 1, 2)) / 3.0
    mean_te = sum(len(windows_by_fold[f]["test"]) for f in (0, 1, 2)) / 3.0
    mean_te_pos = sum(pos_by_fold[f]["test"] for f in (0, 1, 2)) / 3.0

    print("\nECL Mean Counts Across Folds 0--2 (Reconciling Table 2):")
    print(f"  Training (D_train):     {mean_tr:.1f} total | {mean_tr_pos:.1f} pos ({100*mean_tr_pos/mean_tr:.2f}%)")
    print(f"  Early-Stopping (D_es):  {mean_es:.1f} total | {mean_es_pos:.1f} pos ({100*mean_es_pos/mean_es:.2f}%)")
    print(f"  Reserved Cal (D_cal):   {mean_cal:.1f} total | {mean_cal_pos:.1f} pos ({100*mean_cal_pos/mean_cal:.2f}%)")
    print(f"  Held-Out Test (D_test): {mean_te:.1f} total | {mean_te_pos:.1f} pos ({100*mean_te_pos/mean_te:.2f}%)")

def main():
    print("======================================================================")
    print("MAMRD PARTITION DISJOINTNESS & TABLE 2 RECONCILIATION AUDIT")
    print("======================================================================")
    manifest_dir = find_manifest_dir()
    print(f"Using manifest directory: {manifest_dir}")
    audit_sgcc(manifest_dir)
    audit_ecl(manifest_dir)
    print("\n" + "=" * 70)
    print("[PASS] ALL PARTITIONS 100% DISJOINT ACROSS ALL FOLDS (EXIT 0)")
    print("======================================================================")

if __name__ == "__main__":
    main()
