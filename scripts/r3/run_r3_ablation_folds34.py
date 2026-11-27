#!/usr/bin/env python3
"""Revision 3: extend the Table 5 Panel A/B ablation arms from Folds 0-2 to Folds 3-4.

The six controlled ablation arms of Table 5 (S0-S3, A34_no-miss, A32) were trained in Revision 2 on
Folds 0-2 x Seeds 0-2 by `scripts/run_master_retrain.py --stages ablations`, which calls
`scripts/run_n_clean.py`. This driver calls the same, unmodified `run_n_clean.py` with the same
arguments (30 epochs, AdamW lr 3e-4, KFoldSpec(n_folds=5, seed=0), full 42,372 cohort) for
Folds 3-4 x Seeds 0-2. The full model S4 (a34_generic) already exists for Folds 3-4
(results/r3_revision/C1_5fold_and_stats/raw_folds34/a34_cell_{s}_{f}.csv).

Outputs go to results/r3_revision/C9_ablation_folds34/bu/<arm>_{s}_{f}.csv, so the Revision 2 files in
results/phase_n_retrained/ are never touched. Existing cells are skipped unless --force is given.

Usage:  python scripts/r3/run_r3_ablation_folds34.py [--folds 3 4] [--seeds 0 1 2] [--force]
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPTS = THIS.parent.parent
ROOT = SCRIPTS.parent
OUT = ROOT / "results" / "r3_revision" / "C9_ablation_folds34"

ARMS = [  # (Table 5 label, backbone/arm name)
    ("S0",          "a34_bu0_core"),
    ("S1",          "a34_bu1_ccname"),
    ("S2",          "a34_bu2_mask"),
    ("S3",          "a33_generic"),
    ("A34_no-miss", "a34_no_missingness"),
    ("A32",         "a32_generic"),
]


def verify_math_invariants(folds=(3, 4)) -> bool:
    """Pairing invariant: for every requested fold, the test split that run_n_clean.py rebuilds
    (KFoldSpec(n_folds=5, seed=0) + stratified 33/33/34 split with rng(fold)) must be exactly the
    published partition manifest's test rows and must carry the same labels as the stored full-model
    S4 cell (raw_folds34/a34_cell_0_{f}.csv), so every new ablation cell is paired with S4."""
    import csv, json
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(SCRIPTS))
    from repro_config import SGCC_PATH
    from zkmix_lab.data import KFoldSpec
    from run_local_grid import _stratified_three_way_split

    y = pd.read_csv(SGCC_PATH, usecols=["FLAG"])["FLAG"].values.astype(int)
    kfolds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
    man = pd.read_csv(ROOT / "partition_manifests" / "sgcc_partition_manifest.csv")
    raw34 = ROOT / "results" / "r3_revision" / "C1_5fold_and_stats" / "raw_folds34"
    for f in folds:
        _, val = kfolds[f]
        _, _, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34), rng=np.random.default_rng(f))
        te_man = man[(man.fold == f) & (man.split == "test")].row_index.values
        assert np.array_equal(np.sort(te), np.sort(te_man)), f"fold {f}: rebuilt test split != manifest"
        r = next(csv.DictReader(open(raw34 / f"a34_cell_0_{f}.csv")))
        yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
        assert np.array_equal(yt, y[te]), f"fold {f}: S4 cell labels not aligned with rebuilt test split"
        print(f"  fold {f}: test n={len(te)} pos={int(y[te].sum())} == manifest == S4 cell", flush=True)
    assert len(ARMS) == 6 and len({a for _, a in ARMS}) == 6
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--arms", nargs="+", default=[a for _, a in ARMS])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        assert verify_math_invariants(tuple(args.folds))
        print("[SELF-TEST PASS] run_r3_ablation_folds34 pairing invariants verified.", flush=True)
        return

    (OUT / "bu").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MAMRD_RESULTS=str(OUT))
    t0 = time.time()
    for label, arm in ARMS:
        if arm not in args.arms:
            continue
        todo_s = sorted({s for s in args.seeds for f in args.folds
                         if args.force or not (OUT / "bu" / f"{arm}_{s}_{f}.csv").exists()})
        if not todo_s:
            print(f"[skip] {label} ({arm}): all cells exist", flush=True)
            continue
        for s in todo_s:
            todo_f = [f for f in args.folds if args.force or not (OUT / "bu" / f"{arm}_{s}_{f}.csv").exists()]
            cmd = [sys.executable, "run_n_clean.py", "--backbone", arm, "--arm", arm,
                   "--epochs", str(args.epochs), "--seeds", str(s), "--folds", *map(str, todo_f),
                   "--cell_prefix", f"bu/{arm}",
                   "--manifest", str(OUT / "bu" / f"{arm}_manifest_s{s}.csv")]
            print(f"[run] {label} ({arm}) seed {s} folds {todo_f}", flush=True)
            subprocess.run(cmd, cwd=SCRIPTS, env=env, check=True)
    print(f"[done] ablation folds {args.folds} in {(time.time() - t0) / 60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
