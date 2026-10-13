#!/usr/bin/env python3
"""Phase D grid runner — entrypoint for Kaggle / local execution.

Usage
-----
# Full Phase D grid (A1–A5, 5 seeds × 3 folds = 75 cells):
python run_grid_d.py --data_dir ../../data --out_dir ../../results/phase_d_grid

# Quick smoke test (A3 only, 2 seeds, 1 fold):
python run_grid_d.py --data_dir ../../data --out_dir ../../results/phase_d_grid \
    --arms a3_cc_name_ms --seeds 0 1 --n_folds 1

# Headline arm + mechanism ablations only (A1, A2, A3):
python run_grid_d.py --data_dir ../../data --out_dir ../../results/phase_d_grid \
    --arms a1_cc_name_only a2_ms_only a3_cc_name_ms

# Compute verdicts after run (vs Phase C manifest):
python run_grid_d.py --verdicts_only \
    --phase_c_manifest ../../results/phase_c_grid/manifest.csv \
    --out_dir ../../results/phase_d_grid

Pre-registration integrity
--------------------------
This script computes the SHA256 of itself at startup and logs it.  The hash
must match the value recorded in PLAN_D_2026-05-27.md §9 before any cell
is executed.  If the file has been modified since pre-registration, a warning
is printed and the operator must confirm before proceeding.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve package paths.
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).parent
_LAB_DIR = (_HERE / ".." / "zkmix_lab").resolve()
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from phased.runner_d import CellD, phase_d_grid, run_grid, compute_verdicts  # type: ignore
from phased.models import ARM_BUILDERS                                        # type: ignore
from zkmix_lab.training import TrainConfig                                    # type: ignore


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

def _self_sha256() -> str:
    data = Path(__file__).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _check_integrity(plan_path: Path) -> None:
    sha = _self_sha256()
    print(f"[integrity] run_grid_d.py SHA256: {sha}")
    if plan_path.exists():
        content = plan_path.read_text()
        if sha in content:
            print("[integrity] SHA matches pre-registration record. ✓")
        else:
            print(
                "[integrity] WARNING: SHA not found in PLAN_D_2026-05-27.md.\n"
                "  Either the plan has not been updated with the runner SHA,\n"
                "  or the runner was modified after pre-registration.\n"
                "  Update §9 of the plan file before proceeding."
            )
    else:
        print(f"[integrity] Plan file not found at {plan_path}. Skipping check.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D experiment grid runner.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to directory containing data/after_preprocess_data.csv")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for manifest.csv and verdicts.")
    p.add_argument("--arms", nargs="+", default=None,
                   choices=list(ARM_BUILDERS.keys()) + [None],
                   help="Which arms to run (default: all).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to run (default: [0,1,2,3,4]).")
    p.add_argument("--n_folds", type=int, default=3,
                   help="Number of cross-validation folds (default: 3).")
    p.add_argument("--regime", type=str, default="full",
                   choices=["full", "high_miss", "stable"],
                   help="SGCC regime slice.")
    p.add_argument("--split_family", type=str, default="random",
                   choices=["random", "stratified"],
                   help="Cross-validation split family.")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (default: cuda if available, else cpu).")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs (default: 30, matches Phase C).")
    p.add_argument("--verdicts_only", action="store_true",
                   help="Skip training; only compute verdicts from existing manifest.")
    p.add_argument("--phase_c_manifest", type=str, default=None,
                   help="Path to Phase C manifest.csv for paired verdict computation.")
    p.add_argument("--no_integrity_check", action="store_true",
                   help="Skip the SHA integrity check.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    plan_path = _HERE / "PLAN_D_2026-05-27.md"

    if not args.no_integrity_check:
        _check_integrity(plan_path)

    if args.verdicts_only:
        if args.phase_c_manifest is None:
            print("ERROR: --phase_c_manifest required for --verdicts_only.")
            sys.exit(1)
        if args.out_dir is None:
            print("ERROR: --out_dir required.")
            sys.exit(1)
        out_dir = Path(args.out_dir)
        compute_verdicts(
            manifest_d=__import__("pandas").read_csv(out_dir / "manifest.csv"),
            manifest_c_path=args.phase_c_manifest,
            output_path=out_dir / "verdicts_d.md",
        )
        return

    if args.data_dir is None:
        print("ERROR: --data_dir required.")
        sys.exit(1)
    if args.out_dir is None:
        print("ERROR: --out_dir required.")
        sys.exit(1)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Training config — identical to Phase C defaults.
    train_cfg = TrainConfig(
        epochs=args.epochs,
        device=device,
        # All other fields at Phase C defaults:
        batch_size=256,
        pos_fraction=0.5,
        lr=3e-4,
        weight_decay=0.01,
        warmup_frac=0.05,
        min_lr_ratio=0.01,
        grad_clip_norm=1.0,
        patience=8,
    )

    # Build cell list.
    cells = phase_d_grid(
        arms=args.arms,
        seeds=args.seeds,
        n_folds=args.n_folds,
        regime=args.regime,
        split_family=args.split_family,
    )
    print(f"[grid] {len(cells)} cells to run "
          f"(arms={args.arms or 'all'}, "
          f"seeds={args.seeds or [0,1,2,3,4]}, "
          f"n_folds={args.n_folds})")

    manifest_path = Path(args.out_dir) / "manifest.csv"
    run_grid(
        cells=cells,
        data_dir=args.data_dir,
        manifest_path=manifest_path,
        train_cfg=train_cfg,
        verbose=True,
    )

    print(f"\n[done] Manifest written to {manifest_path}")
    if args.phase_c_manifest:
        verdicts_path = Path(args.out_dir) / "verdicts_d.md"
        import pandas as pd
        compute_verdicts(
            manifest_d=pd.read_csv(manifest_path),
            manifest_c_path=args.phase_c_manifest,
            output_path=verdicts_path,
        )


if __name__ == "__main__":
    main()
