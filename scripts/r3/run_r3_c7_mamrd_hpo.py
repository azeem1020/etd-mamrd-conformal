#!/usr/bin/env python3
"""Revision 3, Comment 7: give MAMRD the same validation-based HPO as the tuned baselines.

run_c7_baseline_hpo.py tunes HORLN and Self-Attention (hybridattn) over six (lr, wd) configurations
on Fold 0, Seed 0 D_es and evaluates the D_es winner on the 15 matched cells (Folds 0-4 x Seeds 0-2),
while MAMRD stays at the shared schedule (lr 3e-4, wd 1e-4). This driver applies the identical
protocol to MAMRD (backbone a34_generic) through the identical code path (run_single_cell, same
grid, same selection rule, same cells), so the comparison becomes tuned vs. tuned.

Pre-committed reporting rule (R3/audit/R3_comment_audit/02_*.md, Addendum 2): tuned MAMRD is
reported whatever its value; the Table 3 headline remains default MAMRD.

Outputs: results/r3_revision/C7_baseline_hpo/cells/{hpo_mamrd_*,tuned_mamrd_*}.csv and
results/r3_revision/C7_baseline_hpo/c7_mamrd_tuning_{grid,summary}.csv, c7_mamrd_tuning_summary.json.
Usage: python scripts/r3/run_r3_c7_mamrd_hpo.py [--self-test] [--folds 0 1 2 3 4]
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent))
import run_c7_baseline_hpo as c7  # noqa: E402  (shared grid, cell runner, BCa, paths)

BACKBONE = "a34_generic"
LABEL = "mamrd"
BASELINES = ("horln", "hybridattn")


def default_mamrd_cell(s: int, f: int) -> dict:
    p = (c7.ORIG_RES if f < 3 else c7.RAW34) / f"a34_cell_{s}_{f}.csv"
    return next(csv.DictReader(open(p)))


def verify_math_invariants(folds=(0, 1, 2, 3, 4)) -> bool:
    """(1) The MAMRD grid is exactly the baseline grid and contains the shared schedule.
    (2) Every tuned-baseline cell and every default MAMRD cell for (s, f) carries the same test labels,
        so tuned-MAMRD cells (same split code) are paired with them.
    (3) The selection rule picks argmax D_es AP and ignores test AP."""
    assert len(c7.HPO_GRID) == 6 and (3e-4, 1e-4) in c7.HPO_GRID
    for s in range(3):
        for f in folds:
            ref = json.loads(default_mamrd_cell(s, f)["val_labels_json"])
            for bb in BASELINES:
                r = next(csv.DictReader(open(c7.CELLS_DIR / f"tuned_{bb}_{s}_{f}.csv")))
                assert json.loads(r["val_labels_json"]) == ref, f"label mismatch {bb} s{s} f{f}"
    toy = [{"lr": 1, "wd": 0, "best_val_ap": 0.5, "pr_auc": 0.9},
           {"lr": 2, "wd": 0, "best_val_ap": 0.6, "pr_auc": 0.1}]
    assert select_winner(toy)["lr"] == 2
    print(f"  pairing OK on {3 * len(folds)} cells x {len(BASELINES)} tuned baselines; grid = baseline grid", flush=True)
    return True


def select_winner(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda d: float(d["best_val_ap"]), reverse=True)[0]


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    lo, hi = c7.bca_ci(d)
    p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="exact").pvalue) if np.any(d != 0) else 1.0
    return {"dAP": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "wins": int((d > 0).sum()), "n": len(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    if args.self_test:
        assert verify_math_invariants(tuple(args.folds))
        print("[SELF-TEST PASS] run_r3_c7_mamrd_hpo invariants verified.", flush=True)
        return

    t0 = time.time()
    X_base, y = c7.load_and_preprocess(c7.SGCC_PATH)
    folds = list(c7.KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
    cols = ["label", "backbone", "lr", "wd", "seed", "fold", "best_val_ap", "best_epoch", "wall_s", "pr_auc",
            "roc_auc", "f1", "map_at_100", "p_at_100", "val_labels_json", "val_scores_json"]

    def get_or_run(fname, label, s, f, lr, wd):
        fp = c7.CELLS_DIR / fname
        if fp.exists() and fp.stat().st_size > 100:
            d = pd.read_csv(fp)
            if len(d) == 1 and np.isfinite(float(d["pr_auc"].iloc[0])):
                return d.iloc[0].to_dict(), True
        c7.cool_if_hot()
        r = c7.run_single_cell(X_base, y, folds, label, BACKBONE, s, f, lr, wd, epochs=args.epochs, device=args.device)
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerow(r)
        return r, False

    print("=== HPO grid on D_es (Fold 0, Seed 0) for MAMRD ===", flush=True)
    grid = []
    for lr, wd in c7.HPO_GRID:
        tag = f"hpo_{LABEL}_lr{lr}_wd{wd}_s0_f0"
        r, sk = get_or_run(f"{tag}.csv", tag, 0, 0, lr, wd)
        grid.append(r)
        print(f"  [{'SKIP' if sk else 'RUN'}] lr={lr} wd={wd} val_AP={float(r['best_val_ap']):.4f} "
              f"test_AP={float(r['pr_auc']):.4f} ep={r['best_epoch']}", flush=True)
    pd.DataFrame(grid).drop(columns=["val_labels_json", "val_scores_json"]).to_csv(
        c7.OUT / "c7_mamrd_tuning_grid.csv", index=False)
    win = select_winner(grid)
    w_lr, w_wd = float(win["lr"]), float(win["wd"])
    shared = (w_lr, w_wd) == (3e-4, 1e-4)
    print(f"  -> D_es winner: lr={w_lr} wd={w_wd} val_AP={float(win['best_val_ap']):.4f} (shared schedule: {shared})", flush=True)
    shared_row = next(r for r in grid if (float(r["lr"]), float(r["wd"])) == (3e-4, 1e-4))
    print(f"  consistency: C7-path shared-config F0S0 test AP {float(shared_row['pr_auc']):.4f} vs "
          f"default cell {float(default_mamrd_cell(0, 0)['pr_auc']):.4f}", flush=True)

    tuned = []
    for s in range(3):
        for f in args.folds:
            r, sk = get_or_run(f"tuned_{LABEL}_{s}_{f}.csv", f"tuned_{LABEL}", s, f, w_lr, w_wd)
            assert json.loads(r["val_labels_json"]) == json.loads(default_mamrd_cell(s, f)["val_labels_json"])
            tuned.append(float(r["pr_auc"]))
            print(f"  [{'SKIP' if sk else 'RUN'}] tuned_mamrd s={s} f={f} PR={float(r['pr_auc']):.4f}", flush=True)
    tuned = np.array(tuned)
    default = np.array([float(default_mamrd_cell(s, f)["pr_auc"]) for s in range(3) for f in args.folds])
    base = {bb: np.array([float(next(csv.DictReader(open(c7.CELLS_DIR / f"tuned_{bb}_{s}_{f}.csv")))["pr_auc"])
                          for s in range(3) for f in args.folds]) for bb in BASELINES}
    lo, hi = c7.bca_ci(tuned)
    summary = {
        "selected_lr": w_lr, "selected_wd": w_wd, "selected_is_shared_schedule": shared,
        "selected_des_ap": float(win["best_val_ap"]), "n_cells": len(tuned),
        "tuned_mamrd_ap": float(tuned.mean()), "tuned_mamrd_ci": [lo, hi],
        "default_mamrd_ap": float(default.mean()),
        "tuned_baseline_ap": {bb: float(v.mean()) for bb, v in base.items()},
        "tuned_mamrd_minus_default_mamrd": paired(tuned, default),
        **{f"tuned_mamrd_minus_tuned_{bb}": paired(tuned, v) for bb, v in base.items()},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    pd.DataFrame([{"contrast": k, **v} for k, v in summary.items() if isinstance(v, dict) and "dAP" in v]).to_csv(
        c7.OUT / "c7_mamrd_tuning_summary.csv", index=False)
    (c7.OUT / "c7_mamrd_tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
