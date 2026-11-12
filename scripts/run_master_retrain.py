"""Master Retraining Orchestrator for Scientific Reports Revision 3.

Executes the complete, isolated retraining pipeline from scratch into RESULTS_DIR
(default: results/phase_n_retrained/), ensuring bit-for-bit isolation from older results.

Stages:
  1. Cohort Cleaning: scripts/clean_cohort.py
  2. MAMRD 15-cell Training: a34_generic (seeds 0..4, folds 0..2, 30 epochs)
  3. Full Ablation Suite (9 cells each: seeds 0..2, folds 0..2):
     - S0: a34_bu0_core
     - S1: a34_bu1_ccname
     - S2: a34_bu2_mask
     - S3: a33_generic
     - A32: a32_generic
     - A34_no_miss: a34_no_missingness (controlled S4 ablation)
  4. Baseline detectors (9 cells each: seeds 0..2, folds 0..2):
     - wdcnn, horln, hybridattn, grud, msdm_net, dkgt
  5. ECL Transfer Benchmark:
     - run_ecl_trust.py + run_ecl_trust_analysis.py
  6. Conformal Calibration & Trust Layer:
     - run_ceiling_estimate.py, run_trust_layer.py, run_decision_value.py,
       run_empirical_comparator.py, run_finardi_repro.py
  7. Post-Processing & Manuscript Artifact Generation:
     - compute_r2_revisions.py, compute_table_b.py, compute_table_c.py,
       make_manuscript_figures.py, make_figures.py, make_trust_figure.py
"""
from __future__ import annotations
import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent

# Set default results dir to phase_n_retrained if not specified
if "MAMRD_RESULTS" not in os.environ:
    os.environ["MAMRD_RESULTS"] = str(ROOT / "results" / "phase_n_retrained")

from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH, scrub_paths

OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)
for sub in ["baselines", "bu", "ecl_baselines", "ecl_trust", "figures", "r2_audit", "tableA", "tableB", "tableC", "trust"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "master_retrain.log"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = scrub_paths(f"[{ts}] {msg}")
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def run_cmd(cmd: list[str], desc: str, env=None) -> bool:
    log(f"--- START: {desc} ---")
    log(f"CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    current_env = os.environ.copy()
    current_env["PYTHONPATH"] = f"{THIS}:{ROOT}:{current_env.get('PYTHONPATH', '')}"
    if env:
        current_env.update(env)

    proc = subprocess.Popen(
        cmd,
        cwd=str(THIS),
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with open(LOG_FILE, "a") as f:
        for line in proc.stdout:
            line = scrub_paths(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)

    proc.wait()
    wall = time.time() - t0
    if proc.returncode != 0:
        log(f"FAILED (code {proc.returncode}) after {wall:.1f}s: {desc}")
        return False
    log(f"COMPLETED in {wall:.1f}s: {desc}")
    return True


def cells_exist(dir_path: Path, prefix: str, seeds: list[int], folds: list[int]) -> bool:
    for s in seeds:
        for f in folds:
            p = dir_path / f"{prefix}_{s}_{f}.csv"
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True


def stage_cohort(args):
    log("=== STAGE 1: COHORT PREPARATION & CLEANING ===")
    # Chronological ordering of the raw SGCC columns and the 2016-09-18 fill (idempotent).
    # Training reads columns in file order, so this must run before any model is trained.
    if not run_cmd([sys.executable, "prepare_sgcc_dataset.py"], "SGCC chronological preparation"):
        return False
    keep_fp = OUT / "keep_idx.npy"
    if keep_fp.exists() and not args.force:
        log(f"keep_idx.npy exists at {keep_fp}. Skipping (use --force to rerun).")
        return True
    return run_cmd([sys.executable, "clean_cohort.py"], "Clean cohort filter")


def stage_mamrd(args):
    log("=== STAGE 2: MAMRD a34_generic TRAINING (15 CELLS) ===")
    seeds = [0, 1, 2, 3, 4]
    folds = [0, 1, 2]
    if cells_exist(OUT, "a34_cell", seeds, folds) and not args.force:
        log("All 15 MAMRD a34_cell CSVs already exist. Skipping (use --force to rerun).")
        return True

    cmd = [
        sys.executable, "run_n_clean.py",
        "--backbone", "a34_generic",
        "--arm", "a34_generic",
        "--epochs", str(args.epochs),
        "--seeds", *[str(s) for s in seeds],
        "--folds", *[str(f) for f in folds],
        "--cell_prefix", "a34_cell",
        "--manifest", str(OUT / "a34_manifest.csv"),
    ]
    return run_cmd(cmd, "MAMRD a34_generic 15-cell training")


def stage_ablations(args):
    log("=== STAGE 3: FULL ABLATION SUITE TRAINING (9 CELLS EACH) ===")
    seeds = [0, 1, 2]
    folds = [0, 1, 2]
    ablations = [
        ("S0: a34_bu0_core", "a34_bu0_core", "bu/a34_bu0_core", str(OUT / "bu/a34_bu0_core_manifest.csv")),
        ("S1: a34_bu1_ccname", "a34_bu1_ccname", "bu/a34_bu1_ccname", str(OUT / "bu/a34_bu1_ccname_manifest.csv")),
        ("S2: a34_bu2_mask", "a34_bu2_mask", "bu/a34_bu2_mask", str(OUT / "bu/a34_bu2_mask_manifest.csv")),
        ("S3: a33_generic", "a33_generic", "bu/a33_generic", str(OUT / "a33_sgcc.csv")),
        ("A32: a32_generic", "a32_generic", "bu/a32_generic", str(OUT / "a32_sgcc.csv")),
        ("A34_no_miss: a34_no_missingness", "a34_no_missingness", "bu/a34_no_missingness", str(OUT / "bu/a34_no_missingness_manifest.csv")),
    ]

    for desc, backbone, prefix, manifest in ablations:
        base_prefix = Path(prefix).name
        parent_dir = OUT / Path(prefix).parent
        if cells_exist(parent_dir, base_prefix, seeds, folds) and not args.force:
            log(f"All 9 cells for {desc} already exist in {parent_dir}. Skipping.")
            continue

        cmd = [
            sys.executable, "run_n_clean.py",
            "--backbone", backbone,
            "--arm", backbone,
            "--epochs", str(args.epochs),
            "--seeds", *[str(s) for s in seeds],
            "--folds", *[str(f) for f in folds],
            "--cell_prefix", prefix,
            "--manifest", manifest,
        ]
        if not run_cmd(cmd, f"Ablation {desc}"):
            return False
    return True


def stage_baselines(args):
    log("=== STAGE 4: RIVAL BASELINES TRAINING (9 CELLS EACH) ===")
    seeds = [0, 1, 2]
    folds = [0, 1, 2]
    baselines = ["wdcnn", "horln", "hybridattn", "grud", "msdm_net", "dkgt"]

    for b in baselines:
        if cells_exist(OUT / "baselines", b, seeds, folds) and not args.force:
            log(f"All 9 cells for baseline {b} already exist in baselines/. Skipping.")
            continue

        cmd = [
            sys.executable, "run_n_clean.py",
            "--backbone", b,
            "--arm", b,
            "--epochs", str(args.epochs),
            "--seeds", *[str(s) for s in seeds],
            "--folds", *[str(f) for f in folds],
            "--cell_prefix", f"baselines/{b}",
            "--manifest", str(OUT / "baselines" / f"{b}_manifest.csv"),
        ]
        if not run_cmd(cmd, f"Baseline {b}"):
            return False
    return True


def stage_ecl(args):
    log("=== STAGE 5: ECL TRANSFER BENCHMARK ===")
    ecl_trust_dir = OUT / "ecl_trust"
    if (ecl_trust_dir / "ecl_trust_percell.csv").exists() and not args.force:
        log(f"ECL trust results exist in {ecl_trust_dir}. Skipping.")
        return True

    if not run_cmd([sys.executable, "run_ecl_trust.py"], "ECL Trust Benchmark Training"):
        return False
    if not run_cmd([sys.executable, "run_ecl_trust_analysis.py"], "ECL Trust Analysis"):
        return False
    return True


def stage_trust(args):
    log("=== STAGE 6: CONFORMAL CALIBRATION, CEILINGS & DECISION LAYERS ===")
    trust_dir = OUT / "trust"
    scripts = [
        ("run_ceiling_estimate.py", "Ceiling estimation"),
        ("run_trust_layer.py", "Conformal trust layer evaluation"),
        ("run_decision_value.py", "Decision value economic curves"),
        ("run_empirical_comparator.py", "Empirical comparator evaluation"),
        ("run_finardi_repro.py", "Finardi et al. replication baseline"),
    ]
    for script, desc in scripts:
        if not run_cmd([sys.executable, script], desc):
            return False
    return True


def stage_postproc(args):
    log("=== STAGE 7: POST-PROCESSING, TABLES & FIGURES GENERATION ===")
    # Same post-processing as Tier 1: every table and figure from the freshly cached predictions
    return run_cmd([sys.executable, "reproduce_all_r2.py", "--config", str(THIS.parent / "configs" / "default.yaml")],
                   "Tier 1 master: all tables and figures (reproduce_all_r2.py)")


def main():
    parser = argparse.ArgumentParser(description="Master Retraining Orchestrator")
    parser.add_argument("--stages", nargs="+", default=["all"],
                        choices=["all", "cohort", "mamrd", "ablations", "baselines", "ecl", "trust", "postproc"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Force rerun even if cell files exist")
    args = parser.parse_args()

    stages = args.stages
    if "all" in stages:
        stages = ["cohort", "mamrd", "ablations", "baselines", "ecl", "trust", "postproc"]

    log("=================================================================")
    log("MASTER RETRAINING PIPELINE START")
    log(f"Target Results Directory: {OUT}")
    log(f"Stages to execute: {', '.join(stages)}")
    log(f"Epochs per cell: {args.epochs} | Force: {args.force}")
    log("=================================================================")

    start_time = time.time()
    stage_map = {
        "cohort": stage_cohort,
        "mamrd": stage_mamrd,
        "ablations": stage_ablations,
        "baselines": stage_baselines,
        "ecl": stage_ecl,
        "trust": stage_trust,
        "postproc": stage_postproc,
    }

    for s in stages:
        ok = stage_map[s](args)
        if not ok:
            log(f"Pipeline ABORTED at stage: {s}")
            sys.exit(1)

    total_wall = time.time() - start_time
    log("=================================================================")
    log(f"MASTER RETRAINING PIPELINE COMPLETED SUCCESSFULLY in {total_wall/60:.1f} minutes!")
    log(f"All retrained artifacts, figures, and tables saved in {OUT}")
    log("=================================================================")


if __name__ == "__main__":
    main()

