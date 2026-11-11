#!/usr/bin/env python3
"""Unified Master Reproduction Script for Round-2 Revisions.

Command-line usage:
    python scripts/reproduce_all_r2.py --config configs/default.yaml

Tier-1 reproduction from the cached prediction-level records (no retraining):
  1. verify customer/window disjointness of the partition manifests (Table 2);
  2. run the conformal audit analyses on the shared protocol (conformal_common.py):
     run_decision_value (Table 6, Fig. 5b), run_empirical_comparator (Fig. 5a),
     run_trust_layer (Supp. Fig. S4), run_ecl_trust_analysis (Supp. Table S6);
  3. recompute the supplementary tables: per-group PR-AUC (compute_table_b, Supp. Table S4),
     prevalence-matched contrasts (compute_table_c + table_c_contrasts, Supp. Table S3), ECL per-attack recall
     (compute_table_a, Supp. Table S7 note) and the ECL metric suite (make_ecl_board, Supp. Table S7);
  4. recompute Tables 2-5 of the main text, the statistical tests quoted with them and the
     conformal eligibility ledger (compute_r2_revisions -> r2_audit/table*.csv);
  5. regenerate all data-driven main and supplementary figures (Fig. 4 ECE values ->
     trust/fig4_ece_by_detector.csv).
Independent retraining from raw data is the separate Tier-2 sequence in README.md.
Figure 1 (architecture) is a TikZ schematic and is not data-generated.
"""
from __future__ import annotations
import sys
import os
import argparse
import subprocess
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO_ROOT = THIS.parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_yaml_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        print(f"[warning] Config file {cfg_path} not found. Using default internal parameters.")
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        print("[info] PyYAML not installed; reading plain config text.")
        return {}

def run_step(step_name: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n{'='*75}")
    print(f">> RUNNING STEP: {step_name}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*75}")
    # stream the step's output to the console and to results/.../logs/<script>.log
    from repro_config import RESULTS_DIR
    log_dir = RESULTS_DIR / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{Path(cmd[1]).stem}.log"
    sys.stdout.flush()
    with open(log_path, "w", encoding="utf-8") as lf:
        # Fixed timestamp so matplotlib writes byte-identical PDF figures on every run.
        env = dict(os.environ, SOURCE_DATE_EPOCH=os.environ.get("SOURCE_DATE_EPOCH", "1767225600"))
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line); lf.write(line)
        returncode = proc.wait()
    if returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {returncode}")
        sys.exit(returncode)
    print(f"[PASS] {step_name} completed successfully.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Unified master reproduction runner for MAMRD Scientific Reports Round-2 revisions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure rendering and only compute tabular metrics.",
    )
    args = parser.parse_args()

    cfg_file = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    print("=" * 75)
    print("MAMRD ROUND-2 UNIFIED REPRODUCTION SUITE")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Active Configuration: {cfg_file}")
    print("=" * 75)

    cfg_dict = load_yaml_config(cfg_file)
    if cfg_dict and "training" in cfg_dict:
        t_cfg = cfg_dict["training"]
        print(f"  Training Optimizer: {t_cfg.get('optimizer', 'AdamW')} (lr={t_cfg.get('learning_rate', 3e-4)}, weight_decay={t_cfg.get('weight_decay', 1e-4)})")
        print(f"  Batch Size: {t_cfg.get('batch_size', 128)} (eval: {t_cfg.get('eval_batch_size', 256)}) | Epochs: {t_cfg.get('epochs', 30)}")
    if cfg_dict and "model" in cfg_dict:
        m_cfg = cfg_dict["model"]
        print(f"  Model Architecture: {m_cfg.get('architecture', 'a34_generic')} ({m_cfg.get('parameters', 970817)} params)")
        print(f"  Grid Shapes: SGCC {m_cfg.get('sgcc_input_shape', [147, 7])} | ECL {m_cfg.get('ecl_input_shape', [7, 48])}")

    py_exec = sys.executable

    # 0. Prepare & Verify Chronological Dataset Alignment
    prep_script = THIS / "prepare_sgcc_dataset.py"
    if prep_script.exists():
        run_step(
            "0. Chronological Dataset Alignment & Date Audit",
            [py_exec, str(prep_script)],
            cwd=THIS,
        )

    # 1. Verify Partition Manifests (Customer & Window Disjointness)
    verify_script = THIS / "verify_partition_manifests.py"
    if verify_script.exists():
        run_step(
            "1. Partition manifest disjointness audit",
            [py_exec, str(verify_script)],
            cwd=THIS,
        )

    # 2. Conformal audit analyses on the shared protocol (Table 6, Fig. 5, Supp. Fig. S4, Supp. Table S6)
    for step, script in [
        ("2a. Conformal decision value: all six Table 6 columns + Fig. 5b", "run_decision_value.py"),
        ("2b. Empirical-quantile vs conformal comparator (Fig. 5a)", "run_empirical_comparator.py"),
        ("2c. SGCC trust layer: per-stratum conformal lift (Supp. Fig. S4)", "run_trust_layer.py"),
        ("2d. ECL conformal lift gap (Supp. Table S6)", "run_ecl_trust_analysis.py"),
    ]:
        run_step(step, [py_exec, str(THIS / script)], cwd=THIS)

    # 3. Supplementary tables
    run_step("3a. Per-group PR-AUC across detectors (Supp. Table S4)", [py_exec, str(THIS / "compute_table_b.py")], cwd=THIS)
    run_step("3b. Prevalence-matched per-group contrasts (Supp. Table S3)", [py_exec, str(THIS / "compute_table_c.py")], cwd=THIS)
    run_step("3b'. Matched contrasts and Friedman test (Supp. Table S3)", [py_exec, str(THIS / "table_c_contrasts.py")], cwd=THIS)
    run_step("3c. ECL per-attack recall at 1%/5% FPR (Supp. Table S7 note)", [py_exec, str(THIS / "compute_table_a.py")], cwd=THIS)
    run_step("3d. ECL metric suite (Supp. Table S7)", [py_exec, str(THIS / "make_ecl_board.py")], cwd=THIS)

    # 4. Main-text Tables 2-5, quoted statistical tests, conformal eligibility ledger
    run_step("4. Tables 2-5, statistical tests, eligibility ledger (r2_audit/table*.csv)",
             [py_exec, str(THIS / "compute_r2_revisions.py")], cwd=THIS)

    # 5. Figures
    if not args.skip_figures:
        run_step("5a. Main-text figures (Figures 2, 3, 4, 5)", [py_exec, str(THIS / "make_manuscript_figures.py")], cwd=THIS)
        run_step("5b. Supplementary conformal figure (Supp. Fig. S4)", [py_exec, str(THIS / "make_trust_figure.py")], cwd=THIS)
        run_step("5c. Supplementary Figs. S1, S2 and exploratory figures", [py_exec, str(THIS / "make_figures.py")], cwd=THIS)
        run_step("5d. Supplementary Fig. S3 (illustrative mechanism)", [py_exec, str(THIS / "make_mechanism_figure.py")], cwd=THIS)

    print("\n" + "#" * 75)
    print("# ALL TABLES AND FIGURES REGENERATED FROM CACHED PREDICTIONS (TIER 1)")
    print("#" * 75)
    print("Generated artifacts (results/phase_n_retrained/):")
    print("  - Table 2:  r2_audit/table2_partition_ledger.csv (fold counts: *_fold_counts.csv)")
    print("  - Table 3:  r2_audit/table3_main_results.csv (+ table3_tests.csv)")
    print("  - Table 4:  r2_audit/table4_fpr_budgets.csv (+ table4_prospective.csv, note)")
    print("  - Table 5:  r2_audit/table5_ablation.csv")
    print("  - Table 6:  trust/decision_value_sgcc.csv (+ _resplits.csv, eligibility ledger)")
    print("  - Supp. Table S6: ecl_trust/RESULTS_ECL_TRUST.md")
    print("  - Supp. Tables S3/S4: tableC/tableC_contrasts.csv (+ _friedman.csv), tableB/tableB_agg.csv")
    print("  - Supp. Table S7: ecl_board.csv; per-attack note: tableA/tableA_agg.csv")
    print("  - Text: r2_audit/mask_only_baseline.csv, trust/fig4_ece_by_detector.csv")
    print("  - Figures:  figures/fig2-fig5 (main), fig6_trust_guarantee (Supp. S4), others")
    print("              (Figure 1 is the conceptual architecture schematic in TikZ)")
    print("#" * 75 + "\n")

if __name__ == "__main__":
    main()

