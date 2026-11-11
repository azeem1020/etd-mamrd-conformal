#!/usr/bin/env python3
"""Unified Master Reproduction Script for Round-2 Revisions.

Command-line usage:
    python scripts/reproduce_all_r2.py --config configs/default.yaml

Tier-1 reproduction from the cached prediction-level records (no retraining):
  1. verify customer/window disjointness of the partition manifests (Table 2);
  2. run the conformal audit analyses on the shared protocol (conformal_common.py):
     run_decision_value (Table 6, Fig. 5b), run_empirical_comparator (Fig. 5a),
     run_trust_layer (Fig. 4b calibration CSV, Supp. Fig. S4), run_ecl_trust_analysis (Supp. Table S6);
  3. recompute Tables 3 and 4 and the ECL per-attack recall (compute_table_b,
     compute_table_c, compute_table_a);
  4. recompute partition counts, bootstrap/ablation ledgers and the conformal
     eligibility ledger (compute_r2_revisions: Tables 2 and 5);
  5. regenerate all main and supplementary figures.
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
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
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

    # 1. Verify Partition Manifests (Customer & Window Disjointness, Table 2)
    verify_script = THIS / "verify_partition_manifests.py"
    if verify_script.exists():
        run_step(
            "1. Partition Manifest & Disjointness Audit (Table 2)",
            [py_exec, str(verify_script)],
            cwd=THIS,
        )

    # 2. Conformal audit analyses on the shared protocol (Table 6, Fig. 5, Supp. Fig. S4, Supp. Table S6)
    for step, script in [
        ("2a. Conformal decision value: all six Table 6 columns + Fig. 5b", "run_decision_value.py"),
        ("2b. Empirical-quantile vs conformal comparator (Fig. 5a)", "run_empirical_comparator.py"),
        ("2c. SGCC trust layer: calibration, per-stratum conformal lift (Fig. 4b, Supp. Fig. S4)", "run_trust_layer.py"),
        ("2d. ECL conformal lift gap (Supp. Table S6)", "run_ecl_trust_analysis.py"),
    ]:
        run_step(step, [py_exec, str(THIS / script)], cwd=THIS)

    # 3. Main benchmark (Table 3) and fixed-FPR budgets (Table 4)
    run_step("3a. Protocol-matched benchmark (Table 3)", [py_exec, str(THIS / "compute_table_b.py")], cwd=THIS)
    run_step("3b. Fixed-FPR inspection budgets (Table 4)", [py_exec, str(THIS / "compute_table_c.py")], cwd=THIS)
    run_step("3c. ECL per-attack recall at 1%/5% FPR (Supp. Table S7 note)", [py_exec, str(THIS / "compute_table_a.py")], cwd=THIS)

    # 4. Partition counts, bootstrap and ablation ledgers, conformal eligibility ledger (Tables 2, 5)
    run_step("4. Partition counts, bootstrap/ablation ledgers, eligibility ledger (Tables 2, 5)",
             [py_exec, str(THIS / "compute_r2_revisions.py")], cwd=THIS)

    # 5. Figures
    if not args.skip_figures:
        run_step("5a. Main-text figures (Figures 2, 3, 4, 5)", [py_exec, str(THIS / "make_manuscript_figures.py")], cwd=THIS)
        run_step("5b. Supplementary conformal figure (Supp. Fig. S4)", [py_exec, str(THIS / "make_trust_figure.py")], cwd=THIS)
        run_step("5c. Other supplementary figures", [py_exec, str(THIS / "make_figures.py")], cwd=THIS)

    print("\n" + "#" * 75)
    print("# ALL TABLES AND FIGURES REGENERATED FROM CACHED PREDICTIONS (TIER 1)")
    print("#" * 75)
    print("Generated artifacts (results/phase_n_retrained/):")
    print("  - Table 2:  r2_audit/sgcc_fold_counts.csv, r2_audit/ecl_fold_counts.csv")
    print("  - Table 3:  tableB/tableB_agg.csv            - Table 4: tableC/tableC_agg.csv")
    print("  - Table 5:  r2_audit/table_ablation_provenance.csv")
    print("  - Table 6:  trust/decision_value_sgcc.csv (+ _resplits.csv, eligibility ledger)")
    print("  - Supp. Table S6: ecl_trust/RESULTS_ECL_TRUST.md")
    print("  - Supp. Table S7 per-attack note: tableA/tableA_agg.csv")
    print("  - Figures:  figures/fig2-fig5 (main), fig6_trust_guarantee (Supp. S4), others")
    print("              (Figure 1 is the conceptual architecture schematic in TikZ)")
    print("#" * 75 + "\n")

if __name__ == "__main__":
    main()

