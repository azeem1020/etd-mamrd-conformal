#!/usr/bin/env python3
"""Tier-1 CPU Reproduction Runner for Revision 3 Artifacts (No GPU Retraining).

Regenerates all authoritative 5-fold Revision 3 statistical ledgers, flat CSV tables,
LaTeX supplementary tables (Tables S2-S4), and vector PDF figures from the cached
per-cell predictions in results/phase_n_retrained/ (Folds 0-2) and
results/r3_revision/ (SGCC Folds 3-4, 5-fold client-disjoint ECL, C2/C5/C7 suites).

Steps:
  1. scripts/verify_partition_manifests.py
  2. scripts/r3/compute_r3_ecl_tables.py
  3. scripts/r3/build_r3_5fold_tables_and_figures.py

Usage:
    python scripts/reproduce_all_r3.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO_ROOT = THIS.parent
for p in (str(THIS), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import ECL_PATH, SGCC_PATH, scrub_paths  # noqa: E402


R3_RES = REPO_ROOT / "results" / "r3_revision"
LOG_DIR = R3_RES / "logs"


def run_step(step_name: str, script_path: Path, cwd: Path, combined_lines: list[str]) -> None:
    """Execute one deterministic CPU step, streaming scrubbed output to stdout & log."""
    header = "\n" + "=" * 75 + f"\n>> RUNNING STEP: {step_name}\n   Script: {scrub_paths(str(script_path))}\n" + "=" * 75
    print(header, flush=True)
    combined_lines.append(header)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    step_log = LOG_DIR / f"{script_path.stem}.log"
    env = dict(os.environ, SOURCE_DATE_EPOCH=os.environ.get("SOURCE_DATE_EPOCH", "1767225600"))

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    step_lines = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        clean_line = scrub_paths(raw_line)
        sys.stdout.write(clean_line)
        sys.stdout.flush()
        step_lines.append(clean_line)
        combined_lines.append(clean_line.rstrip("\n"))

    rc = proc.wait()
    step_log.write_text("".join(step_lines), encoding="utf-8")
    if rc != 0:
        err = f"\n[ERROR] Step '{step_name}' failed with exit code {rc}"
        print(err, flush=True)
        combined_lines.append(err)
        (LOG_DIR / "reproduce_all_r3.log").write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
        sys.exit(rc)

    ok_msg = f"[PASS] {step_name} completed."
    print(ok_msg, flush=True)
    combined_lines.append(ok_msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier-1 CPU reproduction suite for MAMRD Revision 3 artifacts."
    )
    parser.parse_args()

    missing = [
        (name, p, env)
        for name, p, env in (
            ("SGCC raw CSV", SGCC_PATH, "MAMRD_SGCC"),
            ("ECL raw TXT", ECL_PATH, "MAMRD_ECL"),
        )
        if not p.exists()
    ]
    if missing:
        for name, p, env in missing:
            print(
                f"[ERROR] Missing required dataset ({name}): {scrub_paths(str(p))}\n"
                f"        Place the public corpus under data/ per data/README.md or set {env}.",
                file=sys.stderr,
                flush=True,
            )
        sys.exit(1)

    t0 = time.time()
    combined_lines = [
        "=" * 75,
        "MAMRD REVISION 3 TIER-1 CPU REPRODUCTION SUITE",
        "Repository Root: .",
        "=" * 75,
    ]
    for line in combined_lines:
        print(line, flush=True)

    run_step(
        "1. Verify 5-fold SGCC and 5-fold client-disjoint ECL manifests",
        THIS / "verify_partition_manifests.py",
        cwd=REPO_ROOT,
        combined_lines=combined_lines,
    )
    run_step(
        "2. Recompute 5-fold client-disjoint ECL tables (Table 2, Supp. Tables S6-S7)",
        THIS / "r3" / "compute_r3_ecl_tables.py",
        cwd=REPO_ROOT,
        combined_lines=combined_lines,
    )
    run_step(
        "3. Rebuild 5-fold verified ledger, flat CSVs, Tables S2-S4, and 7 vector figures",
        THIS / "r3" / "build_r3_5fold_tables_and_figures.py",
        cwd=REPO_ROOT,
        combined_lines=combined_lines,
    )

    elapsed = time.time() - t0
    footer = [
        "\n" + "#" * 75,
        f"# ALL REVISION 3 TABLES AND FIGURES REGENERATED ON CPU IN {elapsed:.1f}s",
        "# Outputs (results/r3_revision/):",
        "#   - r3_manuscript_verified_ledger.json",
        "#   - C3_ecl_client_disjoint/c3_tables_verified.json",
        "#   - tables/{tab_strata,tab_matched,tab_perstratum}.tex + 7 CSV tables",
        "#   - figures/fig{1,2,3,4,5,6}_*.pdf + fig_s6_r3_conformal_robustness.pdf",
        "#" * 75,
    ]
    for line in footer:
        print(line, flush=True)
        combined_lines.append(line)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "reproduce_all_r3.log").write_text("\n".join(combined_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
