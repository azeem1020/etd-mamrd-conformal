# Results

This directory stores the verified experimental outputs and console logs for **Revision 3**:

- [`r3_revision/`](r3_revision/README.md) holds the **authoritative Revision 3** 5-fold SGCC predictions (`Folds 3--4` in `C1_5fold_and_stats/raw_folds34/` paired with `Folds 0--2` in `phase_n_retrained/`), 5-fold client-disjoint `GroupKFold` ECL predictions (`C3_ecl_client_disjoint/`), `C2`/`C5`/`C7` sensitivity suites, the master 5-fold ledger (`r3_manuscript_verified_ledger.json`), flat CSV exports and LaTeX tables (`tables/`), and all regenerated vector PDF figures (`figures/`).
- `phase_n_retrained/` holds the Folds 0–2 SGCC cell predictions, controlled architectural ablation cells (`bu/`, `a32_sgcc.csv`, `a33_sgcc.csv`), and legacy Round-2 outputs read as base inputs for Folds 0–2.
- `_archive_phase_n_pre_retrain/` keeps superseded pre-retrain audit artifacts; no active script reads it.

## Reproducing These Files

See the root [`README.md`](../README.md):

- **Tier 1 — Revision 3 Authoritative Suite (`python scripts/reproduce_all_r3.py`, CPU-only, under 2 minutes)**: Verifies the 5-fold SGCC & client-disjoint ECL manifests, recomputes 5-fold client-disjoint ECL tables, and regenerates `results/r3_revision/r3_manuscript_verified_ledger.json`, `results/r3_revision/tables/`, and all 7 vector PDF figures from cached predictions without retraining.
- **Tier 1 — Legacy Round-2 Folds 0–2 Suite (`python scripts/reproduce_all_r2.py --config configs/default.yaml`, about 3.5 minutes on CPU)**: Regenerates legacy Folds 0–2 outputs in `results/phase_n_retrained/`.
- **Tier 2 — GPU Training Scripts**: `scripts/r3/run_c1_5fold.py`, `run_c2_missingness_sensitivity.py`, `run_c3_ecl_client_disjoint.py`, `run_c5_preprocessing_sensitivity.py`, and `run_c7_baseline_hpo.py` (for Revision 3 additions), paired with `scripts/run_master_retrain.py --force` for Folds 0–2 base models.

The Result Traceability Matrix in [`../README.md`](../README.md) maps every main and supplementary table and figure to its script and verified output file.

## Layout of `phase_n_retrained/` (Folds 0–2 Base Cache & Legacy Round-2 Outputs)

| Path | Contents |
|---|---|
| `a34_cell_<seed>_<fold>.csv`, `a34_manifest.csv` | MAMRD SGCC Folds 0–2 (`5 seeds x 3 folds = 15 cells`): per-cell metrics and held-out test predictions |
| `bu/`, `a32_sgcc.csv`, `a33_sgcc.csv` | Controlled Folds 0–2 architectural ablation suite (`M = 9` cells; Table 5, Figure 3a) |
| `baselines/` | Six deep baselines on SGCC Folds 0–2 (`3 seeds x 3 folds = 9 cells` per baseline) |
| `ecl_trust/`, `ecl_board.csv` | Superseded Folds 0–2 window-level ECL outputs (retained read-only; Revision 3 ECL uses 5-fold client-disjoint `results/r3_revision/C3_ecl_client_disjoint/`) |
| `trust/`, `r2_audit/`, `tableA/`, `tableB/`, `tableC/`, `figures/` | Legacy Round-2 Folds 0–2 post-processed outputs |
| `clean_cohort.json`, `dropped_ids.csv`, `keep_idx.npy`, `conflicting_flag_ids.csv` | Cohort audit (`scripts/clean_cohort.py`); `conflicting_flag_ids.csv` lists the `497` consumers in exact-duplicate clusters with conflicting labels (flagged; kept in the full `42,372`-consumer evaluation cohort) |
| `repro_finardi.json` | Output of `scripts/run_finardi_repro.py` |
| `logs/`, `master_retrain.log`, `master_retrain_r3.log` | Console logs of the Folds 0–2 retraining and Tier-1 Round-2 runs |

## Notes on the Logs

Absolute local paths in released logs are replaced by `.` (repository root), `python` (interpreter), and `<site-packages>`.

Apart from this, `phase_n_retrained/master_retrain.log` is the unedited log of the 9 November 2026 run. Its trust-layer section begins with a header line, `"Pre-reg: PLAN_TRUST_2026-06-02.md"`, printed by the script version used in that run, referring to an internal planning note not part of this repository (removed from the script in commit `982df7e`).
