# Revision 3 Results (`results/r3_revision/`)

This directory contains the authoritative **Revision 3** outputs reported in the manuscript and supplementary information, extending evaluation to all five SGCC outer folds (`Folds 0--4`, `14,420` unique test consumers) and a strictly client-disjoint 5-fold `GroupKFold` ECL benchmark (`370` clients: `296 train / 26 early-stopping / 48 test`, `0` client overlap), alongside the controlled `C2`, `C5`, and `C7` sensitivity suites.

## Reproducing Revision 3 Artifacts (CPU-only GPU-Retrain-Free Suite, Tier 1)

Run from the repository root (reads cached predictions in `results/phase_n_retrained/` for Folds 0–2 and `results/r3_revision/` for Folds 3–4, ECL C3, and C2/C5/C7; requires zero GPU retraining, **provided raw public datasets `data/SGCC.csv` and `data/LD2011_2014.txt` are placed under `data/` per [`../../data/README.md`](../../data/README.md)** for consumer router magnitudes, weekday/weekend masks, and missingness rates):

```bash
python scripts/reproduce_all_r3.py
```

This executes `scripts/verify_partition_manifests.py`, `scripts/r3/compute_r3_ecl_tables.py`, and `scripts/r3/build_r3_5fold_tables_and_figures.py`, regenerating `r3_manuscript_verified_ledger.json`, `C3_ecl_client_disjoint/c3_tables_verified.json`, `tables/`, `figures/`, `fig_previews/`, and scrubbed execution logs in `logs/`.

*(Note: `tables/tab_strata.tex`, `tables/tab_matched.tex`, and `tables/tab_perstratum.tex` are LaTeX `\input{}` table fragments requiring manuscript-level macros `\revthree`, `\name`, `\prauc`, `\rocauc`, and `\ci` defined in the paper preamble.)*

## Directory Layout

| Path | Executing Script | Contents |
|---|---|---|
| `r3_manuscript_verified_ledger.json` | `scripts/r3/build_r3_5fold_tables_and_figures.py` | Single-source-of-truth verified 5-fold JSON ledger across SGCC, ECL, conformal audits, IPW/shift sensitivity, groupings, and constrained recalibration |
| `tables/` | `scripts/r3/build_r3_5fold_tables_and_figures.py` | Authoritative 5-fold flat CSV exports (`table4_5fold_prospective.csv`, `table6_5fold_conformal.csv`, `c4_pooled_5fold_flat.csv`, `c4_ipw_5fold.csv`, `c4_shift_5fold.csv`, `c6_alt_groupings_5fold.csv`, `c8_5fold_constrained_cal.csv`, `table5_ablation_5fold.csv`, `table_s13_tuning_5fold.csv`) and LaTeX tables (`tab_strata.tex`, `tab_matched.tex`, `tab_perstratum.tex`) |
| `figures/` & `fig_previews/` | `scripts/r3/build_r3_5fold_tables_and_figures.py` | Regenerated 5-fold vector PDF figures (`fig1_metric_inflation.pdf`, `fig2_performance_curves.pdf`, `fig3_ablation_bars.pdf`, `fig4_calibration_reliability.pdf`, `fig5_conformal_tradeoffs.pdf`, `fig6_trust_guarantee.pdf`, `fig_s6_r3_conformal_robustness.pdf`) and 220-dpi PNG previews |
| `C1_5fold_and_stats/` | `scripts/r3/run_c1_5fold.py` | SGCC Folds 3–4 trained cell predictions (`raw_folds34/`), 5-fold Table 3 metrics (`25` MAMRD + `15` per baseline), paired tests, and consumer-clustered bootstrap comparisons (3-fold vs. 5-fold; single-model and 3-seed ensemble) |
| `C2_missingness_sensitivity/` | `scripts/r3/run_c2_missingness_sensitivity.py` | Clean-cohort (`N = 40,258`) sensitivity and 3-way protocol-matched neural decomposition (`mask-only`, `linear-interp`, `mean-fill`, `combined MAMRD`) |
| `C3_ecl_client_disjoint/` | `scripts/r3/run_c3_ecl_client_disjoint.py`, `compute_r3_ecl_tables.py` | 5-fold client-disjoint `GroupKFold` ECL predictions (`cells/`, `90` models), partition counts (`c3_ecl_client_disjoint_partitions.csv`), per-attack `FDI1--FDI6` recall, paired contrasts, and `c3_tables_verified.json` (Table 2, Supp. Tables S6–S7) |
| `C5_preprocessing_sensitivity/` | `scripts/r3/run_c5_preprocessing_sensitivity.py` | 5-fold (`M = 15`) sensitivity across Laplacian boundary padding (`replicate`, `reflect`), normalization (`z-score`, global `log1p` min-max, global quantile), and `2016-09-18` reconstruction choices |
| `C7_baseline_hpo/` | `scripts/r3/run_c7_baseline_hpo.py` | 6-configuration `(lr, wd)` validation sweep on Fold 0 Seed 0 `D_es` for HORLN and Self-Attention (`run_c7_baseline_hpo.py`) and for MAMRD (`run_r3_c7_mamrd_hpo.py`), 15-cell tuned runs, WDCNN linear interpolation, and Finardi native quantile transform |
| `C9_ablation_folds34/` | `scripts/r3/run_r3_ablation_folds34.py` | Table 5 ablation arms (S0–S3, A34 no-missingness, A32) on Folds 3–4 x Seeds 0–2 (`36` cells), paired with the Folds 0–2 arms in `results/phase_n_retrained/bu/` |
| `logs/` | `scripts/reproduce_all_r3.py` | Tier-1 CPU step logs and `reproduce_all_r3.log` |
| `_3fold_diagnostics/` | `scripts/r3/run_c2_*.py`, `run_c4_*.py`, `run_c5_*.py`, `run_c6_*.py`, `run_c7_*.py`, `run_c8_*.py` | Earlier Folds 0–2 summaries of C2, C5 and C7 (`*_folds012/`, superseded by the 5-fold runs) and standalone Folds 0–2 diagnostic runs (`C4_conformal_sensitivity/`, `C6_alt_groupings/`, `C8_calibration_and_prospective/`, including the `BFGS` vs. box-constrained `L-BFGS-B` numerical gradient check in `C8`). Kept strictly for audit transparency; all 5-fold manuscript tables and figures use `r3_manuscript_verified_ledger.json` and `tables/*.csv` generated by `build_r3_5fold_tables_and_figures.py`. |

## Note on Console Logs & Verification Audits

Machine-specific absolute paths in `*/run.log`, `_3fold_diagnostics/*/run.log`, and `logs/*.log` are replaced with `.` (repository root), matching [`../README.md`](../README.md); all timestamps, epoch metrics, and statistical summaries are preserved verbatim.

- **ECL C3 Historical Log Note:** `C3_ecl_client_disjoint/run.log` is the unmodified Nov 26 GPU console log. Two baseline cell files (`cells/hybridattn_2_4.csv` and `cells/msdm_2_1.csv`) were overwritten by a concurrent write shortly after that log captured their in-memory summary; all released summary tables (`c3_ecl_client_disjoint_*.csv`, `c3_summary.json`, and `c3_tables_verified.json`) are deterministically computed from the 90 released `cells/*.csv` files using stored unrounded cell AP/ROC values.
- **Conformal BCa Seed Note:** `C1_5fold_and_stats/table6_5fold_conformal.csv` uses `cell_bca` (`seed=0`), whereas the authoritative Revision 3 5-fold tables (`r3_manuscript_verified_ledger.json` and `tables/table6_5fold_conformal.csv`) use `bca_ci` (`seed=12345`).
