# Results

`phase_n_retrained/` holds every result reported in the manuscript (Revision 3). All models were retrained on the chronologically ordered SGCC data and on ECL.

`_archive_phase_n_pre_retrain/` keeps the superseded Revision 2 records for audit only; no script reads it.

## Reproducing these files

See the root [`README.md`](../README.md):

- **Tier 1** (`python scripts/reproduce_all_r2.py --config configs/default.yaml`, about 3.5 minutes on a CPU) regenerates every table and every data-driven figure from the cached predictions in this directory.
- **Tier 2** (`python scripts/run_master_retrain.py`, about 104 minutes on one NVIDIA GeForce RTX 5070 Ti Laptop GPU) retrains all 177 models and then runs Tier 1.

The traceability matrix in the root README maps each table and figure to its script and output file.

## Layout of `phase_n_retrained/`

| Path | Contents |
|---|---|
| `a34_cell_<seed>_<fold>.csv`, `a34_manifest.csv` | MAMRD, 15 seed–fold cells: per-cell metrics and the test-fold labels and scores |
| `bu/`, `a32_sgcc.csv`, `a33_sgcc.csv` | Ablation variants (Table 5, Figure 3) |
| `baselines/` | The six deep baselines on SGCC |
| `ecl_trust/`, `ecl_board.csv` | ECL benchmark (MAMRD and five baselines; DKGT-Net is SGCC-only) and its conformal analysis |
| `trust/` | Conformal and calibration analyses (Table 6, Figures 4–5, Supplementary Figure S4) |
| `r2_audit/` | Tables 2–5, partition ledgers and manifests, conformal eligibility ledger |
| `tableA/`, `tableB/`, `tableC/` | Supplementary tables |
| `figures/` | Data-driven figures |
| `clean_cohort.json`, `dropped_ids.csv`, `keep_idx.npy` | Outputs of the cohort audit (`scripts/clean_cohort.py`) |
| `repro_finardi.json` | Output of `scripts/run_finardi_repro.py` |
| `logs/` | Console log of each Tier 1 step, and of the three Supplementary Table S5 hardware runs (`make_compute_table_run*.log`, GPU in `make_compute_table_gpu.txt`) |
| `master_retrain.log` | Log of the Tier 2 run of 9 November 2026 |

## Notes on the logs

Absolute local paths in the logs are replaced by `.` (repository root), `python` (interpreter) and `<site-packages>`.

Apart from this, `master_retrain.log` is the unedited log of the 9 November 2026 run. Its trust-layer section begins with a header line, "Pre-reg: PLAN_TRUST_2026-06-02.md", printed by the script version used in that run. That line refers to an internal planning note that is not part of this repository. It was removed from the script in commit `982df7e`; the computation is unchanged.
