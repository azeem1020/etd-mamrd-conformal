# b2 — Protocol-matched baselines (LOCKED 2026-05-31)

All baselines re-implemented FAITHFULLY (`phased/baselines.py`) and run under the
IDENTICAL honest protocol as a34 via `run_n_clean.py --backbone X` (same split,
per-user min-max preprocessing, metrics, per-phenotype). Architecture varied,
protocol held fixed (controlled comparison). PDFs in `~/research_papers/IEEE_Papers/`.
WDCNN ported from validated reference implementation (kernel
torch-port UNIT-TESTED bit-equal to numpy ref; reproduces Zheng ROC 0.794).

## SGCC (real theft, n=9 seeds0-2 x folds0-2, BCa 95% CI), --full cohort
| model | PR-AUC | ROC | F1 | MAP@100 | MAP@200 | Δ vs a34 (95% CI) |
|---|---|---|---|---|---|---|
| GRU-D (Che'18)        | 0.201 | 0.674 | 0.243 | 0.489 | 0.427 | -0.367 [-0.39,-0.35]* |
| HybridAttn (Finardi'20)| 0.207 | 0.719 | 0.250 | 0.405 | 0.356 | -0.361 [-0.39,-0.33]* |
| WDCNN (Zheng'18)      | 0.335 | 0.799 | 0.346 | 0.682 | 0.581 | -0.233 [-0.25,-0.21]* |
| MSDM (Wang'25)        | 0.465 | 0.844 | 0.442 | 0.831 | 0.748 | -0.103 [-0.12,-0.09]* |
| HORLN (Zhu'23)        | 0.500 | 0.862 | 0.480 | 0.845 | 0.764 | -0.068 [-0.09,-0.05]* |
| **a34 / NaME (ours, n=15)** | **0.567** | **0.883** | **0.528** | **0.889** | **0.817** | — |

a34 beats EVERY baseline, all Δ significant (CI excludes 0). HORLN strongest competitor,
MSDM next (= the honest 2025 methods, matches lit review). Every model: MAP@100/ROC high
but full PR-AUC low = C3 metric-inflation, universal not architecture-specific.

## ECL (synthetic FDI, 30-min, mean AP over 3 seeds)
| model | AP |
|---|---|
| GRU-D       | 0.561 |
| WDCNN       | 0.836 |
| HybridAttn  | 0.847 |
| HORLN       | 0.852 |
| MSDM        | 0.870 (locked) |
| **a34 (ours)** | **0.880 (locked)** |
a34 best on BOTH datasets.

## Honesty caveat (state in paper)
HybridAttn (0.207) + GRU-D (0.201) underperform their lit ranges. HybridAttn's key
contribution was QUANTILE normalization, deliberately replaced with shared min-max for
protocol-matching → disadvantages it (correct controlled-comparison choice, state explicitly).
HORLN + MSDM (no bespoke preprocessing) land in reported ranges = internal-validity check.

## Files
- Models: `code/phase_d/phased/baselines.py` (build_arm: wdcnn/grud/msdm_net/horln/hybridattn)
- SGCC cells: `results/phase_n/baselines/<arm>_<s>_<f>.csv`
- ECL logs: `results/phase_n/ecl_baselines/<arm>.log` + parsed `ecl_ap.csv`
- Finalizer: `results/phase_n/finalize_b2.py` (regenerates this table)
- Figures: `results/phase_n/figures/fig{1..5}_*.pdf` via `code/phase_d/make_figures.py`
