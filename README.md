# MAMRD: Missingness-Aware Multi-Representation Detector

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23344738.svg)](https://doi.org/10.5281/zenodo.23344738)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Release: v4.0-R3](https://img.shields.io/badge/Release-v4.0--R3-blue.svg)](https://github.com/azeem1020/etd-mamrd-conformal/releases/tag/v4.0-R3)

Official code repository and experiment replication suite for:

> **Trustworthy electricity theft detection integrating informative missingness learning and certified conformal audits**  
> *Hafiz Muhammad Azeem Akram, Siamak Talatahari, and Adil Jhangeer*  
> Under review at *Scientific Reports* (Nature Portfolio).

---

## 1. Overview & Key Contributions

Electricity theft detection (ETD) models frequently report over-optimistic performance due to data leakage and evaluation on artificially balanced test sets. This repository implements:

1. **Native-Prevalence 5-Fold & Client-Disjoint Evaluation Protocol**:
   - **SGCC (5-fold consumer-disjoint)**: Full `42,372`-consumer cohort (`8.53%` theft) evaluated across all five outer `KFold` folds (`Folds 0--4`), yielding `14,420` disjoint held-out test consumers (`1,235` thefts, `76` Flat thefts) with per-consumer normalization and verified zero overlap both within and across test folds (`partition_manifests/sgcc_partition_manifest.csv`). Paired single-model MAMRD reaches `0.608` PR-AUC (`[0.589, 0.628]`; `25`-cell 5-seed `0.610 [0.596, 0.625]`) vs. `0.574 [0.548, 0.601]` for HORLN (`+0.034 [+0.025, +0.045]`, `15/15` cell wins, descriptive Wilcoxon $p = 6.10 \times 10^{-5}$; consumer bootstrap $p < 0.001$ single-model mean, one-sided $p_{\text{boot}} = 0.156$ 3-seed ensemble).
   - **ECL (5-fold client-disjoint `GroupKFold`)**: `370` industrial/commercial profiles grouped strictly by client ID (`296 train / 26 early-stopping / 48 test` clients per fold; `0` client overlap across splits and folds; `10.0%` injected FDI prevalence across `FDI1--FDI6`). Across `90` trained cells (`6` detectors $\times$ `3` seeds $\times$ `5` folds), MAMRD achieves `0.835 [0.801, 0.853]` PR-AUC, `0.911` ROC-AUC, `80.9%` / `84.0%` retrospective (`81.0%` / `84.0%` prospective) recall at `1%` / `5%` FPR, and `98.0%--100.0%` recall on `FDI2--FDI6`.
2. **Missingness-Aware Multi-Representation Detector (MAMRD)**: A compact `0.97M`-trainable-parameter (`975,076` total parameters on SGCC: `974,833` trainable + `243` frozen Laplacian stencil weights) architecture (`a34_generic`) fusing five complementary consumption and missingness representations under binary cross-entropy:
   - Front end: Context-Conditioned Missing-Value Imputation (`CC-MVI` in the manuscript; `CC-NaME` / `cc_name` in the code), which fills missing entries with a learned scalar conditioned on each consumer's observed profile statistics
   - Branch 1: Bidirectional Mamba2 selective state-space sequence backbone ($\mathbb{R}^{90}$)
   - Branch 2: Value 2-D multi-scale discrete Laplacian CNN ($\mathbb{R}^{60}$)
   - Branch 3: Missingness-mask 2-D multi-scale Laplacian CNN ($\mathbb{R}^{30}$)
   - Branch 4: Fixed causal multi-scale lagged differencing 1-D CNN ($s \in \{1, 7, 30\}$, $\mathbb{R}^{32}$)
   - Branch 5: Multi-scale inter- and intra-period dilated 1-D CNN (IIPC, $\mathbb{R}^{32}$)
3. **Group-Conditional (Mondrian) Conformal Audit Layer & Constrained Recalibration**:
   - Post-hoc class-conditional split-conformal framework using $k_\phi = \lceil(n_\phi + 1)(1 - \alpha)\rceil$ across consumption-level strata ($\phi \in \{\text{Heavy, Light, Flat}\}$) at $1 - \alpha = 80\%$. Across all `500` MAMRD resplits (`25` cells $\times$ `20` repeats) over all 5 folds, conditional auditing lowers inspection workload from `47.2%` to `27.6%`, delivering a `35.5%` mean per-split workload reduction (`95%` BCa `[32.0%, 38.8%]`; matched `15`-cell `47.4%` $\to$ `28.0%`, `34.6% [29.8%, 38.8%]`) relative to a global minimax threshold, achieving `40.5%` Heavy hit rate (`3.39x` lift) while lowering Flat target-miss rate from `55.0%` (unadjusted empirical quantile) to `29.0%`.
   - Constrained positive-slope affine logit recalibration ($w = \exp(\eta) > 0$, slope $L_2$ penalty $w^2/(2C)$ with $C = 10^4$, solved via `BFGS` and cross-checked with box-constrained `L-BFGS-B`) across `115` 5-fold fits preserves ranking exactly ($\min \rho_s = 1.0$, $\max|\Delta\text{AP}| = 0.0$, $\hat{w}_{\min} = 0.385$) and reduces MAMRD ECE from `0.111` to `0.017` (`0.109` $\to$ `0.017` matched).

---

## 2. Repository Structure

```
etd-mamrd-conformal/
├── README.md                              # Documentation and reproduction guide
├── LICENSE                                # MIT License
├── CITATION.cff                           # Citation metadata (v4.0.0-R3)
├── requirements.txt / environment.yml     # Python environment specifications
├── pytest.ini                             # Test runner configuration (scripts/tests)
├── .gitignore                             # Git exclusions (releases logs & R3 manifests)
│
├── configs/                               # Benchmark YAML configurations
│   ├── default.yaml                       #   5-fold SGCC benchmark & Mondrian protocol
│   └── ecl_benchmark.yaml                 #   5-fold client-disjoint GroupKFold ECL config
│
├── partition_manifests/                   # 5-fold split manifests & standalone verifier
│   ├── sgcc_partition_manifest.csv        #   211,860 rows (Folds 0-4, 0 overlap)
│   ├── sgcc_fold_counts.csv               #   Per-fold SGCC counts (14,420 unique test)
│   ├── ecl_partition_manifest.csv         #   125,000 rows (Folds 0-4 client-disjoint)
│   ├── ecl_fold_counts.csv                #   Per-fold ECL counts (296/26/48 clients)
│   ├── verify_disjointness.py             #   Standalone 5-fold disjointness verifier
│   └── archive_r2/                        #   Archived Round-2 (3-fold/window) manifests
│
├── scripts/
│   ├── phased/                            # MAMRD core neural architecture & baselines
│   ├── r3/                                # Revision 3 scripts (C1-C8, ECL & 5-fold builder)
│   │   ├── build_r3_partition_manifests.py    # 5-fold SGCC & client-disjoint ECL manifests
│   │   ├── run_c1_5fold.py                    # SGCC Folds 3-4 training + 5-fold stats
│   │   ├── run_c2_missingness_sensitivity.py  # Clean cohort + 3-way neural decomposition
│   │   ├── run_c3_ecl_client_disjoint.py      # 5-fold client-disjoint GroupKFold ECL
│   │   ├── compute_r3_ecl_tables.py           # Verified ECL Table 2 & Supp. Tables S6-S7
│   │   ├── run_c5_preprocessing_sensitivity.py # Padding, normalization & Sep-18 sweeps
│   │   ├── run_c7_baseline_hpo.py             # 6-config HPO, WDCNN-interp, Finardi-QT
│   │   ├── build_r3_5fold_tables_and_figures.py # Authoritative 5-fold ledger, CSVs & PDFs
│   │   ├── run_c4_conformal_sensitivity.py    # Standalone Folds 0-2 C4 diagnostic
│   │   ├── run_c6_alt_groupings.py            # Standalone Folds 0-2 C6 diagnostic
│   │   └── run_c8_calibration_prospective.py  # Standalone Folds 0-2 C8 + L-BFGS-B check
│   ├── reproduce_all_r3.py                # Tier 1 (R3): regenerates all 5-fold ledger/CSVs/PDFs on CPU
│   ├── reproduce_all_r2.py                # Tier 1 (R2 legacy): Folds 0-2 pipeline
│   ├── verify_partition_manifests.py      # 5-fold manifest disjointness verifier
│   ├── run_master_retrain.py              # Tier 2: Folds 0-2 base retraining runner
│   ├── compute_consumer_bootstrap.py      # Customer-clustered bootstrap CI module
│   ├── conformal_common.py                # Shared Mondrian conformal protocol
│   ├── repro_config.py                    # Centralized path & log scrubbing config
│   └── tests/                             # Unit tests (pytest)
│
├── zkmix_lab/                             # Core data loaders, training loops, metrics
├── msdm/                                  # FDI attack simulation (fdi1..fdi6)
├── sct_etd/calibration/                   # Conformal calibration helpers
│
├── results/
│   ├── r3_revision/                       # Authoritative Revision 3 outputs (see README inside)
│   │   ├── r3_manuscript_verified_ledger.json # Master 5-fold single-source-of-truth JSON
│   │   ├── tables/                        # 7 5-fold CSV exports + LaTeX tab_{strata,matched,perstratum}.tex
│   │   ├── figures/ & fig_previews/       # Regenerated vector PDFs & 220-dpi PNG previews
│   │   ├── C1_5fold_and_stats/            # Folds 3-4 raw CSVs, Table 3, consumer bootstraps
│   │   ├── C2_missingness_sensitivity/    # Clean cohort + 3-way neural decomposition
│   │   ├── C3_ecl_client_disjoint/        # 5-fold client-disjoint ECL cells & verified tables
│   │   ├── C5_preprocessing_sensitivity/  # Padding, normalization & Sep-18 sweeps
│   │   ├── C7_baseline_hpo/               # 6-config validation grid, tuned baselines, WDCNN/Finardi
│   │   ├── logs/                          # Tier-1 reproduction logs (reproduce_all_r3.log)
│   │   └── _3fold_diagnostics/            # Early Folds 0-2 standalone C4/C6/C8 diagnostics
│   ├── phase_n_retrained/                 # Folds 0-2 SGCC base predictions & S0-S4 ablations
│   └── _archive_phase_n_pre_retrain/      # Archived pre-retrain audit logs
│
└── data/                                  # Data documentation (raw datasets not shipped)
    └── README.md                          # Instructions to obtain public SGCC & ECL corpora
```

### Baseline detectors

`scripts/phased/baselines.py` contains our PyTorch re-implementations of six published detectors adapted to the `147 x 7` weekly input grid. All seven models, MAMRD included, are trained and evaluated under one common protocol: consumer-disjoint partitions, per-consumer min-max preprocessing, AdamW ($\text{lr}=3\times 10^{-4}$, $\text{wd}=10^{-4}$, batch size `128`, `30` epochs), class-balanced mini-batches, checkpoint selection by validation PR-AUC on $\mathcal{D}_{\text{es}}$, and native-prevalence test sets. Supplementary Table S8 documents all adaptation details; `scripts/r3/run_c7_baseline_hpo.py` additionally evaluates a 6-configuration `(lr, wd)` validation sweep on $\mathcal{D}_{\text{es}}$ for HORLN and Self-Attention, native neighbor interpolation for WDCNN (`0.334` AP vs. `0.352` zero-fill on matched Folds 0–2, `M=9`), and native train-fit 10-quantile normalization for Self-Attention (`0.548` AP vs. `0.522` common preprocessing on matched Folds 0–2, `M=9`).

| Detector | Paper | Core Formulation | Re-implementation Notes |
|---|---|---|---|
| WDCNN | Zheng et al., IEEE TII 2018 | Wide 1-D path + 2-D CNN path (`15` filters, stride-3, `3x3` conv/pool, `64`-unit fusion) | Kernel preprocessing matches authors' released code; `run_c7_baseline_hpo.py` also evaluates linear interpolation |
| HORLN | Zhu et al., IEEE TII 2023 | Two IIWC blocks (`32`, `16` ch.), high-order H x W group convs, 4-head self-dependency, `PReLU` | `147` weeks, value + mask inputs; `run_c7_baseline_hpo.py` evaluates 6 `(lr, wd)` settings |
| Self-Attention | Finardi et al., arXiv 2020 | Two 16-head parallel attention + dilated conv layers, `1x1` conv + LayerNorm + `1,024`-unit head | Common per-consumer scaling in Table 3; `run_c7_baseline_hpo.py` evaluates HPO and 10-quantile scaling |
| GRU-D | Che et al., Sci. Rep. 2018 | Trainable diagonal input decay and hidden-state decay fed by mask (`64` hidden units) | `147` weekly steps of `7` daily variables; per-consumer empirical mean |
| MSDM | Wang et al., IEEE TSG 2025 | Multi-scale differencing (`1, 2, 7, 28, 91, 182` days), adaptive quantization, two MSCBlocks (`48` ch.) | Unspecified details marked `[ASSUMPTION]` in `msdm/model.py`; common balanced sampler |
| DKGT-Net | Tian et al., IEEE TII 2026 | KAN branch (`3x` expansion) + two-layer GRU branch, tanh-sigmoid gating, ECA attention | Trained with single-task BCE without WGAN-GP augmentation; SGCC only |

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10+
- Linux (`x86_64`) recommended
- NVIDIA GPU with CUDA support for GPU training (Reference hardware: RTX 5070 / RTX 5070 Ti Laptop GPU), or standard multi-core CPU for unit tests, manifest checks, and Tier-1 figure/table regeneration.

### Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/azeem1020/etd-mamrd-conformal.git
cd etd-mamrd-conformal

# 2. Create conda environment
conda env create -f environment.yml
conda activate mamrd

# 3. (Optional for training MAMRD selective SSMs) Install CUDA kernels
pip install mamba-ssm causal-conv1d
```

### Quick Verification
Run the unit test suite (`pytest.ini`) and architecture smoke import:
```bash
pytest
PYTHONPATH=scripts:. python -c "from phased.models import build_arm; m = build_arm('a34_generic', weeks=7, days=7); print('MAMRD OK')"
```

---

## 4. Dataset Acquisition

Raw datasets are not shipped due to licensing (see [`data/README.md`](data/README.md)):
1. **SGCC**: State Grid Corporation of China benchmark (`42,372` consumers, `1,035` days), from https://github.com/henryRDlab/ElectricityTheftDetection -> place at `data/SGCC.csv`.
2. **ECL**: UCI `ElectricityLoadDiagrams20112014` (`370` industrial/commercial profiles), https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 -> place at `data/LD2011_2014.txt`.

| File | Bytes | SHA-256 |
| :--- | ---: | :--- |
| SGCC CSV as extracted (before `scripts/prepare_sgcc_dataset.py`) | 175,194,613 | `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7` |
| `data/SGCC.csv` after `scripts/prepare_sgcc_dataset.py` (chronological order, `2016-09-18` filled) | 175,529,148 | `b41e1dca7c40c53f62689a0f9dda3d2a252ebf0311b646e8a95917f032331180` |
| `data/LD2011_2014.txt` | 710,998,915 | `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b` |

Path overrides can be configured without editing any file:
```bash
export MAMRD_SGCC=/path/to/SGCC.csv
export MAMRD_ECL=/path/to/LD2011_2014.txt
export MAMRD_RESULTS=/path/to/results/phase_n_retrained
```

---

## 5. Running Experiments & Generating Paper Artifacts

### Tier 1 (Revision 3 Authoritative Suite): CPU Regeneration Without GPU Retraining (~1 min)

Regenerates all authoritative 5-fold statistical ledgers, 7 flat CSV tables, LaTeX Supplementary Tables S2–S4, and all 7 vector PDF figures from the released per-cell predictions in `results/phase_n_retrained/` (Folds 0–2) and `results/r3_revision/` (Folds 3–4, 5-fold client-disjoint ECL, and C2/C5/C7 sensitivity runs) **without GPU retraining** (requires placing `data/SGCC.csv` and `data/LD2011_2014.txt` per Section 4 to compute consumer router magnitudes, weekday/weekend contrasts, and missingness rates):

```bash
python scripts/reproduce_all_r3.py
```

| Step | Script | Output (`results/r3_revision/` or `partition_manifests/`) |
| :---: | :--- | :--- |
| 1 | `scripts/verify_partition_manifests.py` | Audits `211,860` SGCC rows (`5` folds, `14,420` unique test consumers, `0` overlap) & `125,000` ECL rows (`5` folds, `296/26/48` clients, `0` client overlap) |
| 2 | `scripts/r3/compute_r3_ecl_tables.py` | `C3_ecl_client_disjoint/c3_tables_verified.json` (Table 2 ECL counts, Supp. Tables S6–S7) |
| 3 | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `r3_manuscript_verified_ledger.json`, `tables/{tab_strata,tab_matched,tab_perstratum}.tex`, `tables/*.csv`, `figures/*.pdf`, `fig_previews/*.png` |

*(To regenerate the legacy Round-2 Folds 0–2 outputs in `results/phase_n_retrained/`, run `python scripts/reproduce_all_r2.py --config configs/default.yaml`.)*

---

### Tier 2: Training from Raw Datasets (NVIDIA GPU)

All released prediction CSVs and console logs are already committed. To retrain individual Revision 3 suites on GPU (or Folds 0–2 base models via `python scripts/run_master_retrain.py --force`):

```bash
# 0. Verify SGCC chronological order and 5-fold manifests
python scripts/prepare_sgcc_dataset.py
python scripts/r3/build_r3_partition_manifests.py
python scripts/verify_partition_manifests.py

# 1. Base Folds 0-2 SGCC models + controlled ablations (S0-S4, a32, a34_no_miss)
python scripts/run_master_retrain.py --force

# 2. Revision 3 GPU extensions (Folds 3-4, 5-fold client-disjoint ECL, C2, C5, C7)
python scripts/r3/run_c1_5fold.py                    # SGCC Folds 3-4 (10 MAMRD + 36 baseline cells) & 5-fold stats
python scripts/r3/run_c2_missingness_sensitivity.py  # Clean cohort (N=40,258) & 3-way neural decomposition
python scripts/r3/run_c3_ecl_client_disjoint.py      # 5-fold client-disjoint GroupKFold ECL (90 cells)
python scripts/r3/run_c5_preprocessing_sensitivity.py # Padding, normalization & 2016-09-18 variants
python scripts/r3/run_c7_baseline_hpo.py             # 6-config (lr, wd) HPO, WDCNN-interp, Finardi-QT

# 3. Tier-1 CPU regeneration of all verified ledgers, tables & vector figures
python scripts/reproduce_all_r3.py
```

---

### Result Traceability Matrix

| Manuscript Artifact | Scope & Key Protocol | Primary Script(s) | Verified Output File(s) |
| :--- | :--- | :--- | :--- |
| **Table 2** | 5-fold SGCC (`14,420` test consumers, `76` Flat thefts) & 5-fold client-disjoint ECL (`296/26/48` clients) | `scripts/r3/build_r3_partition_manifests.py`, `compute_r3_ecl_tables.py`, `build_r3_5fold_tables_and_figures.py` | `partition_manifests/*_fold_counts.csv`, `results/r3_revision/r3_manuscript_verified_ledger.json` |
| **Table 3** | 5-fold SGCC main comparison (`15` matched cells per detector + `25`-cell MAMRD); Friedman $\chi^2_6 = 89.26$, Wilcoxon $p = 6.10\times 10^{-5}$; consumer bootstrap ($B=1{,}000$) | `scripts/r3/run_c1_5fold.py` | `results/r3_revision/C1_5fold_and_stats/{table3_5fold_main_results.csv, consumer_bootstrap_3fold_vs_5fold.csv, c1_summary.json}` |
| **Table 4** | 5-fold retrospective & prospective fixed-FPR (`1%, 5%`) operating points on SGCC (`5` folds) & ECL client-disjoint (`5` folds) | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `results/r3_revision/tables/table4_5fold_prospective.csv` |
| **Table 5 & Fig. 3** | Controlled Folds 0–2 sequential & subtractive ablations ($M=9$, $\mathcal{S}_0\to\mathcal{S}_4$, $\mathcal{A}_{32}$, $\mathcal{A}_{34,\text{no-miss}}$) + C2 3-way neural decomposition | `scripts/compute_r2_revisions.py`, `scripts/r3/run_c2_missingness_sensitivity.py`, `build_r3_5fold_tables_and_figures.py` | `results/phase_n_retrained/r2_audit/table5_ablation.csv`, `results/r3_revision/C2_missingness_sensitivity/c2c_three_way_neural.csv`, `results/r3_revision/figures/fig3_ablation_bars.pdf` |
| **Table 6 & Fig. 5** | 5-fold SGCC Mondrian conformal audit ($1-\alpha=0.80$, `500` MAMRD / `300` matched resplits, infinite-quantile fallback when $k_\phi > n_\phi$) | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `results/r3_revision/tables/table6_5fold_conformal.csv`, `results/r3_revision/figures/fig5_conformal_tradeoffs.pdf` |
| **Fig. 2, Fig. 4, Supp. Figs. S2, S4** | 5-fold SGCC PR/ROC curves; 5-fold constrained recalibration ($w = \exp(\eta) > 0$, `BFGS`, $C=10^4$, `115` fits); metric inflation; Heavy−Flat hit-rate gap | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `results/r3_revision/figures/{fig2_performance_curves,fig4_calibration_reliability,fig1_metric_inflation,fig6_trust_guarantee}.pdf`, `tables/c8_5fold_constrained_cal.csv` |
| **Supp. Tables S2–S4** | 5-fold SGCC per-stratum detectability, prevalence-matched contrasts ($\chi^2_4$), cross-detector per-group AP | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `results/r3_revision/tables/{tab_strata,tab_matched,tab_perstratum}.tex` |
| **Supp. Table S5** | Parameters (`975,076` SGCC), RTX 5070 Ti inference latency, peak VRAM | `scripts/make_compute_table.py` | `results/phase_n_retrained/logs/make_compute_table*.log` |
| **Supp. Tables S6–S7** | 5-fold client-disjoint ECL conformal lift gap, overall metric suite, & `FDI1--FDI6` per-attack recall (`90` cells) | `scripts/r3/run_c3_ecl_client_disjoint.py`, `compute_r3_ecl_tables.py` | `results/r3_revision/C3_ecl_client_disjoint/{c3_tables_verified.json, c3_ecl_client_disjoint_per_attack.csv}` |
| **Supp. Table S9** | Consumer-level paired bootstrap (3-fold vs. 5-fold; single-model mean `+0.034`, 3-seed ensemble `+0.010`) | `scripts/r3/run_c1_5fold.py` | `results/r3_revision/C1_5fold_and_stats/consumer_bootstrap_3fold_vs_5fold.csv` |
| **Supp. Table S10** | Clean-cohort (`N = 40,258`, excluding `2,114` zero/NaN consumers) evaluation & retraining + 3-way neural decomposition | `scripts/r3/run_c2_missingness_sensitivity.py` | `results/r3_revision/C2_missingness_sensitivity/{c2a_cohort_exclusion_cached.csv, c2b_clean_cohort_retrain.csv, c2c_three_way_neural.csv}` |
| **Supp. Table S11 & Fig. S6(a)** | 5-fold Flat calibration sweep (`f_cal` $\in \{0.15..0.65\}$, $76$ pooled Flat thefts, $n_{\text{cal}} = 11.1..49.2$); Tibshirani IPW selective inspection (`e(x) = max(0.05, s^\gamma)`) & distribution-shift tests | `scripts/r3/build_r3_5fold_tables_and_figures.py` | `results/r3_revision/tables/{c4_pooled_5fold_flat,c4_ipw_5fold,c4_shift_5fold}.csv`, `results/r3_revision/figures/fig_s6_r3_conformal_robustness.pdf` |
| **Supp. Table S12** | Preprocessing, Laplacian padding (`replicate`, `reflect`), global normalization, & `2016-09-18` sensitivity (`9` cells each) | `scripts/r3/run_c5_preprocessing_sensitivity.py` | `results/r3_revision/C5_preprocessing_sensitivity/{c5_sensitivity_summary.csv, c5_paired_contrasts.csv}` |
| **Supp. Table S13** | 6-config `(lr, wd)` HPO grid on Fold 0 Seed 0 $\mathcal{D}_{\text{es}}$, tuned `9`-cell HORLN/Self-Attn, WDCNN interpolation, Finardi 10-quantile | `scripts/r3/run_c7_baseline_hpo.py` | `results/r3_revision/C7_baseline_hpo/{c7_hpo_validation_grid,c7_tuned_baselines_summary,c7_wdcnn_interp_summary,c7_finardi_native_9cell}.csv` |
| **Supp. Table S14 & Fig. S6(b)** | 5-fold robustness across 4 grouping schemes (`35.5%`, `33.4%`, `24.2%`, `45.3%` workload reduction at $80\%$ target) & constrained calibration (`115` fits, `BFGS` + `L-BFGS-B`) | `scripts/r3/build_r3_5fold_tables_and_figures.py`, `scripts/r3/run_c8_calibration_prospective.py` | `results/r3_revision/tables/{c6_alt_groupings_5fold,c8_5fold_constrained_cal}.csv`, `results/r3_revision/_3fold_diagnostics/C8_calibration_and_prospective/` |

---

## 6. Conformal Audit Protocol & Revision 3 Verification Summary

All 5-fold SGCC and 5-fold ECL conformal audits share `scripts/conformal_common.py`:
- **Resplits**: Each test fold is split `20` times into stratified 50/50 calibration/evaluation halves (`500` splits for 25-cell MAMRD; `300` splits per 15-cell baseline). Every split is retained; zero splits are discarded.
- **Exact Finite-Sample Order Statistic**: $k_\phi = \lceil(n_\phi + 1)(1 - \alpha)\rceil$. Whenever $k_\phi > n_\phi$, $\hat{q}_\phi = +\infty$ and every consumer in stratum $\phi$ is inspected, entering the reported workload. Across all `2,300` 5-fold splits at $\alpha = 0.20$, Flat yields finite quantiles on `96.4%` of splits overall (`96.0%` / `4.0%` infinite fallback on 25-cell MAMRD; pooling out-of-fold calibration across the `76` unique Flat thefts increases $n_{\text{cal,Flat}}$ up to `49.2` at `f_cal = 0.65`, giving `100%` finite quantiles for $\alpha \in \{0.20, 0.10, 0.05\}$).
- **Confidence Intervals & Clustered Testing**: Cell-level `95%` BCa intervals ($B = 2{,}000$, seed `12345`) summarize variation across seed–fold cells; single-model and 3-seed ensemble consumer-clustered bootstraps ($B = 1{,}000$ draws/fold, summed over all `14,420` disjoint SGCC consumers) confirm single-model superiority ($+0.034 [+0.016, +0.052]$, $p < 0.001$) while transparently reporting that the 3-seed ensemble contrast is inconclusive ($+0.010 [-0.008, +0.029]$, $p_{\text{boot}} = 0.156$).

---

## 7. Citation

```bibtex
@article{akram2026mamrd,
  title     = {Trustworthy electricity theft detection integrating informative missingness learning and certified conformal audits},
  author    = {Akram, Hafiz Muhammad Azeem and Talatahari, Siamak and Jhangeer, Adil},
  journal   = {Scientific Reports},
  year      = {2026},
  doi       = {10.5281/zenodo.23344738},
  note      = {Under review (Revision 3, tag v4.0-R3)}
}
```

---

## 8. License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
