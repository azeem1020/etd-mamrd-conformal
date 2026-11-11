# MAMRD: Missingness-Aware Multi-Representation Detector

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23344738.svg)](https://doi.org/10.5281/zenodo.23344738)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Official code repository and experiment replication suite for:

> **Trustworthy electricity theft detection integrating informative missingness learning and certified conformal audits**  
> *Hafiz Muhammad Azeem Akram, Siamak Talatahari, and Adil Jhangeer*  
> Under review at *Scientific Reports* (Nature Portfolio).

---

## 1. Overview & Key Contributions

Electricity theft detection (ETD) models frequently report over-optimistic performance due to temporal data leakage and evaluation on artificially balanced test sets. This repository implements:

1. **Leakage-Free Native-Prevalence Evaluation Protocol**: Standardized evaluation on the State Grid Corporation of China (SGCC, 42,372 consumers) and ElectricityLoadDiagrams (ECL) benchmarks under true operational class imbalance (~8.5% theft) without retrospective or temporal leakage.
2. **Missingness-Aware Multi-Representation Detector (MAMRD)**: A compact 0.97M-parameter deep learning architecture (`a34_generic`) that fuses five complementary consumption and missingness representations under single-task binary cross-entropy:
   - Front end: Context-Conditioned Missing-Value Imputation (CC-MVI in the manuscript; `CC-NaME` / `ccname` in the code), which replaces each missing reading with a learned token conditioned on the consumer's context vector
   - Branch 1: Bidirectional Mamba2 selective state-space sequence backbone ($\mathbb{R}^{90}$)
   - Branch 2: Value 2-D multi-scale discrete Laplacian CNN ($\mathbb{R}^{60}$)
   - Branch 3: Missingness-mask 2-D multi-scale Laplacian CNN ($\mathbb{R}^{30}$)
   - Branch 4: Fixed causal multi-scale lagged differencing 1-D CNN ($s \in \{1, 7, 30\}$, $\mathbb{R}^{32}$)
   - Branch 5: Multi-scale inter- and intra-period dilated 1-D CNN (IIPC, $\mathbb{R}^{32}$)
3. **Group-Conditional (Mondrian) Conformal Audit Layer**: A post-hoc, distribution-free, class-conditional conformal prediction framework using the exact order statistic $k_\phi = \lceil(n_\phi + 1)(1 - \alpha)\rceil$ to target a common $1 - \alpha = 80\%$ marginal theft recall lower bound within each consumption-level group ($\phi \in \{\text{Heavy, Light, Flat}\}$) under within-group exchangeability, reducing the inspected share of consumers from 49.1% to 28.6% (a 35.5% relative workload reduction, 95% BCa interval over seed–fold cells [30.2%, 40.4%]) relative to a single global minimax threshold satisfying the coverage target across all groups. Every calibration/evaluation resplit is used; a group with too few calibration thefts for a finite order statistic is inspected in full and that cost is counted (see Section 6).

---

## 2. Repository Structure

```
etd-mamrd-conformal/
├── README.md                          # Documentation and reproduction guide
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Core Python dependencies
├── pytest.ini                         # Test runner configuration
├── .gitignore                         # Git tracking exclusions
│
├── scripts/                           # Experiment execution and plotting scripts
│   ├── phased/                        # MAMRD core neural architecture package
│   │   ├── models.py                  #   MAMRD (ZKMixV4 / arm 'a34_generic')
│   │   ├── cc_name.py                 #   Context-conditional native missingness embedding
│   │   ├── ms_kernel.py               #   Multi-scale dilated/Laplacian kernels
│   │   ├── baselines.py               #   Rival baselines (GRU-D, WDCNN, Self-Attn, HORLN, MSDM)
│   │   ├── archetype.py               #   Differencing / inter-/intra-period branches
│   │   └── phenotype.py               #   Label-free consumption-level stratification
│   ├── reproduce_all_r2.py            # Master one-command R2 reproduction pipeline
│   ├── compute_r2_revisions.py        # Partition manifests, bootstrap CIs, and R2 audit ledgers
│   ├── compute_consumer_bootstrap.py  # Customer-level clustered bootstrap CIs (B=1,000)
│   ├── verify_partition_manifests.py  # Customer & window disjointness verifier
│   ├── run_empirical_comparator.py    # Group-quantile baseline vs conformal audit comparator
│   ├── run_decision_value.py          # Table 6 (all six columns) + Fig. 5b: conditional vs single-threshold audit
│   ├── conformal_common.py            # Shared conformal protocol (resplits, Mondrian threshold, estimators)
│   ├── make_manuscript_figures.py     # Main-text vector figure generator (Figs. 2, 3, 4, 5)
│   ├── generate_high_contrast_fig2.py # Legacy helper (not used; Fig. 2 is made by make_manuscript_figures.py)
│   ├── make_trust_figure.py           # Supplementary conformal figure generator (Fig. S4)
│   ├── repro_config.py                # Centralized path configuration
│   ├── run_n_clean.py                 # Main benchmark training & evaluation runner
│   ├── run_trust_layer.py             # Conformal audit policy evaluation
│   ├── run_ecl_trust.py               # ECL benchmark training (all 6 detectors, 3 folds x 3 seeds)
│   ├── run_ecl_trust_analysis.py      # ECL conformal audit (Supp. Table S6)
│   └── make_figures.py                # Supplementary figure generators
│
├── zkmix_lab/                         # Core library: data loaders, training loops, metrics
│   ├── data.py                        #   Customer-disjoint fold partitioning
│   ├── training.py                    #   WeeklyGridDataset, training loops, seed discipline
│   └── metrics.py                     #   PR-AUC, ROC-AUC, MAP@K, FPR operating points
│
├── msdm/                              # FDI attack simulation & ECL injection
│   └── fdi.py                         #   Six-attack FDI family (fdi1..fdi6)
│
├── sct_etd/calibration/               # Conformal calibration modules
│   ├── conformal_aps.py               #   Class-conditional Adaptive Prediction Sets
│   └── temperature.py                 #   Affine logit recalibration & temperature scaling
│
├── results/phase_n_retrained/          # All reported results (retrained on chronologically ordered SGCC)
│   ├── a34_cell_*.csv, baselines/, bu/  #   Cached prediction-level records per seed-fold cell
│   ├── r2_audit/                        #   Tables 2–5 (table*.csv), statistical tests, partition manifests, eligibility ledger
│   ├── tableA/, tableB/, tableC/        #   Supp. Table S7 note, S4, S3 (per-attack recall, per-group PR-AUC, matched contrasts)
│   ├── trust/                           #   Table 6, Fig. 5, Supp. Fig. S4 (conformal audit) CSVs; Fig. 4 ECE values
│   ├── ecl_board.csv                    #   Supp. Table S7 (ECL metric suite)
│   ├── ecl_trust/                       #   ECL cached predictions and Supp. Table S6
│   ├── figures/                         #   Regenerated main and supplementary figures
│   └── logs/                            #   Console logs of the runs that produced the outputs
├── results/_archive_phase_n_pre_retrain/ # Superseded pre-retrain results, kept for audit only
├── partition_manifests/               # Customer/window split manifests + standalone verifier
│
└── data/                              # Data documentation (raw datasets not shipped)
    └── README.md                      # Instructions to obtain public SGCC & ECL corpora
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10+
- Linux (x86_64) recommended
- NVIDIA GPU with CUDA support (e.g., RTX 5070 / RTX 4090) for training selective state spaces (`mamba-ssm`), or CPU for figure/table regeneration from precomputed records.

### Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/azeem1020/etd-mamrd-conformal.git
cd etd-mamrd-conformal

# 2. Create conda environment from specification
conda env create -f environment.yml
conda activate mamrd

# Alternatively, manual setup via pip:
# conda create -n mamrd python=3.10 -y && conda activate mamrd
# pip install -r requirements.txt

# 3. (Optional for training MAMRD with selective SSMs) Install hardware-accelerated kernels
pip install mamba-ssm causal-conv1d
```

### Quick Verification
Test that the architecture and modular components import correctly:
```bash
cd scripts
PYTHONPATH=.. python -c "from phased.models import build_arm; m = build_arm('a34_generic', weeks=7, days=7); print('MAMRD Architecture Initialized Successfully')"
```

---

## 4. Dataset Acquisition

Raw datasets are not distributed in this repository due to licensing. Please refer to [`data/README.md`](data/README.md):

1. **SGCC**: State Grid Corporation of China electricity theft benchmark (42,372 consumers, 1,035 days). Place the consumer-by-day CSV at `data/SGCC.csv`.
2. **ECL**: UCI `ElectricityLoadDiagrams20112014` (370 industrial/commercial profiles). Place at `data/LD2011_2014.txt`.

Path overrides can be set using environment variables without modifying any code:
```bash
export MAMRD_SGCC=/path/to/SGCC.csv
export MAMRD_ECL=/path/to/LD2011_2014.txt
export MAMRD_RESULTS=../results/phase_n_retrained
```

---

## 5. Running Experiments & Generating Paper Artifacts

To support transparent peer review and independent replication, the repository implements an explicit two-tier replication architecture:
- **Tier 1 (Regeneration from cached predictions, about 3.5 minutes on a 16-core CPU, no GPU)**: Starting from the cached prediction-level records of every trained model, it verifies the customer-disjoint partition manifests, re-runs every analysis script (Tables 2–6, Supplementary Tables S2–S4, S6, S7 and the statistics quoted in the text), and regenerates all data-driven main and supplementary figures. It does not retrain any model. Every number in Tables 2–6 is the single rounding of a value in a released CSV.
- **Tier 2 (Complete End-to-End Retraining from Scratch, 103.8 minutes on GPU)**: Independently executes all 177 neural-network training runs (on SGCC: 15 MAMRD cells, 54 ablation cells, 54 baseline cells; on ECL: 54 cells for 6 detectors) after cohort cleaning, then runs the Tier 1 post-processing (`reproduce_all_r2.py`). The console log of the reference run is `results/phase_n_retrained/master_retrain.log`.

---

### Tier 1: Master Reproduction Suite from Cached Predictions (about 3.5 minutes)
To regenerate every table (Tables 2–6; Supp. Tables S2–S4, S6, S7) and every data-driven figure (Figures 2–5; Supp. Figures S1–S4) from the cached predictions without retraining; the script runs the steps below in order, stops at the first failure and writes each step's console output to `results/phase_n_retrained/logs/<script>.log`:
```bash
python scripts/reproduce_all_r2.py --config configs/default.yaml
```
| Step | Script(s) | Outputs (`results/phase_n_retrained/`) |
| :--- | :--- | :--- |
| 0 | `prepare_sgcc_dataset.py` | chronological ordering and date audit of `data/SGCC.csv` |
| 1 | `verify_partition_manifests.py` | disjointness check of `partition_manifests/` |
| 2 | `run_decision_value.py`, `run_empirical_comparator.py`, `run_trust_layer.py`, `run_ecl_trust_analysis.py` | `trust/`, `ecl_trust/RESULTS_ECL_TRUST.md` (Table 6, Fig. 5, Supp. Fig. S4, Supp. Table S6) |
| 3 | `compute_table_b.py`, `compute_table_c.py`, `table_c_contrasts.py`, `compute_table_a.py`, `make_ecl_board.py` | `tableB/` (Supp. Table S4, S2 PR-AUC), `tableC/` (Supp. Table S3), `tableA/` (Supp. Table S7 note), `ecl_board.csv` (Supp. Table S7) |
| 4 | `compute_r2_revisions.py` | `r2_audit/table2_partition_ledger.csv` … `table5_ablation.csv`, `table3_tests.csv`, `table4_prospective.csv`, `mask_only_baseline.csv`, `router_params.csv`, conformal eligibility ledger |
| 5 | `make_manuscript_figures.py`, `make_trust_figure.py`, `make_figures.py`, `make_mechanism_figure.py` | `figures/`, `trust/fig4_ece_by_detector.csv` |

Re-running Tier 1 on the released records reproduces every CSV/Markdown output byte-for-byte; console logs of the reference run are in `results/phase_n_retrained/logs/`. Supplementary Table S5 (inference latency and peak memory) is a hardware measurement made once with `scripts/make_compute_table.py` on an RTX 3060 and is not part of Tier 1; its parameter counts are deterministic.
*Note: Figure 1 (MAMRD Architectural Framework) is a conceptual schematic drawn in TikZ; it is not generated from data, and its source is not part of this repository. Figure 2 (Precision-Recall and ROC curves) is dynamically interpolated and rendered by `make_manuscript_figures.py` as part of this routine.*

---

### Tier 2: Complete End-to-End Retraining from Raw Datasets

#### Master One-Command Retraining Runner (103.8 min on NVIDIA GPU):
To independently execute the full end-to-end retraining pipeline from scratch into `results/phase_n_retrained/`:
```bash
python scripts/run_master_retrain.py
```

#### Step-by-Step 7-Stage Retraining Pipeline:
Alternatively, each of the seven stages implemented in `scripts/run_master_retrain.py` can be executed independently from the repository root:

```bash
# ==============================================================================
# Stage 0: Data Acquisition (see data/README.md)
# ==============================================================================
mkdir -p data && cd data
# Download raw SGCC.csv and UCI ElectricityLoadDiagrams20112014 (LD2011_2014.txt)
cd ..

# ==============================================================================
# Stage 1: Cohort Cleaning & Chronological Alignment
# ==============================================================================
# 1. Strict chronological ISO-8601 sort & single-date Sunday 2016-09-18 interpolation (natural NaNs strictly preserved).
#    Required: training reads the date columns in file order and stops unless this step has been run.
python scripts/prepare_sgcc_dataset.py
# 2. Execute pre-registered cohort cleaning (filter uninformative meters nansum(|x|) < 1e-6)
python scripts/clean_cohort.py
# 3. Verify 100% customer-disjoint and window-disjoint partition manifests
python scripts/verify_partition_manifests.py

# ==============================================================================
# Stage 2: MAMRD Architecture Training (15 Cells)
# ==============================================================================
# The commands in Stages 2-4 are exactly those issued by run_master_retrain.py.
# --cell_prefix and --manifest must be given as shown: run_n_clean.py overwrites the
# manifest it is pointed at, and Table 5 / Fig. 3 read results/phase_n_retrained/a32_sgcc.csv.
python scripts/run_n_clean.py --backbone a34_generic --arm a34_generic --epochs 30 --seeds 0 1 2 3 4 --folds 0 1 2 \
    --cell_prefix a34_cell --manifest results/phase_n_retrained/a34_manifest.csv

# ==============================================================================
# Stage 3: Full Architectural Ablation Suite (9 Cells Each)
# ==============================================================================
# S0: Minimal Core (BiMamba2 + 2D Value Laplacian CNN)
python scripts/run_n_clean.py --backbone a34_bu0_core --arm a34_bu0_core --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a34_bu0_core --manifest results/phase_n_retrained/bu/a34_bu0_core_manifest.csv
# S1: + Context vector and CC-MVI imputation token (code name: ccname / CC-NaME)
python scripts/run_n_clean.py --backbone a34_bu1_ccname --arm a34_bu1_ccname --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a34_bu1_ccname --manifest results/phase_n_retrained/bu/a34_bu1_ccname_manifest.csv
# S2: + 2D missingness-mask Laplacian CNN
python scripts/run_n_clean.py --backbone a34_bu2_mask --arm a34_bu2_mask --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a34_bu2_mask --manifest results/phase_n_retrained/bu/a34_bu2_mask_manifest.csv
# S3: + Multi-scale lagged differencing (a33_generic, no IIPC)
python scripts/run_n_clean.py --backbone a33_generic --arm a33_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a33_generic --manifest results/phase_n_retrained/a33_sgcc.csv
# A32: Core + differencing only (no CC-MVI, no Mask CNN, no IIPC)
python scripts/run_n_clean.py --backbone a32_generic --arm a32_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a32_generic --manifest results/phase_n_retrained/a32_sgcc.csv
# A34_no_miss: full S4 with only the missingness pathways disabled (Reviewer 2, Comment 4)
python scripts/run_n_clean.py --backbone a34_no_missingness --arm a34_no_missingness --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
    --cell_prefix bu/a34_no_missingness --manifest results/phase_n_retrained/bu/a34_no_missingness_manifest.csv

# ==============================================================================
# Stage 4: Rival Deep Learning Baselines (9 Cells Each)
# ==============================================================================
for b in wdcnn horln hybridattn grud msdm_net dkgt; do
    python scripts/run_n_clean.py --backbone $b --arm $b --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
        --cell_prefix baselines/$b --manifest results/phase_n_retrained/baselines/${b}_manifest.csv
done

# ==============================================================================
# Stage 5: ECL Industrial Transfer Benchmark
# ==============================================================================
python scripts/run_ecl_trust.py
python scripts/run_ecl_trust_analysis.py

# ==============================================================================
# Stage 6: Certified Conformal Calibration & Trust Layer
# ==============================================================================
python scripts/run_ceiling_estimate.py
python scripts/run_trust_layer.py
python scripts/run_decision_value.py
python scripts/run_empirical_comparator.py
python scripts/run_finardi_repro.py

# ==============================================================================
# Stage 7: Post-Processing — all tables and figures (identical to Tier 1)
# ==============================================================================
python scripts/reproduce_all_r2.py --config configs/default.yaml
```

---

### Result Traceability Matrix

Every result reported in the manuscript is traceable to its corresponding configuration, seed set, execution script, and verified output file in `results/phase_n_retrained/`:

| Manuscript Artifact | Description | Configuration / Hyperparameters | Executing Script | Verified Output File (`results/phase_n_retrained/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Table 2** | Data Partitioning Ledger | `configs/default.yaml`; shuffled 5-fold `KFold` (`random_state=0`), Folds 0–2; held-out fold split by stratified sampling (33/33/34 %); conformal halves on the Table 6 / Supp. Table S6 resplits | `scripts/compute_r2_revisions.py` | `r2_audit/table2_partition_ledger.csv` (per fold: `r2_audit/sgcc_fold_counts.csv`, `r2_audit/ecl_fold_counts.csv`) |
| **Table 3** | Protocol-Matched Main Results | AdamW ($\text{lr}=3\times 10^{-4}$, $\text{wd}=10^{-4}$, $\text{bs}=128$, 30 ep); Seeds 0–4 $\times$ Folds 0–2 (15 cells) for MAMRD; Seeds 0–2 $\times$ Folds 0–2 (9 cells) for baselines; cell BCa ($B=2{,}000$, seed 12345); MAP@100/P@100 averaged over random orderings of tied scores | `scripts/compute_r2_revisions.py` | `r2_audit/table3_main_results.csv`, `r2_audit/table3_tests.csv` (paired tests, Friedman, consumer bootstrap) |
| **Table 4** | Fixed FPR Budgets ($1\%, 5\%$) | Rows: recall at FPR $\in\{0.01, 0.05\}$ by ROC interpolation (`zkmix_lab.metrics.recall_at_fpr`), inspected % $=\pi\cdot$recall$+(1-\pi)\cdot$FPR; note: threshold from conformal-calibration negatives on the Table 6 resplits | `scripts/compute_r2_revisions.py` | `r2_audit/table4_fpr_budgets.csv`, `r2_audit/table4_prospective.csv` |
| **Table 5** | Architectural Ablation Suite | Seeds 0–2 $\times$ Folds 0–2 ($M=9$ matched cells; $M=15$ for $\mathcal{S}_4$ only); $\mathcal{A}_{34,\text{no-miss}}$ (`use_mask=False, use_cc=False`); paired BCa ($B=2{,}000$), exact Wilcoxon | `scripts/compute_r2_revisions.py` | `r2_audit/table5_ablation.csv` |
| **Table 6** | Mondrian Conformal Audit (all six columns) | Target $1-\alpha = 0.80$; exact order stat $k_\phi = \lceil(n_\phi+1)(1-\alpha)\rceil$, $\hat q=+\infty$ (inspect whole group) when $k_\phi > n_\phi$; all 20 stratified 50/50 resplits per cell (300 MAMRD, 180 per baseline); cell-clustered BCa ($B=2{,}000$) | `scripts/run_decision_value.py` | `trust/decision_value_sgcc.csv`, `trust/decision_value_sgcc_resplits.csv` |
| **Figure 2** | Test PR and ROC Curves | 200-point uniform grid interpolation over held-out test predictions; mean $\pm 1\,\sigma$ ribbon | `scripts/make_manuscript_figures.py` | `figures/fig2_performance_curves.pdf` |
| **Figure 3** | Sequential / Subtractive Ablation | Step additions $\mathcal{S}_0 \to \mathcal{S}_4$ & paired drop contrasts ($\Delta\text{AP}$ with 95% BCa intervals) | `scripts/make_manuscript_figures.py` | `figures/fig3_ablation_bars.pdf` |
| **Figure 4** | Probability Calibration & ECE | Alternating split ($\mathcal{I}_{\text{cal}} / \mathcal{I}_{\text{eval}}$) of the test predictions; 10 equal-width bins; $L_2$-logistic fit ($C=10^4$) | `scripts/make_manuscript_figures.py` | `figures/fig4_calibration_reliability.pdf`, `trust/fig4_ece_by_detector.csv` (ECE and Brier score before/after recalibration, fitted $\hat{w}$, $\hat{b}$; also the calibration table of the response letter) |
| **Figure 5** | Conformal Recall & Workload Reduction | (a) empirical vs conformal group thresholds; (b) conditional vs single threshold; same 300 MAMRD resplits as Table 6 (no eligibility gate) | `scripts/run_empirical_comparator.py`, `scripts/run_decision_value.py` → `scripts/make_manuscript_figures.py` | `trust/empirical_comparator_sgcc.csv`, `trust/decision_value_sgcc.csv`, `figures/fig5_conformal_tradeoffs.pdf` |
| **Supp. Fig. S4** | Per-group lift & Heavy−Flat hit-rate gap | Same resplits/protocol as Table 6, all 5 phenotypes | `scripts/run_trust_layer.py` → `scripts/make_trust_figure.py` | `trust/trust_lift_sgcc.csv`, `trust/trust_sgcc_percell.csv`, `figures/fig6_trust_guarantee.pdf` |
| **Supp. Table S7 (per-attack note)** | ECL Recall at 1 % / 5 % FPR per FDI attack type | 9 cells (3 folds × 3 seeds); FPR threshold from evaluation negatives; BCa ($B=2{,}000$) | `scripts/compute_table_a.py` | `tableA/tableA_agg.csv` |
| **Supp. Tables S2, S4** | Per-group PR-AUC of MAMRD and all detectors | 15 MAMRD / 9 baseline cells; BCa; paired gap over 9 common cells | `scripts/compute_table_b.py` | `tableB/tableB_agg.csv`, `tableB/tableB_gap.csv` |
| **Supp. Table S3** | Prevalence-matched per-group contrasts | negatives subsampled to a common prevalence (100 draws); paired BCa over 15 cells; Friedman test | `scripts/compute_table_c.py` → `scripts/table_c_contrasts.py` | `tableC/tableC_contrasts.csv`, `tableC/tableC_friedman.csv` |
| **Supp. Table S7** | ECL metric suite | 9 cells; $F_1$ at the per-fold maximising threshold (oracle); MAP@100 tie-neutral | `scripts/make_ecl_board.py` | `ecl_board.csv` |
| **Supp. Figs. S1, S2, S3** | Detectability ceiling; metric inflation; illustrative mechanism | — | `scripts/make_figures.py`, `scripts/make_mechanism_figure.py` | `figures/fig4_ceiling.pdf`, `figures/fig1_metric_inflation.pdf`, `figures/fig_mechanism.pdf` |
| **Table 1, Supp. Table S1, Figure 1, Supp. Fig. S5** | Layer specification, training hyper-parameters, architecture and workflow schematics | not data-driven: written from the model and training code (`scripts/phased/models.py`, `zkmix_lab/training.py` `TrainConfig`); parameter counts as in Supp. Table S5 | `scripts/make_compute_table.py` (parameter counts) | — (schematics are drawn, not generated) |
| **Supp. Table S5** | Parameters, inference latency, peak memory | hardware measurement (RTX 3060, batch 256); not part of Tier 1 | `scripts/make_compute_table.py` | console output |
| **Text (Intro., Methods)** | Mask-only classifier; missing-rate statistics; router cutoffs | 3 folds | `scripts/compute_r2_revisions.py` | `r2_audit/mask_only_baseline.csv`, `r2_audit/router_params.csv` |
| **Supp. Table S6** | ECL conformal lift gap | Same protocol (`conformal_common.py`) | `scripts/run_ecl_trust_analysis.py` | `ecl_trust/RESULTS_ECL_TRUST.md` |

**Not used by any figure or table.** `scripts/run_trust_layer.py` also writes `trust/calibration_sgcc.csv` and the T1 calibration table of `trust/RESULTS_TRUST.md`. They are a separate calibration diagnostic with a different protocol (20 stratified 50/50 resplits per cell, 15 equal-mass bins, and an additional prior-shift plus temperature-scaling variant). Figure 4 and every calibration value in the paper come from `trust/fig4_ece_by_detector.csv` (alternating split, 10 equal-width bins; see the Figure 4 row).

---

## 6. Conformal Audit Protocol (Table 6, Figure 5, Supp. Figure S4, Supp. Table S6)

All conformal analyses import one module, `scripts/conformal_common.py`, so they use identical calibration/evaluation resplits and the same threshold rule.

- **Resplits.** Each seed–fold test cohort is split 20 times into stratified 50/50 calibration/evaluation halves (`resplit_rng`): 300 resplits for MAMRD (15 cells) and 180 per baseline (9 cells). Every resplit is used; none is excluded.
- **Threshold.** For a group with $n_\phi$ calibration thefts, the threshold is the order statistic $k_\phi = \lceil(n_\phi+1)(1-\alpha)\rceil$. If $k_\phi > n_\phi$ (i.e. $n_\phi < \lceil(1-\alpha)/\alpha\rceil$), the conformal quantile is infinite and every consumer of that group is inspected; the cost is included in the reported workload. At the 80 % target this happens only for the Flat group, in 2.0 % of resplits (`r2_audit/conformal_eligibility_ledger.csv`).
- **Estimates and intervals.** Each quantity is averaged over the 20 resplits of a cell and then over cells. Intervals are BCa bootstrap intervals that resample seed–fold cells; because seeds within a fold share test consumers, they are descriptive.
- **Audit trail.** `trust/decision_value_sgcc_resplits.csv` holds one row per resplit (calibration counts, whether the infinite-quantile rule applied, inspected shares, recall and precision per group); `trust/group_profile_sgcc.csv` holds the per-group theft and consumption shares quoted in the manuscript.

**Corrections made in Revision 3 (commit history on GitHub).** (i) Earlier code discarded every resplit in which any group had fewer than 10 calibration thefts (66 of 300 MAMRD resplits retained); the manuscript described this as a pooled fallback, which was never implemented. The gate is removed as described above. (ii) The Hit Rate / Lift columns of Table 6 were previously transcribed by hand and the High-consumption column reported only the Heavy-Dropper subgroup; all six columns are now written by `run_decision_value.py`. (iii) `partition_manifests/` previously held a pre-retrain copy with the early-stopping and threshold-calibration thirds swapped; it now matches `r2_audit/` and the training code (`run_n_clean.py`). Test sets were unaffected. (iv) The ECL per-attack recall in the note of Supplementary Table S7 had not been recomputed after retraining; `compute_table_a.py` now regenerates it as part of Tier 1 (`tableA/`). (v) Tables 2–5 were assembled from several sources (a ledger script outside Tier 1 that stored 4-decimal values, console logs and hand-entered cells). They are now written unrounded by `compute_r2_revisions.py`, which removes double rounding (e.g. MAMRD AP 0.59746 is reported as 0.597, not 0.598), recomputes the Table 4 inspected share at the same ROC-interpolated operating point as the recall, and evaluates the Table 4 note on the Table 6 resplits. (vi) The SGCC file stores all thefts first, so ranking ties broken by row order favour positives; MAP@100 and P@100 now average over random orderings of tied scores (`zkmix_lab.metrics.topk_tie_neutral`), and recall at a fixed FPR has one definition (`zkmix_lab.metrics.recall_at_fpr`). (vii) On ECL the released pipeline uses the middle held-out third for early stopping and leaves the first third unused; the ECL manifest now labels them accordingly, and the ECL High/Low counts of Table 2 use the observed (post-attack) window load that the ECL conformal analysis uses. (viii) Figure 3(b) labelled the A32 contrast (missingness and periodicity removed, −0.037) as the missingness-only ablation; the label is corrected, and the figure's intervals now use the same bootstrap (B = 2,000, seed 12345) as Table 5. Superseded results are kept in `results/_archive_phase_n_pre_retrain/` and are not read by any script.

---

## 7. Citation

If you use this codebase, benchmark protocols, or models in your research, please cite our manuscript:

```bibtex
@article{akram2026mamrd,
  title     = {Trustworthy electricity theft detection integrating informative missingness learning and certified conformal audits},
  author    = {Akram, Hafiz Muhammad Azeem and Talatahari, Siamak and Jhangeer, Adil},
  journal   = {Scientific Reports},
  year      = {2026},
  doi       = {10.5281/zenodo.23344738},
  note      = {Under review}
}
```

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
