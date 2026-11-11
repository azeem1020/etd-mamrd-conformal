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
   - Branch 1: Bidirectional Mamba2 selective state-space sequence backbone ($\mathbb{R}^{90}$)
   - Branch 2: Value 2-D multi-scale discrete Laplacian CNN ($\mathbb{R}^{60}$)
   - Branch 3: Missingness-mask 2-D multi-scale Laplacian CNN ($\mathbb{R}^{30}$)
   - Branch 4: Fixed causal multi-scale lagged differencing 1-D CNN ($s \in \{1, 7, 30\}$, $\mathbb{R}^{32}$)
   - Branch 5: Multi-scale inter- and intra-period dilated 1-D CNN (IIPC, $\mathbb{R}^{32}$) alongside context-conditioned missing-value fill tokens (`CC-NaME`)
3. **Group-Conditional (Mondrian) Conformal Audit Layer**: A post-hoc, distribution-free, class-conditional conformal prediction framework using the exact order statistic $k_\phi = \lceil(n_\phi + 1)(1 - \alpha)\rceil$ to target a common $1 - \alpha = 80\%$ marginal theft recall lower bound within each consumption-level group ($\phi \in \{\text{Heavy, Light, Flat}\}$) under within-group exchangeability, reducing required inspection capacity by 34.0% (`95% BCa CI: [28.8%, 39.4%]`) relative to a single global minimax threshold satisfying the coverage target across all groups.

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
│   ├── run_decision_value.py          # Group-conditional vs global threshold workload analysis
│   ├── make_manuscript_figures.py     # Main-text vector figure generator (Figs. 1, 3, 4, 5)
│   ├── generate_high_contrast_fig2.py # High-contrast publication PR and ROC curves (Fig. 2)
│   ├── make_trust_figure.py           # Supplementary conformal figure generator (Fig. S4)
│   ├── repro_config.py                # Centralized path configuration
│   ├── run_n_clean.py                 # Main benchmark training & evaluation runner
│   ├── run_trust_layer.py             # Conformal audit policy evaluation
│   ├── run_ecl_ours.py                # ECL generalization evaluation
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
├── results/phase_n/                   # Precomputed evaluation records and CSVs
│   ├── tableA/                        #   Precomputed Table A records
│   ├── tableB/                        #   Precomputed Table B records
│   ├── tableC/                        #   Precomputed Table C records
│   └── trust/                         #   Precomputed conformal audit & comparator CSVs
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

1. **SGCC**: State Grid Corporation of China electricity theft benchmark (~42,372 consumers, 1,034 days). Place the consumer-by-day CSV at `data/SGCC.csv`.
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
- **Tier 1 (Fast Audit & Verification Suite, < 60 seconds on CPU)**: Re-computes all statistical metrics, generates bootstrap confidence intervals ($B=1{,}000$ for customer-clustered CIs, $B=2{,}000$ for paired ablations), verifies customer-disjoint partition manifests, and compiles all empirical publication figures (Figures 2, 3, 4, 5) directly from verified prediction-level records.
- **Tier 2 (Complete End-to-End Retraining from Scratch, 103.8 minutes on GPU)**: Independently executes all 123 neural network training runs (cohort cleaning, 15 MAMRD cells, 54 ablation cells, 54 baseline cells, ECL runs, conformal calibration, and post-processing) from raw CSV data using GPU acceleration.

---

### Tier 1: Fast Master Reproduction Suite (< 60 Seconds)
To verify partition manifests, compute Tables 2, 3, 4, 5, 6, and compile all empirical publication figures (Figures 2–5) without requiring model retraining:
```bash
python scripts/reproduce_all_r2.py --config configs/default.yaml
```
*Note: Figure 1 (MAMRD Architectural Framework) is the conceptual schematic compiled from TikZ (`submission_package/figures/fig0_architecture_src.tex`). Figure 2 (Precision-Recall and ROC curves) is dynamically interpolated and rendered by `make_manuscript_figures.py` as part of this routine.*

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
# 1. Strict chronological ISO-8601 sort & single-date Sunday 2016-09-18 interpolation (natural NaNs strictly preserved)
python scripts/prepare_sgcc_dataset.py
# 2. Execute pre-registered cohort cleaning (filter uninformative meters nansum(|x|) < 1e-6)
python scripts/clean_cohort.py
# 3. Verify 100% customer-disjoint and window-disjoint partition manifests
python scripts/verify_partition_manifests.py

# ==============================================================================
# Stage 2: MAMRD Architecture Training (15 Cells)
# ==============================================================================
# Train MAMRD (a34_generic) across 3 customer-disjoint folds and 5 seeds (15 cells):
python scripts/run_n_clean.py --backbone a34_generic --epochs 30 --seeds 0 1 2 3 4 --folds 0 1 2

# ==============================================================================
# Stage 3: Full Architectural Ablation Suite (9 Cells Each)
# ==============================================================================
# S0: Minimal Core (BiMamba2 + 2D Value Laplacian CNN)
python scripts/run_n_clean.py --backbone a34_bu0_core --cell_prefix bu/a34_bu0_core --epochs 30 --seeds 0 1 2 --folds 0 1 2
# S1: + Context-conditioned fill token (CC-NaME)
python scripts/run_n_clean.py --backbone a34_bu1_ccname --cell_prefix bu/a34_bu1_ccname --epochs 30 --seeds 0 1 2 --folds 0 1 2
# S2: + 2D missingness-mask Laplacian CNN
python scripts/run_n_clean.py --backbone a34_bu2_mask --cell_prefix bu/a34_bu2_mask --epochs 30 --seeds 0 1 2 --folds 0 1 2
# S3: + Multi-scale lagged differencing (a33_generic, no-IIPC)
python scripts/run_n_clean.py --backbone a33_generic --cell_prefix bu/a33_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2
# A32: Generic core + differencing (no CC-NaME, no Mask CNN)
python scripts/run_n_clean.py --backbone a32_generic --cell_prefix bu/a32_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2
# A34_no_miss: Controlled S4 ablation disabling only missingness-related pathways (Reviewer 2, Comment 4)
python scripts/run_n_clean.py --backbone a34_no_missingness --cell_prefix bu/a34_no_missingness --epochs 30 --seeds 0 1 2 --folds 0 1 2

# ==============================================================================
# Stage 4: Rival Deep Learning Baselines (9 Cells Each)
# ==============================================================================
for b in wdcnn horln hybridattn grud msdm_net dkgt; do
    python scripts/run_n_clean.py --arm $b --backbone $b --cell_prefix baselines/$b --epochs 30 --seeds 0 1 2 --folds 0 1 2
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
# Stage 7: Post-Processing, Metrics Ledgers & Manuscript Figures
# ==============================================================================
# 1. Main benchmark metrics across SGCC & ECL (Table 3)
python scripts/compute_table_b.py
# 2. Fixed FPR operating points at 1% and 5% budgets (Table 4)
python scripts/compute_table_c.py
# 3. Customer-level clustered bootstrap CIs and ablation provenance ledgers (Tables 2 & 5)
python scripts/compute_r2_revisions.py
# 4. Empirical publication figures (Figures 2, 3, 4, 5)
python scripts/make_manuscript_figures.py
# 5. Supplementary figures (Figures S1, S2, S3, S4, S6)
python scripts/make_figures.py
python scripts/make_trust_figure.py
```

---

### Result Traceability Matrix

Every result reported in the manuscript is traceable to its corresponding configuration, seed set, execution script, and verified output file in `results/phase_n_retrained/`:

| Manuscript Artifact | Description | Configuration / Hyperparameters | Executing Script | Verified Output File (`results/phase_n_retrained/`) |
| :--- | :--- | :--- | :--- | :--- |
| **Table 2** | Data Partitioning Ledger | `configs/default.yaml`; 5-fold stratified split (`seed=42`); customer-disjoint IDs | `scripts/compute_r2_revisions.py` | `r2_audit/sgcc_fold_counts.csv`, `r2_audit/ecl_fold_counts.csv` |
| **Table 3** | Protocol-Matched Main Results | AdamW ($\text{lr}=3\times 10^{-4}$, $\text{wd}=10^{-4}$, $\text{bs}=128$, 30 ep); Seeds 0–4 $\times$ Folds 0–2 (15 cells) for MAMRD; Seeds 0–2 $\times$ Folds 0–2 (9 cells) for Baselines | `scripts/compute_table_b.py` | `tableB/tableB_recalculated.csv` |
| **Table 4** | Fixed FPR Budgets ($1\%, 5\%$) | Prospective threshold transfer from $\mathcal{D}_{\text{cal}}$; evaluated at $\text{FPR} \in \{0.01, 0.05\}$ across 9 test cells | `scripts/compute_table_c.py` | `tableC/tableC_recalculated.csv` |
| **Table 5** | Architectural Ablation Suite | Seeds 0–2 $\times$ Folds 0–2 ($M=9$ matched cells); $\mathcal{A}_{34,\text{no-miss}}$ (`use_mask=False, use_cc=False`); BCa bootstrap ($B=2{,}000$) | `scripts/compute_r2_revisions.py` | `r2_audit/table_ablation_provenance.csv` |
| **Table 6** | Mondrian Conformal Policies | Marginal target $1-\alpha = 0.80$; exact order stat $k_\phi = \lceil(n_\phi+1)(1-\alpha)\rceil$; 20 stratified 50/50 resplits | `scripts/run_trust_layer.py` | `trust/audit_policies_sgcc.csv` |
| **Figure 2** | Test PR and ROC Curves | 200-point uniform grid interpolation over held-out test predictions; mean $\pm 1\,\sigma$ ribbon | `scripts/make_manuscript_figures.py` | `figures/fig2_performance_curves.pdf` |
| **Figure 3** | Sequential / Subtractive Ablation | Step additions $\mathcal{S}_0 \to \mathcal{S}_4$ & paired drop contrasts ($\Delta\text{AP}$ with 95% BCa intervals) | `scripts/make_manuscript_figures.py` | `figures/fig3_ablation_bars.pdf` |
| **Figure 4** | Probability Calibration & ECE | Alternating split ($\mathcal{I}_{\text{cal}} / \mathcal{I}_{\text{eval}}$); 10 equal-width bins; $L_2$-logistic fit ($C=10^4$) | `scripts/make_manuscript_figures.py` | `figures/fig4_calibration_reliability.pdf` |
| **Figure 5** | Conformal Workload Reduction | Group routing ($\tau_f = 0.0138, \mu^+ = 5.86, \mu^- = 7.31$); evaluated across all 285 valid calibration/evaluation resplits ($n_\phi \ge 5$, $n_{\min}=4$) | `scripts/make_manuscript_figures.py` | `figures/fig5_conformal_tradeoffs.pdf` |

---

## 6. Citation

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

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
