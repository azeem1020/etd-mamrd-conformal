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
3. **Group-Conditional (Mondrian) Conformal Audit Layer**: A post-hoc, distribution-free, class-conditional conformal prediction framework using the exact order statistic $k_\phi = \lceil(n_\phi + 1)(1 - \alpha)\rceil$ to target a common $1 - \alpha = 80\%$ marginal theft recall lower bound within each consumption-level group ($\phi \in \{\text{Heavy, Light, Flat}\}$) under within-group exchangeability, reducing required inspection capacity by 25.3% (`95% BCa CI: [19.5%, 30.7%]`) relative to a single global minimax threshold satisfying the coverage target across all groups.

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
│   ├── run_empirical_comparator.py    # Group-quantile baseline vs conformal audit comparator
│   ├── generate_high_contrast_fig2.py # High-contrast publication PR and ROC curves (Fig. 2)
│   ├── repro_config.py                # Centralized path configuration
│   ├── run_n_clean.py                 # Main benchmark training & evaluation runner
│   ├── run_trust_layer.py             # Conformal audit policy evaluation
│   ├── run_ecl_ours.py                # ECL generalization evaluation
│   └── make_figures.py                # Manuscript figure generators
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
export MAMRD_RESULTS=../results/phase_n
```

---

## 5. Running Experiments & Generating Paper Artifacts

### Unified Master Reproduction (Recommended)
To run the complete Round-2 reproduction suite end-to-end (verifying partition disjointness, computing Tables 2, 3, 4, 5, 6, and generating empirical Figures 2–5 and S1–S2 from raw prediction records; Figure 1 is the conceptual architecture schematic):
```bash
# Execute master reproduction workflow
python scripts/reproduce_all_r2.py --config configs/default.yaml
```

---

### Step-by-Step Individual Workflows

### A. Training & Benchmark Evaluation (Table 2, Table 3, Table 4, Table 5)
Train MAMRD (`a34_generic`) and rival baselines on the full 42,372-consumer SGCC cohort (`--full` is enabled by default, writing both `a34_manifest.csv` and per-cell `a34_cell_{seed}_{fold}.csv` files):
```bash
cd scripts

# Train MAMRD across customer-disjoint folds (0..2) and initialization seeds (0..2)
PYTHONPATH=.. python run_n_clean.py --backbone a34_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2

# Generate Table 3 (formerly Table 2) & Table B performance ledgers (9-cell matched & 15-cell extended)
PYTHONPATH=.. python compute_table_b.py

# Generate Round-2 partition manifests, consumer-level bootstrap CIs, and ablation ledgers
PYTHONPATH=.. python compute_r2_revisions.py

# Generate Table 5 (formerly Table 4) ablations & contrasts
PYTHONPATH=.. python compute_table_c.py
PYTHONPATH=.. python table_c_contrasts.py

# Evaluate external ECL window-level generalization board
PYTHONPATH=.. python run_ecl_ours.py --backbone a34_generic
PYTHONPATH=.. python make_ecl_board.py
```

### B. Manuscript & Supplementary Figures (Figures 1–5 & Figures S1–S6)
Generate all main-text vector PDFs (`fig1_mamrd_architecture.pdf`, `fig3_ablation_bars.pdf`, `fig4_calibration_reliability.pdf`, `fig5_conformal_tradeoffs.pdf`), high-contrast PR/ROC curves (`fig2_performance_curves.pdf`), and supplementary figures (`fig4_ceiling.pdf`, `fig6_trust_guarantee.pdf`):
```bash
cd scripts
PYTHONPATH=.. python make_manuscript_figures.py
PYTHONPATH=.. python generate_high_contrast_fig2.py
PYTHONPATH=.. python make_figures.py
PYTHONPATH=.. python make_trust_figure.py
```

### C. Conformal Audits, Decision Value & Empirical Baseline Comparator
Run the post-hoc group-conditional conformal audit evaluation (`k = ceil((n + 1)(1 - alpha))`), inspection-burden reduction analysis, and 281-resplit empirical group-quantile comparator:
```bash
cd scripts
PYTHONPATH=.. python run_trust_layer.py
PYTHONPATH=.. python run_decision_value.py
PYTHONPATH=.. python run_empirical_comparator.py
PYTHONPATH=.. python run_ecl_trust_analysis.py
```

### D. Full Training Pipeline (Requires GPU & Datasets)
To retrain MAMRD (`a34_generic`) across 3 customer-disjoint folds and 3 (or 5) initialization seeds on the full SGCC cohort:
```bash
cd scripts
PYTHONPATH=.. python run_n_clean.py --backbone a34_generic --epochs 30 --seeds 0 1 2 --folds 0 1 2
```

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
