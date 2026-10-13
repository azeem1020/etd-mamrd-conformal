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
2. **Missingness-Aware Multi-Representation Detector (MAMRD)**: A compact 0.97M-parameter deep learning architecture that fuses four physical consumption representations:
   - Consumption level & behavioral baseline
   - Periodicity & intra-/inter-week regularity
   - Step discontinuities & zero-sum differencing
   - Informative missing-data patterns via context-conditioned fill tokens
3. **Certified Conformal Audit Layer**: A post-hoc, distribution-free, class-conditional conformal prediction framework guaranteeing an 80% expected theft recovery rate across heterogeneous consumption-level groups under within-group exchangeability, while reducing required inspection capacity by 35.3% relative to single-threshold policies.

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

# 2. Create conda environment
conda create -n mamba_gpu python=3.10 -y
conda activate mamba_gpu

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional for training MAMRD with selective SSMs) Install hardware-accelerated kernels
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

To preserve scientific reproducibility and prevent unauthorized pre-publication redistribution of article figures, benchmark outputs and figures are not pre-packaged and must be generated directly from the public datasets.

### A. Training & Benchmark Evaluation (Table 2, Table 3, Table 4)
Train MAMRD (`a34_generic`) and rival baselines under the locked native-prevalence protocol:
```bash
cd scripts

# Train MAMRD across customer-disjoint folds and seeds
PYTHONPATH=.. python run_n_clean.py --backbone a34_generic --epochs 30 --lr 1e-3

# Generate Table 2 & Table B performance ledgers
PYTHONPATH=.. python compute_table_b.py

# Generate Table 4 ablations & contrasts
PYTHONPATH=.. python compute_table_c.py
PYTHONPATH=.. python table_c_contrasts.py

# Evaluate external ECL generalization board
PYTHONPATH=.. python run_ecl_ours.py
PYTHONPATH=.. python make_ecl_board.py
```

### B. High-Contrast Performance Curves (Figure 2)
Generate vector PDF and high-resolution PNG of Precision-Recall and ROC curves across all baselines:
```bash
cd scripts
PYTHONPATH=.. python generate_high_contrast_fig2.py
```
Output files:
- `results/phase_n/figures/fig2_performance_curves.pdf`
- `results/phase_n/figures/fig2_performance_curves.png`

### C. Conformal Audits & Empirical Baseline Comparator
Run the 281-resplit empirical comparator evaluating heuristic group-quantile scoring vs. distribution-free conformal prediction:
```bash
cd scripts
PYTHONPATH=.. python run_empirical_comparator.py
```

Generate the certified safety guarantee visualization (Figure 6):
```bash
cd scripts
PYTHONPATH=.. python make_trust_figure.py
```

### D. Full Training Pipeline (Requires GPU & Datasets)
To retrain MAMRD (`a34_generic`) and all rival baselines across 3 customer-disjoint folds and 3 initialization seeds:
```bash
cd scripts
PYTHONPATH=.. python run_n_clean.py --backbone a34_generic --epochs 30 --lr 1e-3
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
