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

Electricity theft detection (ETD) models frequently report over-optimistic performance due to data leakage and evaluation on artificially balanced test sets. This repository implements:

1. **Native-Prevalence Evaluation Protocol**: Evaluation on the State Grid Corporation of China (SGCC, 42,372 consumers) and ElectricityLoadDiagrams (ECL) benchmarks at the natural class imbalance (8.53% theft on SGCC), with customer-disjoint partitions and per-consumer normalization on SGCC and window-level partitions on ECL.
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
│   │   ├── baselines.py               #   Baseline detectors (GRU-D, WDCNN, Self-Attn, HORLN, MSDM, DKGT-Net)
│   │   ├── archetype.py               #   Differencing / inter-/intra-period branches
│   │   └── phenotype.py               #   Label-free consumption-level stratification
│   ├── reproduce_all_r2.py            # Tier 1: regenerates all tables and figures from cached predictions (no retraining)
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

### Baseline detectors

`scripts/phased/baselines.py` contains our PyTorch re-implementations of six published detectors, written from the architectures described in their papers and adapted to the 147 x 7 weekly input grid. All seven models, MAMRD included, are trained and evaluated under one common protocol: customer-disjoint partitions, per-consumer min-max preprocessing, AdamW (learning rate 3e-4, weight decay 1e-4), 30 epochs, class-balanced mini-batches, checkpoint selection by validation average precision, and native-prevalence test sets. This protocol replaces the preprocessing, sampling and training schedules of the original studies, so the reported scores describe these re-implementations under the common protocol; they are not reproductions of the original experiments. Supplementary Table S8 of the paper lists the details.

| Detector | Paper | As described in the paper | Adapted or chosen here |
|---|---|---|---|
| WDCNN | Zheng et al., IEEE TII 2018 | Wide path (90 units) and CNN path (15 filters, stride-3 3x3 convolution with zero-padding 1, 3x3 max pooling, 60 units), 64-unit fusion layer | Kernel preprocessing reproduces the behaviour of the authors' released code; missing readings set to 0 (the paper interpolates) |
| HORLN | Zhu et al., IEEE TII 2023 | Two IIWC blocks (32 and 16 channels, kernel 3, dilations 1-3), H x W group convolutions (200 first-order features with 4 groups, 300 second-order features with 1 group), self-dependency modeling with 4 heads, normalization and PReLU, two-layer classifier, normal initialization; value and mask input (0.94M parameters, as reported in the paper) | 147 weeks (the paper uses 148); no dropout (none is specified in the paper) |
| Self-Attention | Finardi et al., arXiv 2020 | Two hybrid layers (16 attention heads in parallel with standard and dilated convolutions, 32 channels), 1x1 unifying convolution with layer normalization and PReLU, 1,024-unit dense layer; value and binary-mask input | Quantile normalization replaced by the common preprocessing |
| GRU-D | Che et al., Sci. Rep. 2018 | Trainable diagonal input decay and hidden-state decay, mask fed to the gates, 64 hidden units | 147 weekly steps of 7 daily variables; empirical mean taken per consumer |
| MSDM | Wang et al., IEEE TSG 2025 | SGCC configuration: differencing scales of 1, 2, 7, 28, 91 and 182 days with kernel (-1, 1, 0), adaptive quantization, two MSCBlocks with 48 filters, 128-dimensional features, missing-value indicator channel | Unspecified details (1x1 projections, global max pooling) marked `[ASSUMPTION]` in `msdm/model.py`; class weights replaced by the common sampler |
| DKGT-Net | Tian et al., IEEE TII 2026 | KAN branch on the (weeks x 3, days x 3) expansion and two-layer GRU branch (128 features each), fusion with tanh-sigmoid gating and ECA channel attention | Trained without the WGAN-GP augmentation and the CSSI loss; three KAN layers and a two-layer head; SGCC only |

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

1. **SGCC**: State Grid Corporation of China electricity theft benchmark (42,372 consumers, 1,035 days), released with Zheng et al. (2018) at https://github.com/henryRDlab/ElectricityTheftDetection (download `data.zip`, `data.z01` and `data.z02` and unzip them together). Place the extracted consumer-by-day CSV at `data/SGCC.csv`.
2. **ECL**: UCI `ElectricityLoadDiagrams20112014` (370 industrial/commercial profiles), https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014. Place at `data/LD2011_2014.txt`.

SHA-256 checksums of the files used for the reported results:

| File | Bytes | SHA-256 |
| :--- | ---: | :--- |
| SGCC CSV as extracted from the archive (before `prepare_sgcc_dataset.py`) | 175,194,613 | `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7` |
| `data/SGCC.csv` after `prepare_sgcc_dataset.py` (chronological order, 2016-09-18 filled) | 175,529,148 | `b41e1dca7c40c53f62689a0f9dda3d2a252ebf0311b646e8a95917f032331180` |
| `data/LD2011_2014.txt` | 710,998,915 | `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b` |

`prepare_sgcc_dataset.py` rewrites `data/SGCC.csv` in place; applied to the first file it produces the second byte for byte.

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
- **Tier 2 (Complete End-to-End Retraining from Scratch, 103.8 minutes on one NVIDIA GeForce RTX 5070 Ti Laptop GPU)**: Independently executes all 177 neural-network training runs (on SGCC: 15 MAMRD cells, 54 ablation cells, 54 baseline cells; on ECL: 54 cells for 6 detectors) after the chronological preparation of the SGCC file and a no-signal cohort audit (training uses all 42,372 SGCC consumers), then runs the Tier 1 post-processing (`reproduce_all_r2.py`). The console log of the reference run is `results/phase_n_retrained/master_retrain.log`; HORLN and GRU-D were retrained with the Stage 4 and Stage 5 commands in Revision 3, and their logs are `master_retrain_r3.log` and `logs/retrain_horln_sgcc_r3.log` (the HORLN SGCC cells come from the latter: the HORLN section of `master_retrain_r3.log` ran before HORLN was set to its published configuration, and its SGCC cells were replaced; the ECL HORLN cells in that log use the published configuration). The 103.8 minutes are the wall time of the 9 November 2026 reference run, which used the earlier HORLN and GRU-D versions. The released repository already contains the outputs of all 177 runs, and `run_master_retrain.py` skips every run whose output file exists unless `--force` is given, so the complete retraining command is `python scripts/run_master_retrain.py --force` (Tier 2 below). GPU training is not bit-for-bit deterministic, so re-running Tier 2 reproduces the reported values up to small run-to-run differences; Tier 1 is exact.

---

### Tier 1: Master Reproduction Suite from Cached Predictions (about 3.5 minutes)
To regenerate every table (Tables 2–6; Supp. Tables S2–S4, S6, S7) and every data-driven figure (Figures 2–5; Supp. Figures S1–S4) from the cached predictions without retraining. Tier 1 reads the two raw data files of Section 4: Step 0 stops if `data/SGCC.csv` is missing, and without `data/LD2011_2014.txt` the ECL per-attack recall (`compute_table_a.py`) and the ECL rows of Table 2 (`compute_r2_revisions.py`) are read from the released files instead of being recomputed (the step logs say so). The script runs the steps below in order, stops at the first failure and writes each step's console output to `results/phase_n_retrained/logs/<script>.log`:
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

Re-running Tier 1 on the released records reproduces every CSV/Markdown output and figure byte-for-byte, with one exception that depends on the machine: the mask-only logistic regression (`r2_audit/mask_only_baseline.csv`) changes beyond the third decimal with the number of BLAS threads. In our runs with 1, 8, 16 and 32 threads it gave AP = 0.168 throughout and ROC-AUC = 0.650–0.651, and every other output was identical; the released file was computed with the default 32 threads of the reference machine. Console logs of the reference run are in `results/phase_n_retrained/logs/` (only their load-time lines change between runs). Supplementary Table S5 (inference latency and peak memory) is a hardware measurement made with `scripts/make_compute_table.py` on the Tier 2 machine (NVIDIA GeForce RTX 5070 Ti Laptop GPU, batch 256; latency is the median of three runs, logged in `results/phase_n_retrained/logs/make_compute_table_run1.log` to `_run3.log`, and for the HORLN and GRU-D rows in `make_compute_table_horln_grud_run1.log` to `_run3.log`) and is not part of Tier 1; its parameter counts are deterministic.
*Note: Figure 1 (MAMRD Architectural Framework) is a conceptual schematic drawn in TikZ; it is not generated from data, and its source is not part of this repository. Figure 2 (Precision-Recall and ROC curves) is dynamically interpolated and rendered by `make_manuscript_figures.py` as part of this routine.*

---

### Tier 2: Complete End-to-End Retraining from Raw Datasets

#### Master One-Command Retraining Runner (103.8 min on one NVIDIA GeForce RTX 5070 Ti Laptop GPU):
To retrain all 177 models from scratch into `results/phase_n_retrained/` and then run the Tier 1 post-processing:
```bash
python scripts/run_master_retrain.py --force
```
`--force` is required on the released repository. Without it the script resumes an interrupted run: it skips every SGCC cell and ECL cell whose output file already exists (this is how only HORLN and GRU-D were retrained in Revision 3), and on a fresh clone it therefore trains nothing. With `--force` it retrains every cell and overwrites the released prediction records; to keep them, run it in a separate copy of the repository, or restore them afterwards with `git checkout -- results/`. Because GPU training is not bit-for-bit deterministic, the retrained values differ slightly from the released ones.

#### Step-by-Step 7-Stage Retraining Pipeline:
Alternatively, each of the seven stages implemented in `scripts/run_master_retrain.py` can be executed independently from the repository root:

```bash
# ==============================================================================
# Stage 0: Data Acquisition (see data/README.md)
# ==============================================================================
mkdir -p data && cd data
# SGCC: https://github.com/henryRDlab/ElectricityTheftDetection (unzip data.zip + data.z01 + data.z02) -> data/SGCC.csv
# ECL:  https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 -> data/LD2011_2014.txt
# Check the SHA-256 checksums in Section 4.
cd ..

# ==============================================================================
# Stage 1: Chronological Preparation, Cohort Audit & Partition Check
# ==============================================================================
# 1. Strict chronological ISO-8601 sort & single-date Sunday 2016-09-18 interpolation (natural NaNs strictly preserved).
#    Required: training reads the date columns in file order and stops unless this step has been run.
python scripts/prepare_sgcc_dataset.py
# 2. No-signal cohort audit: lists meters with nansum(|x|) < 1e-6 (clean_cohort.json, keep_idx.npy).
#    The benchmark keeps all 42,372 consumers; run_n_clean.py reads keep_idx.npy only with --filtered,
#    which no reported result uses.
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
# S1: + Context vector and CC-MVI fill value (code name: ccname / CC-NaME)
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
# Stage 4: Baseline Detectors (9 Cells Each)
# ==============================================================================
for b in wdcnn horln hybridattn grud msdm_net dkgt; do
    python scripts/run_n_clean.py --backbone $b --arm $b --epochs 30 --seeds 0 1 2 --folds 0 1 2 \
        --cell_prefix baselines/$b --manifest results/phase_n_retrained/baselines/${b}_manifest.csv
done

# ==============================================================================
# Stage 5: ECL Industrial Transfer Benchmark
# ==============================================================================
python scripts/run_ecl_trust.py --epochs 30 --force   # without --force, cells whose output exists are skipped
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
| **Table 6** | Mondrian Conformal Audit (all six columns) | Target $1-\alpha = 0.80$; exact order stat $k_\phi = \lceil(n_\phi+1)(1-\alpha)\rceil$, score threshold $\hat q = 0$ (infinite conformal quantile; whole group inspected) when $k_\phi > n_\phi$; all 20 stratified 50/50 resplits per cell (300 MAMRD, 180 per baseline); cell-clustered BCa ($B=2{,}000$) | `scripts/run_decision_value.py` | `trust/decision_value_sgcc.csv`, `trust/decision_value_sgcc_resplits.csv` |
| **Figure 2** | Test PR and ROC Curves | 200-point uniform grid interpolation over held-out test predictions; mean curve per detector over cells | `scripts/make_manuscript_figures.py` | `figures/fig2_performance_curves.pdf` |
| **Figure 3** | Sequential / Subtractive Ablation | Step additions $\mathcal{S}_0 \to \mathcal{S}_4$ & paired drop contrasts ($\Delta\text{AP}$ with 95% BCa intervals) | `scripts/make_manuscript_figures.py` | `figures/fig3_ablation_bars.pdf` |
| **Figure 4** | Probability Calibration & ECE | Alternating split ($\mathcal{I}_{\text{cal}} / \mathcal{I}_{\text{eval}}$) of the test predictions; 10 equal-width bins; $L_2$-logistic fit ($C=10^4$) | `scripts/make_manuscript_figures.py` | `figures/fig4_calibration_reliability.pdf`, `trust/fig4_ece_by_detector.csv` (ECE and Brier score before/after recalibration, fitted $\hat{w}$, $\hat{b}$; also the calibration table of the response letter) |
| **Figure 5** | Conformal Recall & Workload Reduction | (a) empirical vs conformal group thresholds; (b) conditional vs single threshold; same 300 MAMRD resplits as Table 6 (no eligibility gate) | `scripts/run_empirical_comparator.py`, `scripts/run_decision_value.py` → `scripts/make_manuscript_figures.py` | `trust/empirical_comparator_sgcc.csv`, `trust/decision_value_sgcc.csv`, `figures/fig5_conformal_tradeoffs.pdf` |
| **Supp. Fig. S4** | Per-group lift & Heavy−Flat hit-rate gap | Same resplits/protocol as Table 6; the five consumption levels pooled into Heavy / Light / Flat | `scripts/run_trust_layer.py` → `scripts/make_trust_figure.py` | `trust/trust_lift_sgcc.csv`, `trust/trust_sgcc_percell.csv`, `figures/fig6_trust_guarantee.pdf` |
| **Supp. Table S7 (per-attack note)** | ECL Recall at 1 % / 5 % FPR per FDI attack type | 9 cells (3 folds × 3 seeds); FPR threshold from evaluation negatives; BCa ($B=2{,}000$) | `scripts/compute_table_a.py` | `tableA/tableA_agg.csv` |
| **Supp. Tables S2, S4** | Per-group PR-AUC of MAMRD and all detectors | 15 MAMRD / 9 baseline cells; BCa; paired gap over 9 common cells | `scripts/compute_table_b.py` | `tableB/tableB_agg.csv`, `tableB/tableB_gap.csv` |
| **Supp. Table S3** | Prevalence-matched per-group contrasts | negatives subsampled to a common prevalence (100 draws); paired BCa over 15 cells; Friedman test | `scripts/compute_table_c.py` → `scripts/table_c_contrasts.py` | `tableC/tableC_contrasts.csv`, `tableC/tableC_friedman.csv` |
| **Supp. Table S7** | ECL metric suite | 9 cells; $F_1$ at the per-fold maximising threshold (oracle); MAP@100 tie-neutral | `scripts/make_ecl_board.py` | `ecl_board.csv` |
| **Supp. Figs. S1, S2, S3** | Detectability ceiling; metric inflation; illustrative mechanism | S1(b): per-branch increments read from the Table 5 ledger `r2_audit/table5_ablation.csv` (no hand-entered values) | `scripts/make_figures.py`, `scripts/make_mechanism_figure.py` | `figures/fig4_ceiling.pdf`, `figures/fig1_metric_inflation.pdf`, `figures/fig_mechanism.pdf` |
| **Table 1, Supp. Tables S1 and S8, Figure 1, Supp. Fig. S5** | Layer specification, training hyper-parameters, baseline re-implementation details, architecture and workflow schematics | not data-driven: written from the model, baseline and training code (`scripts/phased/models.py`, `scripts/phased/baselines.py`, `zkmix_lab/training.py` `TrainConfig`); parameter counts as in Supp. Table S5 | `scripts/make_compute_table.py` (parameter counts) | — (schematics are drawn, not generated) |
| **Supp. Table S5** | Parameters, inference latency, peak memory | hardware measurement on the Tier 2 machine (RTX 5070 Ti Laptop GPU, batch 256, median of three runs); not part of Tier 1 | `scripts/make_compute_table.py` | `logs/make_compute_table_run1.log` to `_run3.log`; HORLN and GRU-D rows: `logs/make_compute_table_horln_grud_run1.log` to `_run3.log` |
| **Text (Intro., Methods)** | Mask-only classifier; missing-rate statistics; router cutoffs | 3 folds | `scripts/compute_r2_revisions.py` | `r2_audit/mask_only_baseline.csv`, `r2_audit/router_params.csv` |
| **Supp. Table S6** | ECL conformal lift gap | Same protocol (`conformal_common.py`) | `scripts/run_ecl_trust_analysis.py` | `ecl_trust/RESULTS_ECL_TRUST.md` |

**Not used by any figure or table.** `scripts/run_trust_layer.py` also writes `trust/calibration_sgcc.csv` and the T1 calibration table of `trust/RESULTS_TRUST.md`. They are a separate calibration diagnostic with a different protocol (20 stratified 50/50 resplits per cell, 15 equal-mass bins, and an additional prior-shift plus temperature-scaling variant). Figure 4 and every calibration value in the paper come from `trust/fig4_ece_by_detector.csv` (alternating split, 10 equal-width bins; see the Figure 4 row).

---

## 6. Conformal Audit Protocol (Table 6, Figure 5, Supp. Figure S4, Supp. Table S6)

All conformal analyses import one module, `scripts/conformal_common.py`, so they use identical calibration/evaluation resplits and the same threshold rule.

- **Resplits.** Each seed–fold test cohort is split 20 times into stratified 50/50 calibration/evaluation halves (`resplit_rng`): 300 resplits for MAMRD (15 cells) and 180 per baseline (9 cells). Every resplit is used; none is excluded.
- **Threshold.** For a group with $n_\phi$ calibration thefts, the threshold is the order statistic $k_\phi = \lceil(n_\phi+1)(1-\alpha)\rceil$. If $k_\phi > n_\phi$ (i.e. $n_\phi < \lceil(1-\alpha)/\alpha\rceil$), the conformal quantile is infinite and every consumer of that group is inspected; the cost is included in the reported workload. At the 80 % target this happens only for the Flat group, in 2.0 % of resplits (`r2_audit/conformal_eligibility_ledger.csv`).
- **Estimates and intervals.** Each quantity is averaged over the 20 resplits of a cell and then over cells. Intervals are BCa bootstrap intervals that resample seed–fold cells; because seeds within a fold share test consumers, they are descriptive.
- **Not used.** `sct_etd/calibration/grcc.py` (`GRCCalibrator`) contains a pooled-threshold fallback for small strata; no script imports it and no reported result uses it. `scripts/run_trust_layer.py` also writes `trust/calibration_sgcc.csv` and the T1 calibration table of `trust/RESULTS_TRUST.md`, which no figure or table uses (see Section 5).
- **Audit trail.** `trust/decision_value_sgcc_resplits.csv` holds one row per resplit (calibration counts, whether the infinite-quantile rule applied, inspected shares, recall and precision per group); `trust/group_profile_sgcc.csv` holds the per-group theft and consumption shares quoted in the manuscript.

**Corrections made in Revision 3 (commit history on GitHub).** (i) The Revision 2 code used a resplit only if every group had at least max(10, ceil(1/alpha)) calibration thefts, so Revision 2's Table 6 covered the 11 of 15 MAMRD cells that kept at least one resplit; applied to the retrained predictions, this gate keeps 66 of 300 MAMRD resplits. The gate is removed as described above. (ii) The Hit Rate / Lift columns of Revision 2's Table 6 came from a different output (`run_trust_layer.py`, with its own resplits) than the workload columns; all six columns are now written by `run_decision_value.py` from the same resplits. (iii) `partition_manifests/` previously held a pre-retrain copy with the early-stopping and threshold-calibration thirds swapped; it now matches `r2_audit/` and the training code (`run_n_clean.py`). Test sets were unaffected. (iv) The ECL per-attack recall in the note of Supplementary Table S7 had not been recomputed after retraining; `compute_table_a.py` now regenerates it as part of Tier 1 (`tableA/`). (v) Tables 2–5 were assembled from several sources (a ledger script outside Tier 1 that stored 4-decimal values, console logs and hand-entered cells). They are now written unrounded by `compute_r2_revisions.py`, which removes double rounding (e.g. MAMRD AP 0.59746 is reported as 0.597, not 0.598), recomputes the Table 4 inspected share at the same ROC-interpolated operating point as the recall, and evaluates the Table 4 note on the Table 6 resplits. (vi) The SGCC file stores all thefts first, so ranking ties broken by row order favor positives; MAP@100 and P@100 now average over random orderings of tied scores (`zkmix_lab.metrics.topk_tie_neutral`), and recall at a fixed FPR has one definition (`zkmix_lab.metrics.recall_at_fpr`). (vii) On ECL the released pipeline uses the middle held-out third for early stopping and leaves the first third unused; the ECL manifest now labels them accordingly, and the ECL High/Low counts of Table 2 use the observed (post-attack) window load that the ECL conformal analysis uses. (viii) Figure 3(b) labeled the A32 contrast (missingness and periodicity removed, −0.037) as the missingness-only ablation; the label is corrected, and the figure's intervals now use the same bootstrap (B = 2,000, seed 12345) as Table 5. Superseded results are kept in `results/_archive_phase_n_pre_retrain/` and are not read by any script.

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
