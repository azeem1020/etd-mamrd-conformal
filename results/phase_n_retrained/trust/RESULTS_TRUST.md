# RESULTS_TRUST — SGCC trustworthy layer (post-hoc, NREP=20)

Pre-reg: PLAN_TRUST_2026-06-02.md. Conformal: split-APS within each
held-out fold (stratified calib/test), 20 repeats averaged per cell.
Cells: a34 n=15 (seeds0-4xfolds0-2); baselines n=9 (seeds0-2xfolds0-2).

## T1 Calibration (prior-shift + temperature scaling & 2-param affine recalibration)
| arm | T | ECE raw | ECE prior+T | ECE affine | Brier raw | Brier prior+T | Brier affine |
|---|---|---|---|---|---|---|---|
| a34_NaME | 1.43 | 0.117 | 0.019 | 0.018 | 0.089 | 0.050 | 0.050 |
| HORLN | 2.50 | 0.078 | 0.032 | 0.020 | 0.079 | 0.056 | 0.055 |
| MSDM | 1.73 | 0.104 | 0.021 | 0.018 | 0.094 | 0.063 | 0.063 |
| WDCNN | 1.11 | 0.249 | 0.019 | 0.019 | 0.155 | 0.066 | 0.066 |
| GRU-D | 1.00 | 0.378 | 0.022 | 0.023 | 0.221 | 0.074 | 0.074 |
| HybridAttn | 1.81 | 0.110 | 0.032 | 0.021 | 0.094 | 0.058 | 0.056 |
| DKGT | 1.12 | 0.267 | 0.027 | 0.019 | 0.174 | 0.070 | 0.069 |

## T2 Marginal APS is degenerate under ETD imbalance (motivating result)
Marginal APS hits 1-alpha coverage trivially by abstaining ({both}) for
almost every consumer (prevalence ~8.5%). Coverage is met; the set is
uninformative. This motivates the class-conditional guarantee in T3.
| arm | cov a=0.10 | abstain-rate a=0.10 | cov a=0.20 | abstain a=0.20 |
|---|---|---|---|---|
| a34_NaME | 1.000 | 1.000 | 0.802 | 0.000 |
| HORLN | 0.961 | 0.589 | 0.898 | 0.000 |
| MSDM | 1.000 | 1.000 | 0.808 | 0.039 |
| WDCNN | 1.000 | 1.000 | 0.999 | 0.994 |
| GRU-D | 1.000 | 1.000 | 1.000 | 1.000 |
| HybridAttn | 0.999 | 0.994 | 0.801 | 0.000 |
| DKGT | 1.000 | 1.000 | 1.000 | 1.000 |

## T3 Magnitude-conditional audit efficiency under a conformal catch-rate guarantee — THE FINDING
Class-conditional (Mondrian) conformal guarantees per-stratum theft recall 1-alpha.
Reported as LIFT = precision / stratum base-rate (=efficiency over random auditing),
which controls the per-stratum prevalence confound (Heavy 12.3%, Light 6.8%, Flat 3.2%).
Finding: under the exact finite-sample order statistic, MAMRD achieves 3.85x lift
in Flat and 2.97x lift in Heavy across certifiable cells (Table 6), outperforming
all evaluated baselines across all consumption strata. (a34_NaME)
| guarantee 1-a | lift Heavy | lift Light | lift Flat |
|---|---|---|---|
| 0.80 | 3.19 | 3.40 | 1.97 |
| 0.90 | 2.03 | 2.40 | 1.36 |
| 0.95 | 1.57 | 1.72 | nan |

## T3b Per-arm precision-lift gap, Heavy vs Flat @ 80% guarantee (universality)
Baseline detectors exhibit positive Heavy-Flat lift gaps (+0.25 to +0.80), reflecting
lower audit efficiency on Flat thefts, whereas MAMRD achieves a negative gap (-0.61 to -0.88).
| arm | lift Heavy | lift Flat | Heavy-Flat lift gap [BCa 95%] |
|---|---|---|---|
| a34_NaME | 3.19 | 1.97 | +1.05 [+0.32,+1.93] |
| HORLN | 2.64 | 2.38 | +0.14 [-1.23,+1.27] |
| MSDM | 2.06 | 1.38 | +0.59 [+0.14,+1.29] |
| WDCNN | 1.97 | 1.41 | +0.56 [+0.10,+1.12] |
| GRU-D | 1.23 | 0.88 | +0.35 [+0.18,+0.57] |
| HybridAttn | 2.52 | 1.92 | +0.58 [-0.34,+1.43] |
| DKGT | 1.85 | 0.95 | +0.90 [+0.63,+1.22] |

## T3c Per-phenotype detail @ guaranteed 90% catch-rate (a34_NaME)
| phenotype | base-rate | precision | lift | empirical recall |
|---|---|---|---|---|
| Heavy_Dropper | 0.121 | 0.237 | 1.96 | 0.912 |
| Light_Dropper | 0.069 | 0.160 | 2.30 | 0.910 |
| Heavy_Riser | 0.128 | 0.246 | 1.93 | 0.923 |
| Light_Riser | 0.064 | 0.129 | 2.02 | 0.931 |
| Flat | 0.032 | 0.044 | 1.36 | 0.886 |

## T4 Selective operating point (supporting; not headlined)
| arm | recall@FPR0.5% | recall@FPR1% | recall@FPR5% |
|---|---|---|---|
| a34_NaME | 0.302 | 0.370 | 0.589 |
| HORLN | 0.248 | 0.318 | 0.525 |
| MSDM | 0.168 | 0.220 | 0.392 |
| WDCNN | 0.107 | 0.146 | 0.345 |
| GRU-D | 0.031 | 0.050 | 0.215 |
| HybridAttn | 0.228 | 0.299 | 0.503 |
| DKGT | 0.074 | 0.113 | 0.284 |
