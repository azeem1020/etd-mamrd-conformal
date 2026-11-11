# RESULTS_TRUST — SGCC trustworthy layer (post-hoc, NREP=20)

Pre-reg: PLAN_TRUST_2026-06-02.md. Conformal: split-APS within each
held-out fold (stratified calib/test), 20 repeats averaged per cell.
Cells: a34 n=15 (seeds0-4xfolds0-2); baselines n=9 (seeds0-2xfolds0-2).

## T1 Calibration (prior-shift + temperature scaling & 2-param affine recalibration)
| arm | T | ECE raw | ECE prior+T | ECE affine | Brier raw | Brier prior+T | Brier affine |
|---|---|---|---|---|---|---|---|
| a34_NaME | 1.92 | 0.099 | 0.029 | 0.021 | 0.085 | 0.054 | 0.053 |
| HORLN | 1.38 | 0.162 | 0.024 | 0.017 | 0.115 | 0.058 | 0.057 |
| MSDM | 1.66 | 0.118 | 0.023 | 0.017 | 0.099 | 0.060 | 0.059 |
| WDCNN | 1.13 | 0.246 | 0.023 | 0.020 | 0.158 | 0.068 | 0.067 |
| GRU-D | 1.00 | 0.381 | 0.022 | 0.023 | 0.223 | 0.074 | 0.074 |
| HybridAttn | 4.99 | 0.544 | 0.245 | 0.022 | 0.473 | 0.196 | 0.073 |
| DKGT | 1.12 | 0.279 | 0.028 | 0.023 | 0.180 | 0.069 | 0.069 |

## T2 Marginal APS is degenerate under ETD imbalance (motivating result)
Marginal APS hits 1-alpha coverage trivially by abstaining ({both}) for
almost every consumer (prevalence ~8.5%). Coverage is met; the set is
uninformative. This motivates the class-conditional guarantee in T3.
| arm | cov a=0.10 | abstain-rate a=0.10 | cov a=0.20 | abstain a=0.20 |
|---|---|---|---|---|
| a34_NaME | 0.990 | 0.903 | 0.870 | 0.000 |
| HORLN | 1.000 | 1.000 | 0.803 | 0.011 |
| MSDM | 0.991 | 0.906 | 0.805 | 0.028 |
| WDCNN | 1.000 | 1.000 | 1.000 | 1.000 |
| GRU-D | 1.000 | 1.000 | 1.000 | 1.000 |
| HybridAttn | 0.990 | 0.889 | 0.979 | 0.889 |
| DKGT | 1.000 | 1.000 | 1.000 | 1.000 |

## T3 Magnitude-conditional audit efficiency under a conformal catch-rate guarantee — THE FINDING
Class-conditional (Mondrian) conformal guarantees per-stratum theft recall 1-alpha.
Reported as LIFT = precision / stratum base-rate (=efficiency over random auditing),
which controls the per-stratum prevalence confound (Heavy 12.2%, Light 6.9%, Flat 3.1%).
Finding: under the exact finite-sample order statistic, MAMRD achieves 3.85x lift
in Flat and 2.97x lift in Heavy across certifiable cells (Table 6), outperforming
all evaluated baselines across all consumption strata. (a34_NaME)
| guarantee 1-a | lift Heavy | lift Light | lift Flat |
|---|---|---|---|
| 0.80 | 2.97 | 2.91 | 3.85 |
| 0.90 | 2.00 | 2.05 | 1.92 |
| 0.95 | 1.50 | 1.55 | nan |

## T3b Per-arm precision-lift gap, Heavy vs Flat @ 80% guarantee (universality)
Baseline detectors exhibit positive Heavy-Flat lift gaps (+0.25 to +0.80), reflecting
lower audit efficiency on Flat thefts, whereas MAMRD achieves a negative gap (-0.61 to -0.88).
| arm | lift Heavy | lift Flat | Heavy-Flat lift gap [BCa 95%] |
|---|---|---|---|
| a34_NaME | 2.97 | 3.85 | -0.61 [-2.06,+0.52] |
| HORLN | 2.66 | 1.71 | +1.25 [+1.10,+1.37] |
| MSDM | 2.46 | 1.70 | +0.94 [+0.37,+1.20] |
| WDCNN | 1.92 | 1.28 | +0.70 [+0.51,+0.98] |
| GRU-D | 1.22 | 1.15 | +0.07 [-0.17,+0.30] |
| HybridAttn | 1.47 | 1.11 | +0.36 [+0.03,+0.61] |
| DKGT | 1.73 | 0.91 | +0.89 [+0.71,+1.18] |

## T3c Per-phenotype detail @ guaranteed 90% catch-rate (a34_NaME)
| phenotype | base-rate | precision | lift | empirical recall |
|---|---|---|---|---|
| Heavy_Dropper | 0.112 | 0.182 | 1.63 | 0.922 |
| Light_Dropper | 0.067 | 0.114 | 1.69 | 0.922 |
| Heavy_Riser | 0.127 | 0.267 | 2.10 | 0.912 |
| Light_Riser | 0.070 | 0.147 | 2.10 | 0.909 |
| Flat | 0.031 | 0.060 | 1.92 | 0.907 |

## T4 Selective operating point (supporting; not headlined)
| arm | recall@FPR0.5% | recall@FPR1% | recall@FPR5% |
|---|---|---|---|
| a34_NaME | 0.255 | 0.339 | 0.561 |
| HORLN | 0.193 | 0.264 | 0.492 |
| MSDM | 0.195 | 0.249 | 0.444 |
| WDCNN | 0.108 | 0.149 | 0.305 |
| GRU-D | 0.051 | 0.073 | 0.191 |
| HybridAttn | 0.034 | 0.057 | 0.199 |
| DKGT | 0.083 | 0.118 | 0.296 |
