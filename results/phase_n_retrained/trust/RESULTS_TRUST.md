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
All 20 resplits per cell are used (no eligibility gate); a stratum with too few
calibration thefts for the order statistic gets q_hat=+inf (all members flagged).
MAMRD (a34_NaME) per-stratum lift by guarantee level:
| guarantee 1-a | lift Heavy | lift Light | lift Flat |
|---|---|---|---|
| 0.80 | 3.19 | 3.40 | 2.44 |
| 0.90 | 2.03 | 2.40 | 1.33 |
| 0.95 | 1.57 | 1.72 | 0.98 |

## T3b Per-arm precision-lift gap, Heavy vs Flat @ 80% guarantee (universality)
Positive gap = audits are more efficient (relative to base rate) in Heavy than in Flat.
| arm | lift Heavy | lift Flat | Heavy-Flat lift gap [BCa 95%] |
|---|---|---|---|
| a34_NaME | 3.19 | 2.44 | +0.75 [+0.01,+1.43] |
| HORLN | 2.64 | 2.66 | -0.02 [-1.05,+0.90] |
| MSDM | 2.06 | 1.67 | +0.39 [-0.10,+0.89] |
| WDCNN | 1.97 | 1.87 | +0.10 [-0.32,+0.65] |
| GRU-D | 1.23 | 1.14 | +0.09 [-0.07,+0.21] |
| HybridAttn | 2.52 | 2.50 | +0.02 [-0.98,+0.90] |
| DKGT | 1.85 | 1.33 | +0.52 [+0.33,+0.74] |

## T3c Per-phenotype detail @ guaranteed 90% catch-rate (a34_NaME)
| phenotype | base-rate | precision | lift | empirical recall |
|---|---|---|---|---|
| Heavy_Dropper | 0.121 | 0.237 | 1.96 | 0.912 |
| Light_Dropper | 0.069 | 0.160 | 2.30 | 0.910 |
| Heavy_Riser | 0.128 | 0.246 | 1.93 | 0.923 |
| Light_Riser | 0.064 | 0.127 | 1.98 | 0.933 |
| Flat | 0.032 | 0.043 | 1.33 | 0.958 |

## T3d Small-sample conformal convention: share of resplits with q_hat=+inf
| arm | 1-a | HEAVY | LIGHT | Flat |
|---|---|---|---|---|
| a34_NaME | 0.80 | 0.000 | 0.000 | 0.020 |
| a34_NaME | 0.90 | 0.000 | 0.000 | 0.557 |
| a34_NaME | 0.95 | 0.000 | 0.000 | 1.000 |
| HORLN | 0.80 | 0.000 | 0.000 | 0.017 |
| HORLN | 0.90 | 0.000 | 0.000 | 0.617 |
| HORLN | 0.95 | 0.000 | 0.000 | 1.000 |
| MSDM | 0.80 | 0.000 | 0.000 | 0.022 |
| MSDM | 0.90 | 0.000 | 0.000 | 0.550 |
| MSDM | 0.95 | 0.000 | 0.000 | 1.000 |
| WDCNN | 0.80 | 0.000 | 0.000 | 0.022 |
| WDCNN | 0.90 | 0.000 | 0.000 | 0.583 |
| WDCNN | 0.95 | 0.000 | 0.000 | 1.000 |
| GRU-D | 0.80 | 0.000 | 0.000 | 0.011 |
| GRU-D | 0.90 | 0.000 | 0.000 | 0.594 |
| GRU-D | 0.95 | 0.000 | 0.000 | 1.000 |
| HybridAttn | 0.80 | 0.000 | 0.000 | 0.017 |
| HybridAttn | 0.90 | 0.000 | 0.000 | 0.589 |
| HybridAttn | 0.95 | 0.000 | 0.000 | 1.000 |
| DKGT | 0.80 | 0.000 | 0.000 | 0.028 |
| DKGT | 0.90 | 0.000 | 0.000 | 0.589 |
| DKGT | 0.95 | 0.000 | 0.000 | 1.000 |

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
