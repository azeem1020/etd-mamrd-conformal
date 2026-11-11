# RESULTS_ECL_TRUST — cross-dataset trustworthy layer (ECL, magnitude strata)

Strata: Heavy/Light by train-fit magnitude median. Base-rate Heavy 7.5% Light 12.7%.
Cells/arm = 9 (3 folds x 3 seeds). NREP=20.

## AP reproduction (sanity vs locked fold-0 numbers)
| arm | mean AP (3x3) | locked AP |
|---|---|---|
| a34_generic | 0.845 | 0.88 |
| horln | 0.826 | 0.852 |
| wdcnn | 0.778 | 0.836 |
| grud | 0.529 | 0.561 |
| hybridattn | 0.809 | 0.847 |
| msdm | 0.838 | 0.87 |

## T1 Calibration (preliminary)
| arm | ECE raw | Brier |
|---|---|---|
| a34_generic | 0.123 | 0.046 |
| horln | 0.099 | 0.044 |
| wdcnn | 0.139 | 0.055 |
| grud | 0.201 | 0.118 |
| hybridattn | 0.132 | 0.056 |
| msdm | 0.026 | 0.027 |

## T2 Marginal-APS degeneracy (abstain-rate @1-a=0.90)
| arm | coverage | abstain |
|---|---|---|
| a34_generic | 0.900 | 0.000 |
| horln | 0.904 | 0.000 |
| wdcnn | 0.903 | 0.000 |
| grud | 1.000 | 1.000 |
| hybridattn | 0.903 | 0.000 |
| msdm | 0.972 | 0.000 |

## T3 Magnitude-conditional audit efficiency (LIFT) @ guaranteed 80% catch-rate — THE FINDING
Lift = precision / stratum base-rate. Gap>0 = Heavy more audit-efficient than Light.
| arm | lift Heavy | lift Light | Heavy-Light lift gap [BCa 95%] |
|---|---|---|---|
| a34_generic | 8.34 | 5.30 | +3.04 [+1.87,+3.84] |
| horln | 7.09 | 5.03 | +2.06 [+1.30,+2.91] |
| wdcnn | 8.22 | 4.83 | +3.38 [+2.71,+4.23] |
| grud | 5.04 | 2.77 | +2.27 [+1.85,+2.71] |
| hybridattn | 8.54 | 4.49 | +4.05 [+3.38,+4.65] |
| msdm | 8.62 | 4.98 | +3.64 [+2.62,+4.60] |

## T4 recall@FPR (supporting)
| arm | recall@FPR1% | recall@FPR5% |
|---|---|---|
| a34_generic | 0.799 | 0.819 |
| horln | 0.795 | 0.809 |
| wdcnn | 0.721 | 0.816 |
| grud | 0.133 | 0.551 |
| hybridattn | 0.737 | 0.814 |
| msdm | 0.793 | 0.806 |
