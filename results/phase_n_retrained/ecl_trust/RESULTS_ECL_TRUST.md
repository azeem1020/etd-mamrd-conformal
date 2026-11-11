# RESULTS_ECL_TRUST — cross-dataset trustworthy layer (ECL, magnitude strata)

Strata: Heavy/Light by train-fit magnitude median. Base-rate Heavy 7.5% Light 12.7%.
Cells/arm = 9 (3 folds x 3 seeds). NREP=20.

## AP reproduction (sanity vs locked fold-0 numbers)
| arm | mean AP (3x3) | locked AP |
|---|---|---|
| a34_generic | 0.843 | 0.88 |
| horln | 0.813 | 0.852 |
| wdcnn | 0.796 | 0.836 |
| grud | 0.541 | 0.561 |
| hybridattn | 0.804 | 0.847 |
| msdm | 0.842 | 0.87 |

## T1 Calibration (preliminary)
| arm | ECE raw | Brier |
|---|---|---|
| a34_generic | 0.121 | 0.046 |
| horln | 0.094 | 0.048 |
| wdcnn | 0.136 | 0.052 |
| grud | 0.195 | 0.109 |
| hybridattn | 0.135 | 0.060 |
| msdm | 0.025 | 0.026 |

## T2 Marginal-APS degeneracy (abstain-rate @1-a=0.90)
| arm | coverage | abstain |
|---|---|---|
| a34_generic | 0.900 | 0.000 |
| horln | 0.917 | 0.000 |
| wdcnn | 0.907 | 0.000 |
| grud | 0.993 | 0.939 |
| hybridattn | 0.901 | 0.000 |
| msdm | 0.972 | 0.000 |

## T3 Magnitude-conditional audit efficiency (LIFT) @ guaranteed 80% catch-rate — THE FINDING
Lift = precision / stratum base-rate. Gap>0 = Heavy more audit-efficient than Light.
| arm | lift Heavy | lift Light | Heavy-Light lift gap [BCa 95%] |
|---|---|---|---|
| a34_generic | 8.06 | 5.15 | +2.91 [+1.90,+3.76] |
| horln | 6.45 | 4.71 | +1.75 [+0.84,+2.61] |
| wdcnn | 8.21 | 5.06 | +3.15 [+2.42,+3.73] |
| grud | 5.41 | 3.03 | +2.38 [+1.87,+2.86] |
| hybridattn | 7.93 | 4.36 | +3.56 [+2.26,+4.41] |
| msdm | 8.94 | 4.94 | +4.00 [+2.89,+5.01] |

## T3b Realized recall of the class-conditional threshold at the 80% target (mean over cells)
| arm | recall Heavy | recall Light | marginal APS coverage |
|---|---|---|---|
| a34_generic | 0.813 | 0.803 | 0.801 |
| horln | 0.823 | 0.806 | 0.800 |
| wdcnn | 0.816 | 0.799 | 0.800 |
| grud | 0.805 | 0.805 | 0.801 |
| hybridattn | 0.812 | 0.813 | 0.800 |
| msdm | 0.807 | 0.811 | 0.953 |

## T4 recall@FPR (supporting)
| arm | recall@FPR1% | recall@FPR5% |
|---|---|---|
| a34_generic | 0.797 | 0.817 |
| horln | 0.779 | 0.799 |
| wdcnn | 0.759 | 0.818 |
| grud | 0.104 | 0.638 |
| hybridattn | 0.738 | 0.807 |
| msdm | 0.795 | 0.810 |
