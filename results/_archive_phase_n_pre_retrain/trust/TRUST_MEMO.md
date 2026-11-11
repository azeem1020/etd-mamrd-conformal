# TRUST_MEMO — rigor record for the SGCC trustworthy layer (2026-06-02)
Companion to RESULTS_TRUST.md. Covers etd-experiment-lab Part X (logic), Part XI (grounding),
§0.3 (red-team), §0.5 (understanding). All numbers post-hoc from LOCKED saved scores; no retrain.

## Finding (one sentence)
A distribution-free class-conditional conformal guarantee turns the bounded (PR-AUC 0.562)
SGCC detector into an auditor-actionable tool: it guarantees a chosen theft catch-rate per
magnitude stratum, and the base-rate-normalized audit EFFICIENCY (lift) under that guarantee
is governed by consumption magnitude — Heavy ~2.8x vs Flat ~2.0x at an 80% guarantee
(gap +1.14 [BCa +0.70,+1.45] for a34; positive & CI-excludes-0 for all six detectors) — while
the low-magnitude majority is floored near random efficiency at every operating point.

## Logic-correctness (Part X)
- X.1 component isolation: marginal APS (T2) vs class-conditional (T3) isolate the imbalance
  degeneracy from the magnitude effect. Direction matches prediction (Heavy > Flat efficiency).
- X.2 temporal/random coherence: N/A here (post-hoc on the locked leakage-controlled cells).
- X.3 metric coherence: empirical recall (eRec 0.92-0.96) confirms the guarantee is honored;
  lift>1 confirms better-than-random; no metric contradicts the headline.
- X.4 negatives reported: (i) marginal APS degenerate under imbalance (T2); (ii) the lift gap
  CLOSES at >=90% guarantees (Flat noisy, undefined at 95% — too few Flat thefts). Both stated.

## Red-team (§0.3) — what I tried to break, what survived
- Confound 1 (base rate): Heavy has 4x the prevalence of Flat -> raw precision gap is partly
  mechanical. DEFEATED by reporting lift = precision/base-rate. Lift gap still significant @80%.
- Confound 2 (operating point): the gap is recall-dependent. DISCLOSED: separation lives at the
  top of the ranking (80% guarantee); at >=90% all strata hit the base-rate floor. Not hidden.
- Confound 3 (instrument): naive marginal conformal cannot express the effect (degenerate).
  Documented as motivating result, not buried (PLAN_TRUST_AMENDMENT_2026-06-02.md).
- Small-sample: Flat pooled thefts ~45; conformal threshold on ~20 calib thefts is noisy ->
  NREP=20 split-averaging + BCa over cells; 95% guarantee on Flat left undefined honestly.

## Alternative explanation (XI.2) — for the manuscript Discussion
"An alternative to the magnitude-efficiency law is per-stratum base-rate variation. We rule
this out by reporting base-rate-normalized lift; the Heavy-Flat lift gap at the 80% guarantee
excludes zero for every detector, so the effect is a detectability property, not a prevalence
artifact."

## Understanding (§0.5) — what the experiment taught
Before: expected conformal to 'fix' the bounded detector into decisive per-consumer decisions.
Learned: (1) marginal conformal is the WRONG tool for rare-event ETD (degenerate); (2) the
correct class-conditional guarantee is always honorable, but its AFFORDABILITY (audit
efficiency) is magnitude-bounded and floored for low-magnitude consumers at every operating
point. What would have changed my mind: a base-rate-normalized lift gap bracketing zero (K2) —
it did not, at the operating point where the magnitude law is active.

## Grounding (Part XV / B1-B4)
- Conformal: B4 §14.3 (split conformal coverage); class-conditional sets = Sadinle et al. 2019;
  APS = Romano-Sesia-Candes 2020; coverage theory = Vovk 2005 / Angelopoulos-Bates 2023.
- Calibration (T1, preliminary): B4 §14.2.2; Guo 2017. a34 best-calibrated (ECE 0.099),
  HybridAttn worst (0.544) — ECE tracks detector quality. NOT headlined (Paper 1 owns it).
- Magnitude strata: P3 own finding (label-free phenotype router, phenotype.py) — characterisation.

## Portfolio boundary (Paper 1, under review RiE)
Conformal coverage guarantee is ABSENT in Paper 1 -> clean. Calibration kept preliminary only.
Net-value/audit-budget NOT headlined (budget reported as confounded context only). P3 inflation
claim = METRIC (not split-leakage). See [[project-paper3-energyAI-pivot]].

## Artefacts
results/phase_n/trust/{RESULTS_TRUST.md, trust_sgcc_percell.csv, calibration_sgcc.csv,
operating_points_sgcc.csv}. Code: code/phase_d/run_trust_layer.py (self-verifies y[te]==saved).
NEXT: figures (per-phenotype lift-vs-guarantee curve; marginal-APS degeneracy; reliability) in
the manuscript task; then ECL cross-dataset (Task #3) + recent 2026 baseline (Task #4).
