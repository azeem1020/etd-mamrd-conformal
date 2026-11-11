# Legacy experimental records (Revision 2 archive)

> **Superseded.** All results reported in Revision 3 are in `results/phase_n_retrained/`. No script reads this directory.

This directory keeps the records produced before the chronological retraining, so that values reported in Revision 2 can be traced.

- **`ecl_ours.csv`:** early exploratory ECL pilot. It used 49-day windows of daily mean load, the prototype architecture `a14_no_mixer`, and 3 seeds × 3 folds, with test folds of 591–592 windows (AP = 0.712 ± 0.060). The pilot is superseded by the benchmark protocol in `results/phase_n_retrained/ecl_trust/` (AP = 0.843 [0.817, 0.864]); see the response to Reviewer 2, Comment 2.
- **`ecl_trust/`:** the ECL cache reported in Revision 2 (AP ≈ 0.845), from before the ECL training settings were aligned with SGCC. It is replaced by the retrained results in `results/phase_n_retrained/ecl_trust/` (AP = 0.843).
- **`tableA/`, `tableB/`, `tableC/`, `trust/` and the remaining files:** outputs of the Revision 2 runs, made under the earlier lexicographic SGCC column order.

Working notes and Markdown summaries that were kept here earlier were removed in Revision 3. They remain in the git history.
