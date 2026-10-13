# Results Directory

This directory is the output destination for experiment scripts, evaluation ledgers, logs, and generated visualizations.

To preserve scientific reproducibility and prevent unauthorized pre-publication redistribution of article figures, results are not distributed in this repository and must be generated directly from data.

### Generating Results

1. **Obtain Data**: Place `SGCC.csv` and `LD2011_2014.txt` in `data/` (see [`data/README.md`](../data/README.md)) or configure environment variables `MAMRD_SGCC` and `MAMRD_ECL`.
2. **Execute Experiment Pipelines**:
   ```bash
   cd scripts
   
   # Train and evaluate MAMRD and baselines
   PYTHONPATH=.. python run_n_clean.py --backbone a34_generic
   
   # Conformal safety audits
   PYTHONPATH=.. python run_trust_layer.py
   
   # Empirical baseline comparator (281 resplits)
   PYTHONPATH=.. python run_empirical_comparator.py
   
   # Generate manuscript tables and figures
   PYTHONPATH=.. python compute_table_b.py
   PYTHONPATH=.. python compute_table_c.py
   PYTHONPATH=.. python make_figures.py
   PYTHONPATH=.. python generate_high_contrast_fig2.py
   ```
Generated CSVs, tables, logs, and figures will be output directly into subdirectories here.
