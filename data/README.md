# Data

The raw datasets are not redistributed here (size and licensing). Obtain them as
below and point the scripts at them (see the *Path configuration* table in the
top-level `README.md`).

## SGCC — State Grid Corporation of China electricity-theft benchmark
- Public benchmark released with Zheng et al., *Wide and Deep Convolutional
  Neural Networks for Electricity-Theft Detection to Secure Smart Grids*,
  IEEE Trans. Industrial Informatics 14(4), 2018 (DOI 10.1109/TII.2017.2785963).
- ~42,372 consumers × 1,034 daily readings (2014-01-01 to 2016-10-31),
  ~8.5% theft prevalence, ~25% missing entries.
- Place the consumer × day CSV where the scripts expect `SGCC.csv`.

## ECL — real load with injected attacks
- Base series: UCI Machine Learning Repository,
  `ElectricityLoadDiagrams20112014` (Trindade, 2015; DOI 10.24432/C58C86),
  370 industrial/commercial clients at 15-min resolution, 2011–2014.
  Download: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- The corpus carries **no native theft**; labels are produced by the six-attack
  FDI family in `msdm/fdi.py` (multiplicative/per-step scaling, intermittent
  zeroing, mean replacement, flattening, time reversal) injected into 10% of
  clients, on observed entries only (missingness structure preserved). The locked
  protocol aggregates to 30-min, 7-day windows (336 readings).
- Place the raw file where the scripts expect `LD2011_2014.txt`.

> Note: the six-attack FDI family is *our* injection choice (the ETD convention,
> after Jokar 2016 / Punmiya 2019). MSDM (Wang et al., 2025) uses the corpus with
> a different (Zanetti) injection model.
