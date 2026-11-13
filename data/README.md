# Data

The raw datasets are not redistributed here (size and licensing). Obtain them as
below and point the scripts at them (see the *Path configuration* table in the
top-level `README.md`).

## SGCC — State Grid Corporation of China electricity-theft benchmark
- Public benchmark released with Zheng et al., *Wide and Deep Convolutional
  Neural Networks for Electricity-Theft Detection to Secure Smart Grids*,
  IEEE Trans. Industrial Informatics 14(4), 2018 (DOI 10.1109/TII.2017.2785963).
- ~42,372 consumers × 1,034 date columns (2014-01-01 to 2016-10-31; the raw columns are
  not in date order and 2016-09-18 is missing). `scripts/prepare_sgcc_dataset.py` sorts the
  columns and fills 2016-09-18 by Sunday-to-Sunday interpolation, giving 1,035 days;
  ~8.5% theft prevalence, ~25% missing entries.
- Download: https://github.com/henryRDlab/ElectricityTheftDetection (`data.zip`,
  `data.z01` and `data.z02`; unzip them together).
- Place the extracted consumer × day CSV where the scripts expect `SGCC.csv`
  (SHA-256 of the extracted file: `99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7`;
  after `prepare_sgcc_dataset.py`: `b41e1dca7c40c53f62689a0f9dda3d2a252ebf0311b646e8a95917f032331180`).

## ECL — real load with injected attacks
- Base series: UCI Machine Learning Repository,
  `ElectricityLoadDiagrams20112014` (Trindade, 2015; DOI 10.24432/C58C86),
  370 industrial/commercial clients at 15-min resolution, 2011–2014.
  Download: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- The corpus carries **no native theft**; labels are produced by the six-attack
  FDI family in `msdm/fdi.py` (multiplicative/per-step scaling, intermittent
  zeroing, mean replacement, flattening, time reversal) injected into 10% of the
  7-day windows (2,500 of 25,000), on observed entries only (missingness structure preserved). The locked
  protocol aggregates to 30-min, 7-day windows (336 readings).
- Place the raw file where the scripts expect `LD2011_2014.txt`
  (SHA-256: `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`).

> Note: the six-attack FDI family is *our* injection choice (the ETD convention,
> after Jokar 2016 / Punmiya 2019). MSDM (Wang et al., 2025) uses the corpus with
> a different (Zanetti) injection model.
