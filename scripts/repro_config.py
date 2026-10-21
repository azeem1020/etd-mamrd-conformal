"""Central path configuration for the MAMRD reproducibility package.

All scripts import their data/result paths from here, so the bundle runs from any
working directory with no edits. Defaults point at this package's own layout;
override any of them with environment variables.

    MAMRD_RESULTS   results directory (default: <bundle>/results/phase_n)
    MAMRD_DATA      raw-data directory (default: <bundle>/data)
    MAMRD_SGCC      SGCC consumer x day CSV (default: <data>/SGCC.csv)
    MAMRD_ECL       ECL LD2011_2014.txt    (default: <data>/LD2011_2014.txt)

Place the raw datasets in <bundle>/data/ (see data/README.md) or point the
variables above at them, e.g.:

    MAMRD_SGCC=/path/SGCC.csv python make_data_figure.py
"""
from __future__ import annotations
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # the reproducibility/ bundle root

RESULTS_DIR = Path(os.environ.get("MAMRD_RESULTS", _ROOT / "results" / "phase_n"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = Path(os.environ.get("MAMRD_DATA",    _ROOT / "data"))
SGCC_PATH   = Path(os.environ.get("MAMRD_SGCC",    DATA_DIR / "SGCC.csv"))
ECL_PATH    = Path(os.environ.get("MAMRD_ECL",     DATA_DIR / "LD2011_2014.txt"))

# Optional author development workstation fallbacks when raw data is not placed in data/
if not SGCC_PATH.exists():
    _dev_sgcc = Path("/home/azeem/papers/All_Work/SGCC.csv")
    if _dev_sgcc.exists():
        SGCC_PATH = _dev_sgcc

if not ECL_PATH.exists():
    _dev_ecl = Path("/home/azeem/papers/Data/ECL/LD2011_2014.txt")
    if _dev_ecl.exists():
        ECL_PATH = _dev_ecl
