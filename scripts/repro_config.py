"""Central path configuration for the MAMRD reproducibility package.

All scripts import their data/result paths from here, so the bundle runs from any
working directory with no edits. Defaults point at this package's own layout;
override any of them with environment variables.

    MAMRD_RESULTS   results directory (default: <bundle>/results/phase_n_retrained)
    MAMRD_DATA      raw-data directory (default: <bundle>/data)
    MAMRD_SGCC      SGCC consumer x day CSV (default: <data>/SGCC.csv)
    MAMRD_ECL       ECL LD2011_2014.txt    (default: <data>/LD2011_2014.txt)

Place the raw datasets in <bundle>/data/ (see data/README.md) or point the
variables above at them, e.g.:

    MAMRD_SGCC=/path/SGCC.csv python make_data_figure.py
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # the reproducibility/ bundle root

RESULTS_DIR = Path(os.environ.get("MAMRD_RESULTS", _ROOT / "results" / "phase_n_retrained"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = Path(os.environ.get("MAMRD_DATA",    _ROOT / "data"))
SGCC_PATH   = Path(os.environ.get("MAMRD_SGCC",    DATA_DIR / "SGCC.csv"))
ECL_PATH    = Path(os.environ.get("MAMRD_ECL",     DATA_DIR / "LD2011_2014.txt"))


def scrub_paths(text: str) -> str:
    """Remove machine-specific absolute paths from console/log text: the repository root
    becomes '.', the Python interpreter 'python', and installed packages '<site-packages>'."""
    text = text.replace(str(_ROOT), ".").replace(sys.executable, "python")
    text = re.sub(r"[^\s\"']*/site-packages/", "<site-packages>/", text)
    return text.replace(str(Path.home()), "~")

