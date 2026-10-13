"""zkmix_lab — Phase A–C lab built parallel to ``sct_etd``.

The lab is intentionally standalone:

* Phase A — ``kernels``:  torch-native anti-mean kernel (Zheng et al. 2018,
  Eqs. 7–9), with frozen and learnable variants and an exact-math test
  battery.  No dependency on the legacy CPU numpy path.

* Phase B — ``models``:  ZKMix v2 (Mixer over 2-D weekly grid) plus
  TSMixer and TimeMixer at training-recipe parity.  All three expose the
  same forward signature so they can be swapped 1:1 in the runner.

* Phase C — ``README.md``:  framing audit (every weak spot from the
  earlier review is mapped to a concrete artefact in this package).

The existing ``sct_etd`` package is not touched.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
