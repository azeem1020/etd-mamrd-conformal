"""GATED CHECK: does model-agnostic per-stratum mutual information track the
per-stratum detectability of tab:strata? If yes, it converts the asserted
information-bound mechanism into measured evidence; if not, we drop it.

MI between a rich set of consumption features and the theft label, estimated
per stratum with the kNN estimator (mutual_info_classif). NO GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

SGCC = SGCC_PATH
KEEP = (RESULTS_DIR / "keep_idx.npy")
PRAUC = {"Heavy_Riser": 0.664, "Heavy_Dropper": 0.631, "Light_Riser": 0.492,
         "Light_Dropper": 0.460, "Flat": 0.402}


def features(raw):
    """Per-consumer summary features computed on valid (non-missing) entries."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(raw, axis=1)
        std = np.nanstd(raw, axis=1)
        med = np.nanmedian(raw, axis=1)
        mx = np.nanmax(raw, axis=1)
        cv = std / (mean + 1e-6)
        miss = np.isnan(raw).mean(axis=1)
        d = np.diff(raw, axis=1)
        dstd = np.nanstd(d, axis=1)          # level-shift volatility
        djump = np.nanmax(np.abs(d), axis=1)  # largest single-day change
        zero = (raw == 0).mean(axis=1)
    F = np.column_stack([mean, std, med, mx, cv, miss, dstd, djump, zero])
    col_med = np.nanmedian(F, axis=0)
    idx = np.where(np.isnan(F))
    F[idx] = np.take(col_med, idx[1])
    return F


def main():
    df = pd.read_csv(SGCC)
    y = df["FLAG"].values.astype(int)
    raw = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    keep = np.load(KEEP)
    raw, y = raw[keep], y[keep]
    nk = (raw.shape[1] // 7) * 7
    g, m = gap_magnitude(raw[:, :nk].reshape(len(raw), nk // 7, 7))
    params = fit_router(g, m)
    ph = assign(g, m, params)
    F = features(raw)

    print(f"{'stratum':<15}{'n':>7}{'theft%':>8}{'totalMI':>10}{'PR-AUC':>9}")
    rows = []
    for s in PHENOTYPES:
        msk = ph == s
        if msk.sum() < 50 or y[msk].sum() < 5:
            continue
        mi = mutual_info_classif(F[msk], y[msk], discrete_features=False,
                                 n_neighbors=5, random_state=0)
        tot = float(mi.sum())
        rows.append((s, int(msk.sum()), 100 * y[msk].mean(), tot, PRAUC[s]))
        print(f"{s:<15}{msk.sum():>7}{100*y[msk].mean():>8.1f}{tot:>10.4f}{PRAUC[s]:>9.3f}")

    # does MI rank-order match PR-AUC rank-order?
    import scipy.stats as st
    mi_v = [r[3] for r in rows]; pr_v = [r[4] for r in rows]
    rho, p = st.spearmanr(mi_v, pr_v)
    print(f"\nSpearman(totalMI, PR-AUC) across strata: rho={rho:.3f}, p={p:.3f}")
    print("VERDICT:", "MI TRACKS detectability -> include" if rho >= 0.8
          else "MI does NOT cleanly track -> DROP (do not force)")


if __name__ == "__main__":
    main()
