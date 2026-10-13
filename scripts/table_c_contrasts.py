"""Table C extension: BCa CIs on the prevalence-MATCHED per-stratum PR-AUC (MAMRD,
SGCC), pairwise contrasts (dissolution test), and Friedman omnibus.

REUSES the exact per-cell matched values already in tableC_tidy.csv (K=100,
MATCH_SEED=777, target=max per-stratum prevalence per cell) — no re-tuning, no
re-scoring. Paired over the n=15 cells (5 seeds x 3 folds); BCa B=2000, seed 12345.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np, pandas as pd
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

OUT = (RESULTS_DIR / "tableC")
N = NormalDist(); BOOT_SEED = 12345; B = 2000
STRATA = ["Heavy_Dropper", "Heavy_Riser", "Light_Dropper", "Light_Riser", "Flat"]
DISP = {"Heavy_Dropper": "Heavy dropper", "Heavy_Riser": "Heavy riser",
        "Light_Dropper": "Light dropper", "Light_Riser": "Light riser", "Flat": "Flat"}
EXPECT = {"Heavy_Riser": 0.664, "Heavy_Dropper": 0.653, "Flat": 0.647,
          "Light_Riser": 0.617, "Light_Dropper": 0.575}


def bca(x, B=B, alpha=0.05, seed=BOOT_SEED):
    x = np.asarray([v for v in x if v == v], float); n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return (float(np.mean(x)) if n else np.nan,)*2
    th = x.mean(); rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = N.inv_cdf(min(max(np.mean(boot < th), 1e-6), 1-1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)]); jb = jk.mean()
    den = 6*(np.sum((jb-jk)**2)**1.5); a = np.sum((jb-jk)**3)/den if den else 0.0
    f = lambda z: N.cdf(z0+(z0+z)/(1-a*(z0+z)))
    return float(np.quantile(boot, f(N.inv_cdf(alpha/2)))), float(np.quantile(boot, f(N.inv_cdf(1-alpha/2))))


def friedman(mat):
    """mat: (n_blocks, k) -> chi2, p (with tie correction). Manual, no scipy dep."""
    from math import erf, sqrt
    n, k = mat.shape
    ranks = np.apply_along_axis(lambda r: pd.Series(r).rank().values, 1, mat)  # rank within each block
    Rj = ranks.sum(0)
    chi2 = 12.0/(n*k*(k+1))*np.sum(Rj**2) - 3*n*(k+1)
    # tie correction
    ties = 0.0
    for b in range(n):
        _, cnts = np.unique(mat[b], return_counts=True)
        ties += np.sum(cnts**3 - cnts)
    corr = 1 - ties/(n*(k**3 - k))
    chi2c = chi2/corr if corr > 0 else chi2
    df = k-1
    # p-value via chi-square survival (regularized upper incomplete gamma, series/Wilson-Hilferty fallback)
    try:
        from scipy.stats import chi2 as _c
        p = float(_c.sf(chi2c, df))
    except Exception:
        # Wilson-Hilferty approximation
        z = ((chi2c/df)**(1/3) - (1 - 2/(9*df)))/sqrt(2/(9*df))
        p = 1 - 0.5*(1+erf(z/sqrt(2)))
    return float(chi2c), float(p), Rj.tolist()


def main():
    df = pd.read_csv(OUT/"tableC_tidy.csv")
    m = df[df.detector == "MAMRD"].copy()
    piv = m.pivot_table(index=["seed", "fold"], columns="stratum", values="matched")
    piv = piv[STRATA]
    assert piv.shape == (15, 5), f"expected 15x5, got {piv.shape}"
    nkeep = m.groupby("stratum").n_kept.mean().round().astype(int)
    pos = m.groupby("stratum").pos.mean().round().astype(int)
    target = 100*m.target.mean()

    print("="*72)
    print("MATCHING CONFIG (reused, not re-tuned):")
    print(f"  common target prevalence (max per-stratum, mean over cells) = {target:.2f}%")
    print(f"  subsampling draws K=100, MATCH_SEED=777 ; BCa B={B}, BOOT_SEED={BOOT_SEED}")
    print(f"  cells: n=15 (5 seeds x 3 folds), paired")
    print("  retained N per stratum (P + kept negatives) | positives/cell:")
    for s in STRATA:
        print(f"    {DISP[s]:14s} N={nkeep[s]:4d}  P={pos[s]:3d}")
    print("="*72)

    # B. per-stratum mean + BCa, verify reproduction
    print("\n[B] Matched PR-AUC per stratum (mean [BCa 95% CI]) — verify vs expected:")
    rows = []
    bad = False
    for s in STRATA:
        v = piv[s].values; mean = float(v.mean()); lo, hi = bca(v)
        dev = abs(mean - EXPECT[s])
        if dev > 0.0015:
            bad = True
        rows.append(dict(stratum=DISP[s], matched_mean=mean, ci_lo=lo, ci_hi=hi,
                         retained_N=int(nkeep[s]), positives=int(pos[s])))
        print(f"  {DISP[s]:14s} {mean:.3f} [{lo:.3f},{hi:.3f}]  (expected {EXPECT[s]:.3f}, dev {dev:.4f})  N={nkeep[s]}")
    if bad:
        print("\n!! MEANS DEVIATE > rounding — STOP (config mismatch)."); return
    pd.DataFrame(rows).to_csv(OUT/"tableC_matched_ci.csv", index=False)

    # C. pairwise contrasts
    pairs = [("Heavy_Riser", "Flat"), ("Heavy_Dropper", "Flat"),
             ("Flat", "Light_Dropper"), ("Flat", "Light_Riser"),
             ("Heavy_Riser", "Light_Dropper")]
    print("\n[C] Pairwise contrasts (paired over 15 cells, BCa 95% CI):")
    crows = []
    for a, b in pairs:
        d = piv[a].values - piv[b].values
        mean = float(d.mean()); lo, hi = bca(d)
        res = (lo > 0) or (hi < 0)
        crows.append(dict(contrast=f"{DISP[a]} - {DISP[b]}", mean=mean, lo=lo, hi=hi, resolved=res))
        print(f"  {DISP[a]:13s} - {DISP[b]:13s}: {mean:+.3f} [{lo:+.3f},{hi:+.3f}]  resolved={res}")
    # pooled heavy vs light
    hp = (piv["Heavy_Dropper"].values + piv["Heavy_Riser"].values)/2
    lp = (piv["Light_Dropper"].values + piv["Light_Riser"].values)/2
    d = hp - lp; mean = float(d.mean()); lo, hi = bca(d); res = (lo > 0) or (hi < 0)
    crows.append(dict(contrast="Heavy (pooled) - Light (pooled)", mean=mean, lo=lo, hi=hi, resolved=res))
    print(f"  {'Heavy(pooled)':13s} - {'Light(pooled)':13s}: {mean:+.3f} [{lo:+.3f},{hi:+.3f}]  resolved={res}")
    pd.DataFrame(crows).to_csv(OUT/"tableC_contrasts.csv", index=False)

    # D. Friedman omnibus
    chi2, p, Rj = friedman(piv.values)
    print(f"\n[D] Friedman omnibus across 5 strata over 15 cells: chi2={chi2:.3f}, df=4, p={p:.4f}")
    print(f"    rank sums {dict(zip([DISP[s] for s in STRATA], [round(r,1) for r in Rj]))}")

    # ---- LaTeX: contrasts table ----
    L = ["% Table C contrasts - matched per-stratum PR-AUC dissolution test (MAMRD, SGCC).",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Prevalence-matched per-stratum detectability of \\name{}: pairwise "
         "contrasts (paired over $n{=}15$ cells, BCa 95\\% CI) and Friedman omnibus. "
         f"Friedman $\\chi^2_{{4}}={chi2:.2f}$, $p={p:.3f}$.}}",
         "\\label{tab:matched-contrasts}",
         "\\begin{tabular}{@{}l c c@{}}", "\\toprule",
         "Contrast & $\\Delta$ matched PR-AUC (95\\% BCa CI) & Resolved \\\\", "\\midrule"]
    for r in crows:
        L.append(f"{r['contrast']} & ${r['mean']:+.3f}$~\\ci{{{r['lo']:+.3f}}}{{{r['hi']:+.3f}}} & "
                 f"{'yes' if r['resolved'] else 'no'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT/"tableC_contrasts.tex").write_text("\n".join(L))

    # ---- LaTeX: updated matched column (mean + CI + retained N) ----
    L2 = ["% Table C matched column with BCa CI and retained N (MAMRD, SGCC).",
          "\\begin{table}[t]\\centering\\small",
          "\\caption{Prevalence-matched per-stratum PR-AUC of \\name{} on SGCC "
          "(negatives subsampled to a common prevalence; mean over $15$ cells, BCa 95\\% CI). "
          "$N$ is the retained per-stratum evaluation size after matching.}",
          "\\label{tab:matched-ci}",
          "\\begin{tabular}{@{}l r r c@{}}", "\\toprule",
          "Stratum & $N_+$ & $N$ & Matched PR-AUC (95\\% BCa CI) \\\\", "\\midrule"]
    for r in rows:
        L2.append(f"{r['stratum']} & {r['positives']} & {r['retained_N']} & "
                  f"{r['matched_mean']:.3f}~\\ci{{{r['ci_lo']:.3f}}}{{{r['ci_hi']:.3f}}} \\\\")
    L2 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT/"tableC_matched_ci.tex").write_text("\n".join(L2))

    # ---- verdict ----
    def find(name):
        return next(r for r in crows if r["contrast"].startswith(name))
    hf = find("Heavy riser - Flat"); hdf = find("Heavy dropper - Flat")
    fld = find("Flat - Light dropper"); flr = find("Flat - Light riser")
    pool = find("Heavy (pooled)")
    print("\n" + "="*72 + "\nVERDICT (no spin):")
    print(f"  Friedman: the 5 matched per-stratum PR-AUCs are "
          f"{'DISTINGUISHABLE' if p < 0.05 else 'NOT distinguishable'} at p<0.05 (p={p:.3f}).")
    print(f"  Heavy-Flat resolved? Heavy riser-Flat: {hf['resolved']} ({hf['mean']:+.3f} [{hf['lo']:+.3f},{hf['hi']:+.3f}]); "
          f"Heavy dropper-Flat: {hdf['resolved']} ({hdf['mean']:+.3f} [{hdf['lo']:+.3f},{hdf['hi']:+.3f}]).")
    print(f"  Flat exceeds light strata with interval excluding zero?  "
          f"Flat-Light dropper: {fld['resolved']} ({fld['mean']:+.3f} [{fld['lo']:+.3f},{fld['hi']:+.3f}]); "
          f"Flat-Light riser: {flr['resolved']} ({flr['mean']:+.3f} [{flr['lo']:+.3f},{flr['hi']:+.3f}]).")
    print(f"  Heavy(pooled)-Light(pooled): {pool['resolved']} ({pool['mean']:+.3f} [{pool['lo']:+.3f},{pool['hi']:+.3f}]).")

    # changelog
    cl = OUT/"CHANGELOG.md"
    entry = (f"\n## 2026-06-04 — Table C matched-column CIs + dissolution test\n"
             f"- Added BCa 95% CIs (B={B}, seed {BOOT_SEED}) to the prevalence-matched per-stratum "
             f"PR-AUC column (MAMRD, SGCC), reusing the exact cached per-cell matched values "
             f"(K=100, MATCH_SEED=777, target=max per-stratum prevalence ~{target:.1f}%). Means reproduce prior estimates.\n"
             f"- Added 6 pairwise contrasts (paired over n=15 cells, BCa) + Friedman omnibus "
             f"(chi2={chi2:.2f}, df=4, p={p:.3f}).\n"
             f"- Artifacts: tableC_matched_ci.{{csv,tex}}, tableC_contrasts.{{csv,tex}}. No retraining; main.tex untouched.\n")
    with open(cl, "a") as fh:
        fh.write(entry)
    print(f"\n[done] artifacts + CHANGELOG in {OUT}")

    # ---- OPTIONAL audit-efficiency lift: SKIPPED (note) ----
    print("\n[optional] audit-efficiency LIFT under matching: SKIPPED.")
    print("  Reason: lift = P_phi/pi_phi needs an operating point phi. The paper's phi is the")
    print("  conformal guaranteed-detection-rate threshold, calibrated on the CALIBRATION split,")
    print("  which is NOT in the cached test cells (only test scores+labels are). Computing it")
    print("  consistently would require the calibration-split scores / conformal machinery, i.e.")
    print("  beyond the cached scores — so per the guardrail it is skipped, not approximated.")


if __name__ == "__main__":
    main()
