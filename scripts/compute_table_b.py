"""TABLE B - Cross-detector per-stratum detectability on SGCC.

PURE CACHE COMPUTATION. No retraining. Uses the precomputed per-stratum PR-AUC
columns (pr_Heavy_Dropper, ...) already stored in each cached cell.

Detectors (7): 6 baselines (n=9 cells, seeds 0-2 x folds 0-2) + MAMRD/a34 (n=15,
seeds 0-4 x folds 0-2). Paired MAMRD-minus-best-baseline gaps are computed on the
9 COMMON cells only (seeds 0-2 x folds 0-2), keyed by (seed,fold) -> identical
test partition (verified: matching n_test/pos).

Outputs (results/phase_n/tableB/):
  tableB_tidy.csv   - raw per-cell per-stratum PR-AUC
  tableB_agg.csv    - means + BCa 95% CI per detector/stratum, gap vs best baseline
  tableB.tex        - booktabs LaTeX using \\ci{}{}
Flags anomalies (NaN cells, near-empty strata, degenerate CIs) rather than smoothing.
"""
from __future__ import annotations
import glob, json, csv
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ROOT = RESULTS_DIR
OUT = ROOT / "tableB"; OUT.mkdir(exist_ok=True)
BOOT_SEED = 12345
B = 2000
NORMAL = NormalDist()

# display order: 6 baselines then MAMRD
DETECTORS = [
    ("wdcnn",      "WDCNN",     ROOT / "baselines"),
    ("horln",      "HORLN",     ROOT / "baselines"),
    ("hybridattn", "Self-Attn", ROOT / "baselines"),
    ("grud",       "GRU-D",     ROOT / "baselines"),
    ("msdm_net",   "MSDM",      ROOT / "baselines"),
    ("dkgt",       "DKGT-Net",  ROOT / "baselines"),
    ("a34",        "MAMRD",     ROOT),            # a34_cell_{seed}_{fold}.csv
]
STRATA = ["Heavy_Dropper", "Light_Dropper", "Heavy_Riser", "Light_Riser", "Flat"]
STRAT_DISPLAY = {"Heavy_Dropper": "Heavy dropper", "Light_Dropper": "Light dropper",
                 "Heavy_Riser": "Heavy riser", "Light_Riser": "Light riser",
                 "Flat": "Flat", "Aggregate": "Aggregate"}
ROW_ORDER = ["Heavy_Dropper", "Heavy_Riser", "Light_Dropper", "Light_Riser", "Flat", "Aggregate"]


def cell_files(key, d):
    pat = f"a34_cell_*.csv" if key == "a34" else f"{key}_[0-9]_[0-9].csv"
    return sorted(glob.glob(str(d / pat)))


def load_cells():
    """Return tidy long df: detector,key,seed,fold,stratum,pr,n_strat,n_test."""
    rows = []
    flags = []
    for key, disp, d in DETECTORS:
        files = cell_files(key, d)
        for f in files:
            r = next(iter(pd.read_csv(f).to_dict("records")))
            seed, fold, n_test = int(r["seed"]), int(r["fold"]), int(r["n_test"])
            for s in STRATA:
                pr = r.get(f"pr_{s}", np.nan)
                nS = int(r.get(f"n_{s}", 0))
                if pr is None or (isinstance(pr, float) and np.isnan(pr)):
                    flags.append(f"NaN pr_{s} in {disp} seed{seed} fold{fold}")
                rows.append(dict(detector=disp, key=key, seed=seed, fold=fold,
                                 stratum=s, pr=float(pr) if pr == pr else np.nan,
                                 n_strat=nS, n_test=n_test))
            rows.append(dict(detector=disp, key=key, seed=seed, fold=fold,
                             stratum="Aggregate", pr=float(r["pr_auc"]),
                             n_strat=n_test, n_test=n_test))
    return pd.DataFrame(rows), flags


def bca_ci(data, B=B, alpha=0.05, seed=BOOT_SEED):
    """Bias-corrected & accelerated bootstrap CI for the mean. Returns (lo,hi,flag)."""
    x = np.asarray([v for v in data if v == v], float)  # drop NaN
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return (float(np.mean(x)) if n else np.nan,
                float(np.mean(x)) if n else np.nan, "degenerate")
    theta = x.mean()
    rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    prop = np.mean(boot < theta)
    z0 = NORMAL.inv_cdf(min(max(prop, 1e-6), 1 - 1e-6))
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    flag = ""
    def adj(z):
        return NORMAL.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    a1 = adj(NORMAL.inv_cdf(alpha / 2)); a2 = adj(NORMAL.inv_cdf(1 - alpha / 2))
    lo, hi = np.quantile(boot, a1), np.quantile(boot, a2)
    if (hi - lo) > 0.30:
        flag = "wide_CI"
    return float(lo), float(hi), flag


def main():
    df, flags = load_cells()
    df.to_csv(OUT / "tableB_tidy.csv", index=False)

    # population %: from MAMRD cells (strata are data-derived; identical across detectors per seed,fold)
    mam = df[(df.detector == "MAMRD") & (df.stratum != "Aggregate")]
    popfrac = {}
    for s in STRATA:
        sub = mam[mam.stratum == s]
        # per-cell n_strat / n_test, averaged
        popfrac[s] = float((sub.n_strat / sub.n_test).mean())
    popfrac["Aggregate"] = 1.0

    # per-detector per-stratum mean + BCa CI (each detector over ITS cells, plus MAMRD_9 over seeds 0-2)
    agg_rows = []
    means = {}  # (detector,stratum)->mean
    for disp in [d[1] for d in DETECTORS]:
        for s in ROW_ORDER:
            vals = df[(df.detector == disp) & (df.stratum == s)].pr.values
            lo, hi, fl = bca_ci(vals)
            m = float(np.nanmean(vals))
            means[(disp, s)] = m
            agg_rows.append(dict(detector=disp, stratum=s, n_cells=int(np.sum(~np.isnan(vals))),
                                 mean=m, ci_lo=lo, ci_hi=hi, flag=fl))
    # Also compute matched 9-cell (seeds 0..2 x folds 0..2) summary for MAMRD
    for s in ROW_ORDER:
        vals9 = df[(df.detector == "MAMRD") & (df.seed.isin([0, 1, 2])) & (df.stratum == s)].pr.values
        lo9, hi9, fl9 = bca_ci(vals9)
        m9 = float(np.nanmean(vals9))
        means[("MAMRD_9", s)] = m9
        agg_rows.append(dict(detector="MAMRD_9", stratum=s, n_cells=int(np.sum(~np.isnan(vals9))),
                             mean=m9, ci_lo=lo9, ci_hi=hi9, flag=fl9))

    # paired MAMRD - best baseline, on the 9 COMMON cells (seed 0-2 x fold 0-2)
    baselines = [d[1] for d in DETECTORS if d[1] != "MAMRD"]
    common = [(s, f) for s in (0, 1, 2) for f in (0, 1, 2)]
    def cell_val(disp, s, seed, fold):
        v = df[(df.detector == disp) & (df.stratum == s) &
               (df.seed == seed) & (df.fold == fold)].pr.values
        return float(v[0]) if len(v) else np.nan
    gap_rows = []
    for s in ROW_ORDER:
        # best baseline = highest mean PR-AUC in this stratum
        best = max(baselines, key=lambda b: means[(b, s)])
        diffs = np.array([cell_val("MAMRD", s, se, fo) - cell_val(best, s, se, fo)
                          for se, fo in common])
        lo, hi, fl = bca_ci(diffs)
        gap_rows.append(dict(stratum=s, best_baseline=best, gap_mean=float(np.nanmean(diffs)),
                             gap_lo=lo, gap_hi=hi, n_pairs=int(np.sum(~np.isnan(diffs))), flag=fl))

    agg = pd.DataFrame(agg_rows)
    gap = pd.DataFrame(gap_rows)
    # attach population fraction
    agg["pop_pct"] = agg.stratum.map(lambda s: 100 * popfrac[s])
    agg.to_csv(OUT / "tableB_agg.csv", index=False)
    gap.to_csv(OUT / "tableB_gap.csv", index=False)

    # ---- LaTeX (booktabs, \ci) ----
    det_order = [d[1] for d in DETECTORS]
    def fmt(m, lo, hi):
        return f"{m:.3f}~\\ci{{{lo:.3f}}}{{{hi:.3f}}}"
    lines = []
    lines.append("% Table B - cross-detector per-stratum PR-AUC on SGCC (cache-computed)")
    lines.append("% baselines n=9 (seeds 0-2), MAMRD n=15 (seeds 0-4); gap on 9 common cells.")
    lines.append("\\begin{table*}[t]\\centering\\small")
    lines.append("\\caption{Per-stratum PR-AUC by detector on SGCC (mean over cells, BCa 95\\% CI). "
                 "Baselines $n{=}9$, \\name{} $n{=}15$; the MAMRD$-$best-baseline gap is paired over "
                 "the $9$ common cells. Pop.\\ \\% is the mean test-set share of each stratum.}")
    lines.append("\\label{tab:perstratum}")
    ncol = "l r " + "c" * len(det_order) + " c"
    lines.append("\\begin{tabular}{@{}" + ncol + "@{}}")
    lines.append("\\toprule")
    head = "Stratum & Pop.\\,\\% & " + " & ".join(det_order) + " & MAMRD$-$best \\\\"
    lines.append(head)
    lines.append("\\midrule")
    for s in ROW_ORDER:
        if s == "Aggregate":
            lines.append("\\midrule")
        cells = []
        for disp in det_order:
            r = agg[(agg.detector == disp) & (agg.stratum == s)].iloc[0]
            cells.append(fmt(r["mean"], r["ci_lo"], r["ci_hi"]))
        g = gap[gap.stratum == s].iloc[0]
        gtxt = f"{g['gap_mean']:+.3f}~\\ci{{{g['gap_lo']:+.3f}}}{{{g['gap_hi']:+.3f}}}"
        lines.append(f"{STRAT_DISPLAY[s]} & {100*popfrac[s]:.1f} & " + " & ".join(cells) + f" & {gtxt} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    (OUT / "tableB.tex").write_text("\n".join(lines))

    # ---- console summary ----
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== TABLE B: per-stratum PR-AUC (mean [BCa 95% CI]) ===")
    print(f"{'stratum':14s} {'pop%':>5s} | " + " ".join(f"{d:>16s}" for d in det_order) + " | MAMRD-best")
    for s in ROW_ORDER:
        cells = []
        for disp in det_order:
            r = agg[(agg.detector == disp) & (agg.stratum == s)].iloc[0]
            cells.append(f"{r['mean']:.3f}[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")
        g = gap[gap.stratum == s].iloc[0]
        print(f"{STRAT_DISPLAY[s]:14s} {100*popfrac[s]:5.1f} | " + " ".join(f"{c:>16s}" for c in cells) +
              f" | {g['gap_mean']:+.3f}[{g['gap_lo']:+.2f},{g['gap_hi']:+.2f}] vs {g['best_baseline']}")

    print("\n--- per-stratum mean test N (min across cells, anomaly check) ---")
    for s in STRATA:
        ns = df[(df.stratum == s)].n_strat
        print(f"  {STRAT_DISPLAY[s]:14s} mean N={ns.mean():.0f}  min N={ns.min()}  (pop {100*popfrac[s]:.1f}%)")

    print("\n--- FLAGS ---")
    allflags = flags + [f"{r['detector']}/{r['stratum']}: {r['flag']}" for _, r in agg.iterrows() if r['flag']]
    allflags += [f"gap {r['stratum']}: {r['flag']}" for _, r in gap.iterrows() if r['flag']]
    print("  (none)" if not allflags else "\n".join("  " + f for f in allflags))

    print("\n--- NOTE: per-stratum theft%% NOT computed: per-stratum positive counts are")
    print("    not cached (n_* are totals); requires the strata reconstruction (Table C path).")
    print(f"\n[done] artifacts in {OUT}")
    print(f"[config] BCa B={B}, boot seed={BOOT_SEED}; baselines n=9, MAMRD n=15, gap pairs=9")


if __name__ == "__main__":
    main()
