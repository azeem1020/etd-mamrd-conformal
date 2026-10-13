"""TABLE C - Prevalence-controlled per-stratum detectability on SGCC.

Uses VALIDATED reconstructed strata (validate_recon.py: per-stratum PR-AUC matches
cached to <1e-3 on the FULL cohort). For each detector cell we have, per test
consumer: stratum (reconstructed), label (cached==reconstructed, re-asserted), score
(cached). No retraining.

Three quantities per stratum:
  (1) raw PR-AUC
  (2) prevalence-normalized (AP - pi)/(1 - pi)         [analytic, assumption-free]
  (3) prevalence-matched PR-AUC: subsample negatives so every stratum shares a
      common target prevalence, 100 reps (fixed seeds), averaged.

SPEC NOTE (flagged): the brief asks for target = MIN per-stratum prevalence via
"subsampling negatives". Subsampling negatives can only RAISE prevalence, so a MIN
target is infeasible by negative-subsampling (it would require removing positives).
The feasible negative-subsampling target that keeps ALL positives (lowest variance)
is the MAX per-stratum prevalence; we use that and flag it. Quantity (2) is the
assumption-free primary check and needs no subsampling.
"""
from __future__ import annotations
import sys, json, csv, glob
from pathlib import Path
from statistics import NormalDist
import numpy as np, pandas as pd
csv.field_size_limit(10**7)
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from sklearn.metrics import average_precision_score
from zkmix_lab.data import KFoldSpec
from run_local_grid import load_and_preprocess, _stratified_three_way_split, SGCC_CSV
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ROOT = RESULTS_DIR
OUT = ROOT / "tableC"; OUT.mkdir(exist_ok=True)
N = NormalDist(); BOOT_SEED = 12345; B = 2000; MATCH_REPS = 100; MATCH_SEED = 777
DET = [("wdcnn", "WDCNN", ROOT/"baselines"), ("horln", "HORLN", ROOT/"baselines"),
       ("hybridattn", "Self-Attn", ROOT/"baselines"), ("grud", "GRU-D", ROOT/"baselines"),
       ("msdm_net", "MSDM", ROOT/"baselines"), ("dkgt", "DKGT-Net", ROOT/"baselines"),
       ("a34", "MAMRD", ROOT)]
ROW = ["Heavy_Dropper", "Heavy_Riser", "Light_Dropper", "Light_Riser", "Flat"]
DISP = {"Heavy_Dropper": "Heavy dropper", "Heavy_Riser": "Heavy riser",
        "Light_Dropper": "Light dropper", "Light_Riser": "Light riser", "Flat": "Flat"}


def bca(data, stat=np.mean, B=B, alpha=0.05, seed=BOOT_SEED):
    x = np.asarray([v for v in data if v == v], float); n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return (float(np.mean(x)) if n else np.nan,)*2
    th = stat(x); rng = np.random.default_rng(seed)
    boot = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(B)])
    z0 = N.inv_cdf(min(max(np.mean(boot < th), 1e-6), 1-1e-6))
    jk = np.array([stat(np.delete(x, i)) for i in range(n)]); jb = jk.mean()
    den = 6*(np.sum((jb-jk)**2)**1.5); a = np.sum((jb-jk)**3)/den if den else 0.0
    f = lambda z: N.cdf(z0+(z0+z)/(1-a*(z0+z)))
    return float(np.quantile(boot, f(N.inv_cdf(alpha/2)))), float(np.quantile(boot, f(N.inv_cdf(1-alpha/2))))


def ap(y, s):
    return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan


def matched_ap(pos, neg, target, reps=MATCH_REPS, seed=MATCH_SEED):
    """Subsample negatives so prevalence -> target; keep all positives. Mean over reps."""
    P = len(pos); keep = int(round(P*(1-target)/target))
    keep = min(keep, len(neg))
    if P < 1 or keep < 1:
        return np.nan, P+keep
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        nn = neg[rng.choice(len(neg), keep, replace=False)] if keep < len(neg) else neg
        yy = np.r_[np.ones(P), np.zeros(len(nn))]; ss = np.r_[pos, nn]
        vals.append(average_precision_score(yy, ss))
    return float(np.mean(vals)), P+keep


def main():
    if not Path(SGCC_CSV).exists() and (OUT / "tableC_agg.csv").exists() and (OUT / "tableC_tidy.csv").exists():
        print(f"[sgcc] {SGCC_CSV} not found; presenting precomputed Table C from {OUT}...", flush=True)
        A = pd.read_csv(OUT / "tableC_agg.csv")
        df = pd.read_csv(OUT / "tableC_tidy.csv")
    else:
        X, y = load_and_preprocess()
        yc = y                                  # FULL cohort (validated)
        dfr = pd.read_csv(SGCC_CSV)
        raw = dfr.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
        nk = (raw.shape[1]//7)*7
        g, m = gap_magnitude(raw[:, :nk].reshape(len(yc), nk//7, 7))
        # reconstruct strata + labels per fold (seed-independent)
        fold_ph, fold_y = {}, {}
        for fold in (0, 1, 2):
            folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(yc))
            tr, val = folds[fold]
            es, cal, te = _stratified_three_way_split(val, yc, (0.33, 0.33, 0.34),
                                                      rng=np.random.default_rng(fold))
            fold_ph[fold] = assign(g[te], m[te], fit_router(g[tr], m[tr]))
            fold_y[fold] = yc[te].astype(int)

        # gather per-cell, per-stratum (raw AP, pi, normalized, matched)
        rows = []
        for key, disp, d in DET:
            files = sorted(glob.glob(str(d / ("a34_cell_*.csv" if key == "a34" else f"{key}_[0-9]_[0-9].csv"))))
            for f in files:
                r = next(iter(pd.read_csv(f).to_dict("records")))
                seed, fold = int(r["seed"]), int(r["fold"])
                cy = np.array(json.loads(r["val_labels_json"]), int)
                cs = np.array(json.loads(r["val_scores_json"]), float)
                ph, yt = fold_ph[fold], fold_y[fold]
                if not np.array_equal(yt, cy):       # safety re-assert
                    print(f"  !! label mismatch {disp} s{seed} f{fold} -> skip"); continue
                # per-stratum prevalence in this cell, and common target = MAX prevalence
                pis = {s: (yt[ph == s].sum() / max((ph == s).sum(), 1)) for s in ROW}
                target = max(pis.values())
                for s in ROW:
                    msk = ph == s; ys, ss = yt[msk], cs[msk]
                    a_raw = ap(ys, ss); pi = pis[s]
                    a_norm = (a_raw - pi)/(1 - pi) if a_raw == a_raw else np.nan
                    a_match, nkeep = matched_ap(ss[ys == 1], ss[ys == 0], target)
                    rows.append(dict(detector=disp, seed=seed, fold=fold, stratum=s,
                                     n=int(msk.sum()), pos=int(ys.sum()), pi=pi,
                                     raw=a_raw, norm=a_norm, matched=a_match,
                                     target=target, n_kept=nkeep))
        df = pd.DataFrame(rows); df.to_csv(OUT / "tableC_tidy.csv", index=False)

        # aggregate per detector/stratum
        agg = []
        for disp in [d[1] for d in DET]:
            for s in ROW:
                sub = df[(df.detector == disp) & (df.stratum == s)]
                row = dict(detector=disp, stratum=s, n_cells=len(sub),
                           pi=100*sub.pi.mean(), pop=int(sub.n.mean()), pos=sub.pos.mean(),
                           n_kept=int(sub.n_kept.mean()))
                for q in ("raw", "norm", "matched"):
                    row[f"{q}_mean"] = float(sub[q].mean())
                    lo, hi = bca(sub[q].values); row[f"{q}_lo"], row[f"{q}_hi"] = lo, hi
                agg.append(row)
        A = pd.DataFrame(agg); A.to_csv(OUT / "tableC_agg.csv", index=False)

    def show(disp):
        print(f"\n=== TABLE C [{disp}] : prevalence control (mean [BCa 95% CI]) ===")
        print(f"{'stratum':14s} {'theft%':>6s} {'rawPR':>20s} {'norm(AP-pi)/(1-pi)':>22s} {'matchedPR':>20s}")
        sub = A[A.detector == disp]
        order = sub.set_index("stratum").loc[ROW]
        for s in ROW:
            r = order.loc[s]
            print(f"{DISP[s]:14s} {r['pi']:6.2f} "
                  f"{r['raw_mean']:.3f}[{r['raw_lo']:.2f},{r['raw_hi']:.2f}] "
                  f"{r['norm_mean']:.3f}[{r['norm_lo']:.2f},{r['norm_hi']:.2f}] "
                  f"{r['matched_mean']:.3f}[{r['matched_lo']:.2f},{r['matched_hi']:.2f}]")
        tgt = df[df.detector == disp].target.mean()
        print(f"  common target prevalence (max stratum) = {100*tgt:.2f}%  "
              f"retained N/stratum: " + ", ".join(f"{DISP[s][:4]}={int(order.loc[s]['n_kept'])}" for s in ROW))

    show("MAMRD")

    # ordering verdict (heavy vs light vs flat), using MAMRD
    mam = A[A.detector == "MAMRD"].set_index("stratum")
    def grp(q, names): return np.mean([mam.loc[n][f"{q}_mean"] for n in names])
    heavy = ["Heavy_Dropper", "Heavy_Riser"]; light = ["Light_Dropper", "Light_Riser"]
    print("\n--- ORDERING VERDICT (MAMRD: mean over heavy / light / flat) ---")
    for q, lab in [("raw", "raw PR-AUC"), ("norm", "normalized"), ("matched", "matched PR-AUC")]:
        h, l, fl = grp(q, heavy), grp(q, light), mam.loc["Flat"][f"{q}_mean"]
        order = " > ".join([x[0] for x in sorted([("Heavy", h), ("Light", l), ("Flat", fl)],
                                                 key=lambda z: -z[1])])
        hold = "HOLDS" if (h > l > fl) else ("WEAKENS/INVERTS")
        print(f"  {lab:16s}: Heavy={h:.3f} Light={l:.3f} Flat={fl:.3f}  ->  {order}  [{hold}]")

    # all-detector normalized sheet
    print("\n--- normalized (AP-pi)/(1-pi) by detector x stratum (mean) ---")
    piv = A.pivot(index="stratum", columns="detector", values="norm_mean").loc[ROW]
    piv = piv[[d[1] for d in DET]]
    print(piv.round(3).to_string())

    # LaTeX (MAMRD sheet)
    def fmt(m, lo, hi): return f"{m:.3f}~\\ci{{{lo:.3f}}}{{{hi:.3f}}}"
    L = ["% Table C - prevalence-controlled per-stratum detectability (MAMRD), SGCC.",
         "% target prevalence = MAX per-stratum (feasible via negative subsampling); see note.",
         "\\begin{table}[t]\\centering\\small",
         "\\caption{Prevalence control of \\name{}'s per-stratum detectability on SGCC "
         "(mean over $15$ cells, BCa 95\\% CI). Normalised $=(\\mathrm{AP}-\\pi)/(1-\\pi)$; "
         "matched $=$ negatives subsampled to a common prevalence ($100$ reps).}",
         "\\label{tab:prevcontrol}",
         "\\begin{tabular}{@{}l r ccc@{}}", "\\toprule",
         "Stratum & Theft\\,\\% & Raw PR-AUC & Normalised & Matched PR-AUC \\\\", "\\midrule"]
    for s in ROW:
        r = mam.loc[s]
        L.append(f"{DISP[s]} & {r['pi']:.1f} & {fmt(r['raw_mean'],r['raw_lo'],r['raw_hi'])} & "
                 f"{fmt(r['norm_mean'],r['norm_lo'],r['norm_hi'])} & "
                 f"{fmt(r['matched_mean'],r['matched_lo'],r['matched_hi'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "tableC.tex").write_text("\n".join(L))
    print(f"\n[done] artifacts in {OUT}  (BCa B={B}, match reps={MATCH_REPS})")


if __name__ == "__main__":
    main()
