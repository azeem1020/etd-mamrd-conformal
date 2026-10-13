"""TABLE A - Per-attack-type detection on the synthetic (ECL) corpus.

Uses VALIDATED reconstructed attack-types (validate_recon.py: reconstructed labels
== cached val_labels_json per fold). No retraining.

Per detector cell (seed,fold): FPR threshold from the evaluation negatives (5% and
1% FPR); per-type recall = fraction of that type's positives scored >= threshold.
Mean over 9 cells (3 folds x 3 seeds) with BCa 95% CI. Companion: paired
MAMRD-minus-best-baseline gap per type. DKGT excluded (SGCC-only).
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
from zkmix_lab.data import KFoldSpec
from run_local_grid import _stratified_three_way_split
from msdm.fdi import ATTACKS
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ROOT = RESULTS_DIR
OUT = ROOT / "tableA"; OUT.mkdir(exist_ok=True)
ECL = ECL_PATH
PPD = 48; WIN = 7*PPD; CAP = 25000
N = NormalDist(); BOOT_SEED = 12345; B = 2000
DET = [("a34_generic", "MAMRD"), ("wdcnn", "WDCNN"), ("horln", "HORLN"),
       ("hybridattn", "Self-Attn"), ("grud", "GRU-D"), ("msdm", "MSDM")]
TYPES = {0: "Multiplicative scaling", 1: "Per-step random scaling", 2: "Intermittent zeroing",
         3: "Mean-scaled replacement", 4: "Flatten to mean", 5: "Time reversal"}


def bca(data, B=B, alpha=0.05, seed=BOOT_SEED):
    x = np.asarray([v for v in data if v == v], float); n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return (float(np.mean(x)) if n else np.nan,)*2
    th = x.mean(); rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = N.inv_cdf(min(max(np.mean(boot < th), 1e-6), 1-1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)]); jb = jk.mean()
    den = 6*(np.sum((jb-jk)**2)**1.5); a = np.sum((jb-jk)**3)/den if den else 0.0
    f = lambda z: N.cdf(z0+(z0+z)/(1-a*(z0+z)))
    return float(np.quantile(boot, f(N.inv_cdf(alpha/2)))), float(np.quantile(boot, f(N.inv_cdf(1-alpha/2))))


def reconstruct():
    df = pd.read_csv(ECL, sep=";", decimal=",", index_col=0); df.index = pd.to_datetime(df.index)
    res = df.resample("30min").mean().values.T.astype(np.float64)
    n, T = res.shape; nw = T//WIN
    W = np.array([res[i, w*WIN:(w+1)*WIN] for i in range(n) for w in range(nw)
                  if np.nansum(np.abs(res[i, w*WIN:(w+1)*WIN])) > 1e-6])
    rng = np.random.default_rng(0)
    if len(W) > CAP:
        W = W[rng.choice(len(W), CAP, False)]
    frng = np.random.default_rng(0); y = np.zeros(len(W), int); types = -np.ones(len(W), int)
    for i in frng.choice(len(W), int(0.10*len(W)), False):
        t = int(frng.integers(6)); types[i] = t; _ = ATTACKS[t](W[i], frng); y[i] = 1
    fold_te = {}
    for fold in (0, 1, 2):
        folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
        tr, val = folds[fold]
        _, es, te = _stratified_three_way_split(val, y, (0.33, 0.33, 0.34),
                                                rng=np.random.default_rng(fold))
        fold_te[fold] = (te, y[te].astype(int), types[te])
    return fold_te


def main():
    if not ECL.exists() and (OUT/"tableA_agg.csv").exists() and (OUT/"tableA_gap.csv").exists():
        print(f"[ecl] {ECL} not found; presenting precomputed Table A from {OUT}...", flush=True)
        A = pd.read_csv(OUT/"tableA_agg.csv")
        G = pd.read_csv(OUT/"tableA_gap.csv")
    else:
        print("[ecl] reconstructing windows + types (~60s)...", flush=True)
        fold_te = reconstruct()

        rows = []
        for key, disp in DET:
            for f in sorted(glob.glob(str(ROOT/"ecl_trust"/f"{key}_[0-9]_[0-9].csv"))):
                r = next(iter(pd.read_csv(f).to_dict("records")))
                seed, fold = int(r["seed"]), int(r["fold"])
                cy = np.array(json.loads(r["val_labels_json"]), int)
                cs = np.array(json.loads(r["val_scores_json"]), float)
                te, yt, tt = fold_te[fold]
                if not np.array_equal(yt, cy):
                    print(f"  !! label mismatch {disp} s{seed} f{fold} -> skip"); continue
                neg = cs[yt == 0]
                t5 = np.quantile(neg, 0.95); t1 = np.quantile(neg, 0.99)   # 5% / 1% FPR
                for ti, tname in TYPES.items():
                    pm = (yt == 1) & (tt == ti); npos = int(pm.sum())
                    rec5 = float((cs[pm] >= t5).mean()) if npos else np.nan
                    rec1 = float((cs[pm] >= t1).mean()) if npos else np.nan
                    rows.append(dict(detector=disp, seed=seed, fold=fold, atype=ti, aname=tname,
                                     n_pos=npos, recall_fpr5=rec5, recall_fpr1=rec1))
        df = pd.DataFrame(rows); df.to_csv(OUT/"tableA_tidy.csv", index=False)

        # aggregate
        agg = []
        for disp in [d[1] for d in DET]:
            for ti, tname in TYPES.items():
                sub = df[(df.detector == disp) & (df.atype == ti)]
                row = dict(detector=disp, atype=ti, aname=tname, n_cells=len(sub),
                           pos_per_cell=sub.n_pos.mean(), pos_total=int(sub.n_pos.sum()))
                for q in ("recall_fpr5", "recall_fpr1"):
                    row[f"{q}_mean"] = float(sub[q].mean()); lo, hi = bca(sub[q].values)
                    row[f"{q}_lo"], row[f"{q}_hi"] = lo, hi
                agg.append(row)
        A = pd.DataFrame(agg); A.to_csv(OUT/"tableA_agg.csv", index=False)

        # paired MAMRD - best baseline per type (recall@5%FPR), on the 9 common cells
        baselines = [d[1] for d in DET if d[1] != "MAMRD"]
        keys = [(s, f) for s in (0, 1, 2) for f in (0, 1, 2)]
        def cv(disp, ti, s, f):
            v = df[(df.detector == disp) & (df.atype == ti) & (df.seed == s) & (df.fold == f)].recall_fpr5.values
            return float(v[0]) if len(v) else np.nan
        gaps = []
        for ti, tname in TYPES.items():
            best = max(baselines, key=lambda b: A[(A.detector == b) & (A.atype == ti)].recall_fpr5_mean.values[0])
            d = np.array([cv("MAMRD", ti, s, f) - cv(best, ti, s, f) for s, f in keys])
            lo, hi = bca(d)
            gaps.append(dict(atype=ti, aname=tname, best=best, gap=float(np.nanmean(d)), lo=lo, hi=hi))
        G = pd.DataFrame(gaps); G.to_csv(OUT/"tableA_gap.csv", index=False)

    det_order = [d[1] for d in DET]
    print("\n=== TABLE A: per-type RECALL @ 5% FPR (mean [BCa 95% CI]) ===")
    print(f"{'attack type':24s} {'npos':>4s} | " + " ".join(f"{d:>15s}" for d in det_order) + " | MAMRD-best")
    for ti, tname in TYPES.items():
        cells = []
        for disp in det_order:
            r = A[(A.detector == disp) & (A.atype == ti)].iloc[0]
            cells.append(f"{r['recall_fpr5_mean']:.2f}[{r['recall_fpr5_lo']:.2f},{r['recall_fpr5_hi']:.2f}]")
        g = G[G.atype == ti].iloc[0]
        npos = int(A[(A.detector == 'MAMRD') & (A.atype == ti)].iloc[0]['pos_total'])
        print(f"{tname:24s} {npos:4d} | " + " ".join(f"{c:>15s}" for c in cells) +
              f" | {g['gap']:+.2f}[{g['lo']:+.2f},{g['hi']:+.2f}] vs {g['best']}")

    print("\n=== TABLE A: per-type RECALL @ 1% FPR (mean) ===")
    print(f"{'attack type':24s} | " + " ".join(f"{d:>10s}" for d in det_order))
    for ti, tname in TYPES.items():
        cells = [f"{A[(A.detector==disp)&(A.atype==ti)].iloc[0]['recall_fpr1_mean']:.2f}" for disp in det_order]
        print(f"{tname:24s} | " + " ".join(f"{c:>10s}" for c in cells))

    # LaTeX (recall @ 5% FPR)
    def fmt(m, lo, hi): return f"{m:.2f}~\\ci{{{lo:.2f}}}{{{hi:.2f}}}"
    L = ["% Table A - per-attack-type recall @ 5% FPR on synthetic ECL (cache+validated types).",
         "\\begin{table*}[t]\\centering\\small",
         "\\caption{Per-attack-type recall at $5\\%$ FPR on the synthetic-attack corpus "
         "(mean over $9$ cells, BCa 95\\% CI). DKGT-Net is SGCC-only and excluded.}",
         "\\label{tab:perattack}",
         "\\begin{tabular}{@{}l r " + "c"*len(det_order) + " c@{}}", "\\toprule",
         "Attack type & $N_+$ & " + " & ".join(det_order) + " & MAMRD$-$best \\\\", "\\midrule"]
    for ti, tname in TYPES.items():
        cells = []
        for disp in det_order:
            r = A[(A.detector == disp) & (A.atype == ti)].iloc[0]
            cells.append(fmt(r['recall_fpr5_mean'], r['recall_fpr5_lo'], r['recall_fpr5_hi']))
        g = G[G.atype == ti].iloc[0]
        npos = int(A[(A.detector == 'MAMRD') & (A.atype == ti)].iloc[0]['pos_total'])
        L.append(f"{tname} & {npos} & " + " & ".join(cells) +
                 f" & ${g['gap']:+.2f}$~\\ci{{{g['lo']:+.2f}}}{{{g['hi']:+.2f}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (OUT/"tableA.tex").write_text("\n".join(L))

    # blind-spot summary
    print("\n--- baseline blind spots (recall@5%FPR < 0.50) ---")
    for disp in [d for d in det_order if d != "MAMRD"]:
        blind = [TYPES[ti] for ti in TYPES if A[(A.detector==disp)&(A.atype==ti)].iloc[0]['recall_fpr5_mean'] < 0.50]
        print(f"  {disp:10s}: " + (", ".join(blind) if blind else "(none < 0.50)"))
    mamblind = [TYPES[ti] for ti in TYPES if A[(A.detector=='MAMRD')&(A.atype==ti)].iloc[0]['recall_fpr5_mean'] < 0.50]
    print(f"  {'MAMRD':10s}: " + (", ".join(mamblind) if mamblind else "(none < 0.50)"))
    print(f"\n[done] artifacts in {OUT}  (BCa B={B})")


if __name__ == "__main__":
    main()
