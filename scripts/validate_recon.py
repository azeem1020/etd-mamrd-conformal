"""VALIDATION GATE for Tables A & C reconstruction. Computes NOTHING for the paper;
only checks that deterministically reconstructed (strata / attack-types) align with
the cached cells. Reuses the EXACT harness functions (no reimplementation).

PASS criteria:
  SGCC (C): reconstructed labels y[te] == cached val_labels_json, AND recomputed
            per-stratum PR-AUC == cached pr_<stratum> (to 1e-3).
  ECL  (A): reconstructed labels y[te] == cached val_labels_json (per fold).
If anything fails -> print FAIL and exit 1 (do NOT proceed to table computation).
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
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
from msdm.fdi import ATTACKS
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ROOT = RESULTS_DIR
ECL = ECL_PATH
PPD = 48; WIN = 7 * PPD; CAP = 25000
ok = True


def load_cell(p):
    r = next(iter(pd.read_csv(p).to_dict("records")))
    return r, np.array(json.loads(r["val_labels_json"]), int), np.array(json.loads(r["val_scores_json"]), float)


# ---------------- SGCC strata (Table C) ----------------
print("="*70, "\n[SGCC / Table C] reconstructing strata + labels", flush=True)
X, y = load_and_preprocess()
keep = np.load(ROOT / "keep_idx.npy")
Xc, yc = X[keep], y[keep]
dfr = pd.read_csv(SGCC_CSV)
raw = dfr.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
nk = (raw.shape[1] // 7) * 7
g, m = gap_magnitude(raw[keep][:, :nk].reshape(len(keep), nk // 7, 7))
print(f"  cohort kept={len(keep)} theft%={100*yc.mean():.2f}")

for fold in (0, 1, 2):
    kf = 0
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(yc))
    tr, val = folds[fold]
    es, cal, te = _stratified_three_way_split(val, yc, (0.33, 0.33, 0.34),
                                              rng=np.random.default_rng(kf*7+fold))
    params = fit_router(g[tr], m[tr])
    ph = assign(g[te], m[te], params)
    yt = yc[te].astype(int)
    cell = ROOT / f"a34_cell_0_{fold}.csv"          # seed0, this fold
    r, cy, cs = load_cell(cell)
    lab_ok = np.array_equal(yt, cy)
    print(f"  fold{fold}: len te={len(te)} cached={len(cy)}  LABELS_MATCH={lab_ok}")
    if not lab_ok:
        ok = False; print("    !! label mismatch -> STOP"); continue
    for p in PHENOTYPES:
        msk = ph == p
        rec = (average_precision_score(yt[msk], cs[msk])
               if 0 < yt[msk].sum() < msk.sum() else float("nan"))
        cap = r[f"pr_{p}"]; cn = r[f"n_{p}"]
        match = (np.isnan(rec) and (cap != cap)) or abs(rec - cap) < 1e-3
        flag = "OK" if (match and msk.sum() == cn) else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"      {p:14s} n={msk.sum():4d}(cache {cn:4d}) recomputed_PR={rec:.4f} cached={cap:.4f} [{flag}]")

# ---------------- ECL attack-types (Table A) ----------------
print("="*70, "\n[ECL / Table A] reconstructing attack-types + labels", flush=True)
print("  loading ECL windows (710MB, ~30-60s)...", flush=True)
dfe = pd.read_csv(ECL, sep=";", decimal=",", index_col=0); dfe.index = pd.to_datetime(dfe.index)
res = dfe.resample("30min").mean().values.T.astype(np.float64)
n, T = res.shape; nw = T // WIN
W = np.array([res[i, w*WIN:(w+1)*WIN] for i in range(n) for w in range(nw)
              if np.nansum(np.abs(res[i, w*WIN:(w+1)*WIN])) > 1e-6])
print(f"  raw windows N={len(W)} (CAP={CAP})")
rng = np.random.default_rng(0)
if len(W) > CAP:
    W = W[rng.choice(len(W), CAP, False)]
frng = np.random.default_rng(0)
y_ecl = np.zeros(len(W), int); types = -np.ones(len(W), int)
inj = frng.choice(len(W), int(0.10 * len(W)), False)
for i in inj:
    t = int(frng.integers(6)); types[i] = t
    _ = ATTACKS[t](W[i], frng)                      # advance rng identically
    y_ecl[i] = 1
print(f"  N={len(W)} injected={y_ecl.sum()} theft%={100*y_ecl.mean():.2f}")
print(f"  type histogram (global): {np.bincount(types[types>=0], minlength=6).tolist()}")

for fold in (0, 1, 2):
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y_ecl))
    tr, val = folds[fold]
    _, es, te = _stratified_three_way_split(val, y_ecl, (0.33, 0.33, 0.34),
                                            rng=np.random.default_rng(fold))
    yt = y_ecl[te].astype(int)
    cell = ROOT / "ecl_trust" / f"a34_generic_0_{fold}.csv"   # seed0 this fold
    r, cy, cs = load_cell(cell)
    lab_ok = np.array_equal(yt, cy)
    tt = types[te]; pos_types = tt[yt == 1]
    print(f"  fold{fold}: len te={len(te)} cached={len(cy)} pos={yt.sum()}(cache {int(cy.sum())}) "
          f"LABELS_MATCH={lab_ok} pos_type_hist={np.bincount(pos_types, minlength=6).tolist()}")
    if not lab_ok:
        ok = False; print("    !! label mismatch -> STOP")

print("="*70)
print("VALIDATION:", "ALL GATES PASSED ✓" if ok else "FAILED ✗ (do not compute tables)")
sys.exit(0 if ok else 1)
