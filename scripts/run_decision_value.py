"""Paper 3 — Move B: decision-value of the CONSUMPTION-CONDITIONAL conformal guarantee.

NEW, post-hoc, no retraining. Reuses the LOCKED conformal protocol of
run_trust_layer.py VERBATIM (cc_recall_threshold, stratified_half, recon_fold,
3-group collapse high/low/flat) so numbers are consistent with sec:trust.

Question (the deployment decision the paper currently lacks): to GUARANTEE a
per-group theft detection rate of 1-alpha in EVERY consumption group, how many
consumers must an auditor inspect under
  (i)  a CLASS-CONDITIONAL conformal threshold (one threshold per group), vs
  (ii) a SINGLE (marginal) threshold that still honours the same per-group
       guarantee -> it must be set to the most lenient per-group threshold,
       thr* = min_g thr_g, which over-flags the easy groups.
Metric: audit-set size (fraction of consumers flagged) for (i) and (ii), and the
reduction (ii)->(ii) = 1 - cond/single = wasted inspections avoided by conditioning.

BOUNDARY (Paper 1 owns selective-prediction + net-value): this is framed strictly
on the CONFORMAL GUARANTEE's CONDITIONING (coverage validity per group), measured
as audit-set efficiency. NOT $ net-value, NOT generic audit-a-subset.

Run (CPU, ~minutes, env tgpu):  python run_decision_value.py
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
from collections import defaultdict
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, pandas as pd
from run_local_grid import load_and_preprocess, _stratified_three_way_split, SGCC_CSV
from zkmix_lab.data import KFoldSpec
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

csv.field_size_limit(10**7)
RES = RESULTS_DIR
OUT = RES / "trust"; OUT.mkdir(parents=True, exist_ok=True)
NREP = 20
ALPHAS = [0.20, 0.10]                       # guaranteed detection rate 0.80, 0.90
HEAVY = ("Heavy_Riser", "Heavy_Dropper")
LIGHT = ("Light_Riser", "Light_Dropper")
ARMS = {
    "a34_NaME":   (str(RES / "a34_cell_{s}_{f}.csv"), range(5), range(3)),
    "HORLN":      (str(RES / "baselines/horln_{s}_{f}.csv"), range(3), range(3)),
    "MSDM":       (str(RES / "baselines/msdm_net_{s}_{f}.csv"), range(3), range(3)),
    "WDCNN":      (str(RES / "baselines/wdcnn_{s}_{f}.csv"), range(3), range(3)),
    "GRU-D":      (str(RES / "baselines/grud_{s}_{f}.csv"), range(3), range(3)),
    "HybridAttn": (str(RES / "baselines/hybridattn_{s}_{f}.csv"), range(3), range(3)),
    "DKGT":       (str(RES / "baselines/dkgt_{s}_{f}.csv"), range(3), range(3)),
}
_ARM_ORDER = {a: i for i, a in enumerate(ARMS)}

# ---- helpers copied VERBATIM from run_trust_layer.py (identical conformal protocol) ----
def stratified_half(y, rng):
    idx = np.arange(len(y)); cal = []
    for c in (0, 1):
        ic = idx[y == c]; rng.shuffle(ic); cal.append(ic[: len(ic) // 2])
    cal = np.concatenate(cal)
    mask = np.zeros(len(y), bool); mask[cal] = True
    return np.where(mask)[0], np.where(~mask)[0]

def cc_recall_threshold(s_cal, y_cal, alpha):
    pos = s_cal[y_cal == 1]; n = len(pos)
    if n < max(10, int(np.ceil(1.0 / alpha))):
        return None
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        tau = 1.0
    else:
        tau = float(np.sort(1.0 - pos)[k - 1])
    return 1.0 - tau

def bca_ci(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); th = x.mean()
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(B)])
    from scipy.stats import norm
    z0 = norm.ppf(np.clip((boot < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)])
    jbar = jk.mean(); num = ((jbar - jk) ** 3).sum(); den = 6.0 * ((jbar - jk) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0.0
    def adj(q):
        z = norm.ppf(q); return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, [adj(alpha / 2), adj(1 - alpha / 2)]); return (float(lo), float(hi))

# ---- data + folds (identical to run_trust_layer) ----
_data_cache = {}
def _get_data():
    if "data" not in _data_cache:
        print("[load] SGCC ...", flush=True)
        X, y = load_and_preprocess()
        raw = pd.read_csv(SGCC_CSV).drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
        nk = (raw.shape[1] // 7) * 7
        g, m = gap_magnitude(raw[:, :nk].reshape(len(y), nk // 7, 7))
        _data_cache["data"] = (X, y, g, m)
    return _data_cache["data"]

_fold_cache = {}
def recon_fold(fold, kf=0):
    if (fold, kf) in _fold_cache: return _fold_cache[(fold, kf)]
    X, y, g, m = _get_data()
    folds = list(KFoldSpec(family="random", n_folds=5, seed=kf).iter_folds(y))
    tr, val = folds[fold]
    _, _, te = _stratified_three_way_split(val, y, fractions=(0.33, 0.33, 0.34),
                                           rng=np.random.default_rng(kf * 7 + fold))
    ph = assign(g[te], m[te], fit_router(g[tr], m[tr]))
    _fold_cache[(fold, kf)] = (te, ph); return te, ph

def three_group(ph):
    """Collapse 5 phenotypes -> {high, low, flat} exactly as sec:trust does."""
    grp = np.empty(len(ph), dtype=object)
    for i, p in enumerate(ph):
        grp[i] = "high" if p in HEAVY else ("low" if p in LIGHT else "flat")
    return grp

def main():
    X, y, g, m = _get_data()
    rows = []
    for arm, (patt, seeds, folds) in ARMS.items():
        for a in ALPHAS:
            cond_fracs, single_fracs, reductions = [], [], []
            rec_high, rec_low, rec_flat = [], [], []   # realized per-group recall under conditional
            for s in seeds:
                for f in folds:
                    fp = Path(patt.format(s=s, f=f))
                    if not fp.exists(): continue
                    r = pd.read_csv(fp).iloc[0]
                    yt = np.array(json.loads(r["val_labels_json"]))
                    st = np.array(json.loads(r["val_scores_json"]))
                    te, ph = recon_fold(int(r["fold"]))
                    assert np.array_equal(yt, y[te]), f"label mismatch {arm} {s} {f}"
                    grp = three_group(ph)
                    cc, sc, rd = [], [], []
                    rh, rl, rfl = [], [], []
                    for rep in range(NREP):
                        rng = np.random.default_rng((_ARM_ORDER[arm] * 9973 + int(s) * 97 + int(f) * 7 + rep))
                        ci, ti = stratified_half(yt, rng)
                        sc_ci, sc_ti = st[ci], st[ti]; y_ci, y_ti = yt[ci], yt[ti]
                        gp_ci, gp_ti = grp[ci], grp[ti]
                        thr = {}
                        for gname in ("high", "low", "flat"):
                            cm = gp_ci == gname
                            thr[gname] = cc_recall_threshold(sc_ci[cm], y_ci[cm], a)
                        if any(thr[gn] is None for gn in thr):  # need all 3 groups certifiable
                            continue
                        # conditional: per-group threshold
                        flag_cond = np.array([sc_ti[i] >= thr[gp_ti[i]] for i in range(len(ti))])
                        thr_star = min(thr.values())            # single threshold honouring all groups
                        flag_single = sc_ti >= thr_star
                        cc.append(flag_cond.mean()); sc.append(flag_single.mean())
                        rd.append(1.0 - flag_cond.mean() / max(flag_single.mean(), 1e-9))
                        for gname, store in (("high", rh), ("low", rl), ("flat", rfl)):
                            gm = (gp_ti == gname) & (y_ti == 1)
                            if gm.sum() > 0: store.append(flag_cond[gm].mean())
                    if cc:
                        cond_fracs.append(np.mean(cc)); single_fracs.append(np.mean(sc)); reductions.append(np.mean(rd))
                        if rh: rec_high.append(np.mean(rh))
                        if rl: rec_low.append(np.mean(rl))
                        if rfl: rec_flat.append(np.mean(rfl))
            if not reductions:
                print(f"  {arm:11s} a={a}: no certifiable cells"); continue
            rlo, rhi = bca_ci(reductions)
            rows.append(dict(arm=arm, guarantee=round(1 - a, 2), n_cells=len(reductions),
                             audit_cond=np.mean(cond_fracs), audit_single=np.mean(single_fracs),
                             reduction=np.mean(reductions), red_lo=rlo, red_hi=rhi,
                             rec_high=np.mean(rec_high) if rec_high else np.nan,
                             rec_low=np.mean(rec_low) if rec_low else np.nan,
                             rec_flat=np.mean(rec_flat) if rec_flat else np.nan))
            print(f"  {arm:11s} guar={1-a:.2f} n={len(reductions):2d} | audit cond={np.mean(cond_fracs):.3f} "
                  f"single={np.mean(single_fracs):.3f} | REDUCTION={np.mean(reductions)*100:4.1f}% "
                  f"[{rlo*100:.1f},{rhi*100:.1f}] | realized recall H/L/F="
                  f"{np.mean(rec_high) if rec_high else float('nan'):.2f}/"
                  f"{np.mean(rec_low) if rec_low else float('nan'):.2f}/"
                  f"{np.mean(rec_flat) if rec_flat else float('nan'):.2f}")
    pd.DataFrame(rows).to_csv(OUT / "decision_value_sgcc.csv", index=False)
    print(f"\n[done] wrote {OUT/'decision_value_sgcc.csv'}")

if __name__ == "__main__":
    main()
