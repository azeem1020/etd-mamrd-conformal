"""Decision value of the consumption-conditional (Mondrian) conformal audit — Table 6, Fig. 5b.

Post-hoc, no retraining. Reads the saved held-out test-fold scores in RESULTS_DIR
(results/phase_n_retrained/) and applies the shared protocol in conformal_common.py.

Question: to guarantee a marginal theft recall of 1-alpha in EVERY consumption group
(high / low / flat), how many consumers must be inspected under
  (i)  group-conditional conformal thresholds (one per group), vs
  (ii) a single threshold that honours the same per-group guarantee, i.e. the most
       lenient group threshold thr* = min_g thr_g?
Workload reduction = 1 - |(i)| / |(ii)|.

Every cell (seed x fold) contributes all NREP = 20 stratified 50/50 resplits of its
test fold. There is no eligibility gate: a group with too few calibration thefts for
the order statistic (n < ceil((1-alpha)/alpha)) receives q_hat = +inf, i.e. every
member is inspected (see conformal_common.py). How often that happens is reported per
group ("trivial_*" columns) and its cost is included in the workload.

All six Table 6 columns come from this script on identical resplits:
  hit rate (precision) in High / Flat, lift = hit rate / group base rate pi_g,
  inspected % (conditional / single), workload reduction [cell-clustered BCa 95 %].
Estimand: mean over the 20 resplits within a cell, then mean over cells (balanced
design, identical to the pooled mean over all resplits — asserted below). Intervals
resample cells (seed-fold units); because seeds within a fold share test consumers
and folds share training data, they are descriptive, not independent-sample CIs.

Outputs (RESULTS_DIR/trust/):
  decision_value_sgcc.csv          one row per arm x guarantee
  decision_value_sgcc_resplits.csv one row per arm x guarantee x seed x fold x resplit
  group_profile_sgcc.csv           theft-case and metered-kWh shares per group

Run (CPU, minutes):  agy-py scripts/run_decision_value.py
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, pandas as pd
from run_local_grid import load_and_preprocess, _stratified_three_way_split, SGCC_CSV
from zkmix_lab.data import KFoldSpec
from phased.phenotype import gap_magnitude, fit_router, assign
from repro_config import RESULTS_DIR
from conformal_common import (NREP, n_min, resplit_rng, stratified_half, cc_recall_threshold,
                              bca_ci, cell_summary, cell_bca)

csv.field_size_limit(10**7)
RES = RESULTS_DIR
OUT = RES / "trust"; OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.20, 0.10, 0.05]                 # target marginal recall 0.80 (primary), 0.90, 0.95
GROUPS = ("high", "low", "flat")
HEAVY = ("Heavy_Riser", "Heavy_Dropper")
LIGHT = ("Light_Riser", "Light_Dropper")
ARMS = {  # same order as run_trust_layer.py -> identical resplit seeds
    "a34_NaME":   (str(RES / "a34_cell_{s}_{f}.csv"), range(5), range(3)),
    "HORLN":      (str(RES / "baselines/horln_{s}_{f}.csv"), range(3), range(3)),
    "MSDM":       (str(RES / "baselines/msdm_net_{s}_{f}.csv"), range(3), range(3)),
    "WDCNN":      (str(RES / "baselines/wdcnn_{s}_{f}.csv"), range(3), range(3)),
    "GRU-D":      (str(RES / "baselines/grud_{s}_{f}.csv"), range(3), range(3)),
    "HybridAttn": (str(RES / "baselines/hybridattn_{s}_{f}.csv"), range(3), range(3)),
    "DKGT":       (str(RES / "baselines/dkgt_{s}_{f}.csv"), range(3), range(3)),
}
_ARM_ORDER = {a: i for i, a in enumerate(ARMS)}

# ---- data + folds (identical to run_trust_layer.py) ----
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
    """Collapse 5 phenotypes -> {high, low, flat} (high = Heavy_Riser + Heavy_Dropper)."""
    grp = np.empty(len(ph), dtype=object)
    for i, p in enumerate(ph):
        grp[i] = "high" if p in HEAVY else ("low" if p in LIGHT else "flat")
    return grp

def group_base_rates():
    """Theft prevalence per group on the SGCC test folds 0-2 (consumer-pooled)."""
    _, y, _, _ = _get_data()
    pos = {gn: 0 for gn in GROUPS}; tot = {gn: 0 for gn in GROUPS}
    for f in range(3):
        te, ph = recon_fold(f); grp = three_group(ph); yt = y[te]
        for gn in GROUPS:
            m = grp == gn; pos[gn] += int(yt[m].sum()); tot[gn] += int(m.sum())
    return {gn: pos[gn] / tot[gn] for gn in GROUPS}


def group_profile():
    """Where thefts and metered consumption sit across the three groups (test folds 0-2).

    Reports, among theft-labelled consumers: share of cases, share of metered kWh
    (mean daily kWh summed over consumers) and median mean-daily kWh; plus the median
    for all consumers. SGCC records post-tampering metered consumption, not stolen
    energy, so these describe exposure, not recovered revenue.
    """
    _, y, _, _ = _get_data()
    raw = pd.read_csv(SGCC_CSV).drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    daily = np.nanmean(raw, axis=1)
    parts = []
    for f in range(3):
        te, ph = recon_fold(f)
        parts.append(pd.DataFrame(dict(group=three_group(ph), y=y[te], kwh=daily[te])))
    d = pd.concat(parts, ignore_index=True)
    t = d[d.y == 1]
    out = pd.DataFrame({
        "theft_cases": t.groupby("group").size(),
        "theft_case_share": t.groupby("group").size() / len(t),
        "theft_kwh_share": t.groupby("group").kwh.sum() / t.kwh.sum(),
        "theft_median_daily_kwh": t.groupby("group").kwh.median(),
        "all_median_daily_kwh": d.groupby("group").kwh.median(),
    }).reindex(list(GROUPS)).reset_index().rename(columns={"index": "group"})
    return out


def evaluate_resplit(st, yt, grp, ci, ti, a):
    """All per-resplit quantities for one calibration/evaluation resplit."""
    sc_ci, sc_ti = st[ci], st[ti]; y_ci, y_ti = yt[ci], yt[ti]
    gp_ci, gp_ti = grp[ci], grp[ti]
    rec = {}
    thr = {}
    for gn in GROUPS:
        cm = gp_ci == gn
        thr[gn], trivial = cc_recall_threshold(sc_ci[cm], y_ci[cm], a)
        rec[f"ncal_{gn}"] = int(y_ci[cm].sum())
        rec[f"trivial_{gn}"] = bool(trivial)
    thr_vec = np.array([thr[g] for g in gp_ti])
    flag_cond = sc_ti >= thr_vec
    flag_single = sc_ti >= min(thr.values())
    rec["audit_cond"] = float(flag_cond.mean())
    rec["audit_single"] = float(flag_single.mean())
    rec["reduction"] = 1.0 - rec["audit_cond"] / rec["audit_single"]
    for gn in GROUPS:
        gm = gp_ti == gn
        pos = gm & (y_ti == 1)
        rec[f"neval_{gn}"] = int(pos.sum())
        rec[f"rec_{gn}"] = float(flag_cond[pos].mean()) if pos.any() else np.nan
        fl = gm & flag_cond
        rec[f"prec_{gn}"] = float(y_ti[fl].mean()) if fl.any() else np.nan
    return rec


def main():
    X, y, g, m = _get_data()
    PI = group_base_rates()
    print("[base rates] " + ", ".join(f"{k}={v:.4f}" for k, v in PI.items()), flush=True)
    prof = group_profile()
    prof.to_csv(OUT / "group_profile_sgcc.csv", index=False)
    print("[group profile]\n" + prof.to_string(index=False), flush=True)
    long_rows, rows = [], []
    for arm, (patt, seeds, folds) in ARMS.items():
        for a in ALPHAS:
            for s in seeds:
                for f in folds:
                    fp = Path(patt.format(s=s, f=f))
                    if not fp.exists():
                        raise FileNotFoundError(f"missing cell {fp}")
                    r = pd.read_csv(fp).iloc[0]
                    yt = np.array(json.loads(r["val_labels_json"]))
                    st = np.array(json.loads(r["val_scores_json"]), float)
                    te, ph = recon_fold(int(r["fold"]))
                    assert int(r["fold"]) == f and np.array_equal(yt, y[te]), f"label mismatch {arm} {s} {f}"
                    grp = three_group(ph)
                    for rep in range(NREP):
                        ci, ti = stratified_half(yt, resplit_rng(_ARM_ORDER[arm], s, f, rep))
                        rec = evaluate_resplit(st, yt, grp, ci, ti, a)
                        long_rows.append(dict(arm=arm, guarantee=round(1 - a, 2), seed=s, fold=f, rep=rep, **rec))
            d = pd.DataFrame([lr for lr in long_rows if lr["arm"] == arm and lr["guarantee"] == round(1 - a, 2)])
            cells = [grp_df for _, grp_df in d.groupby(["seed", "fold"])]
            def summ(col):
                grand, pooled, nc, nv = cell_summary([c[col].values for c in cells])
                if col in ("audit_cond", "audit_single", "reduction"):
                    assert abs(grand - pooled) < 1e-12, f"unbalanced design for {col}"
                return grand
            red_lo, red_hi = cell_bca([c["reduction"].values for c in cells], seed=0)
            fold_red = d.groupby("fold")["reduction"].mean()
            out = dict(arm=arm, guarantee=round(1 - a, 2), n_min=n_min(a),
                       n_cells=len(cells), n_resplits=len(d),
                       audit_cond=summ("audit_cond"), audit_single=summ("audit_single"),
                       reduction=summ("reduction"), red_lo=red_lo, red_hi=red_hi,
                       red_fold_min=float(fold_red.min()), red_fold_max=float(fold_red.max()))
            for gn in GROUPS:
                out[f"trivial_{gn}"] = float(d[f"trivial_{gn}"].mean())
                out[f"ncal_{gn}_mean"] = float(d[f"ncal_{gn}"].mean())
                out[f"rec_{gn}"] = summ(f"rec_{gn}")
                out[f"rec_{gn}_below_target"] = float((d[f"rec_{gn}"].dropna() < 1 - a).mean())
                out[f"prec_{gn}"] = summ(f"prec_{gn}")
                out[f"pi_{gn}"] = PI[gn]
                out[f"lift_{gn}"] = out[f"prec_{gn}"] / PI[gn]
            rows.append(out)
            print(f"  {arm:10s} 1-a={1-a:.2f} cells={out['n_cells']:2d} resplits={out['n_resplits']:3d} "
                  f"q=inf H/L/F={out['trivial_high']:.2f}/{out['trivial_low']:.2f}/{out['trivial_flat']:.2f} | "
                  f"inspect cond={out['audit_cond']*100:5.1f}% single={out['audit_single']*100:5.1f}% "
                  f"red={out['reduction']*100:5.1f}% [{red_lo*100:.1f},{red_hi*100:.1f}] | "
                  f"hit H/F={out['prec_high']*100:5.1f}/{out['prec_flat']*100:4.1f}% "
                  f"lift H/F={out['lift_high']:.2f}/{out['lift_flat']:.2f} | "
                  f"recall H/L/F={out['rec_high']:.3f}/{out['rec_low']:.3f}/{out['rec_flat']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "decision_value_sgcc.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUT / "decision_value_sgcc_resplits.csv", index=False)
    print(f"\n[done] wrote {OUT/'decision_value_sgcc.csv'} and decision_value_sgcc_resplits.csv")


if __name__ == "__main__":
    main()
