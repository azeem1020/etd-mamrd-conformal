"""ECL trustworthy-layer analysis (cross-dataset confirmation). Post-hoc from the
per-cell dumps in results/phase_n/ecl_trust/ (produced by run_ecl_trust.py).

Mirrors the SGCC layer (run_trust_layer.py) but stratifies by MAGNITUDE ONLY
(Heavy/Light by the leakage-safe TRAIN-fit median), since ECL has no calendar
phenotype. Reports: AP reproduction vs locked, T1 calibration, T2 marginal-APS
degeneracy, T3 class-conditional recall guarantee with base-rate-normalized LIFT
(Heavy vs Light), T4 recall@FPR. Same instruments + stats as SGCC.
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
from collections import defaultdict
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))                      # code/ -> sct_etd
import numpy as np, pandas as pd
from sct_etd.calibration.conformal_aps import APSCalibrator
from sct_etd.calibration.temperature import TemperatureCalibrator
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
csv.field_size_limit(10**7)

IN = (RESULTS_DIR / "ecl_trust")
ALPHAS = [0.05, 0.10, 0.20]
NREP = 20
EPS = 1e-6
ARMS = ["a34_generic", "horln", "wdcnn", "grud", "hybridattn", "msdm"]
# Deterministic, process-independent arm index for conformal seeding (see run_trust_layer.py).
_ARM_ORDER = {a: i for i, a in enumerate(ARMS)}
LOCKED_AP = {"a34_generic": 0.880, "horln": 0.852, "wdcnn": 0.836,
             "grud": 0.561, "hybridattn": 0.847, "msdm": 0.870}


def stratified_half(y, rng):
    idx = np.arange(len(y)); cal = []
    for c in (0, 1):
        ic = idx[y == c]; rng.shuffle(ic); cal.append(ic[: len(ic) // 2])
    cal = np.concatenate(cal); mask = np.zeros(len(y), bool); mask[cal] = True
    return np.where(mask)[0], np.where(~mask)[0]


def ece_brier(p, y, n_bins=15):
    p = np.clip(p, 0, 1); y = y.astype(float); brier = float(np.mean((p - y) ** 2))
    o = np.argsort(p); p, yy = p[o], y[o]; ece = 0.0; n = len(p)
    for b in np.array_split(np.arange(n), n_bins):
        if len(b): ece += (len(b) / n) * abs(p[b].mean() - yy[b].mean())
    return float(ece), brier


def cc_recall_threshold(s_cal, y_cal, alpha):
    pos = s_cal[y_cal == 1]; n = len(pos)
    if n < max(10, int(np.ceil(1.0 / alpha))): return None
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    tau = 1.0 if k > n else float(np.sort(1.0 - pos)[k - 1])
    return 1.0 - tau


def recall_at_fpr(y, s, cap):
    neg, pos = s[y == 0], s[y == 1]
    if not len(neg) or not len(pos): return float("nan")
    return float((pos > np.quantile(neg, 1 - cap)).mean())


def bca_ci(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3: return (float("nan"), float("nan"))
    from scipy.stats import norm
    rng = np.random.default_rng(seed); th = x.mean()
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((boot < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.delete(x, i).mean() for i in range(n)]); jb = jk.mean()
    den = 6.0 * ((jb - jk) ** 2).sum() ** 1.5; a = ((jb - jk) ** 3).sum() / den if den else 0.0
    def adj(qq):
        z = norm.ppf(qq); return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boot, [adj(alpha / 2), adj(1 - alpha / 2)])
    return float(lo), float(hi)


def load_cells(arm):
    out = []
    for fp in sorted(IN.glob(f"{arm}_*.csv")):
        r = next(csv.DictReader(open(fp)))
        out.append((int(r["seed"]), int(r["fold"]),
                    np.array(json.loads(r["val_labels_json"]), int),
                    np.clip(np.array(json.loads(r["val_scores_json"]), float), EPS, 1 - EPS),
                    np.array(json.loads(r["val_mag_json"]), float),
                    float(r["mag_median_tr"]), float(r["ap"])))
    return out


# population base-rate per magnitude stratum (pooled over cells of arm a34)
def main():
    rows = []; prevpos = defaultdict(int); prevn = defaultdict(int)
    for arm in ARMS:
        cells = load_cells(arm)
        if not cells:
            print(f"[skip] no cells for {arm}"); continue
        for s, f, y, sc, mag, med, ap in cells:
            heavy = mag > med
            grp = {"HEAVY": heavy, "LIGHT": ~heavy}
            if arm == "a34_generic":
                for gn, gm in grp.items():
                    prevpos[gn] += int(y[gm].sum()); prevn[gn] += int(gm.sum())
            probs = np.column_stack([1 - sc, sc]); logit = np.column_stack([np.log(1 - sc), np.log(sc)])
            accc = defaultdict(list); acc = defaultdict(lambda: defaultdict(list))
            for rep in range(NREP):
                rng = np.random.default_rng(((_ARM_ORDER[arm] * 97 + s * 31 + f) * 23 + rep) & 0xFFFFFFFF)
                ci, ti = stratified_half(y, rng)
                T = TemperatureCalibrator().fit(logit[ci], y[ci])
                e0, b0 = ece_brier(sc[ti], y[ti]); e1, b1 = ece_brier(T.transform(logit[ti])[:, 1], y[ti])
                accc["ece_raw"].append(e0); accc["ece_cal"].append(e1); accc["brier_raw"].append(b0)
                gci = {"HEAVY": heavy[ci], "LIGHT": ~heavy[ci]}
                gti = {"HEAVY": heavy[ti], "LIGHT": ~heavy[ti]}
                for a in ALPHAS:
                    cal = APSCalibrator(alpha=a).fit(probs[ci], y[ci])
                    ind = cal.predict_sets_indicator(probs[ti])
                    cov = np.where(y[ti] == 1, ind[:, 1], ind[:, 0]); abst = ind[:, 0] & ind[:, 1]
                    acc[a]["marg_cov"].append(float(cov.mean())); acc[a]["marg_abst"].append(float(abst.mean()))
                    for gn in ("HEAVY", "LIGHT"):
                        thr = cc_recall_threshold(sc[ci][gci[gn]], y[ci][gci[gn]], a)
                        tm = gti[gn]
                        if thr is None or tm.sum() == 0: continue
                        fl = sc[ti][tm] >= thr; yy = y[ti][tm]
                        acc[a][f"prec_{gn}"].append(float(yy[fl].mean()) if fl.sum() else np.nan)
                        acc[a][f"erec_{gn}"].append(float(fl[yy == 1].mean()) if (yy == 1).sum() else np.nan)
            base = dict(arm=arm, seed=s, fold=f, ap=ap,
                        ece_raw=np.mean(accc["ece_raw"]), ece_cal=np.mean(accc["ece_cal"]),
                        brier=np.mean(accc["brier_raw"]),
                        recall_fpr01=recall_at_fpr(y, sc, 0.01), recall_fpr05=recall_at_fpr(y, sc, 0.05))
            for a in ALPHAS:
                for k, v in acc[a].items():
                    base[f"a{a}_{k}"] = float(np.nanmean(v))
            rows.append(base)
    df = pd.DataFrame(rows); df.to_csv(IN / "ecl_trust_percell.csv", index=False)
    PREV = {g: prevpos[g] / prevn[g] for g in prevn}

    L = ["# RESULTS_ECL_TRUST — cross-dataset trustworthy layer (ECL, magnitude strata)",
         "", f"Strata: Heavy/Light by train-fit magnitude median. Base-rate Heavy "
         f"{100*PREV.get('HEAVY',float('nan')):.1f}% Light {100*PREV.get('LIGHT',float('nan')):.1f}%.",
         f"Cells/arm = {len(load_cells('a34_generic'))} (3 folds x 3 seeds). NREP={NREP}.", ""]
    L += ["## AP reproduction (sanity vs locked fold-0 numbers)",
          "| arm | mean AP (3x3) | locked AP |", "|---|---|---|"]
    for arm in ARMS:
        d = df[df.arm == arm]
        if len(d): L.append(f"| {arm} | {d.ap.mean():.3f} | {LOCKED_AP.get(arm,'-')} |")
    L += ["", "## T1 Calibration (preliminary)", "| arm | ECE raw | Brier |", "|---|---|---|"]
    for arm in ARMS:
        d = df[df.arm == arm]
        if len(d): L.append(f"| {arm} | {d.ece_raw.mean():.3f} | {d.brier.mean():.3f} |")
    L += ["", "## T2 Marginal-APS degeneracy (abstain-rate @1-a=0.90)", "| arm | coverage | abstain |", "|---|---|---|"]
    for arm in ARMS:
        d = df[df.arm == arm]
        if len(d) and "a0.1_marg_cov" in d: L.append(f"| {arm} | {d['a0.1_marg_cov'].mean():.3f} | {d['a0.1_marg_abst'].mean():.3f} |")
    L += ["", "## T3 Magnitude-conditional audit efficiency (LIFT) @ guaranteed 80% catch-rate — THE FINDING",
          "Lift = precision / stratum base-rate. Gap>0 = Heavy more audit-efficient than Light.",
          "| arm | lift Heavy | lift Light | Heavy-Light lift gap [BCa 95%] |", "|---|---|---|---|"]
    for arm in ARMS:
        d = df[df.arm == arm]
        ph, pl = f"a0.2_prec_HEAVY", f"a0.2_prec_LIGHT"
        if len(d) and ph in d and pl in d and PREV.get("HEAVY") and PREV.get("LIGHT"):
            lH = d.set_index(["seed", "fold"])[ph] / PREV["HEAVY"]
            lL = d.set_index(["seed", "fold"])[pl] / PREV["LIGHT"]
            gaps = (lH - lL).values; lo, hi = bca_ci(gaps, seed=1)
            L.append(f"| {arm} | {lH.mean():.2f} | {lL.mean():.2f} | {np.nanmean(gaps):+.2f} [{lo:+.2f},{hi:+.2f}] |")
    L += ["", "## T4 recall@FPR (supporting)", "| arm | recall@FPR1% | recall@FPR5% |", "|---|---|---|"]
    for arm in ARMS:
        d = df[df.arm == arm]
        if len(d): L.append(f"| {arm} | {d.recall_fpr01.mean():.3f} | {d.recall_fpr05.mean():.3f} |")
    (IN / "RESULTS_ECL_TRUST.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\n[done] {IN}/RESULTS_ECL_TRUST.md")


if __name__ == "__main__":
    main()
