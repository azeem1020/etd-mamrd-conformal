"""Script: run_empirical_comparator.py
Evaluates the empirical group-quantile baseline vs. the Mondrian conformal prediction policy
on SGCC detection tensors across seeds, folds, and repeated calibration/test resplits.

Isolates:
1. Base detector ranking efficiency.
2. Group-specific threshold flexibility.
3. Conformal prediction finite-sample coverage validity.

Uses the shared protocol in conformal_common.py on exactly the resplits of
run_decision_value.py (Table 6), so Fig. 5a and Table 6 share one population:
all 15 MAMRD cells x 20 resplits, no eligibility gate.
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPRO_DIR = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(REPRO_DIR), str(REPRO_DIR / "zkmix_lab")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_decision_value import ARMS, _ARM_ORDER, recon_fold, three_group
from conformal_common import NREP, resplit_rng, stratified_half, cc_recall_threshold
from repro_config import RESULTS_DIR

OUT = RESULTS_DIR / "trust"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.20  # Nominal target recall = 80%

def run_comparator(arm="a34_NaME", alpha=ALPHA, n_rep=NREP):
    patt, seeds, folds = ARMS[arm]
    
    emp_inspected, conf_inspected = [], []
    emp_rec_h, conf_rec_h = [], []
    emp_rec_l, conf_rec_l = [], []
    emp_rec_f, conf_rec_f = [], []
    emp_rec_all, conf_rec_all = [], []
    emp_viol_f, conf_viol_f = [], []
    trivial_f = []
    
    for s in seeds:
        for f in folds:
            fp = Path(patt.format(s=s, f=f))
            if not fp.exists():
                raise FileNotFoundError(f"missing cell {fp}")
            r = pd.read_csv(fp).iloc[0]
            yt = np.array(json.loads(r["val_labels_json"]))
            st = np.array(json.loads(r["val_scores_json"]))
            te, ph = recon_fold(int(r["fold"]))
            grp = three_group(ph)
            
            # Same resplits as run_decision_value.py / run_trust_layer.py (conformal_common.resplit_rng).
            # No eligibility gate: every resplit is evaluated. A group whose calibration thefts are too
            # few for the conformal order statistic gets q_hat = +inf (flag all members); the empirical
            # comparator uses the plain empirical (1-alpha) quantile of the calibration thefts, which is
            # defined for n >= 1 (n = 0 -> flag all members, the same convention).
            for rep in range(n_rep):
                ci, ti = stratified_half(yt, resplit_rng(_ARM_ORDER[arm], s, f, rep))
                sc_ci, sc_ti = st[ci], st[ti]
                y_ci, y_ti = yt[ci], yt[ti]
                gp_ci, gp_ti = grp[ci], grp[ti]

                thr_conf, thr_emp = {}, {}
                for gname in ("high", "low", "flat"):
                    cm = gp_ci == gname
                    thr_conf[gname], trivial = cc_recall_threshold(sc_ci[cm], y_ci[cm], alpha)
                    if gname == "flat":
                        trivial_f.append(trivial)
                    pos = sc_ci[cm & (y_ci == 1)]
                    thr_emp[gname] = (1.0 - float(np.quantile(1.0 - pos, 1.0 - alpha))) if len(pos) else 0.0
                
                flag_c = np.array([sc_ti[i] >= thr_conf[gp_ti[i]] for i in range(len(ti))])
                flag_e = np.array([sc_ti[i] >= thr_emp[gp_ti[i]] for i in range(len(ti))])
                
                conf_inspected.append(flag_c.mean())
                emp_inspected.append(flag_e.mean())
                
                for gn, (el, cl) in [("high", (emp_rec_h, conf_rec_h)),
                                     ("low", (emp_rec_l, conf_rec_l)),
                                     ("flat", (emp_rec_f, conf_rec_f))]:
                    gm = (gp_ti == gn) & (y_ti == 1)
                    if gm.sum() > 0:
                        er = flag_e[gm].mean()
                        cr = flag_c[gm].mean()
                        el.append(er)
                        cl.append(cr)
                        if gn == "flat":
                            emp_viol_f.append(er < (1 - alpha))
                            conf_viol_f.append(cr < (1 - alpha))
                
                gm_all = (y_ti == 1)
                emp_rec_all.append(flag_e[gm_all].mean())
                conf_rec_all.append(flag_c[gm_all].mean())
                
    summary = {
        "arm": arm,
        "n_resplits": len(conf_inspected),
        "conf_flat_trivial_pct": float(np.mean(trivial_f) * 100),
        "conf_inspected_mean": float(np.mean(conf_inspected)),
        "conf_inspected_lo": float(np.percentile(conf_inspected, 2.5)),
        "conf_inspected_hi": float(np.percentile(conf_inspected, 97.5)),
        "emp_inspected_mean": float(np.mean(emp_inspected)),
        "emp_inspected_lo": float(np.percentile(emp_inspected, 2.5)),
        "emp_inspected_hi": float(np.percentile(emp_inspected, 97.5)),
        "conf_rec_flat_mean": float(np.mean(conf_rec_f)),
        "emp_rec_flat_mean": float(np.mean(emp_rec_f)),
        "conf_flat_viol_pct": float(np.mean(conf_viol_f) * 100),
        "emp_flat_viol_pct": float(np.mean(emp_viol_f) * 100),
        "conf_rec_heavy_mean": float(np.mean(conf_rec_h)),
        "emp_rec_heavy_mean": float(np.mean(emp_rec_h)),
        "conf_rec_light_mean": float(np.mean(conf_rec_l)),
        "emp_rec_light_mean": float(np.mean(emp_rec_l)),
        "conf_rec_overall_mean": float(np.mean(conf_rec_all)),
        "emp_rec_overall_mean": float(np.mean(emp_rec_all)),
        "n_flat_eval_resplits": len(conf_rec_f),
    }
    return summary

if __name__ == "__main__":
    print("[running] Empirical Comparator Benchmark on MAMRD...")
    res = run_comparator("a34_NaME", alpha=0.20, n_rep=20)
    df = pd.DataFrame([res])
    out_csv = OUT / "empirical_comparator_sgcc.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] Saved to {out_csv}")
    print(df.to_string())
