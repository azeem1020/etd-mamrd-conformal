"""Paper 3 — Move A: model-free, representation-AGNOSTIC ceiling estimate.

PRE-REGISTRATION (dated 2026-06-05, before running):
  CLAIM under test: the 0.567 detectability ceiling is SIGNAL-limited, not MODEL-limited.
  ESTIMATOR: HistGradientBoosting on hand-crafted, model-free per-consumer features
    (mean/std/cov/min/max/quantiles/missing-rate/zero-fraction/weekday-weekend gap/
     lag-1,lag-7 autocorr/trend slope) -- NO learned/deep representation -- evaluated
     per stratum under the SAME folds as the detectors.
  PREDICTION: model-free AP (i) rank-tracks observed per-stratum detector AP (Spearman>0)
     and (ii) aggregate sits near observed 0.567.
  DECISION RULE (asymmetric, honest):
     model-free ~= deep  -> signal-limited corroborated (simple features already reach deep)
     model-free  < deep  -> inconclusive, KEEP the hedge (deep extracts more)
     model-free  > deep  -> STOP & flag (ceiling would be model-limited; detectors suboptimal)
  CAVEAT (pre-stated): model-free features are themselves a representation choice, so this is a
     LOWER bound on the true achievable; a match supports signal-limitation, a shortfall is
     inconclusive (not disproof). NOT a representation-free Bayes error.

Observed MAMRD per-stratum AP (a34, n=15): HR .664 HD .631 LR .492 LD .460 Flat .402; agg .567.
Run (env tgpu, CPU minutes):  python run_ceiling_estimate.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from run_local_grid import load_and_preprocess, _stratified_three_way_split, SGCC_CSV
from zkmix_lab.data import KFoldSpec
from phased.phenotype import gap_magnitude, fit_router, assign, PHENOTYPES

OBS = {"Heavy_Riser":0.664,"Heavy_Dropper":0.631,"Light_Riser":0.492,"Light_Dropper":0.460,"Flat":0.402}
AGG_OBS = 0.567
STR5 = list(OBS.keys())

print("[load] SGCC raw ...", flush=True)
df = pd.read_csv(SGCC_CSV); y = df["FLAG"].astype(int).values
date_cols = [c for c in df.columns if c not in ("CONS_NO","FLAG")]
dates = pd.to_datetime(pd.Series(date_cols), format="%Y/%m/%d", errors="coerce")
order = np.argsort(dates.values); date_cols = [date_cols[i] for i in order]
dow = dates.iloc[order].dt.dayofweek.values            # 0..6
V = df[date_cols].values.astype(float)                 # N x T, NaN = missing
N, T = V.shape; print(f"  {N} x {T}")

# ---- model-free per-consumer features (hand-crafted; no learned representation) ----
def feats(V):
    with np.errstate(all="ignore"):
        mean=np.nanmean(V,1); std=np.nanstd(V,1); cov=std/np.maximum(mean,1e-9)
        mn=np.nanmin(V,1); mx=np.nanmax(V,1)
        qs=np.nanpercentile(V,[10,25,50,75,90],axis=1)
        miss=np.isnan(V).mean(1); zero=np.nan_to_num((V==0).sum(1)/np.maximum((~np.isnan(V)).sum(1),1))
        wd=np.nanmean(np.where(dow<5,V,np.nan),1); we=np.nanmean(np.where(dow>=5,V,np.nan),1)
        wdwe=wd-we
        # lag-1 / lag-7 autocorr and linear trend, on each row's observed values
        Vc=V-mean[:,None]
        def autocorr(lag):
            a=Vc[:,:-lag]; b=Vc[:,lag:]; num=np.nansum(a*b,1); den=np.nansum(Vc*Vc,1)
            return num/np.maximum(den,1e-9)
        ac1=autocorr(1); ac7=autocorr(7)
        t=np.arange(T); tc=t-t.mean()
        trend=np.nansum((V-mean[:,None])*tc[None,:],1)/ (np.nansum(np.where(np.isnan(V),np.nan,tc**2),1)+1e-9)
    F=np.column_stack([mean,std,cov,mn,mx,*qs,miss,zero,wd,we,wdwe,ac1,ac7,trend])
    return np.nan_to_num(F, nan=0.0)
F = feats(V); print(f"  features: {F.shape[1]} per consumer")
g, m = gap_magnitude(V[:, :(T//7)*7].reshape(N, T//7, 7))

def recon(fold, kf=0):
    folds=list(KFoldSpec(family="random",n_folds=5,seed=kf).iter_folds(y))
    tr,val=folds[fold]
    _,_,te=_stratified_three_way_split(val,y,fractions=(0.33,0.33,0.34),rng=np.random.default_rng(kf*7+fold))
    ph=assign(g[te],m[te],fit_router(g[tr],m[tr]))
    return tr,te,ph

# ---- fit model-free GBM per fold x seed; per-stratum + aggregate AP ----
per_str={s:[] for s in STR5}; agg=[]
for fold in range(3):
    tr,te,ph=recon(fold)
    for seed in range(3):
        clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,
            max_leaf_nodes=31,l2_regularization=1.0,random_state=seed,
            class_weight="balanced",early_stopping=True,validation_fraction=0.15)
        clf.fit(F[tr],y[tr]); p=clf.predict_proba(F[te])[:,1]
        agg.append(average_precision_score(y[te],p))
        for s in STR5:
            mk=ph==s
            if mk.sum()>0 and y[te][mk].sum()>0:
                per_str[s].append(average_precision_score(y[te][mk],p[mk]))
print(f"\n=== MODEL-FREE (HistGBM on hand-crafted features), n={len(agg)} cells ===")
print(f"{'stratum':14s} {'model-free AP':>13s} {'observed (MAMRD)':>16s} {'diff':>7s}")
mf=[]; ob=[]
for s in STR5:
    v=np.mean(per_str[s]); mf.append(v); ob.append(OBS[s])
    print(f"{s:14s} {v:13.3f} {OBS[s]:16.3f} {v-OBS[s]:+7.3f}")
print(f"{'AGGREGATE':14s} {np.mean(agg):13.3f} {AGG_OBS:16.3f} {np.mean(agg)-AGG_OBS:+7.3f}")
rho,pv=spearmanr(mf,ob)
print(f"\nSpearman(model-free, observed) rho={rho:+.2f} p={pv:.3f}")
print(f"mean |per-stratum diff| = {np.mean(np.abs(np.array(mf)-np.array(ob))):.3f}")
verdict = ("model-free ~= deep -> SIGNAL-LIMITED corroborated" if abs(np.mean(agg)-AGG_OBS)<0.03 and rho>0
           else "model-free > deep -> STOP/FLAG (model-limited)" if np.mean(agg)-AGG_OBS>0.03
           else "model-free < deep -> inconclusive, KEEP HEDGE")
print("VERDICT:", verdict)
pd.DataFrame({"stratum":STR5+["AGGREGATE"],"model_free_ap":mf+[np.mean(agg)],
             "observed_ap":ob+[AGG_OBS]}).to_csv(THIS.parent.parent/"results/phase_n/trust/ceiling_estimate.csv",index=False)
print("[done] wrote results/phase_n/trust/ceiling_estimate.csv")
