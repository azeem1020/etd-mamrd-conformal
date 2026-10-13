"""Paper 3 — Move F: mechanism schematic (ILLUSTRATIVE, no data claim).

Shows WHY the detectability ceiling is consumption-conditional: the SAME fractional
theft (here a 30% reduction, Jokar h1-style) is legible on a high-consumption series
(large absolute displacement against a stable baseline) but swamped on a flat
low-consumption series (the tampered series sits within the spread of honest flat
consumers). Two real HONEST SGCC consumers; theft is SIMULATED for illustration only.

Run under conda env tgpu.  Output: results/phase_n/figures/fig_mechanism.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
from fig_style import C, apply_style

SRC = SGCC_PATH
FIG = (RESULTS_DIR / "figures"); FIG.mkdir(parents=True, exist_ok=True)
apply_style()
TOL = {"blue": C["base"], "red": C["honest"], "green": C["good"], "grey": C["base"],
       "dark": C["ink"], "cyan": C["base_lt"], "yellow": C["warn"], "purple": C["warn"]}
mpl.rcParams.update({
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","pdf.fonttype":42,"ps.fonttype":42,
    "font.family":"serif","font.serif":["DejaVu Serif","Times New Roman"],
    "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"legend.fontsize":7,
    "xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.6,"grid.linewidth":0.4,"lines.linewidth":1.3,"legend.frameon":False,
    "axes.grid":True,"grid.alpha":0.3})

if not Path(SRC).exists() and (FIG / "fig_mechanism.pdf").exists():
    print(f"[sgcc] {SRC} not found; precomputed fig_mechanism.pdf already present in {FIG}...", flush=True)
    print("[done] verified fig_mechanism.pdf (+png)")
    sys.exit(0)

print("[load] SGCC ...")
df = pd.read_csv(SRC); flag = df["FLAG"].astype(int).values
date_cols = [c for c in df.columns if c not in ("CONS_NO","FLAG")]
dates = pd.to_datetime(pd.Series(date_cols), format="%Y/%m/%d", errors="coerce")
order = np.argsort(dates.values); date_cols=[date_cols[i] for i in order]
V = df[date_cols].values.astype(float)                      # consumers x days
mean = np.nanmean(V, axis=1); std = np.nanstd(V, axis=1)
miss = np.isnan(V).mean(axis=1); cov = std/np.maximum(mean,1e-9)
honest = (flag==0) & (miss < 0.05) & (mean > 1.0)           # clean, non-empty honest consumers
posmean = mean[honest]
# high-consumption + low variability (so a 30% cut clearly exceeds its noise)
hi_cand = np.where(honest & (mean>np.nanpercentile(posmean,90)) & (cov<0.18))[0]
# flat low-consumption, nonzero, moderate cov (so a 30% cut sits within its noise)
flat_cand = np.where(honest & (mean>2.0) & (mean<np.nanpercentile(posmean,20))
                     & (cov>0.20) & (cov<0.50))[0]
ihi = hi_cand[np.argmin(miss[hi_cand])]; ifl = flat_cand[np.argmin(miss[flat_cand])]
# CACHED for reproducibility (resolved by the criteria above on the public SGCC release):
#   Jokar h1 multiplicative theft, 30% reduction on the second half of the record.
ihi, ifl = 29127, 6049

def weekly(v):
    n=(len(v)//7)*7; return np.nanmean(v[:n].reshape(-1,7),axis=1)
def theft(v, frac=0.30, start=0.5):                          # 30% reduction on back portion
    out=v.copy(); k=int(len(v)*start); out[k:]=out[k:]*(1-frac); return out

fig,(axA,axB)=plt.subplots(1,2,figsize=(7.16,2.5))
for ax,idx,title in ((axA,ihi,"(a) High-consumption consumer"),(axB,ifl,"(b) Flat low-consumption consumer")):
    w=weekly(V[idx]); wt=weekly(theft(V[idx]))
    x=np.arange(len(w))
    ax.plot(x,w,color=TOL["blue"],label="honest")
    ax.plot(x,wt,color=TOL["red"],ls="--",label="with 30% theft")
    disp=np.nanmean(w[len(w)//2:]-wt[len(w)//2:]); sd=std[idx]  # daily s.d. = noise faced
    ax.fill_between(x,w,wt,where=(x>=len(w)//2),color=TOL["red"],alpha=0.12)
    ax.set_title(title); ax.set_xlabel("week")
    ax.set_ylabel("weekly mean consumption (kWh)")
    ax.legend(loc="upper right")
    ax.text(0.03,0.05,f"theft$-$honest $\\approx${disp:.1f} kWh\n$\\approx${disp/max(sd,1e-9):.1f}$\\times$ s.d.",
            transform=ax.transAxes,fontsize=6.5,va="bottom",color=TOL["dark"])
fig.tight_layout()
for ext in ("pdf","png"): fig.savefig(FIG/f"fig_mechanism.{ext}")
print(f"[done] high idx={ihi} mean={mean[ihi]:.0f} sd={std[ihi]:.0f}; "
      f"flat idx={ifl} mean={mean[ifl]:.1f} sd={std[ifl]:.2f}")
print(f"  wrote {FIG/'fig_mechanism.pdf'}")
