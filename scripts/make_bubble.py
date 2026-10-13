"""Pilot: senior-grade detector-comparison bubble (SGCC).
x = ROC-AUC (prevalence-insensitive), y = full-prevalence PR-AUC,
bubble area = MAP@100. MAMRD = accent; baselines = neutral. Replaces a dense table.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from fig_style import C, MODEL_COLOR, MODEL_NAME, apply_style, SINGLE
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH
apply_style()

RES = RESULTS_DIR
ARMS = {"ours":(str(RES/"a34_cell_{s}_{f}.csv"),range(5),range(3))}
for b in ["horln","msdm_net","wdcnn","grud","hybridattn","dkgt"]:
    ARMS[b]=(str(RES/f"baselines/{b}_{{s}}_{{f}}.csv"),range(3),range(3))

def mapk(y,s,K=100):
    o=np.argsort(-s); y=y[o][:K]
    if y.sum()==0: return np.nan
    p=np.cumsum(y)/(np.arange(len(y))+1); return p[y==1].mean()

rows=[]
for arm,(patt,seeds,folds) in ARMS.items():
    pr,roc,mp=[],[],[]
    for s in seeds:
        for f in folds:
            fp=Path(patt.format(s=s,f=f))
            if not fp.exists(): continue
            r=pd.read_csv(fp).iloc[0]
            y=np.array(json.loads(r["val_labels_json"])); sc=np.array(json.loads(r["val_scores_json"]))
            pr.append(r["pr_auc"]); roc.append(r["roc_auc"]); mp.append(mapk(y,sc,100))
    key="msdm_net" if arm=="msdm_net" else arm
    rows.append(dict(arm=arm,name=MODEL_NAME[arm],pr=np.mean(pr),roc=np.mean(roc),
                     mp=np.nanmean(mp),color=MODEL_COLOR[arm]))
df=pd.DataFrame(rows)

fig,ax=plt.subplots(figsize=(SINGLE+0.4,3.1))
def area(m): return 70+(m-0.40)/0.55*660          # MAP 0.40..0.95 -> bubble area
bl=df[df.arm!="ours"]; me=df[df.arm=="ours"]
ax.scatter(bl["roc"].to_numpy(),bl["pr"].to_numpy(),s=[area(m) for m in bl["mp"]],
           c=list(bl["color"]),edgecolor=C["ink"],linewidth=0.5,alpha=0.85,zorder=2)
ax.scatter(me["roc"].to_numpy(),me["pr"].to_numpy(),s=[area(m) for m in me["mp"]],
           c=list(me["color"]),edgecolor=C["ink"],linewidth=1.2,alpha=0.95,zorder=3)
# labels (offset, leader-free for clarity)
off={"MAMRD":(0.009,0.0,"left"),"HORLN":(0.010,0.004,"left"),"MSDM":(-0.010,0.004,"right"),
     "WDCNN":(0.010,0.0,"left"),"DKGT-Net":(-0.010,-0.002,"right"),
     "Self-attn":(0.0,-0.028,"center"),"GRU-D":(-0.010,0.0,"right")}
for _,r in df.iterrows():
    dx,dy,ha=off.get(r["name"],(0.009,0.0,"left"))
    lab=r"$\bf{MAMRD}$ (ours)" if r["arm"]=="ours" else r["name"]
    ax.annotate(lab,(r["roc"],r["pr"]),xytext=(r["roc"]+dx,r["pr"]+dy),
                fontsize=7.2 if r["arm"]=="ours" else 6.6,ha=ha,
                color=C["ours"] if r["arm"]=="ours" else C["ink"],va="center")
ax.axhline(0.50,ls=(0,(4,3)),lw=0.7,color=C["base"]); ax.text(0.665,0.505,
    "  no baseline $>0.50$",fontsize=6,color=C["base"],va="bottom")
ax.set_xlabel("ROC-AUC  (prevalence-insensitive)")
ax.set_ylabel("full-prevalence PR-AUC")
ax.set_xlim(0.63,0.935); ax.set_ylim(0.15,0.62)
# bubble-size legend (MAP@100) via proxy handles
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',linestyle='None',markerfacecolor=C["base"],
                markeredgecolor=C["ink"],markeredgewidth=0.5,
                markersize=np.sqrt(area(mv))*0.55,label=f"{mv:.2f}") for mv in (0.45,0.65,0.85)]
leg=ax.legend(handles=handles,title="MAP@100",loc="lower right",labelspacing=1.4,
              borderpad=0.9,handletextpad=1.0,fontsize=6.5,title_fontsize=6.5)
fig.tight_layout()
FIG=RES/"figures"
for ext in ("pdf","png"): fig.savefig(FIG/f"fig_bubble.{ext}")
print("[done] per-detector:");
for _,r in df.iterrows(): print(f"  {r['name']:12s} ROC={r['roc']:.3f} PR={r['pr']:.3f} MAP@100={r['mp']:.3f}")
print(f"  wrote {FIG}/fig_bubble.pdf")
