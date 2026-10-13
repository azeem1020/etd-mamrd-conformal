"""Architecture diagram for the MAMRD detector (Paper 3, Energy and AI).

Six branches fused into a classification head, plus the post-hoc class-conditional
conformal audit layer. Vector PDF, Tol palette, serif — matches make_figures.py.
Names branches by function; scientific grounding goes in the caption.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

FIG = (RESULTS_DIR / "figures"); FIG.mkdir(exist_ok=True)
TOL = {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833", "yellow": "#CCBB44",
       "red": "#EE6677", "purple": "#AA3377", "grey": "#BBBBBB", "dark": "#222222",
       "lblue": "#CFE3F2", "lgreen": "#D5E8D4", "lyellow": "#FCF3CF", "lpurple": "#EAD9F0",
       "lgrey": "#EEEEEE", "lred": "#FADBD8"}
mpl.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"], "font.size": 8,
})
DOUBLE = 7.16
fig, ax = plt.subplots(figsize=(DOUBLE, 3.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, text, fc, ec=TOL["dark"], fs=7.0, lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=3)

def arrow(x1, y1, x2, y2, ls="-", c=TOL["dark"], lw=0.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                 lw=lw, color=c, ls=ls, shrinkA=1, shrinkB=1, zorder=1))

# --- input ---
box(1, 23, 13, 10, "Smart-meter\nseries $\\mathbf{x}_i$\n+ mask $\\mathbf{m}_i$", TOL["lgrey"], fs=7)
# --- context-conditional missing-value front end ---
box(18, 24.5, 14, 7, "context-cond.\nmissing-value\nembedding", TOL["lred"], fs=6.6)

# --- six branches (value branches fed by the embedding; mask branch fed by mask) ---
bx, bw, bh = 38, 23, 6.2
ys = [47, 38.5, 30, 21.5, 13]
blab = [
    ("BiMamba2 state-space backbone", TOL["lblue"]),
    ("Multi-scale dilated convolution", TOL["lgreen"]),
    ("Multi-scale differencing", TOL["lyellow"]),
    ("Inter/intra-period convolution", TOL["lpurple"]),
    ("Mask-channel convolution", TOL["lred"]),
]
for (lab, fc), y in zip(blab, ys):
    box(bx, y, bw, bh, lab, fc, fs=6.8)

# --- fusion + head + score ---
box(66, 27, 13, 8, "Concat\n+ MLP\nhead", TOL["lgrey"], fs=6.8)
box(83, 28, 14, 6, "suspicion\nscore $s_i$", TOL["lgrey"], fs=6.8)
# --- conformal audit (post-hoc, dashed) ---
box(64, 5, 33, 9, "Class-conditional conformal audit\nguaranteed per-stratum detection rate $1-\\alpha$",
    TOL["lyellow"], ec=TOL["purple"], lw=1.0, ls="--", fs=6.5)

# arrows: input -> embedding; embedding -> value branches (top 4); input(mask) -> mask branch
arrow(14, 28, 18, 28)
for y in ys[:4]:
    arrow(32, 28, bx, y + bh/2, c=TOL["red"])
arrow(14, 25, bx, ys[4] + bh/2, c=TOL["dark"])   # mask path straight to mask branch
# branches -> fusion
for y in ys:
    arrow(bx + bw, y + bh/2, 66, 31)
arrow(79, 31, 83, 31)                     # head -> score
arrow(90, 28, 90, 14, ls="--", c=TOL["purple"])  # score -> conformal (post-hoc)

ax.text(90, 53, "detector (trained end-to-end)", ha="center", fontsize=6.5, style="italic", color=TOL["dark"])
ax.text(91.5, 20.5, "post-hoc\n(no retraining)", ha="left", fontsize=5.8, style="italic", color=TOL["purple"])

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig0_architecture.{ext}")
print(f"[done] wrote {FIG}/fig0_architecture.pdf (+png)")
