import json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Load extracted vector data
SCRIPTS_DIR = Path(__file__).resolve().parent
REPRO_DIR = SCRIPTS_DIR.parent
json_path = REPRO_DIR / "results" / "phase_n" / "figures" / "curves_dump.json"
if not json_path.exists():
    json_path = REPRO_DIR.parent / "figures" / "curves_dump.json"

with open(json_path) as f:
    data = json.load(f)

# Styles
styles_pr = {
    "MAMRD":      {"color": "#1B5E94", "lw": 2.2, "ls": "-",   "zorder": 6, "label": "MAMRD (0.57)"},
    "HORLN":      {"color": "#D33F23", "lw": 1.4, "ls": (0, (5, 2)), "zorder": 5, "label": "HORLN (0.50)"},
    "MSDM":       {"color": "#6E388C", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2)), "zorder": 4, "label": "MSDM (0.47)"},
    "WDCNN":      {"color": "#C97A10", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2, 1.5, 2)), "zorder": 3, "label": "WDCNN (0.34)"},
    "Self-attn.": {"color": "#207D53", "lw": 1.5, "ls": (0, (1.5, 2)), "zorder": 2, "label": "Self-attn. (0.21)"},
    "GRU-D":      {"color": "#2D3436", "lw": 1.3, "ls": (0, (3, 2, 1.5, 2)), "zorder": 1, "label": "GRU-D (0.20)"},
}

styles_roc = {
    "MAMRD":      {"color": "#1B5E94", "lw": 2.2, "ls": "-",   "zorder": 6, "label": "MAMRD (0.88)"},
    "HORLN":      {"color": "#D33F23", "lw": 1.4, "ls": (0, (5, 2)), "zorder": 5, "label": "HORLN (0.86)"},
    "MSDM":       {"color": "#6E388C", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2)), "zorder": 4, "label": "MSDM (0.84)"},
    "WDCNN":      {"color": "#C97A10", "lw": 1.4, "ls": (0, (5, 2, 1.5, 2, 1.5, 2)), "zorder": 3, "label": "WDCNN (0.80)"},
    "Self-attn.": {"color": "#207D53", "lw": 1.5, "ls": (0, (1.5, 2)), "zorder": 2, "label": "Self-attn. (0.72)"},
    "GRU-D":      {"color": "#2D3436", "lw": 1.3, "ls": (0, (3, 2, 1.5, 2)), "zorder": 1, "label": "GRU-D (0.67)"},
}

order = ["GRU-D", "WDCNN", "Self-attn.", "MSDM", "HORLN", "MAMRD"]

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E8EC",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
})

fig, (axP, axR) = plt.subplots(1, 2, figsize=(7.16, 2.8), layout="constrained")

# Panel A: Precision-Recall
for name in order:
    c = data[name]
    st = styles_pr[name]
    axP.plot(c["pr_x"], c["pr_y"], color=st["color"], lw=st["lw"], ls=st["ls"],
             label=st["label"], zorder=st["zorder"])

axP.axhline(0.0853, ls=":", lw=0.9, color="#636E72", alpha=0.8, label="Base rate (0.085)", zorder=0)
axP.set_xlim(-0.02, 1.02)
axP.set_ylim(-0.02, 1.02)
axP.set_xlabel("Recall")
axP.set_ylabel("Precision")
axP.set_title("(a) Precision–Recall (AP)")
axP.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DCE1E7", framealpha=0.95)

# Panel B: ROC
for name in order:
    c = data[name]
    st = styles_roc[name]
    axR.plot(c["roc_x"], c["roc_y"], color=st["color"], lw=st["lw"], ls=st["ls"],
             label=st["label"], zorder=st["zorder"])

axR.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#636E72", alpha=0.7, label="Chance (0.50)", zorder=0)
axR.set_xlim(-0.02, 1.02)
axR.set_ylim(-0.02, 1.02)
axR.set_xlabel("False Positive Rate")
axR.set_ylabel("True Positive Rate")
axR.set_title("(b) ROC (AUC)")
axR.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DCE1E7", framealpha=0.95)

# Save to destination paths
destinations = [
    REPRO_DIR / "results" / "phase_n" / "figures" / "fig2_performance_curves.pdf",
    REPRO_DIR / "results" / "phase_n" / "figures" / "fig2_performance_curves.png",
    REPRO_DIR.parent / "figures" / "fig2_performance_curves.pdf",
    REPRO_DIR.parent / "figures" / "fig2_performance_curves.png",
    REPRO_DIR.parent / "submission_package" / "figures" / "fig2_performance_curves.pdf",
]

for dst in destinations:
    if dst.parent.exists():
        fig.savefig(dst, dpi=300 if dst.suffix == '.png' else None)
        print(f"Saved: {dst}")

plt.close(fig)
