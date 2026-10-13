"""Shared senior-grade figure style for Paper 3 (Energy & AI).

ONE palette, ONE rcParams, color = meaning, fixed across every figure.
Import in every make_*.py:  from fig_style import C, MODEL_COLOR, MODEL_NAME, strata_ramp, apply_style, SINGLE, DOUBLE

Design rules (the "senior, not childish" contract):
  - baselines are a restrained NEUTRAL grey ramp; MAMRD is the lone saturated accent.
  - strata use a single-hue SEQUENTIAL ramp (heavy = dark -> flat = light) = "ordered".
  - no long in-figure titles (the caption carries the description); short axis labels only.
  - muted, colourblind-safe; thin spines (top/right off); subtle grid below the data.
"""
import matplotlib as mpl

C = dict(
    ink     = "#1F2430",   # text / spines / arrows
    ours    = "#2A5784",   # MAMRD — the single strong accent
    ours_lt = "#A9C2DA",   # MAMRD light fill
    honest  = "#B23A48",   # "honest / headline" contrast (muted brick)
    good    = "#3F7A57",   # conditional / good (trust)
    warn    = "#C58A2E",   # secondary warm accent (use sparingly)
    base    = "#8A929B",   # baselines — neutral
    base_lt = "#CFD3D8",   # inflated / easy metrics — light neutral
    grid    = "#DEE1E5",
)

# baselines = graded neutral (light->dark by strength); MAMRD = accent. NO rainbow.
MODEL_COLOR = {
    "grud":"#CBD0D6", "hybridattn":"#AEB5BD", "dkgt":"#949BA4",
    "wdcnn":"#7B838C", "msdm_net":"#5E666F", "horln":"#454C54",
    "ours":C["ours"], "a34":C["ours"], "a34_NaME":C["ours"],
}
MODEL_NAME = {
    "grud":"GRU-D", "hybridattn":"Self-attn.", "dkgt":"DKGT-Net",
    "wdcnn":"WDCNN", "msdm_net":"MSDM", "horln":"HORLN",
    "ours":"MAMRD", "a34":"MAMRD", "a34_NaME":"MAMRD",
}

def strata_ramp():  # heavy_riser, heavy_dropper, light_riser, light_dropper, flat
    return ["#1F4E79", "#356EA6", "#5B91C2", "#9DBAD8", "#CFDDEC"]

def apply_style():
    mpl.rcParams.update({
        "figure.dpi":150, "savefig.dpi":600, "savefig.bbox":"tight", "savefig.pad_inches":0.02,
        "pdf.fonttype":42, "ps.fonttype":42,
        "font.family":"serif", "font.serif":["DejaVu Serif","Times New Roman"],
        "font.size":8, "axes.titlesize":8.5, "axes.labelsize":8,
        "legend.fontsize":7, "xtick.labelsize":7.2, "ytick.labelsize":7.2,
        "text.color":C["ink"], "axes.labelcolor":C["ink"], "axes.edgecolor":C["ink"],
        "xtick.color":C["ink"], "ytick.color":C["ink"],
        "axes.linewidth":0.7, "axes.spines.top":False, "axes.spines.right":False,
        "axes.grid":True, "axes.axisbelow":True, "grid.color":C["grid"],
        "grid.linewidth":0.5, "grid.alpha":0.9,
        "lines.linewidth":1.4,
        "legend.frameon":False, "figure.facecolor":"white", "savefig.facecolor":"white",
    })

SINGLE, DOUBLE = 3.5, 7.16
