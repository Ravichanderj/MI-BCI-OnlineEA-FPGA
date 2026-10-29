"""Block diagrams for the paper: Fig. 1 system architecture, Fig. 2 online-EA datapath; Fig. 5 PhysioNet scatter."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
BLUE_F, BLUE_E = "#e3eefb", "#2a78d6"     # PL blocks
GRAY_F, GRAY_E = "#f1f0ec", "#8a8984"     # PS / memory
ORNG_F, ORNG_E = "#fdebe3", "#eb6834"     # online-EA (contribution)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5})

def box(ax, x, y, w, h, text, fc, ec, bold=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, x0, y0, x1, y1, text=None, dy=0.12):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.0,
                                                         shrinkA=0, shrinkB=0, mutation_scale=8))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.5, color=MUTED)

# ---------------- Fig. 1: system
fig, ax = plt.subplots(figsize=(7.16, 2.9)); ax.set_xlim(0, 14.3); ax.set_ylim(0, 5.8); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.1, 0.2), 3.4, 5.35, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=GRAY_E, lw=0.8, ls="--"))
ax.text(1.8, 5.22, "Processing system (ARM)", ha="center", fontsize=7, color=MUTED)
box(ax, 0.35, 3.75, 2.9, 1.05, "EEG stream, 22 ch\n250 Hz (µV)", GRAY_F, GRAY_E)
box(ax, 0.35, 2.15, 2.9, 1.2, "Causal band-pass 4–40 Hz\n(2 biquads), ↓2 → 125 Hz\n4-s window, int32 Q.10", GRAY_F, GRAY_E, fs=7)
box(ax, 0.35, 0.8, 2.9, 1.0, "DDR buffer 22×500;\nPYNQ driver", GRAY_F, GRAY_E)
arrow(ax, 1.8, 3.75, 1.8, 3.35); arrow(ax, 1.8, 2.15, 1.8, 1.8)
ax.add_patch(FancyBboxPatch((3.75, 0.2), 10.45, 5.35, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=BLUE_E, lw=0.8, ls="--"))
ax.text(8.97, 5.22, "Programmable logic (XC7Z020): mi_bci_top at 50 MHz", ha="center", fontsize=7, color=MUTED)
box(ax, 4.0, 0.8, 1.9, 1.0, "AXI master\nburst load", BLUE_F, BLUE_E)
arrow(ax, 3.25, 1.3, 4.0, 1.3, "HP0", dy=0.05)
box(ax, 4.0, 2.15, 4.0, 2.65, "", ORNG_F, ORNG_E)
ax.text(6.0, 4.5, "Online EA (label-free)", ha="center", fontsize=7.2, fontweight="bold", color=INK)
sub = [("Trial covariance  P = XXᵀ/2⁹", 3.95), ("R ← R + αₙ(P − R),  αₙ: 1/n → 2⁻⁵", 3.5),
       ("Shrink  S = R + ε·tr(R)/C·I", 3.05), ("r = tr(S)^(−1/2)  (LOD + Newton)", 2.6)]
for t, yy in sub: ax.text(4.15, yy, t, fontsize=6.5, color=INK, va="center")
arrow(ax, 4.95, 1.8, 4.95, 2.15)
box(ax, 8.25, 3.55, 2.65, 1.25, "Coupled Newton–Schulz\nZ = (r²S)^(−1/2)\nK = 18, matmul/add", ORNG_F, ORNG_E, fs=6.8)
box(ax, 8.25, 2.15, 2.65, 1.1, "Whiten + int8\nq = sat(r·Z·x / s_in)", ORNG_F, ORNG_E, fs=6.8)
arrow(ax, 8.0, 4.15, 8.25, 4.15); arrow(ax, 9.57, 3.55, 9.57, 3.25)
box(ax, 11.15, 2.15, 2.85, 2.65, "int8 decoder (2,420 params)\n\nspatial 16×22\ntemporal 64-tap, HW-ELU\navg-pool 4, DW 16-tap\nPW 16×16, HW-ELU\navg-pool 8, FC 240×4", BLUE_F, BLUE_E, fs=6.6)
arrow(ax, 10.9, 2.7, 11.15, 2.7)
box(ax, 11.15, 0.8, 2.85, 1.0, "Integer argmax\n→ class, logits", BLUE_F, BLUE_E)
arrow(ax, 12.57, 2.15, 12.57, 1.8)
box(ax, 7.0, 0.8, 3.2, 1.0, "AXI-Lite: start/done,\nsession reset, class", BLUE_F, BLUE_E)
arrow(ax, 11.15, 1.3, 10.2, 1.3)
ax.plot([8.6, 8.6, 1.8], [0.8, 0.45, 0.45], color=LINE, lw=1.0)
arrow(ax, 1.8, 0.45, 1.8, 0.8)
ax.text(6.4, 0.5, "GP0", fontsize=6.5, color=MUTED, ha="center")
fig.tight_layout(pad=0.1)
for e in ("pdf", "png"): fig.savefig(f"figures/fig1_system.{e}", dpi=300)

# ---------------- Fig. 2: EA datapath / schedule (snake layout)
fig, ax = plt.subplots(figsize=(7.16, 3.0)); ax.set_xlim(0, 14.3); ax.set_ylim(0, 6.0); ax.axis("off")
steps = ["(1) Trial covariance\n253 upper-triangle entries\n× 500 samples, 64-bit acc.",
         "(2) Recursive reference update\nαₙ = 1/n from 33-entry ROM,\nthen 2⁻⁵; mirror to full matrix",
         "(3) Shrinkage and trace\nS = R + ε·tr(R)/C·I\nε = 0.01 (constant multiply)",
         "(4) Scalar inverse square root\nleading-one detector → shift,\n26-bit Newton on the mantissa",
         "(5) Coupled Newton–Schulz, K = 18\nT = ½(3I − ZY);  Y ← YT;  Z ← TZ\n32-bit, unroll 2",
         "(6) Whiten and quantise\n22×22 MAC per sample\nround, saturate to ±127"]
w, h, g = 4.3, 1.6, 0.45
pos = [(0.2, 3.9), (0.2 + w + g, 3.9), (0.2 + 2 * (w + g), 3.9),
       (0.2 + 2 * (w + g), 1.75), (0.2 + w + g, 1.75), (0.2, 1.75)]
for i, (t, (x, y)) in enumerate(zip(steps, pos)):
    blue = i == 5
    box(ax, x, y, w, h, t, BLUE_F if blue else ORNG_F, BLUE_E if blue else ORNG_E, fs=7)
arrow(ax, pos[0][0] + w, 4.7, pos[1][0], 4.7); arrow(ax, pos[1][0] + w, 4.7, pos[2][0], 4.7)
arrow(ax, pos[2][0] + w / 2, 3.9, pos[3][0] + w / 2, 1.75 + h)
arrow(ax, pos[3][0], 2.55, pos[4][0] + w, 2.55); arrow(ax, pos[4][0], 2.55, pos[5][0] + w, 2.55)
ax.text(0.2, 1.15, "No eigendecomposition, divider or square-root unit: every step is a multiply, add, shift or ROM look-up; "
        "latency is fixed per trial.", fontsize=6.8, color=MUTED)
ax.text(0.2, 0.55, "Formats: x Q14.10 (24 b) · R, S Q24.24 (48 b) · Y, Z, T Q8.24 (32 b) · NS accumulators 56 b. "
        "A session reset clears R and n.", fontsize=6.8, color=MUTED)
fig.tight_layout(pad=0.1)
for e in ("pdf", "png"): fig.savefig(f"figures/fig2_ea_datapath.{e}", dpi=300)

# ---------------- Fig. 5: PhysioNet per-subject scatter
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for a, (t, ch) in zip(axs, (("P4", 25), ("P2", 50))):
    on = pd.read_csv(f"results/table{t}_online_ea.csv", index_col=0)
    no = pd.read_csv(f"results/table{t}_no_ea.csv", index_col=0)
    cols = [c for c in on.columns if c.startswith("S")]
    x, y = no.loc["Prop", cols].astype(float), on.loc["Prop", cols].astype(float)
    lo, hi = min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3
    a.plot([lo, hi], [lo, hi], color=LINE, lw=0.8, ls="--")
    a.scatter(x, y, s=12, color="#2a78d6", alpha=0.75, edgecolor="white", lw=0.4)
    a.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="no alignment (%)", ylabel="online EA (%)")
    n = int((y > x).sum())
    a.set_title(f"({'a' if t=='P4' else 'b'}) PhysioNet {'4' if t=='P4' else '2'}-class: {n}/{len(cols)} subjects above diagonal",
                fontsize=7.5, loc="left")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.grid(color="#e4e3df", lw=0.5)
fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(f"figures/fig5_physionet_scatter.{e}", dpi=300)
print("ok")
