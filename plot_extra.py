"""Fig.: decoder dataflow with operator reordering; Fig.: IV-2a per-subject alignment comparison."""
import re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
def box(ax, x, y, w, h, t, fc, ec, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2)
def arr(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1, mutation_scale=8))

# ---- decoder dataflow, trained vs deployed order
fig = plt.figure(figsize=(7.16, 2.9)); ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_xlim(0, 14.3); ax.set_ylim(0, 6); ax.axis("off")
G, GE, B, BE, O, OE = "#f1f0ec", "#8a8984", "#e3eefb", "#2a78d6", "#fdebe3", "#eb6834"
ax.text(0.1, 5.6, "(a) Trained network (EEGNet order)", fontsize=7.5, fontweight="bold")
row = [("Input\n22 × 500", G, GE), ("Temporal conv\n8 × 64 taps\n5.63 M MAC", O, OE), ("BN", G, GE),
       ("Depthwise spatial\n16 × 22\n0.18 M MAC", B, BE), ("BN, ELU,\navg-pool 4", G, GE)]
x = 0.1
for i, (t, f, e) in enumerate(row):
    w = 2.6 if i in (1, 3) else 1.75
    box(ax, x, 3.65, w, 1.6, t, f, e)
    if i: arr(ax, x - 0.3, 4.45, x, 4.45)
    x += w + 0.3
ax.text(x + 0.1, 4.45, "front end:\n5.81 M MAC", fontsize=7, color=OE, va="center", fontweight="bold")
ax.text(0.1, 2.95, "(b) Deployed integer network (BN folded, spatial and temporal swapped)", fontsize=7.5, fontweight="bold")
row = [("int8 input\n22 × 500", G, GE), ("Spatial mix\n16 × 22, requant\n0.18 M MAC", B, BE),
       ("Temporal conv\n16 × 64 taps (+bias)\n0.51 M MAC", B, BE), ("HW-ELU (base-2\nPWL), avg-pool 4", G, GE),
       ("DW 16 · PW 16×16\nHW-ELU, pool 8\n64 k MAC", B, BE), ("FC 240 → 4\ninteger\nargmax", B, BE)]
x = 0.1
for i, (t, f, e) in enumerate(row):
    w = 2.05
    box(ax, x, 0.85, w, 1.65, t, f, e, fs=6.8)
    if i: arr(ax, x - 0.3, 1.67, x, 1.67)
    x += w + 0.3
ax.text(0.1, 0.3, "Front end 5.81 M → 0.69 M MAC (−88.2%); whole decoder 0.75 M MAC per 4-s trial.\nSwapping is exact: both "
        "convolutions are linear and the folded BN between them is a per-map affine map.", fontsize=6.8, color=MUTED, va="center")
for e in ("pdf", "png"): fig.savefig(f"figures/fig_decoder_reorder.{e}", dpi=300)

# ---- IV-2a per-subject: online vs oracle vs none (Proposed, 3 seeds)
runs, cur = {}, []
for line in open("results/kaggle_v2_log.txt"):
    m = re.search(r"S(\d+) \[loso\].*online-EA ([\d.]+).*oracle ([\d.]+) \| no-EA ([\d.]+)", line)
    if m: cur.append([float(v) for v in m.groups()[1:]])
    m = re.search(r"== (\S+) done", line)
    if m:
        if len(cur) == 9: runs[m.group(1)] = np.array(cur) * 100
        cur = []
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
cols = {"no alignment": "#c3c2b7", "online EA (hardware)": "#2a78d6", "offline oracle": "#eb6834"}
for a, (pre, ttl, ch) in zip(axs, (("A4_prop", "(a) IV-2a, 4-class", 25), ("A2_prop", "(b) IV-2a, 2-class", 50))):
    A = np.mean([runs[f"{pre}_s{s}"] for s in (42, 7, 123)], 0)       # 9 x [online, oracle, none]
    x = np.arange(9); w = 0.27
    for j, (lab, idx) in enumerate((("no alignment", 2), ("online EA (hardware)", 0), ("offline oracle", 1))):
        a.bar(x + (j - 1) * (w + 0.02), A[:, idx], w, color=cols[lab], label=lab)
    a.axhline(ch, color=MUTED, lw=0.8, ls="--")
    a.set_xticks(x, [f"S{i+1}" for i in range(9)]); a.set_ylabel("accuracy (%)")
    a.set_ylim(0, 100); a.set_title(ttl + f": {A[:,2].mean():.1f} → {A[:,0].mean():.1f} (oracle {A[:,1].mean():.1f})",
                                     fontsize=7.5, loc="left")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.grid(axis="y", color="#e4e3df", lw=0.5); a.set_axisbelow(True)
axs[0].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)
fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(f"figures/fig_iv2a_alignment.{e}", dpi=300)
print("ok")
