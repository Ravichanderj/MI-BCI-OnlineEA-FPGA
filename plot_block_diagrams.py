"""Block diagrams with a bold title in every box: Fig. 1 system, Fig. 2 online-EA datapath, Fig. 3 decoder reordering."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#6f6e69"
G, GE = "#f1f0ec", "#8a8984"        # processing system / neutral
B, BE = "#e3eefb", "#2a78d6"        # PL decoder & interfaces
O, OE = "#fdebe3", "#eb6834"        # online EA (main contribution)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


def block(ax, x, y, w, h, title, body, fc, ec, tfs=7.4, bfs=6.7):
    """Rounded box: bold title band on top, body text centred below."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    tb = 0.42                                                    # title band height
    ax.plot([x + 0.08, x + w - 0.08], [y + h - tb, y + h - tb], color=ec, lw=0.6)
    ax.text(x + w / 2, y + h - tb / 2, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + (h - tb) / 2, body, ha="center", va="center", fontsize=bfs, color=INK, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, label=None, dx=0, dy=0.1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.0, mutation_scale=8,
                                                         shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=6.4, color=MUTED)


def region(ax, x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc="none", ec=ec, lw=0.9, ls="--"))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=MUTED)


def canvas(w_in, h_in, X, Y):
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.axis("off")
    return fig, ax


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{e}", dpi=300)
    plt.close(fig)


# ============================ Fig. 1: system architecture
fig, ax = canvas(7.16, 3.35, 14.3, 6.7)
region(ax, 0.08, 0.15, 3.5, 6.4, "Processing System (ARM)", GE)
region(ax, 3.78, 0.15, 10.44, 6.4, "Programmable Logic (XC7Z020): mi_bci_top @ 50 or 75 MHz", BE)

block(ax, 0.3, 4.55, 3.05, 1.35, "EEG Acquisition", "22 channels, 250 Hz\nsamples in µV", G, GE)
block(ax, 0.3, 2.45, 3.05, 1.75, "Pre-processing", "causal band-pass 4–40 Hz\n(2 biquads), decimate ×2\n→ 125 Hz, 4-s window\nint32, Q.10 format", G, GE)
block(ax, 0.3, 0.55, 3.05, 1.5, "DDR Buffer & Driver", "22 × 500 trial buffer\nPYNQ: start, session reset", G, GE)
arrow(ax, 1.82, 4.55, 1.82, 4.2); arrow(ax, 1.82, 2.45, 1.82, 2.05)

block(ax, 4.0, 0.55, 2.15, 1.5, "AXI4 Master", "burst load of\n22 × 500 samples", B, BE)
arrow(ax, 3.35, 1.3, 4.0, 1.3, "HP0", dy=0.06)

block(ax, 4.0, 2.45, 4.15, 3.45, "Online Euclidean Alignment",
      "(label-free, per trial)\n\n" r"trial covariance  $P = XX^{T}/2^{9}$" "\n"
      r"$R \leftarrow R + \alpha_n (P - R)$,  $\alpha_n$: 1/n ROM → $2^{-5}$" "\n"
      r"shrinkage  $S = R + \varepsilon\,\mathrm{tr}(R)/C \cdot I$" "\n"
      r"$r = \mathrm{tr}(S)^{-1/2}$  (LOD + Newton)", O, OE, bfs=6.6)
arrow(ax, 5.07, 2.05, 5.07, 2.45)

block(ax, 8.35, 4.25, 2.9, 1.65, "Coupled Newton–Schulz", r"$Z = (r^{2}S)^{-1/2}$" "\nK = 18 iterations\nmultiply/add only", O, OE)
block(ax, 8.35, 2.45, 2.9, 1.5, "Whitening & int8", r"$q = \mathrm{sat}_8(r\,Z\,x\,/\,s_{in})$" "\nround, clip ±127", O, OE)
arrow(ax, 8.15, 5.07, 8.35, 5.07); arrow(ax, 9.8, 4.25, 9.8, 3.95)

block(ax, 11.45, 2.45, 2.6, 3.45, "int8 CNN Decoder",
      "2,420 parameters\n\nspatial 16 × 22\ntemporal 64-tap\nHW-ELU, avg-pool 4\nDW 16-tap, PW 16 × 16\nHW-ELU, avg-pool 8\nFC 240 → 4", B, BE, bfs=6.5)
arrow(ax, 11.25, 3.2, 11.45, 3.2)
block(ax, 11.45, 0.55, 2.6, 1.5, "Integer Argmax", "Q15 class scales\n→ class, logits l0–l3", B, BE)
arrow(ax, 12.75, 2.45, 12.75, 2.05)
block(ax, 7.0, 0.55, 3.6, 1.5, "AXI4-Lite Control", "ap_start / ap_done\nsession reset, class out", B, BE)
arrow(ax, 11.45, 1.3, 10.6, 1.3)
ax.plot([8.8, 8.8, 1.82], [0.55, 0.33, 0.33], color=LINE, lw=1.0)
arrow(ax, 1.82, 0.33, 1.82, 0.55)
ax.text(5.6, 0.36, "GP0", fontsize=6.4, color=MUTED, ha="center", va="bottom")
save(fig, "fig1_system")

# ============================ Fig. 2: online-EA datapath
fig, ax = canvas(7.16, 3.3, 14.3, 6.6)
steps = [("1. Trial Covariance", "row-wise: 22 accumulators\n11 multipliers, 500 steps/row\n64-bit, upper triangle kept"),
         ("2. Recursive Reference Update", r"$R \leftarrow R + \alpha_n (P - R)$" "\n" r"$\alpha_n$ = 1/n (33-entry ROM), then $2^{-5}$" "\nmirror to full matrix"),
         ("3. Shrinkage & Trace", r"$S = R + \varepsilon\,\mathrm{tr}(R)/C \cdot I$" "\n" r"$\varepsilon$ = 0.01" "\nconstant multiply"),
         ("4. Scalar Inverse Square Root", "leading-one detector → shift\n26-bit Newton on mantissa\n" r"$r = \mathrm{tr}(S)^{-1/2}$"),
         ("5. Coupled Newton–Schulz (K = 18)", r"$T = \frac{1}{2}(3I - ZY)$" "\n" r"$Y \leftarrow YT,\;\; Z \leftarrow TZ$" "\n32-bit operands, 11 multipliers, II = 2"),
         ("6. Whitening & int8 Quantization", "22 × 22 MAC per sample\n× gain " r"$r/s_{in}$" "\nround, saturate to ±127")]
w, h, g = 4.3, 2.0, 0.45
pos = [(0.2, 4.35), (0.2 + w + g, 4.35), (0.2 + 2 * (w + g), 4.35),
       (0.2 + 2 * (w + g), 1.85), (0.2 + w + g, 1.85), (0.2, 1.85)]
for i, ((t, body), (x, y)) in enumerate(zip(steps, pos)):
    blue = i == 5
    block(ax, x, y, w, h, t, body, B if blue else O, BE if blue else OE)
yt, yb = 4.35 + (h - 0.42) / 2, 1.85 + (h - 0.42) / 2
arrow(ax, pos[0][0] + w, yt, pos[1][0], yt); arrow(ax, pos[1][0] + w, yt, pos[2][0], yt)
arrow(ax, pos[2][0] + w / 2, 4.35, pos[3][0] + w / 2, 1.85 + h)
arrow(ax, pos[3][0], yb, pos[4][0] + w, yb); arrow(ax, pos[4][0], yb, pos[5][0] + w, yb)
ax.text(0.2, 1.2, "No eigendecomposition, divider or square-root unit: every step is a multiply, add, shift or ROM look-up; "
        "latency is fixed per trial.", fontsize=6.7, color=MUTED)
ax.text(0.2, 0.6, "Formats: x Q14.10 (24 b) · R, S Q24.24 (48 b) · Y, Z, T Q8.24 (32 b) · NS accumulators 56 b. "
        "A session reset clears R and n.", fontsize=6.7, color=MUTED)
save(fig, "fig2_ea_datapath")

# ============================ Fig. 3: decoder, trained vs deployed order
fig, ax = canvas(7.16, 3.25, 14.3, 6.5)
ax.text(0.15, 6.15, "(a) Trained network (EEGNet order)", fontsize=7.6, fontweight="bold", va="center")
row = [("Input", "22 × 500\nfloat", G, GE, 1.7), ("Temporal Conv.", "8 filters × 64 taps\n5.63 M MAC", O, OE, 2.5),
       ("Batch Norm", "per filter", G, GE, 1.75), ("Depthwise Spatial", "16 × 22\n0.18 M MAC", B, BE, 2.5),
       ("BN + ELU + Pool", "avg-pool 4", G, GE, 2.35)]
x = 0.15
for i, (t, body, f, e, w) in enumerate(row):
    block(ax, x, 3.85, w, 1.75, t, body, f, e)
    if i: arrow(ax, x - 0.3, 4.52, x, 4.52)
    x += w + 0.3
ax.text(x + 0.05, 4.72, "front end:\n5.81 M MAC", fontsize=7.2, color=OE, va="center", fontweight="bold")
ax.text(0.15, 3.3, "(b) Deployed integer network (BN folded; spatial and temporal stages swapped)", fontsize=7.6,
        fontweight="bold", va="center")
row = [("int8 Input", "22 × 500", G, GE), ("Spatial Mixing", "16 × 22, requant\n0.18 M MAC", B, BE),
       ("Temporal Conv.", "16 × 64 taps + bias\n0.51 M MAC", B, BE), ("HW-ELU + Pool", "base-2 PWL\navg-pool 4", G, GE),
       ("Separable Conv.", "DW 16 · PW 16×16\nHW-ELU, pool 8", B, BE), ("Classifier", "FC 240 → 4\ninteger argmax", B, BE)]
x = 0.15
for i, (t, body, f, e) in enumerate(row):
    block(ax, x, 1.0, 2.05, 1.75, t, body, f, e, tfs=7.2)
    if i: arrow(ax, x - 0.3, 1.67, x, 1.67)
    x += 2.05 + 0.3
ax.text(0.15, 0.45, "Front end 5.81 M → 0.69 M MAC (−88.2%); whole decoder 0.75 M MAC per 4-s trial.\nThe swap is exact: both "
        "stages are linear and the folded BN between them is a per-map affine map.", fontsize=6.6, color=MUTED, va="center")
save(fig, "fig_decoder_reorder")
print("ok")
