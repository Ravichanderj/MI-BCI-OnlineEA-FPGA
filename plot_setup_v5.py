"""Fig. 7: labelled photo of the measurement setup (PYNQ-Z2, host laptop, inline USB power meter)."""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
X0, Y0, X1, Y1 = 330, 0, 1560, 1120
im = Image.open(sys.argv[1]).convert("RGB").rotate(90, expand=True).crop((X0, Y0, X1, Y1))
plt.rcParams.update({"font.family": "serif"})
fig = plt.figure(figsize=(7.16, 7.16 * im.size[1] / im.size[0]))
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis("off")
def lab(pt, txt, at):
    ax.annotate(txt, (pt[0] - X0, pt[1] - Y0), xytext=(at[0] - X0, at[1] - Y0), fontsize=7.6, fontweight="bold",
                color="#0b0b0b", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2a78d6", lw=1.0, alpha=0.95),
                arrowprops=dict(arrowstyle="-|>", color="#eb6834", lw=1.6, mutation_scale=11))
lab((1000, 960), "PYNQ-Z2 board\n(Zynq XC7Z020)", (1360, 1000))
lab((1240, 515), "Inline USB power meter\n(5 V; 1 mV / 1 mA)", (1360, 330))
lab((795, 1015), "micro-USB: 5-V supply\nthrough the meter", (560, 1080))
lab((760, 972), "Ethernet link\n(PYNQ, Jupyter)", (560, 880))
lab((800, 270), "Host laptop: Jupyter driver,\naccuracy and latency logging", (1080, 455))
fig.savefig("figures/fig_setup.png", dpi=250); fig.savefig("figures/fig_setup.pdf")
