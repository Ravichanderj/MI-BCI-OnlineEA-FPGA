"""Fig. 8: measured latency and energy per trial, ARM Cortex-A9 vs. accelerator at 50 / 75 MHz (PYNQ-Z2, S1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
cats = ["ARM\nCortex-A9", "FPGA\n50 MHz", "FPGA\n75 MHz"]
cols = ["#8a8984", "#2a78d6", "#eb6834"]           # gray = software baseline
lat = [21.37, 3.59, 2.34]; en = [3.33, 0.57, 0.77]
lat_lab = ["21.37 ms", "3.59 ms\n(5.96×)", "2.34 ms\n(9.13×)"]
en_lab = ["3.33 mJ", "0.57 mJ", "0.77 mJ"]
plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2))
for ax, v, lb, t, yl in [(axs[0], lat, lat_lab, "(a) Latency per trial", "ms"), (axs[1], en, en_lab, "(b) Energy per trial", "mJ")]:
    b = ax.bar(range(3), v, width=0.62, color=cols, edgecolor="white", linewidth=1.0)
    for i, (x, s) in enumerate(zip(v, lb)):
        ax.text(i, x + max(v) * 0.03, s, ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=6.6, color=INK)
    ax.set_ylim(0, max(v) * 1.32); ax.set_ylabel(yl, color=MUTED)
    ax.set_title(t, fontsize=7.6, fontweight="bold", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#9b9a96")
    ax.tick_params(colors=MUTED, length=2)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig("figures/fig_board_results.png", dpi=300); fig.savefig("figures/fig_board_results.pdf")
