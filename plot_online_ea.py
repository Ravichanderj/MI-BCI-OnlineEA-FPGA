"""Fig.: online-EA warm-up (a) and Newton-Schulz iterations (b) from results/online_ea_analysis.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

R = json.load(open("results/online_ea_analysis.json"))
seqs = np.array([s for v in R["curve"].values() for s in v], float)          # 18 x 288
B = 8
blk = seqs[:, :seqs.shape[1] // B * B].reshape(len(seqs), -1, B).mean(2) * 100  # 18 x 36
m, se = blk.mean(0), blk.std(0, ddof=1) / np.sqrt(len(blk))
first, rest = seqs[:, :32].mean(1) * 100, seqs[:, 32:].mean(1) * 100
p = wilcoxon(first, rest).pvalue
oracle = 100 * np.mean(list(R["oracle"].values()))
K = sorted(R["ns"], key=int)
accK = [100 * np.mean(list(R["ns"][k].values())) for k in K]
print(f"first 32: {first.mean():.2f}  rest: {rest.mean():.2f}  Wilcoxon p={p:.3f} (n={len(seqs)} sessions)")

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})

x = (np.arange(len(m)) + 0.5) * B
a.fill_between(x, m - se, m + se, color=BLUE, alpha=0.18, lw=0)
a.plot(x, m, color=BLUE, lw=1.5, marker="o", ms=3, label="online EA (mean ± s.e., 18 sessions)")
a.axhline(oracle, color=MUTED, ls="--", lw=1, label="offline-EA oracle")
a.axvspan(0, 32, color=GRID, alpha=0.6, lw=0)
a.text(16, 74, "first 32\ntrials", ha="center", va="top", color=MUTED, fontsize=7)
a.legend(loc="lower center", ncol=2, frameon=False, fontsize=7)
a.set(xlim=(0, x[-1] + 4), ylim=(25, 75), xlabel="trial index within session", ylabel="accuracy (%)")
a.set_title(f"(a) warm-up: first 32 trials {first.mean():.1f}% vs later {rest.mean():.1f}% (p = {p:.2f})",
            fontsize=8, loc="left", color=INK)
a.grid(axis="y", color=GRID, lw=0.6)

b.plot([int(k) for k in K], accK, color=BLUE, lw=1.5, marker="o", ms=4)
b.axhline(oracle, color=MUTED, ls="--", lw=1)
for k, v in zip(K, accK):
    if int(k) in (4, 12, 18):
        b.annotate(f"{v:.1f}", (int(k), v), textcoords="offset points", xytext=(9, -3) if int(k) == 4 else (0, 5),
                   ha="left" if int(k) == 4 else "center", fontsize=7, color=INK)
b.set(xticks=[int(k) for k in K], ylim=(42, 54), xlabel="Newton–Schulz iterations K", ylabel="accuracy (%)")
b.set_title("(b) whitening accuracy vs. iterations", fontsize=8, loc="left", color=INK)
b.grid(axis="y", color=GRID, lw=0.6)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig_online_ea.{ext}", dpi=300)
print("saved figures/fig_online_ea.pdf/.png")
