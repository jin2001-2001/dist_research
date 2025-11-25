import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.linewidth"] = 0.8

FONT = 4

devices = ["S25 Ultra", "Xiaomi 15", "RTX 4050", "RTX 4060", "V100", "A40"]
memory  = ["12GB",      "12GB",      "6GB",     "8GB",      "32GB", "48GB"]
latency = np.array([
    36.00,
    38.00,
    32.08,
    28.70,
    42.00 * 0.497,
    42.00 * 0.497 * 424.55 / 761,
])

fig, ax = plt.subplots(figsize=(3.45/2, 1.35))
labels = [f"{d}" for d, m in zip(devices, memory)]
x = np.arange(len(labels))

bar_width = 0.60

bars = ax.bar(
    x,
    latency,
    width=bar_width,
    color="#1BA9CD",
    edgecolor="black",
    linewidth=0.6,
)

# Y Axis Label
ax.set_ylabel("Latency (ms)", fontsize=FONT)

# X Axis Label
ax.set_xticks(x)
ax.set_xticklabels(
    labels,
    rotation=38,
    ha="right",
    fontsize=FONT,
)

# Tick Font Size
ax.tick_params(axis="y", direction="in", labelsize=FONT)
ax.tick_params(axis="x", direction="in", labelsize=FONT)

ax.set_ylim(0, max(latency) * 1.25)

# Bar Top Numbers
for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.5,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=FONT,
        rotation=0,
    )

# Remove top/right borders
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout(pad=0.2)
fig.savefig("graph2_l.pdf", bbox_inches="tight", pad_inches=0)
#fig.savefig("graph2.png", dpi=350, bbox_inches="tight", pad_inches=0)
plt.close(fig)
