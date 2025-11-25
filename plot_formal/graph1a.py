import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Serif"
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0

# Data
models_mmlu = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
eng = np.array([25.5, 40.5, 64.0, 63.0, 67.0])
bus = np.array([28.5, 46.0, 65.0, 76.0, 78.5])
health = np.array([26.0, 45.0, 65.5, 75.5, 76.5])

# color_eng = "#4C78A8"
# color_bus = "#F58518"
# color_health = "#54A24B"

color_eng = "#90C9E7"
color_bus = "#219EBC"
color_health = "#0D6783"

fig, ax = plt.subplots(figsize=(4.2, 3.0))

# ✨ 去掉右边 & 上边边框
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

width = 0.22
x = np.arange(len(models_mmlu))

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

b1 = ax.bar(x - width, eng, width,
            color=color_eng, edgecolor="black", linewidth=0.8, label="Engineering")
b2 = ax.bar(x, bus, width,
            color=color_bus, edgecolor="black", linewidth=0.8, label="Business")
b3 = ax.bar(x + width, health, width,
            color=color_health, edgecolor="black", linewidth=0.8, label="Health")

ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models_mmlu, rotation=30, ha="right", fontsize=9.5)
ax.set_ylim(0, 90)

ax.tick_params(axis="both", direction="in", top=True, right=True)
ax.tick_params(right=False, top=False)
ax.legend(fontsize=8, frameon=False)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h + 0.8,
                f"{h:.1f}", ha="center", va="bottom", fontsize=4)

plt.tight_layout()
fig.savefig("graph1a.png", dpi=300, bbox_inches="tight")

plt.close(fig)
