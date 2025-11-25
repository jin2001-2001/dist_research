import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Global Style
# ===========================
plt.rcParams["font.family"] = "DejaVu Serif"
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0

# ===========================
# Data (MMMU)
# ===========================
models_mmmu = ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "Qwen3-Omni-30B"]
marketing = np.array([50.9, 64.6, 77.1])
medicine_ai = np.array([34.2, 56.1, 64.3])
digital_farm = np.array([37.0, 41.3, 65.8])

# ===========================
# Research Colors (Tableau)
# ===========================
# color_marketing   = "#4C78A8"   # blue
# color_medicine_ai = "#F58518"   # orange
# color_digital     = "#54A24B"   # green

color_marketing = "#90C9E7"
color_medicine_ai = "#219EBC"
color_digital = "#0D6783"

# ===========================
# Figure
# ===========================
fig, ax = plt.subplots(figsize=(4.2, 3.0))

width = 0.22
x = np.arange(len(models_mmmu))

# Background dashed grid
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)

# Remove upper & right spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Bars with black border
c1 = ax.bar(x - width, marketing, width,
            color=color_marketing, edgecolor="black", linewidth=0.8,
            label="Marketing")
c2 = ax.bar(x, medicine_ai, width,
            color=color_medicine_ai, edgecolor="black", linewidth=0.8,
            label="MedicineAI")
c3 = ax.bar(x + width, digital_farm, width,
            color=color_digital, edgecolor="black", linewidth=0.8,
            label="Digital Farming")

# Axis labels
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models_mmmu, rotation=30, ha="right", fontsize=7)
ax.set_ylim(0, 90)

# Tick settings
ax.tick_params(axis="both", direction="in")   # ticks pointing inward
ax.tick_params(right=False, top=False)        # remove top & right ticks

# Legend
ax.legend(fontsize=8, frameon=False)

# Small top labels
for bars in [c1, c2, c3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=4
        )

plt.tight_layout()
fig.savefig("graph1b.png", dpi=300, bbox_inches="tight")

plt.close(fig)
