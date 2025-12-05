import matplotlib.pyplot as plt
import numpy as np

# Values
values = [10.6, 24.5, 9.0, 13.4, 29.6, 11.4]
group_indices = [[0, 1, 2], [3, 4, 5]]
group_names = ["Qwen-0.6", "Qwen-1.7"]

color_map = {
    "cat1": "#90C9E7",
    "cat2": "#219EBC",
    "cat3": "#FB973A",
}
colors = [
    color_map["cat1"], color_map["cat2"], color_map["cat3"],
    color_map["cat1"], color_map["cat2"], color_map["cat3"]
]
legend_items = [
    ("D2D", color_map["cat1"]),
    ("Shared", color_map["cat2"]),
    ("Optimal", color_map["cat3"])
]

# Bar layout within groups
group_centers = [0, 1.0]
width = 0.25
gap = 0.05

positions = []
for gi, idxs in enumerate(group_indices):
    n = len(idxs)
    offsets = np.linspace(-(n-1)*(width+gap)/2, (n-1)*(width+gap)/2, n)
    positions.extend(group_centers[gi] + offsets)

fig, ax = plt.subplots(figsize=(6,5))

bars = ax.bar(
    positions, values, width=width,
    color=colors, edgecolor="black", linewidth=0.6
)

# --- remove upper + right borders ---
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Light grid
ax.grid(True, axis='y', linestyle='--', alpha=0.25)

ylim = 29.9
ax.set_ylim(0, ylim)
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in', labelsize=20)

# --- Put labels INSIDE the bars ---
for bar, val in zip(bars, values):
    x = bar.get_x() + bar.get_width() / 2
    h = bar.get_height()
    ax.text(
        x, h -3,              # middle of the bar height
        f"{val}",
        ha="center", va="center",
        color="black",
        fontsize=20,
        rotation=90
    )

# Group labels
ax.set_xticks(group_centers)
ax.set_xticklabels(group_names,fontsize=20)

ax.set_ylabel("Latency per iteration(s)",fontsize=20)
ax.set_title("", pad=12)

# --- Legend on top in ONE ROW ---
handles = [
    plt.Rectangle((0,0), 0.5, 1, facecolor=c, edgecolor="black", linewidth=0.8)
    for _, c in legend_items
]
ax.legend(
    handles,
    [name for name, _ in legend_items],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=3,           # <-- one row
    frameon=False,
    fontsize=18
)

plt.tight_layout()
plt.show()
fig.savefig("graph2_train.pdf", bbox_inches="tight", pad_inches=0)
