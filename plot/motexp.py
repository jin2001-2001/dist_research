import matplotlib.pyplot as plt
import numpy as np

# Values
values = [297, 25.6, 24.08, 545, 64.62, 43.53]
# Grouping: Group 1 -> bars 1&2; Group 2 -> bars 3,4,5
group_indices = [[0, 1, 2], [3, 4, 5]]
group_names = ["Cloud Network Setup", "Edge Network Setup"]

# Colors (close shades) and legend mapping:



color_map = {
    "cat1": "#4e79a7",  # Bars 1 & 3
    "cat2": "#6b9acb",  # Bars 2 & 4
    "cat3": "#9fbbe3",  # Bar 5
}
colors = [color_map["cat1"], color_map["cat2"], color_map["cat3"],
          color_map["cat1"], color_map["cat2"], color_map["cat3"]]
legend_items = [("Metis", color_map["cat1"]),
                ("Astroid", color_map["cat2"]),
                ("Oracle",      color_map["cat3"])]

# Bar layout within groups
group_centers = [0, 1.0]   # spacing between groups
width = 0.25
gap = 0.05

positions = []
for gi, idxs in enumerate(group_indices):
    n = len(idxs)
    offsets = np.linspace(-(n-1)*(width+gap)/2, (n-1)*(width+gap)/2, n)
    positions.extend(group_centers[gi] + offsets)

fig, ax = plt.subplots(figsize=(8,5))

bars = ax.bar(positions, values, width=width,
              color=colors, edgecolor="black", linewidth=0.6)

# Light mesh/grid
ax.grid(True, axis='y', linestyle='--', alpha=0.25)

# Zoomed view; keep small bars readable
ylim = 110
ax.set_ylim(0, ylim)

# Annotate values; clamp tall bars just below the top so they don't hit the title
for bar, val in zip(bars, values):
    x = bar.get_x() + bar.get_width() / 2
    h = bar.get_height()
    if h > ylim:
        ax.annotate(f'{val}', xy=(x, ylim - 10), xycoords='data',
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom')
        ax.annotate("↑", xy=(x, ylim - 1.2), ha='center', va='bottom')
    else:
        ax.annotate(f'{val}', xy=(x, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom')

# One x label per group
ax.set_xticks(group_centers)
ax.set_xticklabels(group_names)

ax.set_ylabel("Latency per iteration(s)")
ax.set_title("Time latency under different network setup", pad=12)

# Legend
handles = [plt.Rectangle((0,0), 1, 1, color=c) for _, c in legend_items]
ax.legend(handles, [name for name, _ in legend_items],
          loc="upper right", frameon=False,
          fontsize=9)

plt.tight_layout()
plt.show()
