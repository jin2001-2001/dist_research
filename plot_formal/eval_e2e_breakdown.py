import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ----------------------------
# 1) Put your data here
#    Shape: 4 panels × (4 groups × 3 bars)
#    For each panel: a (4, 3) array-like where axis=0 is groups, axis=1 is bar index in group
# ----------------------------
panel_data = [
    np.array([[84.26,  15.10,  2.01],
              [ 586,  32.25,  11.408],
              [545,  64.62,  43.5],
              [  0,  0,  0]], dtype=float),

    np.array([[ 53.02,  9.444 , 2.01],
              [ 370,  20.23 ,  9.252],
              [ 350,  40.42 ,  27],
              [ 0,  0 ,  0]], dtype=float),

    np.array([[ 0,  0,   0],
              [0,  0,  0],
              [ 0,  0,  0],
              [ 0,  0, 0]], dtype=float),

    np.array([[ 0,  0,  0],
              [ 0,  0,  0],
              [0,  0,  0],
              [ 0, 0,  0]], dtype=float),
]
ymax_manual_list = [100,100,100,100]

group_names = ["Bert-Base","Qwen3-0.6B","Qwen3-1.7B", "Qwen-MLLM-7B"]     # one label per group
series_names = ["Original", "Average_strategy", "OPT-2"]  # bar i across groups

# Close shades for the 3 series (consistent across subplots)
series_colors = ["#4e79a7", "#6b9acb", "#9fbbe3"]

# Grid style / legend font size
GRID_ALPHA = 0.25
LEGEND_FONTSIZE = 8

# How aggressive to clip tall bars so small ones are visible.
# We'll set ylim to 1.1 * percentile (e.g., 90th), but not below a minimum.
PERCENTILE = 90
YMIN = 0
MIN_YMAX = 30  # never set ylim upper below this to ensure some headroom

canvas_counter = -1
def plot_panel(ax, values_4x3, title=None):
    global canvas_counter
    canvas_counter+=1
    """
    values_4x3: array-like shape (4, 3), rows are groups, cols are series/bars within group
    """

    values = np.asarray(values_4x3, dtype=float)
    G, S = values.shape
    assert G == 4 and S == 3, "Expecting 4 groups × 3 bars."

    # x layout: 4 groups centered at [0, 1, 2, 3]
    group_centers = np.arange(G, dtype=float)
    width = 0.22
    gap = 0.04
    # offsets for 3 bars around each group center
    offsets = np.linspace(-(S-1)*(width+gap)/2, (S-1)*(width+gap)/2, S)

    # Build bars
    bars = []
    for s in range(S):
        x = group_centers + offsets[s]
        bar = ax.bar(x, values[:, s], width=width,
                     color=series_colors[s],
                     edgecolor="black", linewidth=0.6,
                     label=series_names[s])
        bars.append(bar)

    # Light mesh/grid behind
    ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)

    # Decide y-limit to ignore (clip) very large values, but keep small bars visible
    flat_vals = values.flatten()
    # choose an upper bound from percentile, make sure it's at least MIN_YMAX
    ymax = max(MIN_YMAX, np.percentile(flat_vals, PERCENTILE) * 1.1)
    ymax = ymax_manual_list[canvas_counter]
    ax.set_ylim(YMIN, ymax)


    # Annotate each bar; for bars taller than ymax, put the label beside the bar
    # so it never collides with the title or gets clipped.
    for s in range(S):
        for rect, val in zip(bars[s], values[:, s]):
            x = rect.get_x() + rect.get_width() / 2
            h = rect.get_height()
            if h > ymax:
                # Place label to the right side of the bar, vertically centered in the visible area
                ax.annotate(f"{val:g}",
                            xy=(x + width/2 + 0.06, (YMIN + ymax) * 0.95),
                            ha='left', va='center')
                # Optional: little up arrow at the top to indicate clipping
                ax.annotate("↑", xy=(x, ymax - 0.8), ha='center', va='bottom')
            else:
                ax.annotate(f"{val:g}",
                            xy=(x, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    # One x label per group (at group centers)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names)
    #ax.set_yticklabels("Iteration Latency (s)")

    # Compact legend
    ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FONTSIZE)

    if title:
        ax.set_title(title, pad=8)

# ----------------------------
# 2) Draw the 4 panels
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
axes = axes.flatten()

panel_titles = ["Smart home 1", "Smart home 2", "Traffic monitor", "Edge cluster"]

for ax, pdata, ttl in zip(axes, panel_data, panel_titles):
    plot_panel(ax, pdata, title=ttl)
    #add labels...
    ax.set_ylabel("Per-iteration time (s)")


# Shared y-label for the whole figure (optional)
fig.suptitle("End to End latency performance", y=0.995)

plt.show()
