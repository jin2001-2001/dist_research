import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ----------------------------
# 1) Data
# ----------------------------
panel_data = [
    np.array([[58.0, 11.51, 11.4, 14.8, 2.66],
              [16.44, 16.5, 16.5, 16.4, 8.5],
              [20.1, 38.1, 37.9, 20.1, 10.4],
              [29.9, 97, 96, 29, 22.2]], dtype=float),

    np.array([[96.7, 37.0, 18.75, 24.1, 3.0],
              [27.2, 27.0, 27.03, 26.0, 7.88],
              [32.8, 172.7, 62.7, 32.4, 17.0],
              [36.17, 153.5, 154.2, 35.1, 25.64]], dtype=float),

    np.array([[853, 211.6, 210.6, 210, 25.4],
              [243, 745.8, 745.88, 240.42, 68.5],
              [291, 1958, 1959, 292.0, 87.88],
              [0, 1839, 1839, 1240, 133.12]], dtype=float),

    np.array([[1.14, 0.69, 0.63, 0.56, 0.51],
              [2.99, 2.84, 2.59, 2.10, 1.89],
              [4.37, 4.21, 3.85, 3.67, 2.58],
              [11.5, 9.9, 7.8, 6.8, 5.57]], dtype=float),
]

group_names = ["Bert", "Qwen-0.6", "Qwen-1.7", "Omni"]
series_names = ["EdgeShard", "Alpa", "Metis", "Asteroid", "Dora"]
series_colors = ["#90C9E7", "#219EBC", "#0D6783", "#DD552F", "#E8A85B"]

GRID_ALPHA = 0.25
LEGEND_FONTSIZE = 14

panel_titles = ["Smart home 1", "Smart home 2", "Traffic monitor", "Edge cluster"]

# ------------------------------------------------
# 2) USE A 2×4 GRID: Top = plots, Bottom = legend
# ------------------------------------------------
fig = plt.figure(figsize=(18, 5), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[20, 1])

axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis("off")

# ------------------------------------------------
# 3) Plotting function for each panel
# ------------------------------------------------
def plot_panel(ax, values, title=None):

    values = np.asarray(values, float)
    G, S = values.shape
    group_centers = np.arange(G)
    width = 0.15
    gap = 0.02
    offsets = np.linspace(-(S-1)*(width+gap)/2, (S-1)*(width+gap)/2, S)

    bars = []
    for s in range(S):
        x = group_centers + offsets[s]
        bar = ax.bar(x, values[:, s], width=width,
                     color=series_colors[s],
                     edgecolor="black", linewidth=0.6)
        bars.append(bar)

    # Grid
    ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)

    # -----------------------------
    # Linear y-limit (except log panel)
    # -----------------------------
    if  0:   # not the log panel
        flat_vals = values.flatten()
        ymax_lin = np.percentile(flat_vals[flat_vals > 0], 50) * 1.15
        ymax_lin = max(9, ymax_lin)
        ax.set_ylim(0, ymax_lin)

    # -----------------------------
    # SAFE ANNOTATION (for linear only)
    # -----------------------------
    if  0:
        for s in range(S):
            for rect, val in zip(bars[s], values[:, s]):
                h = rect.get_height()
                x = rect.get_x() 
                #ax.annotate(f"{val:g}",
                #            xy=(x, h),
                #            xytext=(0, 3),
                #            textcoords="offset points",
                #            ha="center", va="bottom", fontsize=8)

    # -----------------------------
    # For LOG PANEL (3rd subplot)
    # -----------------------------
    if 1:  # index 2
        if ax is axes[3]:
            ax.set_yscale("log", base =2)
        else:
            ax.set_yscale("log")

        # Remove zeros (log invalid)
        nonzero = values[values > 0]
        ymin = nonzero.min() * 0.8
        ymax = nonzero.max() * 1.4
        ax.set_ylim(ymin, ymax)

        # Log-safe annotations
        for s in range(S):
            for rect, val in zip(bars[s], values[:, s]):
                if val > 0:
                    x = rect.get_x() + rect.get_width() / 2 
                    #ax.text(x, val, f"{val:g}",
                    #        ha="center", va="bottom",
                    #        fontsize=7)
                if val == 0:
                    x = rect.get_x() + rect.get_width() / 2
                    # Draw a small bar so log scale doesn't break
                    rect.set_height(1e-9)

                    # Draw a cross at the baseline (log y-min)
                    ymin = ax.get_ylim()[0]
                    ax.text(
                        x, ymin+1.5,
                        "×",
                        ha="center", va="center",
                        fontsize=12, color="black"
                    )


    # X ticks
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names, fontsize=12)

    if title:
        ax.set_title(title, fontsize=15, pad=3)

    # y label
    ax.set_ylabel("Per-iteration time (s)", fontsize=15)


# ------------------------------------------------
# 4) Draw all panels
# ------------------------------------------------
for ax, pdata, ttl in zip(axes, panel_data, panel_titles):
    plot_panel(ax, pdata, ttl)

# ------------------------------------------------
# 5) SHARED LEGEND (centered)
# ------------------------------------------------
legend_ax.legend(
    handles=[plt.Rectangle((0,0),1,1,color=c,ec='black') for c in series_colors],
    labels=series_names,
    loc="center",
    ncol=5,
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
)

# ------------------------------------------------
# 6) Save & show
# ------------------------------------------------
fig.savefig("training.pdf", dpi=300)
plt.show()
