import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# ----------------------------
# 1) Data
# ----------------------------
panel_data = [
    np.array([[37.2, 39.1, 38.7, 35.66, 35.6],
              [154.3, 177.5, 177, 153.4, 152],
              [219, 245, 245, 219.9, 215],
              [762, 928, 932, 745, 555]], dtype=float),

    np.array([[23.1, 20.3, 25.6, 20.34, 3.6],
              [82, 122, 108, 116, 14.61],
              [120.8, 87, 146, 167, 20.62],
              [505, 254, 272, 482, 108.5]], dtype=float),

    np.array([[34.2, 62, 62.6, 62, 26.4],
              [133, 188,188.3, 133, 103],
              [184.6+20, 648.4/2, 648.4/2, 184.6+20, 184.62],
              [0, 567, 578, 612, 284.5]], dtype=float),

    np.array([[49.7, 54, 53,50.0, 48],
              [208, 239, 235, 226, 196],
              [294, 328, 324, 321, 277],
              [698, 638, 5545, 512, 458]], dtype=float),
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
            ax.set_yscale("log", base =10)
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
    ax.set_ylabel("Per-iteration Energy saving(J)", fontsize=15)


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
fig.savefig("training_energy.pdf", dpi=300)
plt.show()
