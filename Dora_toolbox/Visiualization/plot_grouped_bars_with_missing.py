
import math
from typing import Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_bars_with_missing(
    title: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    ylabel: str = "Throughput (sample/s)",
    hatch_cycle: Sequence[str] = ("", "//", "xx", "\\\\", "++", ".."),
    bar_width: float = 0.15,
    x_cross: str = "X",
    x_y_offset: float = 0.00,
):
    """
    Plot a grouped bar chart. If a value is None or NaN, draw an 'X' at the baseline.

    Args:
        title: Plot title.
        categories: Labels on the x-axis (one group per item).
        series: List of (label, values). 'values' length must match len(categories).
                Missing entries can be None or NaN.
        ylabel: Y-axis label.
        hatch_cycle: Hatching patterns cycled across series for visual distinction.
        bar_width: Width of each bar.
        x_cross: String drawn for missing values at the baseline.
        x_y_offset: Fraction of y-range to nudge the 'X' above the axis line.
    """
    n_groups = len(categories)
    n_series = len(series)
    x = np.arange(n_groups, dtype=float)
    #x = x + bar_width

    fig, ax = plt.subplots(figsize=(12.0, 4.0), dpi=160)

    # Determine offsets so bars are centered within each group
    total_width = n_series * bar_width
    start = - (total_width - bar_width) / 2.0

    # Compute y-limit candidate to place X slightly above 0
    max_val = 0.0
    for _, vals in series:
        for v in vals:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                max_val = max(max_val, float(v))
    if max_val <= 0:
        max_val = 1.0

    # Plot bars and handle missing values
    for i, (label, vals) in enumerate(series):
        offsets = x + start + i * bar_width
        hatch = hatch_cycle[i % len(hatch_cycle)]
        sanitized = []
        for v in vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                sanitized.append(np.nan)  # skip drawing the bar
            else:
                sanitized.append(v)
        bars = ax.bar(offsets, sanitized, width=bar_width, hatch=hatch, edgecolor="black")
        # Draw 'X' for missing entries
        for xi, v in zip(offsets, vals):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ymin, ymax = ax.get_ylim()
                y = (ymax - ymin) * x_y_offset  # small offset above baseline
                ax.text(xi, y, x_cross, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend([s[0] for s in series], loc="upper left", frameon=False)
    ax.set_xlim(-0.44,6)
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    # Example usage
    categories = ["0.6B-P100","1.7B-P100",
                  "0.6B-5070","1.7B-5070",
                  "0.6B-H100","1.7B-H100",
                  ]
    series = [
        ("offloading", [0.185,None, 0.592,None, 0.268,0.167]),
        ("offloading+acp",[0.126,0.084, 0.421,None, 0.224,0.158]),
        ("acp",         [2.203,None, 3.156,None, 3.363,1.396]),
        ("original", [2.733,None, None,None, 4.103, 1.769]),
        ("3-device dp", [4.644,None, 5.556,None, 7.799,4.248]),
    ]
    series1 = [
        ("offloading+acp",[0.126,0.084, 0.421,0.2981, 0.224,0.158]),
        ("offloading", [0.185,0.1159, 0.592,0.3711, 0.268,0.167]),
        ("acp",         [2.203,0.9213, 3.156,1.3169, 3.363,1.396]),
        ("original", [2.733,1.1675, 2.9307*1.1,1.2636*1.1, 4.103, 1.769]),
        ("3-device dp", [4.644,2.50632, 5.556,3.0343, 7.799,4.248]),
    ]
    fig, ax = plot_grouped_bars_with_missing(
        title="different technique on edge device(Qwen3)",
        categories=categories,
        series=series,
        ylabel="training speed(steps/s)",
    )
    fig.savefig("motiv_L4.png")
    plt.show()
