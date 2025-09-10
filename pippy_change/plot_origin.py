# Re-import libraries after reset
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


with open("timeline_batch0_all.json", "r") as f:
    records = json.load(f)

def smooth_series(values, window_size=5):
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='same')

# Regenerate DataFrame with separated SEND/RECV per rank
rows = []
base_ns = records[0]["start_ns"]
for record in records:
    mb = record.get("mb_idx", -1)
    mb_str = ",".join(map(str, mb)) if isinstance(mb, list) else str(mb)
    action = record["action"]
    rank = record["rank"]
    y_tag = f"{rank}-{action}" if action in {"SEND_F", "RECV_F", "SEND_B", "RECV_B"} else f"{rank}-MAIN"
    rows.append({
        "y_tag": y_tag,
        "rank": rank,
        "action": action,
        "start": (record["start_ns"] - base_ns) / 1e6,
        "end": (record["end_ns"] - base_ns) / 1e6,
        "mb_idx": mb_str
    })

df = pd.DataFrame(rows)

# Assign operation colors
op_colors = {
    "FORWARD": "skyblue",
    "FULL_BACKWARD": "lightcoral",
    "ALL_REDUCE": "orange",
    "SEND_F": "gray",
    "RECV_F": "green",
    "SEND_B": "darkgray",
    "RECV_B": "lightgreen"
}

# Assign vertical positions
y_tags = sorted(df["y_tag"].unique())
y_map = {tag: i for i, tag in enumerate(y_tags)}

# Begin plot
fig, ax = plt.subplots(figsize=(16, 10))
for _, row in df.iterrows():
    y_pos = y_map[row["y_tag"]]
    width = row["end"] - row["start"]
    left = row["start"]
    ax.barh(
        y=y_pos,
        width=width,
        left=left,
        height=0.6,
        color=op_colors.get(row["action"], "black"),
        edgecolor="black"
    )
    ax.text(
        left + width / 2,
        y_pos,
        f'MB{row["mb_idx"]}',
        va='center',
        ha='center',
        fontsize=12,
        weight='bold',
        color='black'
    )

# Overlay smoothed bandwidth lines
for record in records:
    if record["action"].startswith("RECV") and record["net_series"]:
        y_tag = f"{record['rank']}-{record['action']}"
        y_base = y_map[y_tag]
        x_vals = [(ts - base_ns) / 1e6 for ts, _, _ in record["net_series"]]
        up_vals = [up for _, up, _ in record["net_series"]]
        down_vals = [down for _, _, down in record["net_series"]]
        up_smooth = smooth_series(up_vals, window_size=5)
        down_smooth = smooth_series(down_vals, window_size=5)
        bw_max = max(max(up_smooth), max(down_smooth), 1)
        up_scaled = [y_base + 0.3 * (u / bw_max) for u in up_smooth]
        down_scaled = [y_base - 0.3 * (d / bw_max) for d in down_smooth]
        ax.plot(x_vals, up_scaled, color='red', linewidth=1.5, label="Up Mbps" if 'Up Mbps' not in ax.get_legend_handles_labels()[1] else "")
        ax.plot(x_vals, down_scaled, color='blue', linewidth=1.5, label="Down Mbps" if 'Down Mbps' not in ax.get_legend_handles_labels()[1] else "")

# Final touches
ax.set_yticks(list(y_map.values()))
ax.set_yticklabels(y_tags, fontsize=11)
ax.set_xlabel("Time (ms)", fontsize=13)
ax.set_title("Hybrid PP Gantt Chart (Smoothed Bandwidth Overlay)", fontsize=15)

# Build legend
handles = [mpatches.Patch(color=c, label=t) for t, c in op_colors.items()]
handles += [
    plt.Line2D([0], [0], color='red', lw=1.5, label="Up Mbps"),
    plt.Line2D([0], [0], color='blue', lw=1.5, label="Down Mbps"),
]
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()
