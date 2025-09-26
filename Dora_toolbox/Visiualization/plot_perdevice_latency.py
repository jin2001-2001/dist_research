import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------
# OPTION A: Put your data here.
#   time latency unit: milliseconds
#   mem_gb: "common memory size" you want to display (GPU or CPU or hybrid)
#           If you have both, you can choose one or sum them before putting here.
# ---------------------------------------------------------------------
data = [
    {"device": "Jetson Nano", "model": "qwen-0.6B", "tbt_latency_ms": 16, "mem_gb": 8},
    {"device": "Jetson Nano", "model": "qwen-1.7B", "tbt_latency_ms": 45.3, "mem_gb": 8},
    {"device": "Jetson Nano", "model": "qwen-4B", "tbt_latency_ms": 106.7, "mem_gb": 8},

#    {"device": "Raspberry Pi 4", "model": "qwen-0.6B", "tbt_latency_ms": 0, "mem_gb": 8},
#    {"device": "Raspberry Pi 4", "model": "qwen-1.7B", "tbt_latency_ms": 0, "mem_gb": 8},
#    {"device": "Raspberry Pi 4", "model": "qwen-4B", "tbt_latency_ms": 0, "mem_gb": 8},

    {"device": "Desktop AMD9600X", "model": "qwen-0.6B", "tbt_latency_ms": 18.36, "mem_gb": 12},
    {"device": "Desktop AMD9600X", "model": "qwen-1.7B", "tbt_latency_ms": 23.66, "mem_gb": 12},
    {"device": "Desktop AMD9600X", "model": "qwen-4B", "tbt_latency_ms": 0, "mem_gb": 12},

    {"device": "xiaomi12s", "model": "qwen-0.6B", "tbt_latency_ms": 36.866, "mem_gb": 12},
    {"device": "xiaomi12s", "model": "qwen-1.7B", "tbt_latency_ms": 58.44, "mem_gb": 12},
    {"device": "xiaomi12s", "model": "qwen-4B", "tbt_latency_ms": 97.7, "mem_gb": 12},
]
df = pd.DataFrame(data)

# ---------------------------------------------------------------------
# Filter for desired models
# ---------------------------------------------------------------------
models_order = ["qwen-0.6B", "qwen-1.7B", "qwen-4B"]
df = df[df["model"].isin(models_order)].copy()

# Memory per device (max across models)
mem_per_device = df.groupby("device")["mem_gb"].max()

# Pivot latency table
latency_wide = df.pivot_table(index="device",
                              columns="model",
                              values="tbt_latency_ms",
                              aggfunc="mean").reindex(columns=models_order)

devices = latency_wide.index.tolist()
n_devices = len(devices)
n_groups = len(models_order)

# Bar positions
bar_width = 0.15
x = range(n_devices)
offsets = [(-2*bar_width),(-bar_width), 0]  # Two latency bars per device; memory bar will be offset right

fig, ax = plt.subplots(figsize=(10, 6))

# Plot latency bars
for i, model in enumerate(models_order):
    heights = latency_wide[model].values
    bars_x = [xi + offsets[i] for xi in x]
    ax.bar(bars_x, heights, width=bar_width, label=model, alpha=0.8)

ax.set_ylabel("TBT Latency (ms)", color="tab:blue")
ax.set_xlabel("Device")
ax.set_title("TBT Latency and Memory Size per Device")
ax.tick_params(axis='y', labelcolor="tab:blue")

ax.set_ylim(0, 120)

# Memory bar on secondary axis
ax2 = ax.twinx()
mem_x = [xi + bar_width for xi in x]  # offset to right of latency bars
ax2.bar(mem_x, mem_per_device.values, width=bar_width, label="Memory Size (GB)",
        color="tab:orange", alpha=0.6,hatch="xx")
ax2.set_ylabel("Memory Size (GB)", color="tab:orange")
ax2.tick_params(axis='y', labelcolor="tab:orange")

ax2.set_ylim(0, 30)
# X-axis device labels
ax.set_xticks(list(x))
ax.set_xticklabels(devices, rotation=20, ha="right")

# Legends: combine from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()