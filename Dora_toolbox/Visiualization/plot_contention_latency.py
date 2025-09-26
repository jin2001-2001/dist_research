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
    {"device": "d2d", "model": "oracle", "tbt_latency_ms": 30, "mem_gb": 8},
    {"device": "d2d", "model": "Astroid", "tbt_latency_ms": 45.3, "mem_gb": 8},
    {"device": "d2d", "model": "Metis", "tbt_latency_ms": 55, "mem_gb": 8},
    {"device": "d2d", "model": "Pipedream", "tbt_latency_ms": 100, "mem_gb": 8},

    {"device": "sharedBW", "model": "oracle", "tbt_latency_ms":45, "mem_gb": 8},
    {"device": "sharedBW", "model": "Astroid", "tbt_latency_ms": 90, "mem_gb": 8},
    {"device": "sharedBW", "model": "Metis", "tbt_latency_ms": 110, "mem_gb": 8},
    {"device": "sharedBW", "model": "Pipedream", "tbt_latency_ms": 180, "mem_gb": 8},

]
df = pd.DataFrame(data)

# ---------------------------------------------------------------------
# Filter for desired models
# ---------------------------------------------------------------------
models_order = ["oracle", "Astroid", "Metis", "Pipedream"]
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
offsets = [(-2*bar_width),(-bar_width), 0, bar_width]  # Two latency bars per device; memory bar will be offset right

fig, ax = plt.subplots(figsize=(10, 6))

# Plot latency bars
for i, model in enumerate(models_order):
    heights = latency_wide[model].values
    bars_x = [xi + offsets[i] for xi in x]
    ax.bar(bars_x, heights, width=bar_width, label=model, alpha=0.8)

ax.set_ylabel("Inference/Training Latency (ms)")
ax.set_xlabel("Setup Env")
ax.set_title("Latency of current SoR methods under different set up(Qwen3-0.6B):Example")
ax.tick_params(axis='y')

ax.set_ylim(0, 300)

# X-axis device labels
ax.set_xticks(list(x))
ax.set_xticklabels(devices, rotation=20, ha="right")

# Legends: combine from both axes
lines1, labels1 = ax.get_legend_handles_labels()

ax.legend(lines1, labels1, loc="upper left")

ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()