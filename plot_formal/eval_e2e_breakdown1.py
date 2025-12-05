import matplotlib.pyplot as plt

# ----------------------------
methods = ["Stage1 only", "Stage2 only", "Stage1+2"]

model1_g1 = [182.2, 173.6, 133.1]
model2_g1 = [109.2, 99.3, 87]

model1_g2 = [83.85, 54.7, 52.0]
model2_g2 = [51.2, 31.2, 23.2]

markers = ["o", "s"]
colors = ["#219EBC", "#0D6783"]
models = ["Qwen-omni", "Qwen-1.7"]

# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# ---- Graph 1 ----
line1, = axes[0].plot(methods, model1_g1, marker=markers[0], color=colors[0], linewidth=2)
line2, = axes[0].plot(methods, model2_g1, marker=markers[1], color=colors[1], linewidth=2)

axes[0].set_title("Training",fontsize=20)
axes[0].set_ylabel("Time Latency (iteration/s)",fontsize=15)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Remove top/right spines
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add value labels
for x, y in zip(methods, model1_g1):
    axes[0].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(8, 6), ha='center', fontsize=11)
for x, y in zip(methods, model2_g1):
    axes[0].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(8, 6), ha='center', fontsize=11)

# ---- Graph 2 ----
axes[1].plot(methods, model1_g2, marker=markers[0], color=colors[0], linewidth=2)
axes[1].plot(methods, model2_g2, marker=markers[1], color=colors[1], linewidth=2)

axes[1].set_title("Inference",fontsize=20)
axes[1].grid(True, linestyle="--", alpha=0.4)

axes[0].tick_params(axis='x', labelsize=11)
axes[1].tick_params(axis='x', labelsize=11)

axes[0].tick_params(axis='y', labelsize=11)
axes[1].tick_params(axis='y', labelsize=11)
# Remove top/right spines
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add value labels
for x, y in zip(methods, model1_g2):
    axes[1].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(8, 6), ha='center', fontsize=11)
for x, y in zip(methods, model2_g2):
    axes[1].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(8, 6), ha='center', fontsize=11)

# ----------------------------
# Reserve top space for legend
fig.subplots_adjust(top=0.78)

# Shared Legend (top center)
fig.legend(
    handles=[line1, line2],
    labels=models,
    loc="upper center",
    ncol=2,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.92),
    frameon=False
)
fig.savefig("breakdown1.pdf", dpi=300)
plt.show()
