import matplotlib.pyplot as plt
import numpy as np

# X-axis categories
methods = ["Metis", "Astroid", "Dora"]
x = np.arange(len(methods))

# -----------------------------
# Replace with your real values
# -----------------------------
# Example values for:
# model × setting × method

# Qwen1.7
q17_s1 = np.array([12, 15, 10])   # setting 1
q17_s2 = np.array([11, 13,  9])   # setting 2

# Qwen-Omni
qo_s1  = np.array([22, 30, 20])   # setting 1
qo_s2  = np.array([21, 28, 19])   # setting 2

# -----------------------------
# Plot config
# -----------------------------
colors = {
    "Qwen-1.7": "#1f77b4",
    "Qwen-Omni": "#ff7f0e"
}

markers = {
    "Setting1": "o",
    "Setting2": "^"
}

plt.figure(figsize=(9,6))

# Qwen1.7
plt.plot(x, q17_s1, color=colors["Qwen1.7"], marker=markers["Setting1"], linewidth=2, markersize=8, label="Qwen1.7 - Setting1")
plt.plot(x, q17_s2, color=colors["Qwen1.7"], marker=markers["Setting2"], linewidth=2, markersize=8, label="Qwen1.7 - Setting2")

# Qwen-Omni
plt.plot(x, qo_s1,  color=colors["Qwen-Omni"], marker=markers["Setting1"], linewidth=2, markersize=8, label="Qwen-Omni - Setting1")
plt.plot(x, qo_s2,  color=colors["Qwen-Omni"], marker=markers["Setting2"], linewidth=2, markersize=8, label="Qwen-Omni - Setting2")

# Aesthetics
plt.xticks(x, methods, fontsize=14)
plt.ylabel("Plan latency", fontsize=14)
plt.xlabel("Method", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.3)

plt.legend(ncol=1, fontsize=12, frameon=False, loc="upper left")

plt.tight_layout()
plt.show()
