import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting
import numpy as np

# Lists to store data
x_vals = []
y_vals = []
z_vals = []
ratios = []

# Read the CSV file
with open("../record/test_improve.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        x_vals.append(float(row["ratio_of_compute"]))
        y_vals.append(float(row["ratio_of_comm"]))
        z_vals.append(float(row["ratio_of_between"]))
        ratios.append(float(row["ratio"]))

# Create 3D scatter plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
#print(x_vals)
# Scatter with color mapped to ratio
sc = ax.scatter(
    x_vals, 
    y_vals, 
    z_vals, 
    c=ratios, 
    cmap="viridis", 
    s=50
)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Ratio")

# Labels
ax.set_xlabel("Ratio of Compute")
ax.set_ylabel("Ratio of Comm")
ax.set_zlabel("Ratio of Between")
ax.set_title("3D Scatter with Gradient Color")

plt.tight_layout()
plt.show()
