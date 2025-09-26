
import csv
import matplotlib.pyplot as plt

# Read data
x_indices = []
ratios = []
compute_labels = []

with open("../record/test_improve_b2_sub8.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    idx = 0
    prev_ratio_of_compute = None

    for row in reader:

        current_compute = row["ratio_of_compute"]
        ratio_of_comm = row["ratio_of_comm"]
        ratio_of_between = row["ratio_of_between"]
        if int(current_compute)%40 !=0 or int(ratio_of_comm)%40 !=0 or int(ratio_of_between)%40!=0:
            continue

        ratios.append(float(row["ratio"]))
        x_indices.append(idx)

        # If this is a new ratio_of_compute value, label it
        if current_compute != prev_ratio_of_compute:
            compute_labels.append(current_compute)
            prev_ratio_of_compute = current_compute
        else:
            compute_labels.append("")

        idx += 1

# Plot
plt.figure(figsize=(12, 5))
#plt.plot(x_indices, ratios, marker="o")
plt.plot(x_indices, ratios, linewidth=1)

plt.xticks(
    ticks=x_indices,
    labels=compute_labels,
    rotation=45,
    ha='right'
)
plt.xlabel("ratio_of_compute (only shown when it changes)")
plt.ylabel("Ratio")
plt.title("Ratio over entries")
plt.grid(True)
plt.tight_layout()
plt.show()
