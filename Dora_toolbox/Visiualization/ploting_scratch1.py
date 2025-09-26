import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_csv(filepath, step=1):
    """Load ratio values and index labels from a CSV file."""
    x_indices = []
    ratios = []
    compute_labels = []

    with open(filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        idx = 0
        prev_ratio_of_compute = None
        for row in reader:

            current_compute = row["ratio_of_compute"]
            ratio_of_comm = row["ratio_of_comm"]
            ratio_of_between = row["ratio_of_between"]
            if (int(current_compute)-100)%40 !=0 or (int(ratio_of_comm)-100)%40 !=0 or (int(ratio_of_between)-100)%40!=0:
                continue
            ratios.append(float(row["ratio"]))
            x_indices.append(idx)

            current_compute = row["ratio_of_compute"]
            if current_compute != prev_ratio_of_compute:
                compute_labels.append(current_compute)
                prev_ratio_of_compute = current_compute
            else:
                compute_labels.append("")

            idx += 1

    # Subsample if needed
    x_indices = x_indices[::step]
    ratios = ratios[::step]
    compute_labels = compute_labels[::step]

    return x_indices, ratios, compute_labels

# Load both datasets
step = 2  # Subsample every 10 rows
x1, y1, labels1 = load_csv("../record/test_improve_b2_sub8.csv", step=step)
x2, y2, labels2 = load_csv("../record/test_improve_b4_sub8.csv", step=step)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(x1, y1, marker=".", markersize=2, linestyle="None", label="microbatch 2")
plt.plot(x2, y2, marker=".", markersize=2, linestyle="None", label="microbatch 4")

# Configure x-axis ticks: use labels from the first dataset
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer=True))
plt.xticks(
    ticks=x1,
    labels=labels1,
    rotation=45,
    ha="right"
)

plt.xlabel("index(only i1 value marked)")
plt.ylabel("Ratio")
plt.title("Comparison of Ratios from mbatch2 to mbatch4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
