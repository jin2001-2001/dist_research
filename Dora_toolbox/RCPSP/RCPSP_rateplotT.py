import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


x_vals = []
y_vals = []

x1_vals = []
y1_vals = []

x2_vals = []
y2_vals = []

plt.figure(figsize=(8, 5))


with open('./RCPSPprofile10.txt', 'r') as f:
    for line in f:
        if ',' not in line:
            continue  # skip malformed lines
        parts = line.strip().split(',')
        input_vals = list(map(int, parts[0].strip().split()))
        output_val = float(parts[1].strip())

        x = input_vals[0] * input_vals[1] * input_vals[2]
        y = output_val

        x1_vals.append(x)
        y1_vals.append(y)
#plt.plot(x1_vals, y1_vals, 'o', color=(0.6, 1.0, 0.6, 0.5), markersize=2)  
#smoothed1 = lowess(y1_vals, x1_vals, frac=0.1)
#plt.plot(smoothed1[:, 0], smoothed1[:, 1], color='green', linewidth=2, label='10 sec bound trend')

with open('./RCPSPprofile30.txt', 'r') as f:
    for line in f:
        if ',' not in line:
            continue  # skip malformed lines
        parts = line.strip().split(',')
        input_vals = list(map(int, parts[0].strip().split()))
        output_val = float(parts[1].strip())
        x = input_vals[0] * input_vals[1] * input_vals[2]
        y = output_val

        x2_vals.append(x)
        y2_vals.append(y)

#plt.plot(x_vals, y_vals, 'o', color =(0.3, 0.5, 1.0, 0.4), markersize=2)  
#smoothed = lowess(y_vals, x_vals, frac=0.1)
#plt.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2, label='60 sec bound trend')


with open('./RCPSPprofile60.txt', 'r') as f:
    for line in f:
        if ',' not in line:
            continue  # skip malformed lines
        parts = line.strip().split(',')
        input_vals = list(map(int, parts[0].strip().split()))
        output_val = float(parts[1].strip())
        output_t = float(parts[2].strip())

        x = input_vals[0] * input_vals[1] * input_vals[2]
        y = output_t
        if y<100:
            x_vals.append(x)
            y_vals.append(y)


plt.plot(x_vals, y_vals, 'o',color=(1.0, 0.0, 0.0, 0.3), markersize=2)  
smoothed2 = lowess(y_vals, x_vals, frac=0.1)
plt.plot(smoothed2[:, 0], smoothed2[:, 1], color='red', linewidth=2, label='60 sec bound trend')
# Plot
#plt.plot(x_vals, y_vals, 'bo', markersize=3)  # blue dots
plt.legend()
plt.xlabel('# of tasks (= stage × microbatch × subtasks)')
plt.ylabel('running time of the tasks')
plt.title('running time vs. # of tasks')
plt.grid(True)
plt.tight_layout()
plt.show()
