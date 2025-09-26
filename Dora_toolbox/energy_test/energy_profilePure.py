import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.signal import savgol_filter

def data_predeal(df):

    agg_results = []

    for gid, group in df.groupby("id"):
        cpu_energy_vals = (group["selfWEnergy"]/group['total_new_tokens']).to_numpy()
        throughput_vals = group["throughput"].values
        tokens= group['total_new_tokens'].values

        # Median absolute deviation (MAD)
        median_val = np.median(cpu_energy_vals)
        mad = np.median(np.abs(cpu_energy_vals - median_val))

        # Keep only values within k*MAD of median
        k = 1  # can adjust sensitivity
        mask = np.abs(cpu_energy_vals - median_val) <= k * mad
        filtered_cpu_energy = cpu_energy_vals[mask]

        # Aggregate
        robust_avg_cpu_energy = np.mean(filtered_cpu_energy)
        avg_throughput = np.mean(throughput_vals[mask])  # keep throughput aligned
        avg_tokens = np.mean(tokens[mask])

        agg_results.append((gid, avg_throughput,avg_tokens, robust_avg_cpu_energy))

    # Convert to DataFrame
    agg_df = pd.DataFrame(agg_results, columns=["id", "throughput","total_new_tokens", "PerselfWEnergy"])
    return agg_df



# Load CSV
df = pd.read_csv("formal_summerize_ftest.csv", skip_blank_lines=True)
df = df.dropna(how='all')


# Convert types for numerical plotting
df['total_new_tokens'] = pd.to_numeric(df['total_new_tokens'], errors='coerce')
df['throughput'] = pd.to_numeric(df['throughput'], errors='coerce')
#df['cpu_increment_power'] = pd.to_numeric(df['cpu_increment_power'], errors='coerce')
df['cpu_power'] = pd.to_numeric(df['cpu_power'], errors='coerce')
df['cpu_energy'] = pd.to_numeric(df['cpu_energy'], errors='coerce')
df['cpu_idle_energy'] = pd.to_numeric(df['cpu_idle_energy'], errors='coerce')
df['selfWEnergy'] = pd.to_numeric(df['selfWEnergy'], errors='coerce')
df['selfIEnergy'] = pd.to_numeric(df['selfIEnergy'], errors='coerce')

df["id"] = [i % 46 for i in range(len(df))]

# Filter for two models
models = ["Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B"
         ]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

unitbgE = np.mean(df['selfIEnergy'].values)/10.00
tokens = int(np.mean(df['total_new_tokens'].values))
print(unitbgE, tokens)

# Filter for Raspberry Pi and maxt=64
filtered = df[(df['name'] == 'amdcpu')]
filtered = filtered.dropna(subset=['id', 'throughput','total_new_tokens', 'selfWEnergy', 'selfIEnergy'])

#filtered1 = df[(df['name'] == 'desktop_ubuntuo') & (df['maxt'] == 64)]
#filtered1 = filtered1.dropna(subset=['throughput', 'cpu_energy', 'cpu_idle_energy'])

for model_id in models:
    sub_df = filtered[filtered['model_id'] == model_id].sort_values(by='throughput')
    agg_df = data_predeal(sub_df)
    agg_df = agg_df.sort_values(by='throughput')

    ax1.plot(
        agg_df['throughput'],
        (agg_df["PerselfWEnergy"] - 1/(agg_df['throughput'])*unitbgE),        
        marker='o',
        label=model_id+ "cpuUtil: Energy consumption"
    )

    #y_raw = (agg_df['cpu_energy'] - tokens/(agg_df['throughput'])*unitbgE).values
    #y_smooth = savgol_filter(y_raw, window_length=5, polyorder=2)

    #ax1.plot(agg_df['throughput'], y_smooth, marker='o', label=model_id + "cpuUtil: Energy consumption")



ax1.set_xlabel("Throughput (tokens/sec)")
ax1.set_ylabel("energy (kWh)")
#ax1.set_ylim(0, 5e-6)
ax1.set_title("Throughput vs energy consumption")
ax1.legend()
ax1.grid(True)

# Filter for Desktop and maxt=64
#filtered = df[(df['name'] == 'desktop_ubuntu') & (df['maxt'] == 64)]
#filtered = filtered.dropna(subset=['throughput', 'cpu_power', 'cpu_increment_power'])
#
#filtered1 = df[(df['name'] == 'desktop_ubuntuo') & (df['maxt'] == 64)]
#filtered1 = filtered1.dropna(subset=['throughput', 'cpu_power', 'cpu_increment_power'])
#
#for model_id in models:
#    sub_df = filtered[filtered['model_id'] == model_id].sort_values(by='throughput')
#    sub_df1 = filtered1[filtered1['model_id'] == model_id].sort_values(by='throughput')
#    ax2.plot(
#        sub_df['throughput'],
#        #sub_df['energy_consumed']-sub_df['gpu_energy'],
#        sub_df['cpu_increment_power'],
#        marker='o',
#        label=model_id+"cpuUtil: increment(pure) inference Power"
#    )
#    ax2.plot(
#        sub_df1['throughput'],
#        #sub_df['energy_consumed']-sub_df['gpu_energy'],
#        sub_df1['cpu_increment_power'],
#        marker='o',
#        label=model_id+"cpuFreq: increment(pure) inference Power"
#    )
#    #ax2.plot(
#    #    sub_df['throughput'],
#    #    #sub_df['energy_consumed']-sub_df['gpu_energy'],
#    #    sub_df['cpu_power']-sub_df['cpu_increment_power'],
#    #    marker='o',
#    #    label=model_id+" background Power"
#    #)

ax2.set_xlabel("Throughput (tokens/sec)")
ax2.set_ylabel("Power (W)")
ax2.set_title("Throughput vs Power Consumed for Qwen Models on Desktop")
ax2.legend()
ax2.grid(True)


plt.tight_layout()
plt.show()