import pandas as pd
import matplotlib.pyplot as plt


# Load CSV
df = pd.read_csv("summerize2.csv", skip_blank_lines=True)
df = df.dropna(how='all')

# Convert types for numerical plotting
df['throughput'] = pd.to_numeric(df['throughput'], errors='coerce')
df['cpu_increment_power'] = pd.to_numeric(df['cpu_increment_power'], errors='coerce')
df['cpu_power'] = pd.to_numeric(df['cpu_power'], errors='coerce')
df['cpu_energy'] = pd.to_numeric(df['cpu_energy'], errors='coerce')
df['cpu_idle_energy'] = pd.to_numeric(df['cpu_idle_energy'], errors='coerce')

# Filter for two models
models = ["Qwen/Qwen3-0.6B"]
          #, "Qwen/Qwen3-1.7B"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


# Filter for Raspberry Pi and maxt=64
filtered = df[(df['name'] == 'desktop_ubuntu') & (df['maxt'] == 64)]
filtered = filtered.dropna(subset=['throughput', 'cpu_energy', 'cpu_idle_energy'])

filtered1 = df[(df['name'] == 'desktop_ubuntuo') & (df['maxt'] == 64)]
filtered1 = filtered1.dropna(subset=['throughput', 'cpu_energy', 'cpu_idle_energy'])

for model_id in models:
    sub_df = filtered[filtered['model_id'] == model_id].sort_values(by='throughput')
    sub_df1 = filtered1[filtered1['model_id'] == model_id].sort_values(by='throughput')
    #ax1.plot(
    #    sub_df['throughput'],
    #    sub_df['cpu_energy'],
    #    marker='o',
    #    label=model_id+" Total inference energy"
    #)
    #ax1.plot(
    #    sub_df['throughput'],
    #    sub_df['cpu_idle_energy'],
    #    marker='o',
    #    label=model_id+ " Background energy"
    #)
    ax1.plot(
        sub_df['throughput'],
        sub_df['cpu_energy']- sub_df['cpu_idle_energy'],
        marker='o',
        label=model_id+ "cpuUtil: Total inference energy - Background energy"
    )
    ax1.plot(
        sub_df1['throughput'],
        sub_df1['cpu_energy']- sub_df1['cpu_idle_energy'],
        marker='o',
        label=model_id+ "cpuFreq: Total inference energy - Background energy"
    )

ax1.set_xlabel("Throughput (tokens/sec)")
ax1.set_ylabel("energy (kWh)")
ax1.set_title("Throughput vs energy consumption")
ax1.legend()
ax1.grid(True)

# Filter for Desktop and maxt=64
filtered = df[(df['name'] == 'desktop_ubuntu') & (df['maxt'] == 64)]
filtered = filtered.dropna(subset=['throughput', 'cpu_power', 'cpu_increment_power'])

filtered1 = df[(df['name'] == 'desktop_ubuntuo') & (df['maxt'] == 64)]
filtered1 = filtered1.dropna(subset=['throughput', 'cpu_power', 'cpu_increment_power'])

for model_id in models:
    sub_df = filtered[filtered['model_id'] == model_id].sort_values(by='throughput')
    sub_df1 = filtered1[filtered1['model_id'] == model_id].sort_values(by='throughput')
    ax2.plot(
        sub_df['throughput'],
        #sub_df['energy_consumed']-sub_df['gpu_energy'],
        sub_df['cpu_increment_power'],
        marker='o',
        label=model_id+"cpuUtil: increment(pure) inference Power"
    )
    ax2.plot(
        sub_df1['throughput'],
        #sub_df['energy_consumed']-sub_df['gpu_energy'],
        sub_df1['cpu_increment_power'],
        marker='o',
        label=model_id+"cpuFreq: increment(pure) inference Power"
    )
    #ax2.plot(
    #    sub_df['throughput'],
    #    #sub_df['energy_consumed']-sub_df['gpu_energy'],
    #    sub_df['cpu_power']-sub_df['cpu_increment_power'],
    #    marker='o',
    #    label=model_id+" background Power"
    #)

ax2.set_xlabel("Throughput (tokens/sec)")
ax2.set_ylabel("Power (W)")
ax2.set_title("Throughput vs Power Consumed for Qwen Models on Desktop")
ax2.legend()
ax2.grid(True)


plt.tight_layout()
plt.show()