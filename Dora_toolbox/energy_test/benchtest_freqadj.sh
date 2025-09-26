#!/bin/bash

#work for linux, ubuntu on respberry pi...
#must have the administration ability to use cgroup to modify the cpu frequency

# Activate your environment
source ~/anaconda3/etc/profile.d/conda.sh  # adjust path to your conda
conda info --envs
#conda activate qwen  # Replace `myenv` with your actual environment name
#conda create -n qwen python=3.10
conda activate qwen
#pip install -r requirements.txt


# List of model IDs
model_ids=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-14B"
)

# Get number of CPU cores
max_core=$(nproc --all)
echo "Detected $max_core CPU cores"

# Get list of available CPU frequencies from CPU0
freqs=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
freq_array=($freqs)

# Loop through each model
for model_id in "${model_ids[@]}"; do
  echo "Running benchmark for model: $model_id"

  # Loop through each available frequency
  for freq in "${freq_array[@]}"; do
    echo "  -> Setting CPU frequency to: $freq Hz"

    # Apply frequency to all cores
    for ((core=0; core<max_core; core++)); do
      sudo cpufreq-set -c "$core" -f "$freq"
    done

    # Wait a moment to let frequency take effect
    sleep 1

    # Run the benchmark
    python3 bench_energy.py --model_id "$model_id" --name "pi"

    echo "  -> Done with frequency $freq Hz"
  done

  echo "Finished all frequency configurations for $model_id"
done
