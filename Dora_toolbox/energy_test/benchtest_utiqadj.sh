#!/bin/bash


# Ask for password once
sudo -v


# List of model IDs
model_ids=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  #"Qwen/Qwen3-4B"
)

# Utilization levels
utilization_levels=($(seq 15 15 700))


# Get the number of CPU cores
max_core=$(nproc --all)

echo "Detected CPU cores: $max_core"

for model_id in "${model_ids[@]}"; do
  echo "==============================="
  echo "Benchmarking model: $model_id"
  echo "==============================="

  for util in "${utilization_levels[@]}"; do
    echo "-------------------------------"
    echo "Target utilization: ${util}%"
    echo "-------------------------------"

    # Create cgroup
    sudo mkdir -p /sys/fs/cgroup/mygroup


    # Calculate quota:
    # quota = utilization% * (cores * period)
    # period = 1,000,000 microseconds
    total_period=$(( 1000000 ))
    quota=$((util * total_period / 100 ))
    sudo bash -c "echo \"$quota $total_period\" > /sys/fs/cgroup/mygroup/cpu.max"
    echo "Set cpu.max to $quota $total_period"


    # Move this shell into the cgroup
    echo $$ | sudo tee /sys/fs/cgroup/mygroup/cgroup.procs >/dev/null

    # Run your workload
    python bench_energy.py --model_id "$model_id" --name "amdcpu" --output "formal_summerize_ftest.csv"

    # Reset CPU limit
    sudo bash -c "echo max > /sys/fs/cgroup/mygroup/cpu.max"

    # Remove cgroup
    sudo rmdir /sys/fs/cgroup/mygroup
  done
done
