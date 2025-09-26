#!/bin/bash

# List of model IDs
model_ids=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-14B"
)

# Get number of CPU cores (0-based index)
max_core=$(nproc --all)
let max_core_index=max_core-1

# Loop through each model
for model_id in "${model_ids[@]}"; do
  echo "Running benchmark for model: $model_id"

  # Loop through core ranges: 0, 0-1, 0-2, ..., 0-max
  for ((end_core=0; end_core<=max_core_index; end_core++)); do
    core_range="0-$end_core"
    echo "  -> Using cores: $core_range"
    
    # Run with taskset binding
    taskset -c $core_range python3 bench_energy.py --model_id "$model_id"

    echo "  -> Done with cores $core_range"
  done

  echo "Finished all core configurations for $model_id"
done
