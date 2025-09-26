#!/bin/bash

set -e
# Ask for password once
sudo -v

# List of model IDs
model_ids=(
  "Qwen/Qwen3-0.6B"
#  "Qwen/Qwen3-1.7B"
)

# Utilization levels
utilization_levels=($(seq 10 5 70))

for model_id in "${model_ids[@]}"; do
  echo "==============================="
  echo "Benchmarking model: $model_id"
  echo "==============================="

  for util in "${utilization_levels[@]}"; do
    echo "-------------------------------"
    echo "Background CPU load: ${util}%"
    echo "-------------------------------"

    # Start stress-ng in background
    stress-ng --cpu "$(nproc)" --cpu-load "$util" --cpu-method matrixprod &
    STRESS_PID=$!

    # Wait for load to stabilize
    sleep 5

    # Run your benchmark
    python3 bench_energy.py --model_id "$model_id" --name "desktop_ubuntu" --output "summerize3.csv"

    # Stop the stress
    kill $STRESS_PID
    echo "Finished benchmark for utilization ${util}%"
  done
done
