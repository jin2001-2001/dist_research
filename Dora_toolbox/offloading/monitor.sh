#!/bin/bash

# === CONFIG ===
GPU_LOG="gpu_memoryH100_1.7_offloading.log"
CPU_LOG="cpu_memoryH100_1.7_offloading.log"
TRAIN_CMD="accelerate launch offloading_hf.py"

# === 1. Start GPU memory monitor ===
echo "Starting GPU memory logging..."
(
    echo "Time, GPU Memory Usage (MiB)"
    while true; do
        echo "$(date +%T), $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd ',' -)"
        sleep 0.2
    done
) >> "$GPU_LOG" &
GPU_MON_PID=$!

# === 2. Start CPU memory monitor ===
echo "Starting CPU memory logging..."
(
    while true; do
        echo -n "$(date +%T), "
        free -m | awk '/^Mem:/ {print $3 " MiB used / " $2 " MiB total"}'
        sleep 1
    done
) >> "$CPU_LOG" &
CPU_MON_PID=$!

# === 3. Run training ===
echo "Running training script..."
$TRAIN_CMD

# === 4. Cleanup ===
echo "Training complete. Killing loggers..."
kill $GPU_MON_PID
kill $CPU_MON_PID
echo "Done. Logs saved to $GPU_LOG and $CPU_LOG."
