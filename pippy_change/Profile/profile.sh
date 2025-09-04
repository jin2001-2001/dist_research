#!/usr/bin/env bash
set -euo pipefail

# Ask for sudo once
sudo -v

# -------- Config --------
BATCH_SIZES=(1 2 3 4 5)
UTILS=(100 70)                 # target CPU utilization percentages
CGROUP_NAME="pybench_group"
PERIOD_US=1000000              # 1,000,000 µs (1s) period, matches your example
# ------------------------

# Make sure cgroup v2 cpu controller is available
if [ ! -d /sys/fs/cgroup ]; then
  echo "cgroup fs not found at /sys/fs/cgroup"; exit 1
fi

# Detect cores
CORES=$(nproc --all)
echo "Detected CPU cores: ${CORES}"

# Ensure clean start
if [ -d "/sys/fs/cgroup/${CGROUP_NAME}" ]; then
  echo "Removing existing cgroup ${CGROUP_NAME}..."
  sudo bash -c "echo max > /sys/fs/cgroup/${CGROUP_NAME}/cpu.max" || true
  sudo rmdir "/sys/fs/cgroup/${CGROUP_NAME}" || true
fi

for util in "${UTILS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    echo "=============================================="
    echo "Run: CPU util ${util}%  |  batch ${bs}"
    echo "=============================================="

    # Create cgroup
    sudo mkdir -p "/sys/fs/cgroup/${CGROUP_NAME}"

    # Compute quota:
    # For cgroup v2, cpu.max = "<quota> <period>"
    # quota is microseconds of CPU time across *all* cores per period.
    # To target X% of full machine: quota = (X/100) * (CORES * PERIOD)
    QUOTA=$(( util * CORES * PERIOD_US / 100 ))
    printf "%d %d\n" "$QUOTA" "$PERIOD" | sudo tee "/sys/fs/cgroup/${CGROUP_NAME}/cpu.max" >/dev/null
    echo "Set cpu.max to ${QUOTA} ${PERIOD_US}"


    # Move this shell into the cgroup (children inherit it)
    echo $$ | sudo tee "/sys/fs/cgroup/${CGROUP_NAME}/cgroup.procs" >/dev/null

    # Run workload
    OUTFILE="./cpu${util}_bs${bs}.json"
    echo "Output -> ${OUTFILE}"
    python measure_layers.py --batch "${bs}" --out "${OUTFILE}" --host "cpu${util}"

    # Reset CPU limit and remove cgroup
    sudo bash -c "echo max > /sys/fs/cgroup/${CGROUP_NAME}/cpu.max"
    sudo rmdir "/sys/fs/cgroup/${CGROUP_NAME}"
  done
done

echo "All runs completed."
