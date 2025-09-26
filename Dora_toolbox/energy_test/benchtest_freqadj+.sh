#!/bin/bash

#work for linux, ubuntu on respberry pi...
#must have the administration ability to use cgroup to modify the cpu frequency

# Activate your environment
#source ~/anaconda3/etc/profile.d/conda.sh  # adjust path to your conda
#conda info --envs
#conda activate qwen  # Replace `myenv` with your actual environment name
#conda create -n qwen python=3.10
#conda activate qwen
#pip install -r requirements.txt


# List of model IDs
#!/bin/bash

# List of model IDs
model_ids=(
  "Qwen/Qwen3-0.6B"
#  "Qwen/Qwen3-1.7B"
#  "Qwen/Qwen3-4B"
)

# Ask for password once
sudo -v

# Get number of CPU cores
max_core=$(nproc --all)
echo "Detected $max_core CPU cores"

# Frequency sweep from 600MHz to 5000MHz in 200MHz steps
for model_id in "${model_ids[@]}"; do
  echo "Running benchmark for model: $model_id"

  for ((mhz=600; mhz<=5000; mhz+=200)); do
    freq="${mhz}MHz"
    echo "  -> Setting CPU frequency to: $freq"

    # Set both upper and lower bound to the same frequency (fixed)
    sudo cpupower frequency-set -d "$freq" -u "$freq"

    # Wait a bit for the setting to apply
    sleep 1

    # Run benchmark
    sudo env "PATH=$PATH" "CONDA_PREFIX=$CONDA_PREFIX" python3 bench_energy.py --model_id "$model_id" --name "desktop_ubuntu"

    echo "  -> Done with frequency $freq"
  done

  echo "Finished all frequency configurations for $model_id"
done


