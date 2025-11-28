#!/bin/bash

# ========================================================
# Usage:
#   ./run.sh <mbps> <#mbatch> <name1> <name2> ...
#
# Example:
#   ./run.sh 500 6 A6000_bs8 4090_bs4 Pi5_bs1
#
# This will pass:
#   --mbps 500   → to parse_bandwidth_csv.py
#   --nbatch 6   → to plan_from_measured.py
# ========================================================

if [ $# -lt 3 ]; then
  echo "Usage: $0 <mbps> <#mbatch> <name1> <name2> ..."
  exit 1
fi

# First argument = MBPS bandwidth
MBPS=$1
shift

# Second argument = NBATCH (microbatch count)
NBATCH=$1
shift

# Remaining args = device profile names
inputs=""
pureinputs=""
for f in "$@"; do
  pureinputs="$pureinputs ${f}"
  inputs="$inputs ../Profile_exp_1.7/${f}.json"
done

# Last device is used as layer profile
LAST_INPUT=${!#}
lastinput="../Profile_exp_1.7/${LAST_INPUT}.json"

echo "=== Arguments ==="
echo "MBPS (bandwidth): $MBPS"
echo "NBATCH:           $NBATCH"
echo "inputs:           $inputs"
echo "last input:       $lastinput"
echo "pureinputs:       $pureinputs"
echo "==================="

echo "Running..."

python3 aggregate_capacities.py --inputs $inputs

# Pass MBPS to parse_bandwidth_csv.py
python3 parse_bandwidth_csv.py --inputs $inputs --mbps $MBPS

python3 aggregate_mem_profile.py --inputs $pureinputs

python3 plan_from_measured.py \
  --layers $lastinput \
  --capacities capacities.json \
  --bandwidth bandwidth_matrix.json \
  --memory_model memory_model.json \
  --devices devices.json \
  --tied_embed false \
  --max_stages None \
  --out_dir out_measured \
  --nbatch $NBATCH
