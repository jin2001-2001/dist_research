#!/bin/bash

# Set a unique log file for power measurements
POWERLOG="powerjoular_log1.txt"

# Start PowerJoular in the background
sudo powerjoular -f $POWERLOG > /dev/null 2>&1 &
PJ_PID=$!

echo "PowerJoular started with PID $PJ_PID"

#conda init
#conda activate qwencpu
# Run your Python script
python3 bench_energy.py

# Stop PowerJoular after Python script finishes
sudo kill -SIGINT $PJ_PID

echo "PowerJoular stopped. Results saved to $POWERLOG"
