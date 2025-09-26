#!/bin/bash

# Clear or create the log file
#echo "Input1 Input2 Input3, Output" > log.txt

# Triple nested loop for all combinations from (1,1,1) to (40,40,40)
for i in $(seq 2 10); do
  for j in $(seq $i 10); do
    for k in $(seq 3 10); do
      # Run setup script before running the main one
      python3 sample_generation.py $i $j $k

      # Run the main script and capture the output
      output=$(python3 classical_jobshop.py $i $j $k)


      # Extract the first word or line
      first_output=$(echo "$output" | head -n 1 | awk '{print $1}')
      second_output=$(echo "$output" | sed -n 2p| awk '{print $1}')
      thrid_output=$(echo "$output" | sed -n 3p| awk '{print $1}')
      # Check for the "infeasible" string
      if [[ "$first_output" == "SolveStatus.INFEASIBLE" ]]; then
        echo "Infeasible solution detected at input $i $j $k. Stopping..."
        exit 1
      fi

      # Append input and output to the log file
      echo "$i $j $k, $second_output, $thrid_output" >> RCPSPprofile.txt
    done
  done
done
