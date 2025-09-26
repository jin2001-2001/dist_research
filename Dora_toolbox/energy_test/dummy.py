from huggingface_hub import login
from datasets import load_dataset
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import track_emissions
import csv
import argparse
import os
import subprocess

@track_emissions()
def idle_bench(duration=10):
    print(f"Simulating idle task for {duration} seconds...")
    time.sleep(duration)
    print("Idle task finished.")


if __name__ == "__main__":
    #idle_bench()
    energy = int(subprocess.check_output(["sudo", "./read_rapl.sh"]))
    print("Energy (µJ):", energy)