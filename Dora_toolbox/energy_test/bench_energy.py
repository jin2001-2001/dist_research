#import promptbench as pb
from huggingface_hub import login
from datasets import load_dataset
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import track_emissions
import csv
import argparse
import os
import subprocess



max_Erange = int(subprocess.check_output(["sudo", "./read_rapl_max.sh"]))

def calE(pre, cur):
    if cur > pre:
        return cur-pre
    else:
        return max_Erange-pre+cur

def idle_cooldown(duration=5):
    time.sleep(duration)

@track_emissions(measure_power_secs=3)
def idle_bench(duration=60):
    print(f"Simulating idle task for {duration} seconds...")
    PreE = int(subprocess.check_output(["sudo", "./read_rapl.sh"]))
    time.sleep(duration)
    CurE = int(subprocess.check_output(["sudo", "./read_rapl.sh"]))
    print("Idle task finished.")
    return calE(PreE,CurE)

@track_emissions(measure_power_secs=3)
def per_bench(maxt, model,prompts, tokenizer):
        real_start = time.time()
        PreE = int(subprocess.check_output(["sudo", "./read_rapl.sh"]))
        total_new_tokens, total_time = 0, 0.0
        for i in range(10):
            if time.time() - real_start > 30:
                break
            for prompt in prompts:
                message = [{"role": "user", "content": prompt}]

                #    text = tokenizer.apply_chat_template(
                #    message,
                #    tokenize=False,
                #    add_generation_prompt=True,
                #    enable_thinking=False #enable between thinking and non-thinking modes. Default is True.
                #    )
                #
                ##templete need more latest version of the lib

                model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=200).to(model.device)

                #model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=maxt
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

                thinking_content = tokenizer.decode(output_ids[:], skip_special_tokens=True).strip("\n")

                #outputs.append(out)
                new_tokens = len(output_ids)
                total_new_tokens += new_tokens
                print('BEGIN', 'READY', thinking_content,'END')

        CurE = int(subprocess.check_output(["sudo", "./read_rapl.sh"]))
        real_total = time.time() - real_start
        return total_new_tokens, real_total, calE(PreE,CurE)

def per_bench_warmup(maxt, model,prompts, tokenizer):
        total_new_tokens, total_time = 0, 0.0
        for prompt in prompts:
            message = [{"role": "user", "content": prompt}]
            model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=16).to(model.device)
            start = time.time()
            #print(len(model_inputs["input_ids"][0]))
            generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=maxt
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            thinking_content = tokenizer.decode(output_ids[:], skip_special_tokens=True).strip("\n")
            duration = time.time() - start
            new_tokens = len(output_ids)
            total_new_tokens += new_tokens
            total_time += duration
            print('BEGIN', 'READY', thinking_content,'END')
        return total_new_tokens, total_time

def time_bench(maxt =50, device = "cpu" , model_id = "Qwen/Qwen3-0.6B"):
    ## Load dataset — here using MMLU (can select any supported dataset)
    #dataset = pb.DatasetLoader.load_dataset("mmlu")
    ##print("Available splits:", dataset.keys())
    #
    #prompts = []
    #for idx, example in enumerate(dataset):
    #    if idx >= 10:
    #        break
    #    prompts.append(example)
    #
    #print(prompts)

    #extract out first num prompts from 2wikimqa
    sample_n = 9
    prompts = []
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    for idx, example in enumerate(ds):
        if idx >= sample_n:
            break
        prompts.append(example['prompt'])



    #model_id = "Qwen/Qwen3-0.6B"
    #device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, force_download=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto"
        #,device_map="cpu"
    ).to(device)

    total_new_tokens, total_time = 0, 0.0

    print("warm up begin:\n")
    per_bench_warmup(maxt, model,prompts[0:2], tokenizer)
    print("warm up over\n")

    print("begin real loading tests:\n")
    total_new_tokens, total_time, wE = per_bench(maxt, model,prompts[sample_n//3:sample_n], tokenizer)
    #avg_latency = total_time / len(prompts)
    throughput = total_new_tokens / total_time
    #print(f"Avg latency per prompt: {avg_latency:.3f}s")
    print(f"Overall throughput: {throughput:.2f} tokens/sec\n")

    print("cool down begin:\n")
    idle_cooldown()
    print("cool down over\n")

    print("begin idle test based on the same time span:\n")
    iE = idle_bench(duration=10)

    return throughput, total_new_tokens, total_time, iE, wE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark time latency, and write to summerize.txt")
    parser.add_argument("--name", type=str, default="desktop_5070", help="Device name, e.g., 'respberry pi'")
    parser.add_argument("--maxt", type=int, default=1, help="Max tokens, e.g., 50")
    parser.add_argument("--device", type=str, default="cpu", help="Device type, e.g., 'cpu'")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B", help="Model ID, e.g., 'Qwen/Qwen3-0.6B'")
    #parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="final_summerize1.csv", help="Path to output file")

    args = parser.parse_args()



    throughput, total_new_tokens, total_time, iE, wE = time_bench(args.maxt, args.device , args.model_id)
    #carbon will generate a emissions.csv file at the same location folder.

    # Path to your CSV and output file
    csv_file_path = "emissions.csv"  # change to your actual CSV file path
    output_file_path = args.output
    print("beginning writing to the recoder file...")
    # Read the CSV and extract the last row's required fields
    with open(csv_file_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))
        if reader:  # ensure there's at least one row
            lastlast_row = reader[-2]
            last_row = reader[-1]
            # Extract relevant values
            duration = lastlast_row['duration']
            cpu_energy = lastlast_row['cpu_energy']
            cpu_power2 = lastlast_row['cpu_power']
            cpu_idle_energy = last_row['cpu_energy']
            gpu_energy = lastlast_row['gpu_energy']
            ram_energy = lastlast_row['ram_energy']
            #energy_consumed = last_row['energy_consumed']
            #cpu_power = last_row['cpu_power']
            #gpu_power = last_row['gpu_power']
            #ram_power = last_row['ram_power']
            cpu_power1 = float(cpu_energy)/float(duration)*3600*1000
            cpu_increment_power = (float(cpu_energy) - float(cpu_idle_energy))/float(duration)*3600*1000

            selfWEnergy = float(wE)/ 3.6e12
            selfIEnergy = float(iE)/ 3.6e12
            # Format the line
            headers = ["name", "maxt", "device", "model_id", "throughput", "total_new_tokens", "total_time", "duration",'cpu_power','cpu_power2','cpu_increment_power', "cpu_energy","cpu_idle_energy","selfWEnergy","selfIEnergy", "ram_energy"]
            #line = f"{args.device}, {args.model_id}, {args.maxt}, {throughput}, {total_new_tokens}, {total_time}, {duration}, {cpu_energy}, {gpu_energy}, {ram_energy}, {energy_consumed}\n"
            row = [args.name, args.maxt, args.device, args.model_id, throughput, total_new_tokens, total_time, duration, cpu_power1, cpu_power2, cpu_increment_power, cpu_energy, cpu_idle_energy,selfWEnergy,selfIEnergy, ram_energy]
            write_header = not os.path.exists(output_file_path)

            with open(output_file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)



            print(f"Data written to {output_file_path} successfully.")
        else:
            print("CSV file is empty.")
