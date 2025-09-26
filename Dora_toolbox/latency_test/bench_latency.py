#import promptbench as pb
from huggingface_hub import login
from datasets import load_dataset
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import argparse
import os


def idle_cooldown(duration=5):
    time.sleep(duration)


def idle_bench(duration=60):
    print(f"Simulating idle task for {duration} seconds...")
    time.sleep(duration)
    print("Idle task finished.")


def per_bench(maxt, model,prompts, tokenizer , max_length = 16):
        total_new_tokens, total_time = 0, 0.0
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

            model_inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

            #model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            start = time.time()

            generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=maxt
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            thinking_content = tokenizer.decode(output_ids[:], skip_special_tokens=True).strip("\n")

            #outputs.append(out)
            duration = time.time() - start
            new_tokens = len(output_ids)
            total_new_tokens += new_tokens
            total_time += duration
            print('BEGIN', 'READY', thinking_content,'END')
        return total_new_tokens, total_time

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

def time_bench(maxt = 64, device = "cpu" , model_id = "Qwen/Qwen3-0.6B"):
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
    sample_n = 6
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
        ,device_map="auto"
    )

    total_new_tokens, total_time = 0, 0.0

    print("warm up begin:\n")
    per_bench_warmup(maxt, model,prompts[0:sample_n//3], tokenizer)
    print("warm up over\n")

    print("begin real loading tests:\n")
    total_new_tokens, total_time = per_bench(1, model,prompts[sample_n//3:sample_n], tokenizer, max_length = 50)
    #avg_latency = total_time / len(prompts)
    #throughput = total_new_tokens / total_time
    #print(f"Avg latency per prompt: {avg_latency:.3f}s")
    print(f"TTFT: {total_time:.2f} sec\n")
    total_new_tokens, total_time1 = per_bench(maxt, model,prompts[sample_n//3:sample_n], tokenizer, max_length = 50)
    tbt_t = (total_time1-total_time)/(total_new_tokens-(sample_n-sample_n//3))
    print(f"TBT: {tbt_t:.2f} sec\n")

    return total_time, tbt_t, total_time1, total_new_tokens
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark time latency, and write to summerize.txt")
    parser.add_argument("--name", type=str, default="Rpi", help="Device name, e.g., 'respberry pi'")
    parser.add_argument("--maxt", type=int, default=50, help="Max tokens, e.g., 64")
    parser.add_argument("--device", type=str, default="cuda", help="Device type, e.g., 'cpu'")
    #parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B", help="Model ID, e.g., 'Qwen/Qwen3-0.6B'")
    #parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="rpi_inference.csv", help="Path to output file")

    args = parser.parse_args()

    #model_lists = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]
    model_lists = ["drush8/Qwen3-0.6B", "drush8/Qwen3-1.7B"]
    for model in model_lists:
        TTFT, TBT, total_time, total_new_tokens = time_bench(args.maxt, args.device , model)
        #carbon will generate a emissions.csv file at the same location folder.

        output_file_path = args.output
        print("beginning writing to the recoder file...")

        # Format the line
        headers = ["name", "maxt", "device", "model_id", "TTFT", "TBT","total_new_tokens", "total_time"]
        #line = f"{args.device}, {args.model_id}, {args.maxt}, {throughput}, {total_new_tokens}, {total_time}, {duration}, {cpu_energy}, {gpu_energy}, {ram_energy}, {energy_consumed}\n"
        row = [args.name, args.maxt, args.device, model, TTFT,TBT, total_new_tokens, total_time]
        write_header = not os.path.exists(output_file_path)
        with open(output_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)

    print(f"Data written to {output_file_path} successfully.")

