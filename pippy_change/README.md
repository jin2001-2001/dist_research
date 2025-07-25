# Training Guide

## Environment Setup

3 Raspberry Pi devices  
Minimum memory: 8GB  
Test system: **Ubuntu 22.04**

1. Install Miniconda

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```

2. Create the `qwencpu` environment and configure it

```sh
conda create -n qwencpu python=3.11
conda activate qwencpu

pip install torch==2.7.1+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install psutil transformers datasets
```

Locate the `pipelining_source_code` folder:

```sh
find $(python -c "import site; print(site.getsitepackages()[0])") -type d -name pipelining
```

Replace all files in the `pipelining_source_code` folder with those from `pipelining_source_code`.

---

## Running the Experiment

### Training Commands

Run the following on **rank 0 (master)** — the master **must** be started first:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=0 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 16 --microbatch_num 4 --profile_batch 1
```

Run the following on **rank 1**:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=1 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 16 --microbatch_num 4 --profile_batch 1
```

Run the following on **rank 2**:

```sh
torchrun --nproc-per-node=1 --nnodes=3 --node-rank=2 --master-addr=[replace with master IP] --master-port=29500 three_stages_simple_pp.py --train_steps 5 --batch_size 16 --microbatch_num 4 --profile_batch 1
```

---

### Parameter Explanation

| Parameter          | Data Type   | Description                                                                                                                                      | Default Value                                         | Typical Usage & Notes                                                                                                                                                       |
| ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--train_steps`    | `int`       | **Total number of training steps** (i.e., optimizer updates).<br>If omitted or set to `None`, the script runs a full epoch using the DataLoader. | `None`                                                | For *testing/debugging*, set a small number like `--train_steps 200`.<br>If set **greater than** DataLoader length, `StopIteration` will be raised after data is exhausted. |
| `--batch_size`     | `int`       | **Number of samples each rank (process) processes per forward/backward pass**.<br>Global batch size = `batch_size × number of DP ranks`.         | Reads env variable `BATCH_SIZE`, defaults to `16`     | Increasing it improves throughput but also memory usage.<br>Command-line argument overrides environment variable, e.g., `--batch_size 32` overrides `BATCH_SIZE=16`.        |
| `--microbatch_num` | `int`       | **How many micro-batches to split each batch into for pipeline training**.<br>Affects pipeline bubbles, peak memory, and efficiency.             | Reads env variable `MICROBATCH_NUM`, defaults to `4`  | Should satisfy `batch_size % microbatch_num == 0`.<br>Larger values reduce memory use but increase communication/scheduling overhead.                                       |
| `--profile_batch`  | `str / int` | **Which batch index to profile for performance**.<br>This value is written back to env variable `PROFILE_BATCH` for internal use.                | Reads env variable `PROFILE_BATCH`, defaults to `"0"` | - `0`: profile the 1st batch (commonly used for warm-up).<br>- `k`: profile the (k+1)-th batch.                                                                             |

If you encounter **OOM**, adjust `batch_size` and `microbatch_num`.

---

## Data Collection

You can change the profiled batch by modifying `profile_batch`.

After script execution, files like `timeline_batch0_all.json` will be generated.


