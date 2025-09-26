# General

To install the required packages, run:

```bash
pip install -r requirements.txt
```

To enable cgroup2 on Rpi, please refer to the OS version of Rpi, to install cgroup. I installed it based on chatGPT result.

---

# Latency Test

To run the latency benchmark:

```bash
cd latency_test
python3 bench_latency.py
```

## Parser Configuration

* **Line 132**: Adjusts the input token size (default is `50`).
  If inference runs too slowly, consider decreasing this value.

* **Line 136**: Specifies the file name for storing the latency results.
  Please set this to any filename you prefer.

## Model Selection

* **Lines 140–141**: By default, the script will test all models continuously.
  If you'd like to test models individually, modify the list to contain only one model ID at a time.

## Time refer
you can read time recorded in rpirpi_inference.csv(or the file name you specified) if you want.


---

# Energy Test

To run the latency benchmark:

```bash
cd energy_test
./benchtest_utiqadj.sh
```

## Configuration

* how to adjust cpu utilization level? first, please run 
```bash
nproc --all
```
to get maximum cores the Rpi has. Then, the maximum utilization level should be 100*#cpus. For example, the original maximum utilization level is 1000 beacuse my computer has 10 virtual cores.

* the script will test all different cpu utility levels automatically(at line 16). If you want to do exp with level one by one, please set only one value in **utilization_levels** list each time. By default, the lowest cpu utilization level should be 20 and the granularity is also 20. If one Rpi it runs too slow, consider raise the lowest bound to 40 or higher...

* model selection: at line 10 of the shell, change the model id to higher if you decide to task larger model...

## Warm-up Phase

* Please note that the script includes a **warm-up run**, during which we **do not collect energy data**.

* We only begin recording energy after the following message is printed:

```text
begin real loading tests:
```

* If it's difficult to monitor the exact start of the real test and read the electric meter in time,
  you can **comment out lines 115–117** of energy_test/bench_energy.py to skip the warm-up phase altogether.


## Time refer
you can read time recorded in rpi.csv if you want. Noticed that the energy recorded in this file is not accurate...

---
