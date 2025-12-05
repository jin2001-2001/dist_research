import os
import json

folder = "profile"  # <-- change to your folder name
num = 100*5
def extend_to_100(lst):
    """Extend list to length 100 by repeating the last element."""
    if len(lst) == 0:
        return [0] * num  # safety case
    if len(lst) >= num:
        return lst[:num]
    return lst + [lst[-1]] * (num - len(lst))

for fname in os.listdir(folder):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(folder, fname)

    with open(path, "r") as f:
        data = json.load(f)

    # ---- modify num_layers ----
    data["model"]["num_layers"] = num

    # ---- extend lists ----
    params_list = data["model"]["parameters"]["parameters_per_layer_bytes"]
    compute_list = data["execution_time"]["layer_compute_total_ms"]
    mem_list = data["execution_memory"]["layer_memory_total_mb"]

    data["model"]["parameters"]["parameters_per_layer_bytes"] = extend_to_100(params_list)
    data["execution_time"]["layer_compute_total_ms"] = extend_to_100(compute_list)
    data["execution_memory"]["layer_memory_total_mb"] = extend_to_100(mem_list)

    # ---- save back ----
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated: {fname}")
