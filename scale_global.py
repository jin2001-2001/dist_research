#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from typing import Any
import shutil

# Regex for time fields
MS_PAT = re.compile(r"(?:^|_)time_?ms$", re.IGNORECASE)
S_PAT  = re.compile(r"(?:^|_)time_?s$",  re.IGNORECASE)

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def scale_value(val: Any, ratio: float) -> Any:
    return float(val) * float(ratio)

def scale_time_in_dict(d: dict, ratio_f: float, ratio_b: float) -> None:
    """
    Recursively scale time-like fields, using ratio_f for forward times
    and ratio_b for backward times.
    """
    for k, v in d.items():
        # Recurse
        if isinstance(v, dict):
            scale_time_in_dict(v, ratio_f, ratio_b)
            continue
        if isinstance(v, list):
            if all(isinstance(x, dict) for x in v):
                for elem in v:
                    scale_time_in_dict(elem, ratio_f, ratio_b)
            else: # ok, so all is 
                # Scale lists of times uniformly (use forward ratio by default)
                if "forward" in k.lower():
                    d[k] = [float(x) * ratio_f for x in v]
                elif "backward" in k.lower():
                    d[k] = [float(x) * ratio_b for x in v]
                else:
                    d[k] = [float(x) * ratio_f for x in v]  # default: forward ratio
            continue

        # Scalars
        if "forward" in k.lower() and "backward" in k.lower():
            d[k] = scale_value(v, ratio_f*0.333+ratio_b*0.667)        
        elif "forward" in k.lower():
            d[k] = scale_value(v, ratio_f)
        elif "backward" in k.lower():
            d[k] = scale_value(v, ratio_b)
        elif MS_PAT.search(k) or S_PAT.search(k):
            # Generic time fields (not forward/backward specific)
            d[k] = scale_value(v, ratio_f)  # default: forward ratio

def process_one_file(in_path: str, out_path: str, ratio_f: float, ratio_b: float) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        scale_time_in_dict(data, ratio_f, ratio_b)
    elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
        for elem in data:
            scale_time_in_dict(elem, ratio_f, ratio_b)

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote: {out_path}")



def main():
    ap = argparse.ArgumentParser(description="Batch-scale forward/backward times in JSON profiles, with name replacement.")
    ap.add_argument("--input-dir", required=True, help="Folder containing JSON profiles.")
    ap.add_argument("--out-dir", required=True, help="Folder for scaled outputs.")
    #ap.add_argument("--model_name", required=True, help="Folder for scaled outputs.")
    ap.add_argument("--find", default="CPU50", help="Substring to find in filename.")
    #ap.add_argument("--replace", required=True, help="Replacement string.")
    args = ap.parse_args()

    ratio_dict = {
        "Samsung": (19.04,13.35,            1),
        "SamsungINT8": (19.04*2.2,13.35*2.2,            1),
        "Xiaomi": (18.00, 12.65,            1),
        "XiaomiINT8": (18.00*2.2, 12.65*2.2,            1),
        "2630": (1,1,                       1),
        "4050":(21.366,13.36,               1),
        "4050INT8":(21.366*1.8,13.36*1.8,               1),
        "4060":(21.366*1.117,13.36*1.117,   1),
        "4060INT8":(21.366*1.117*1.8,13.36*1.117*1.8,   1),
        "V100": (32.77,23.033,               1),
        "V100INT8": (32.77*3,23.033*3,               1),
        "A40": (50, 34.55495,                  1),
        "A40INT8": (50*3, 34.55495*3,                  1),
        "Camera": (2.285, 1.602,            1),
        "CameraINT8": (2.285*3, 1.602*3,            1),

    }

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if not fname.endswith(".json"):
            continue
        in_path = os.path.join(args.input_dir, fname)
        # Replace substring in filename
        for device_name, ratio_pair in ratio_dict.items():

            new_name = fname.replace(args.find, device_name)
            out_path = os.path.join(args.out_dir, new_name)
            process_one_file(in_path, out_path, 1/ratio_pair[0], 1/ratio_pair[1])
    #duplicate for metis
    for fname in os.listdir(args.input_dir + "/metisProfile"):
        if not fname.endswith(".json"):
            continue
        in_path = os.path.join(args.input_dir + "/metisProfile", fname)
        # Replace substring in filename
        for device_name, ratio_pair in ratio_dict.items():

            new_name = fname.replace(args.find, device_name)
            out_path = os.path.join(args.out_dir+"/metisProfile", new_name)
            process_one_file(in_path, out_path, 1/ratio_pair[0], 1/ratio_pair[1])

    ##finally, move the file:
    src_dir = args.out_dir+"/metisProfile"
    dst_dir = "./metis/profile"
    for fname in os.listdir(src_dir):
        if fname.endswith(".json"):
            fnewname = fname.replace("4060", "G4060")
            fnewname = fnewname.replace("4050", "G4050")
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fnewname)
            shutil.move(src_path, dst_path)  # move file
            print(f"Moved: {fname}")
if __name__ == "__main__":
    main()