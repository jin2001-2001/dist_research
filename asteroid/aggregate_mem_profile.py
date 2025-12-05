#!/usr/bin/env python3
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="List of names, e.g., CPU100 CPU100 CPU100")
    parser.add_argument("--out", default="devices.json",
                        help="Output JSON file")
    parser.add_argument("--value", type=int, default=55,
                        help="Value to assign to each entry (default: 60)")
    args = parser.parse_args()

    data = {}
    # enumerate all inputs, add suffix
    # for final test, we use above dictionary for mem searching...:
    mem_dict = {
        "SamsungINT8": 12*200,
        "Samsung": 12*200,
        "Xiaomi": 12*200,
        "XiaomiINT8": 12*200,
        "2630": 32*200,
        "4050":12*200,
        "4050INT8":12*200,
        "4060":12*200,
        "4060INT8":12*200,
        "V100": 32*200,
        "V100INT8": 32*200,
        "A40": 48*200,
        "A40INT8": 48*200,
        "Camera": 16*200,
        "CameraINT8": 16*200
    }


    for i, name in enumerate(args.inputs):
        base = name.split("_")[0]   # take only the first part
        key = f"{base}_{i}"
        data[key] = mem_dict[base]

    # write JSON
    with open(args.out, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()