#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a CSV of pairwise link rates to planner's bandwidth_matrix.json.

CSV format (no header):
  src_host,dst_host,gbps

Usage:
  python parse_bandwidth_csv.py --csv links.csv --out bandwidth_matrix.json

Tip: gather with iperf3 like:
  # On dst: iperf3 -s
  # On src: iperf3 -c DST_IP -f g -J | jq '.end.sum_received.bits_per_second'
  # Put gbps into CSV (divide by 1e9). You should measure both directions.
"""

import argparse, csv, json
import itertools
import os


def name_extractor(name_list):
    datas = []
    prefixes = []

    for p in name_list:
        # load JSON
        with open(p, "r") as f:
            datas.append(json.load(f))

        # filename without folder path
        fname = os.path.basename(p)        # e.g. "CPU100_bs5.json"

        # remove extension
        stem, _ = os.path.splitext(fname)  # e.g. "CPU100_bs5"

        # take prefix before first "_"
        prefix = stem.split("_", 1)[0]     # e.g. "CPU100"

        prefixes.append(prefix)

    return prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs",nargs="+", required=True)
    ap.add_argument("--mbps", default=400)
    ap.add_argument("--out", default="bandwidth_matrix.json")
    args = ap.parse_args()


    bm = {}
    print(args.inputs)
    datas = [json.load(open(p,"r")) for p in args.inputs]
    name_list = name_extractor(args.inputs)
    host_name = []
    for i in range(len(datas)):
        name = str(name_list[i]) + f'_{i}'
        host_name.append(name)

    #print(host_name)
    for pair in list(itertools.permutations(host_name, 2)):
        bm[pair] = float(args.mbps) * 1e6 /8.0

    #with open(args.csv, newline="") as f:
    #    for row in csv.reader(f):
    #        if not row or row[0].startswith("#"): 
    #            continue
    #        src, dst, gbps = row[0].strip(), row[1].strip(), float(row[2])
    #        bm[(src, dst)] = gbps * 1e9 / 8.0  # bytes/sec

    # Serialize keys as "src|dst"
    out = {"bytes_per_sec": {f"{k[0]}|{k[1]}": v for k,v in bm.items()}}
    json.dump(out, open(args.out,"w"), indent=2)
    print("bandwidth dump finished...")

if __name__ == "__main__":
    main()
