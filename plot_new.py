# Re-import libraries after reset
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Optional
import numpy as np

def first_peak_end_time_raw(
    series,
    start_t,
    low_ok: float = 1,          # ignore values <= this (e.g., ~10 noise)
    high_ok: float = 1.2e4,          # ignore values >= this (e.g., 10000+ spikes)
    baseline_ns: int = 1e8,        # samples from start to estimate baseline
    amp_thresh: float = 0,     # enter peak when value >= baseline + amp_thresh
    end_tol: float = 1,         # exit peak when value <  baseline + end_tol
    end_hold: int = 1,            # need N consecutive exit samples
    inverted: bool = False,        # True if peaks are downward dips
    min_duration: Optional[float] = None,  # minimum time the peak must last
    min_samples: Optional[int] = 1      # minimum number of samples in the peak
):
    """
    Returns the timestamp of the end of the first 'normal' peak, or None.
    - No smoothing. Outliers are ignored (treated as NaN).
    - Peaks that are shorter than min_duration or min_samples are skipped.
    """

    arr = np.asarray(series, dtype=object)
    t = arr[:, 0].astype(int)
    v = arr[:, 2].astype(float)

    # Apply start_time filter
    if start_t is not None:
        mask = t >= start_t
        t = t[mask]
        v = v[mask]
        if len(t) == 0:
            return None
    print(t[0])

    # Mark outliers as NaN so they don't count toward continuity.
    valid = (v > low_ok) & (v < high_ok)
    v_use = v.astype(float)
    #v_use[~valid] = np.nan
    if inverted:
        v_use = -v_use

    # Baseline from early valid points.
    head = v_use[:min(baseline_ns, len(v_use))]
    mask = (head<high_ok) & (head > low_ok)
    baseline = np.nanmedian(head[mask])
    #print(baseline)
    if np.isnan(baseline):
        print("baseline error")
        baseline = high_ok
    #print(baseline)
    baseline= baseline/10


    enter_thr = baseline + amp_thresh
    exit_thr  = baseline + end_tol

    i = 0
    prev = v_use[0] if len(v_use) > 0 else np.nan

    while i < len(v_use):
        cur = v_use[i]
        # Find entry into a peak (crossing from below to >= enter_thr)
        if (not np.isnan(cur)) and (not np.isnan(prev)) and (enter_thr <= cur):
            start_idx = i
            # Now find end: first time we stay below exit_thr for end_hold consecutive valid points
            below_run = 0
            end_idx = None
            j = start_idx
            while j < len(v_use):
                x = v_use[j]
                if np.isnan(x):
                    below_run = 0
                elif x < exit_thr:
                    below_run += 1
                    if below_run >= end_hold:
                        end_idx = j
                        break
                else:
                    below_run = 0
                j += 1

            if end_idx is None:
                end_idx = end_hold-1+j-1 # last till...

            # Duration/sample-width checks
            duration_ok = (min_duration is None) or ((t[end_idx] - t[start_idx]) >= min_duration)
            # Count samples considered "in-peak" as those from start to the first index
            # where we began the exit run (end_idx - end_hold + 1) or end_idx if we prefer inclusive.
            first_exit_idx = end_idx - end_hold + 1
            width_samples = max(0, first_exit_idx - start_idx+1)
            samples_ok = (min_samples is None) or (width_samples >= min_samples)

            if duration_ok and samples_ok:
                return int(t[end_idx])  # end time of the first normal peak

            # Too narrow → skip this peak and continue scanning after it
            i = end_idx + 1
            prev = v_use[i-1] if i-1 < len(v_use) else np.nan
            continue

        prev = cur
        i += 1
    print("error, no peak end found")
    return None




with open("./record/timeline_batch1_all.json", "r") as f:
    records = json.load(f)

def smooth_series(values, window_size=5):
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='same')


mbatch_size = 5

# Regenerate DataFrame with separated SEND/RECV per rank
rows = []
stage_info = {}


base_ns = records[0]["start_ns"]
for record in records:
    mb = record.get("mb_idx", -1)
    mb_str = ",".join(map(str, mb)) if isinstance(mb, list) else str(mb)
    action = record["action"]
    rank = record["rank"]
    stage_idx = record["stage_idx"]
    y_tag = f"{rank}-{action}" if action in {"SEND_F", "RECV_F", "SEND_B", "RECV_B"} else f"{rank}-MAIN"

    #we use a dictionary to record the stage_info(mainly which rank belongs to which stage)
    if stage_idx in stage_info:
        if rank not in stage_info[stage_idx]:
            stage_info[stage_idx].append(rank)
    else:
        stage_info[stage_idx] = [rank]

    send_start = record["start_ns"]
    real_end = record["end_ns"]

    if action in {"RECV_F", "RECV_B"}:
        #print(record["net_series"])
        #print(y_tag, mb_str)
        for start_record in records:
            cand_mb = start_record.get("mb_idx", -1)
            cand_action = start_record["action"]
            cand_stage_idx = start_record["stage_idx"]
            h,_,t0 = cand_action.partition('_')
            _,_, t1 = action.partition('_')
            if h == 'SEND' and t0==t1 and cand_mb == mb:
                if (t1 == 'F' and cand_stage_idx == stage_idx-1) or (t1 == 'B' and cand_stage_idx == stage_idx+1):
                    send_start = start_record["start_ns"]
                    break
        #real_end = first_peak_end_time_raw(series = record["net_series"],start_t = send_start)

    if action not in {"SEND_F", "SEND_B"}:
        rows.append({
        "y_tag": y_tag,
        "stage_idx": stage_idx,
        "rank": rank,
        "action": action,
        "start": (send_start - base_ns) / 1e9,
        "end": (real_end - base_ns) / 1e9,
        "mb_idx": mb_str,
        "net_series": [[s, u ,d] for s, u, d in record["net_series"] if s>send_start and s< real_end]
        })
        if action in {"RECV_F","RECV_B"}:
            print(stage_idx,rank,action, mb_str, (real_end-send_start)/1e9, real_end/1e9, send_start/1e9)

df = pd.DataFrame(rows)

#
base_loc_dic = {}
base_loc_idx = 0
for i in range(len(stage_info)):
    base_loc_dic[i] = base_loc_idx
    base_loc_idx+=1+len(stage_info[i])




# Assign operation colors
op_colors = {
    "FORWARD": "skyblue",
    "FULL_BACKWARD": "lightcoral",
    "ALL_REDUCE": "orange",
    "SEND_F": "gray",
    "RECV_F": "green",
    "SEND_B": "darkgray",
    "RECV_B": "lightgreen"
}

# Assign vertical positions
y_tags = sorted(df["y_tag"].unique())
y_map_ = {tag: i for i, tag in enumerate(y_tags)}
#print(y_map)

#additional ploting rephase
# we put receive backward to above line while receive forward to beneath line
y_map = {}

highest = 0
for tag, i in y_map_.items():
    rank, _, action = tag.partition('-')
    rank = int(rank)

    stage= None
    offset = None
    for can_stage, can_rank_list in stage_info.items():
        if rank in can_rank_list:
            offset = can_rank_list.index(rank)
            stage = can_stage
            break
    base_loc = base_loc_dic[stage]
    if action == "RECV_F":
        base_loc-=1
    elif action == "RECV_B":
        base_loc+=len(stage_info[stage])
    else:
        base_loc+=offset
    y_map[tag] = base_loc

    if base_loc>highest:
        highest = base_loc


# Begin plot
fig, ax = plt.subplots(figsize=(16, 10))
for _, row in df.iterrows():
    y_pos = highest-y_map[row["y_tag"]]
    width = row["end"] - row["start"]
    left = row["start"]
    ax.barh(
        y=y_pos,
        width=width,
        left=left,
        height=0.6,
        color=op_colors.get(row["action"], "black"),
        edgecolor="black"
    )

    kk, _, _ = row["mb_idx"].partition(',')
    reduced_mb_index = int(int(kk)/mbatch_size)

    ax.text(
        left + width / 2,
        y_pos,
        f'MB{reduced_mb_index}',
        va='center',
        ha='center',
        fontsize=3,
        weight='bold',
        color='black'
    )

# Overlay smoothed bandwidth lines
for record in rows:
    if record["action"].startswith("RECV") and record["net_series"]:
        y_tag = f"{record['rank']}-{record['action']}"
        y_base = highest-y_map[y_tag]
        x_vals = [(ts - base_ns) / 1e9 for ts, _, _ in record["net_series"]]
        #up_vals = [up for _, up, _ in record["net_series"]]
        down_vals = [down for _, _, down in record["net_series"]]
        #up_smooth = smooth_series(up_vals, window_size=5)
        down_smooth = smooth_series(down_vals, window_size=5)
        #bw_max = max(max(up_smooth), max(down_smooth), 1)
        bw_max = max(max(down_smooth), 1)
        #up_scaled = [y_base + 0.3 * (u / bw_max) for u in up_smooth]
        down_scaled = [y_base - 0.3 * (d / bw_max) for d in down_smooth]
        #ax.plot(x_vals, up_scaled, color='red', linewidth=1.5, label="Up Mbps" if 'Up Mbps' not in ax.get_legend_handles_labels()[1] else "")
        ax.plot(x_vals, down_scaled, color='blue', linewidth=1.5, label="Down Mbps" if 'Down Mbps' not in ax.get_legend_handles_labels()[1] else "")

# Final touches
ax.set_yticks(list(y_map.values()))
ax.set_yticklabels(y_tags, fontsize=11)
ax.set_xlabel("Time (s)", fontsize=13)
ax.set_title("Hybrid PP Gantt Chart (Smoothed Bandwidth Overlay)", fontsize=15)

# Build legend
handles = [mpatches.Patch(color=c, label=t) for t, c in op_colors.items()]
handles += [
    plt.Line2D([0], [0], color='red', lw=1.5, label="Up Mbps"),
    plt.Line2D([0], [0], color='blue', lw=1.5, label="Down Mbps"),
]
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()
