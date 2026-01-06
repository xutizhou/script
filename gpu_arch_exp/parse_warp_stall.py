#!/usr/bin/env python3
"""Parse NCU warp stall statistics from CSV."""

import csv
import sys

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "ncu_reports/warp_stall_raw.csv"
    
    # 读取CSV
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 要提取的stall metrics
    stall_metrics = [
        ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "long_sb"),
        ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "short_sb"),
        ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "barrier"),
        ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "math_thr"),
        ("smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "not_sel"),
        ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "lg_thr"),
        ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "mio_thr"),
        ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "wait"),
        ("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio", "membar"),
        ("smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio", "branch"),
        ("smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio", "imc_miss"),
        ("smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio", "no_instr"),
    ]

    print(f"{'Kernel':<50} | Top Stall Reasons (ratio)")
    print("=" * 120)

    seen = set()
    for row in rows:
        kernel = row.get("Kernel Name", "Unknown")
        if kernel in seen or not kernel:
            continue
        seen.add(kernel)
        
        # 获取各个stall的值
        stall_vals = {}
        for metric_key, short_name in stall_metrics:
            try:
                val = float(row.get(metric_key, 0))
                stall_vals[short_name] = val
            except:
                stall_vals[short_name] = 0.0
        
        # 排序找top 4 stall
        sorted_stalls = sorted(stall_vals.items(), key=lambda x: x[1], reverse=True)[:4]
        top_stalls = ", ".join([f"{k}:{v:.2f}" for k, v in sorted_stalls if v > 0.01])
        
        short_kernel = (kernel[:48] + "..") if len(kernel) > 50 else kernel
        print(f"{short_kernel:<50} | {top_stalls}")

    print()
    print("=" * 80)
    print("Stall Type Legend:")
    print("  long_sb  = long_scoreboard (waiting for global/texture memory L1TEX)")
    print("  short_sb = short_scoreboard (waiting for shared memory/L1)")
    print("  barrier  = __syncthreads() synchronization")
    print("  math_thr = math_pipe_throttle (compute units busy)")
    print("  not_sel  = not_selected (warp ready but scheduler chose another)")
    print("  lg_thr   = lg_throttle (local/global memory queue full)")
    print("  mio_thr  = mio_throttle (MIO queue full)")
    print("  wait     = wait (waiting for atomics or other dependencies)")
    print("  membar   = membar (__threadfence)")
    print("  branch   = branch_resolving (waiting for branch result)")
    print("  imc_miss = instruction cache miss")
    print("  no_instr = no_instruction (no instruction to issue)")

if __name__ == "__main__":
    main()

