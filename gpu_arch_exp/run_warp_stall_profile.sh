#!/bin/bash

# Warp Stall Analysis Script
# 使用NCU收集warp state statistics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 编译
echo "=== Compiling warp_stall_benchmark ==="
nvcc -O3 -arch=sm_80 warp_stall_benchmark.cu -o warp_stall_benchmark -lcublas
echo "Compilation done."

# 创建输出目录
OUTPUT_DIR="ncu_reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "=== Running NCU Profiling ==="
echo "Output directory: $OUTPUT_DIR"

# Warp State Statistics 相关指标
WARP_STALL_METRICS="
smsp__warp_issue_stalled_barrier_per_issue_active.pct,
smsp__warp_issue_stalled_branch_resolving_per_issue_active.pct,
smsp__warp_issue_stalled_dispatch_stall_per_issue_active.pct,
smsp__warp_issue_stalled_drain_per_issue_active.pct,
smsp__warp_issue_stalled_imc_miss_per_issue_active.pct,
smsp__warp_issue_stalled_lg_throttle_per_issue_active.pct,
smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct,
smsp__warp_issue_stalled_math_pipe_throttle_per_issue_active.pct,
smsp__warp_issue_stalled_membar_per_issue_active.pct,
smsp__warp_issue_stalled_mio_throttle_per_issue_active.pct,
smsp__warp_issue_stalled_misc_per_issue_active.pct,
smsp__warp_issue_stalled_no_instruction_per_issue_active.pct,
smsp__warp_issue_stalled_not_selected_per_issue_active.pct,
smsp__warp_issue_stalled_selected_per_issue_active.pct,
smsp__warp_issue_stalled_short_scoreboard_per_issue_active.pct,
smsp__warp_issue_stalled_sleeping_per_issue_active.pct,
smsp__warp_issue_stalled_tex_throttle_per_issue_active.pct,
smsp__warp_issue_stalled_wait_per_issue_active.pct
"

# 移除换行符和空格
WARP_STALL_METRICS=$(echo "$WARP_STALL_METRICS" | tr -d '\n' | tr -d ' ')

# 方式1: 只收集warp stall指标 (快速)
echo ""
echo "=== Method 1: Warp Stall Metrics Only ==="
ncu --metrics "$WARP_STALL_METRICS" \
    --csv \
    -o "$OUTPUT_DIR/warp_stall_only" \
    ./warp_stall_benchmark 2>&1 | tee "$OUTPUT_DIR/warp_stall_only.log"

# 导出CSV
ncu --import "$OUTPUT_DIR/warp_stall_only.ncu-rep" --csv > "$OUTPUT_DIR/warp_stall_only.csv" 2>/dev/null || true

# 方式2: 完整profiling (较慢但信息全面)
echo ""
echo "=== Method 2: Full Profiling ==="
ncu --set full \
    -o "$OUTPUT_DIR/full_profile" \
    ./warp_stall_benchmark 2>&1 | tee "$OUTPUT_DIR/full_profile.log"

echo ""
echo "=== Profiling Complete ==="
echo ""
echo "Reports saved to: $OUTPUT_DIR"
echo ""
echo "To view reports:"
echo "  ncu-ui $OUTPUT_DIR/warp_stall_only.ncu-rep"
echo "  ncu-ui $OUTPUT_DIR/full_profile.ncu-rep"
echo ""
echo "CSV data:"
echo "  cat $OUTPUT_DIR/warp_stall_only.csv"
echo ""

# 解析并显示warp stall统计摘要
echo "=== Warp Stall Summary ==="
if [ -f "$OUTPUT_DIR/warp_stall_only.csv" ]; then
    python3 << 'EOF'
import csv
import sys
from collections import defaultdict

csv_file = sys.argv[1] if len(sys.argv) > 1 else "warp_stall_only.csv"

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        kernel_stats = defaultdict(dict)
        
        for row in reader:
            kernel_name = row.get('Kernel Name', 'Unknown')
            
            # 提取stall相关的metrics
            for key, value in row.items():
                if 'stalled' in key.lower():
                    try:
                        val = float(value) if value else 0.0
                        # 简化metric名称
                        short_name = key.replace('smsp__warp_issue_stalled_', '').replace('_per_issue_active.pct', '')
                        kernel_stats[kernel_name][short_name] = val
                    except:
                        pass
        
        print(f"\n{'Kernel':<40} | Top 3 Stall Reasons")
        print("-" * 100)
        
        for kernel, stats in kernel_stats.items():
            if not stats:
                continue
            # 排序找出top 3
            sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
            top_stalls = ", ".join([f"{k}: {v:.1f}%" for k, v in sorted_stats if v > 0])
            
            # 截断kernel名称
            short_kernel = kernel[:38] + ".." if len(kernel) > 40 else kernel
            print(f"{short_kernel:<40} | {top_stalls}")

except Exception as e:
    print(f"Error parsing CSV: {e}")
EOF
fi

echo ""
echo "Done!"

