#!/bin/bash
# Quick NCU Experiment Runner V2
# 增加了详细日志输出，用于捕获memory restore信息
#
# 用法: ./quick_experiment_v2.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v2_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "Quick NCU Timing Experiment V2"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

# 编译
echo "Compiling..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"

# 测试ncu版本
echo ""
echo "NCU Version:"
ncu --version | head -1
echo ""

run_experiment() {
    local mode=$1
    local ncu_set=$2
    local clock=$3
    local name="${mode}_${ncu_set}_${clock}"
    
    echo ""
    echo "=============================================="
    echo "Experiment: $name"
    echo "=============================================="
    
    local cmd="ncu --set $ncu_set"
    if [ "$clock" != "none" ]; then
        cmd="$cmd --clock-control $clock"
    fi
    
    # 添加详细输出选项来捕获memory restore信息
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --print-details all"
    cmd="$cmd --print-summary per-kernel"
    cmd="$cmd --nvtx"
    cmd="$cmd --force-overwrite"
    # 添加verbose选项来捕获replay和memory信息
    cmd="$cmd --verbose"
    cmd="$cmd $KERNEL_BIN $mode"
    
    echo "CMD: $cmd"
    echo ""
    
    # 运行并捕获所有输出
    eval "$cmd" 2>&1 | tee "$OUTPUT_DIR/${name}_stdout.txt"
    
    echo ""
    echo "--- Extracting kernel time from report ---"
    # 直接从报告提取时间
    ncu --import "$OUTPUT_DIR/${name}.ncu-rep" --page raw 2>&1 | head -50 | tee "$OUTPUT_DIR/${name}_raw.txt"
    
    echo ""
}

# Baseline (无NCU)
echo ""
echo "=============================================="
echo "Running Baselines (without NCU)"
echo "=============================================="
echo "--- Method 1: CUDA binary with cudaEvent ---"
"$KERNEL_BIN" short 2>&1 | tee "$OUTPUT_DIR/baseline_short.log"
echo ""
"$KERNEL_BIN" long 2>&1 | tee "$OUTPUT_DIR/baseline_long.log"

echo ""
echo "--- Method 2: Torch Profiler (if available) ---"
if command -v python &> /dev/null; then
    python "$SCRIPT_DIR/timing_kernel_torch.py" short --runs 5 2>&1 | tee "$OUTPUT_DIR/baseline_short_torch_profiler.log" || echo "Torch profiler baseline failed for short kernel"
    echo ""
    python "$SCRIPT_DIR/timing_kernel_torch.py" long --runs 5 2>&1 | tee "$OUTPUT_DIR/baseline_long_torch_profiler.log" || echo "Torch profiler baseline failed for long kernel"
else
    echo "Python not available, skipping torch profiler baseline"
fi

# 核心实验: 2 modes × 2 sets × 2 clocks = 8 experiments
for mode in short long; do
    for ncu_set in basic full; do
        for clock in base none; do
            run_experiment "$mode" "$ncu_set" "$clock"
        done
    done
done

echo ""
echo "=============================================="
echo "Experiments completed!"
echo "=============================================="
echo ""
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Summary: Extracting kernel times from all reports"
echo "=============================================="
echo ""

# 汇总所有kernel时间
echo "Report                              Kernel Time (from NCU)"
echo "--------------------------------------------------------------"

for rep in "$OUTPUT_DIR"/*.ncu-rep; do
    if [ -f "$rep" ]; then
        name=$(basename "$rep" .ncu-rep)
        # 尝试提取Duration
        time_info=$(ncu --import "$rep" --page raw 2>/dev/null | grep -i "duration\|time" | head -3 || echo "N/A")
        echo "[$name]"
        echo "$time_info"
        echo ""
    fi
done

echo ""
echo "=============================================="
echo "Baseline Timing Summary"
echo "=============================================="
echo ""
echo "--- CUDA Binary (cudaEvent) ---"
grep -h "Kernel duration" "$OUTPUT_DIR"/baseline_*.log 2>/dev/null | head -4 || echo "N/A"

echo ""
echo "--- Torch Profiler ---"
for log in "$OUTPUT_DIR"/baseline_*_torch_profiler.log; do
    if [ -f "$log" ]; then
        echo "$(basename $log):"
        grep -E "Mean:|Profiler vs CudaEvent" "$log" 2>/dev/null | head -4 || echo "  N/A"
    fi
done

echo ""
echo "To analyze memory restore info:"
echo "  cat $OUTPUT_DIR/*_stdout.txt | grep -i 'restore\|replay\|memory'"
echo ""
echo "To view full report:"
echo "  ncu-ui $OUTPUT_DIR/*.ncu-rep"

