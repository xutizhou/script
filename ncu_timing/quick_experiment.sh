#!/bin/bash
# Quick NCU Experiment Runner
# 快速运行关键实验组合，用于验证NCU计时准确性
#
# 用法: ./quick_experiment.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./quick_results_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "Quick NCU Timing Experiment"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

# 编译
echo "Compiling..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"

# 关键实验组合 (减少实验数量，聚焦关键对比)
# 只测试: basic vs full, base vs none(boost), short vs long

echo ""
echo "=============================================="
echo "Running experiments..."
echo "=============================================="

run_experiment() {
    local mode=$1
    local ncu_set=$2
    local clock=$3
    local name="${mode}_${ncu_set}_${clock}"
    
    echo ""
    echo "--- $name ---"
    
    local cmd="ncu --set $ncu_set"
    if [ "$clock" != "none" ]; then
        cmd="$cmd --clock-control $clock"
    fi
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --print-summary per-kernel"
    cmd="$cmd $KERNEL_BIN $mode"
    
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee -a "$OUTPUT_DIR/${name}.log"
}

# Baseline (无NCU)
echo ""
echo "--- Baselines ---"
"$KERNEL_BIN" short 2>&1 | tee "$OUTPUT_DIR/baseline_short.log"
"$KERNEL_BIN" long 2>&1 | tee "$OUTPUT_DIR/baseline_long.log"

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
echo "To analyze results:"
echo "  python $SCRIPT_DIR/parse_ncu_results.py $OUTPUT_DIR"
echo "  python $SCRIPT_DIR/analyze_memory_restore.py $OUTPUT_DIR"

