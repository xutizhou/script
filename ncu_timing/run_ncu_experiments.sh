#!/bin/bash
# NCU Timing Accuracy Experiments
# 
# 这个脚本运行一系列NCU实验来验证不同配置对计时准确性的影响
# 
# 实验变量:
# 1. NCU set: basic, detailed, full, nvlink, pmsampling, roofline
# 2. Clock control: base, boost
# 3. Kernel duration: short (~10us), long (~10ms)
#
# 用法: ./run_ncu_experiments.sh [output_dir]

set -e

# 输出目录
OUTPUT_DIR=${1:-"./ncu_results_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

# CUDA编译
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "NCU Timing Accuracy Experiments"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# 编译kernel
echo "Compiling CUDA kernel..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"
echo "Compilation successful: $KERNEL_BIN"
echo ""

# 定义实验参数
NCU_SETS=("basic" "detailed" "full" "roofline")
CLOCK_CONTROLS=("base" "none")  # none表示boost/默认
KERNEL_MODES=("short" "long")

# 先运行一次不带ncu的baseline
echo "=============================================="
echo "Running baseline (without NCU)..."
echo "=============================================="
for mode in "${KERNEL_MODES[@]}"; do
    echo "--- Baseline: $mode kernel ---"
    "$KERNEL_BIN" "$mode" 2>&1 | tee "$OUTPUT_DIR/baseline_${mode}.log"
    echo ""
done

# 运行NCU实验
echo "=============================================="
echo "Running NCU experiments..."
echo "=============================================="

experiment_count=0
total_experiments=$((${#NCU_SETS[@]} * ${#CLOCK_CONTROLS[@]} * ${#KERNEL_MODES[@]}))

for ncu_set in "${NCU_SETS[@]}"; do
    for clock in "${CLOCK_CONTROLS[@]}"; do
        for mode in "${KERNEL_MODES[@]}"; do
            experiment_count=$((experiment_count + 1))
            
            experiment_name="${mode}_${ncu_set}_clock${clock}"
            report_file="$OUTPUT_DIR/${experiment_name}.ncu-rep"
            log_file="$OUTPUT_DIR/${experiment_name}.log"
            
            echo "=============================================="
            echo "Experiment $experiment_count/$total_experiments: $experiment_name"
            echo "=============================================="
            echo "  Set: $ncu_set"
            echo "  Clock: $clock"
            echo "  Mode: $mode"
            echo "  Report: $report_file"
            echo ""
            
            # 构建NCU命令
            ncu_cmd="ncu --set $ncu_set"
            
            # 添加clock control
            if [ "$clock" != "none" ]; then
                ncu_cmd="$ncu_cmd --clock-control $clock"
            fi
            
            # 添加详细输出和报告文件
            ncu_cmd="$ncu_cmd -o $report_file"
            ncu_cmd="$ncu_cmd --log-file $log_file"
            ncu_cmd="$ncu_cmd --print-summary per-kernel"
            
            # 添加kernel目标
            ncu_cmd="$ncu_cmd $KERNEL_BIN $mode"
            
            echo "Command: $ncu_cmd"
            echo ""
            
            # 运行实验
            start_time=$(date +%s.%N)
            eval "$ncu_cmd" 2>&1 | tee -a "$log_file"
            end_time=$(date +%s.%N)
            
            wall_time=$(echo "$end_time - $start_time" | bc)
            echo ""
            echo "Wall clock time: ${wall_time}s"
            echo "Wall clock time: ${wall_time}s" >> "$log_file"
            echo ""
        done
    done
done

echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of generated files:"
ls -la "$OUTPUT_DIR"
echo ""

# 生成汇总报告
summary_file="$OUTPUT_DIR/summary.txt"
echo "Generating summary report: $summary_file"
echo ""

{
    echo "NCU Timing Accuracy Experiments Summary"
    echo "========================================"
    echo "Date: $(date)"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Experiment Configuration:"
    echo "  NCU Sets: ${NCU_SETS[*]}"
    echo "  Clock Controls: ${CLOCK_CONTROLS[*]}"
    echo "  Kernel Modes: ${KERNEL_MODES[*]}"
    echo ""
    echo "========================================"
    echo "Results:"
    echo ""
    
    for log in "$OUTPUT_DIR"/*.log; do
        if [ -f "$log" ]; then
            echo "--- $(basename "$log") ---"
            # 提取关键信息
            grep -E "(Kernel duration|Wall clock|Duration|Time)" "$log" 2>/dev/null || echo "(no timing info found)"
            echo ""
        fi
    done
} > "$summary_file"

cat "$summary_file"

echo ""
echo "To view detailed NCU reports, use:"
echo "  ncu-ui $OUTPUT_DIR/*.ncu-rep"
echo ""
echo "Or export to CSV:"
echo "  ncu --import $OUTPUT_DIR/<report>.ncu-rep --csv > output.csv"

