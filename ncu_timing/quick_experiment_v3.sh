#!/bin/bash
# Quick NCU Experiment Runner V3
# 
# 对比 NCU Duration vs Torch Profiler Duration
# 测试 base / none(default) / boost 三种 clock-control 模式
#
# 用法: ./quick_experiment_v3.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v3_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "NCU vs Torch Profiler Experiment V3"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

# 编译
echo "Compiling CUDA kernel..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"

# 测试ncu版本
echo ""
echo "NCU Version:"
ncu --version | head -1
echo ""

# ========== Torch Profiler Baseline ==========
echo "=============================================="
echo "Step 1: Torch Profiler Baseline (Real GPU Time)"
echo "=============================================="

echo ""
echo "--- Short Kernel ---"
python "$SCRIPT_DIR/timing_kernel_torch.py" short --runs 10 --output-json "$OUTPUT_DIR/torch_profiler_short.json" 2>&1 | tee "$OUTPUT_DIR/torch_profiler_short.log"

echo ""
echo "--- Long Kernel ---"
python "$SCRIPT_DIR/timing_kernel_torch.py" long --runs 10 --output-json "$OUTPUT_DIR/torch_profiler_long.json" 2>&1 | tee "$OUTPUT_DIR/torch_profiler_long.log"

# ========== NCU Experiments ==========
echo ""
echo "=============================================="
echo "Step 2: NCU Profiling with Different Clock Controls"
echo "=============================================="

run_ncu_experiment() {
    local mode=$1
    local ncu_set=$2
    local clock=$3
    local name="${mode}_${ncu_set}_${clock}"
    
    echo ""
    echo "--- Experiment: $name ---"
    
    local cmd="ncu --set $ncu_set"
    
    # clock control: base, none (default), boost
    if [ "$clock" == "base" ]; then
        cmd="$cmd --clock-control base"
    elif [ "$clock" == "boost" ]; then
        cmd="$cmd --clock-control none"  # none 应该是 boost
        # 尝试用 nvml 设置最大频率
    fi
    # none 不加参数，使用默认
    
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --print-summary per-kernel"
    cmd="$cmd --force-overwrite"
    cmd="$cmd $KERNEL_BIN $mode"
    
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "$OUTPUT_DIR/${name}_stdout.txt"
}

# 测试 3 种 clock control 模式: base, none (default), boost
# 只用 basic set 来加快实验
for mode in short long; do
    for clock in base none; do
        run_ncu_experiment "$mode" "basic" "$clock"
    done
done

# ========== 结果汇总 ==========
echo ""
echo "=============================================="
echo "Step 3: Results Summary"
echo "=============================================="

# 提取 torch profiler 结果
echo ""
echo "--- Torch Profiler Results (Reference) ---"
for mode in short long; do
    json_file="$OUTPUT_DIR/torch_profiler_${mode}.json"
    if [ -f "$json_file" ]; then
        mean=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['mean_us']:.2f}\")" 2>/dev/null || echo "N/A")
        echo "  $mode: ${mean} us"
    fi
done

# 提取 NCU Duration
echo ""
echo "--- NCU Duration Results ---"
for rep in "$OUTPUT_DIR"/*.ncu-rep; do
    if [ -f "$rep" ]; then
        name=$(basename "$rep" .ncu-rep)
        # 使用 ncu --import 提取 Duration
        duration=$(ncu --import "$rep" --csv 2>/dev/null | grep "Duration" | head -1 | awk -F',' '{print $15}' | tr -d '"' || echo "N/A")
        unit=$(ncu --import "$rep" --csv 2>/dev/null | grep "Duration" | head -1 | awk -F',' '{print $14}' | tr -d '"' || echo "")
        echo "  $name: $duration $unit"
    fi
done

# 调用 Python 脚本进行详细对比
echo ""
echo "=============================================="
echo "Step 4: Detailed Comparison (NCU vs Torch Profiler)"
echo "=============================================="
python "$SCRIPT_DIR/compare_ncu_torch.py" "$OUTPUT_DIR" 2>&1 || echo "Comparison script not available or failed"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

