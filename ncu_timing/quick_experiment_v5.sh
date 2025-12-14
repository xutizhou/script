#!/bin/bash
# NCU Timing Experiment V5
# 
# 只测量 CUDA kernel (timing_kernel.cu)
# - Baseline: cudaEvent (运行在 boost 频率)
# - NCU: base 和 none 两种 clock-control 模式
#
# 目标:
# - short kernel: ~10us
# - long kernel: ~10ms
#
# 用法: ./quick_experiment_v5.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v5_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "NCU Timing Experiment V5"
echo "=============================================="
echo "Measuring CUDA kernel: timing_kernel.cu"
echo "Output: $OUTPUT_DIR"
echo ""

# 编译
echo "Compiling CUDA kernel..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"
echo "Done."
echo ""

# NCU 版本
echo "NCU Version:"
ncu --version | head -1
echo ""

# ========== Step 1: cudaEvent Baseline ==========
echo "=============================================="
echo "Step 1: cudaEvent Baseline (Boost Frequency)"
echo "=============================================="

echo ""
echo "--- Short Kernel ---"
$KERNEL_BIN short 2>&1 | tee "$OUTPUT_DIR/baseline_short.log"

echo ""
echo "--- Long Kernel ---"
$KERNEL_BIN long 2>&1 | tee "$OUTPUT_DIR/baseline_long.log"

# ========== Step 2: NCU Experiments ==========
echo ""
echo "=============================================="
echo "Step 2: NCU Profiling"
echo "=============================================="

run_ncu_experiment() {
    local mode=$1
    local clock=$2
    local name="${mode}_${clock}"
    
    echo ""
    echo "=== Experiment: $name ==="
    
    local cmd="ncu --set basic"
    
    # clock control: base 或 none
    if [ "$clock" == "base" ]; then
        cmd="$cmd --clock-control base"
    fi
    # none 不加 --clock-control，使用默认
    
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --force-overwrite"
    cmd="$cmd $KERNEL_BIN $mode"
    
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "$OUTPUT_DIR/${name}_stdout.txt"
}

# 测试 base 和 none (default/boost) 两种模式
for mode in short long; do
    for clock in base none; do
        run_ncu_experiment "$mode" "$clock"
    done
done

# ========== Step 3: 提取 NCU Duration ==========
echo ""
echo "=============================================="
echo "Step 3: Extract NCU Duration"
echo "=============================================="

for rep in "$OUTPUT_DIR"/*.ncu-rep; do
    if [ -f "$rep" ]; then
        name=$(basename "$rep" .ncu-rep)
        echo ""
        echo "[$name]"
        # 使用 ncu --import 提取 timed_rw_kernel 的指标
        ncu --import "$rep" --page raw --csv 2>/dev/null | \
            grep "timed_rw_kernel" | \
            grep -E "gpu__time_duration|sm__frequency" | \
            head -4
    fi
done

# ========== Step 4: 结果汇总 ==========
echo ""
echo "=============================================="
echo "Step 4: Results Summary (using Python script)"
echo "=============================================="

python3 "$SCRIPT_DIR/extract_results.py" "$OUTPUT_DIR"

echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
