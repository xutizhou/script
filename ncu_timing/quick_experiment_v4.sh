#!/bin/bash
# NCU vs Torch Profiler Experiment V4
# 
# 关键：NCU 和 Torch Profiler 测量同一个 Triton kernel
# 两者使用完全相同的配置：
#   - short: N=4096, iterations=1, blocks=1, threads=128
#   - long:  N=16M, iterations=100, blocks=128, threads=128
#
# 用法: ./quick_experiment_v4.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v4_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "NCU vs Torch Profiler Experiment V4"
echo "=============================================="
echo "Both NCU and Torch Profiler measure the SAME Triton kernel"
echo "with identical configuration as timing_kernel.cu"
echo "Output: $OUTPUT_DIR"
echo ""

# NCU 版本
echo "NCU Version:"
ncu --version | head -1
echo ""

# ========== Step 1: Torch Profiler Baseline ==========
echo "=============================================="
echo "Step 1: Torch Profiler Baseline (Boost Frequency)"
echo "=============================================="

for mode in short long; do
    echo ""
    echo "--- $mode kernel ---"
    python "$SCRIPT_DIR/timing_kernel_torch.py" $mode --runs 10 \
        --output-json "$OUTPUT_DIR/torch_profiler_${mode}.json" \
        2>&1 | tee "$OUTPUT_DIR/torch_profiler_${mode}.log"
done

# ========== Step 2: NCU Experiments (Same Triton Kernel) ==========
echo ""
echo "=============================================="
echo "Step 2: NCU Profiling (Same Triton Kernel)"
echo "=============================================="

run_ncu_experiment() {
    local mode=$1
    local clock=$2
    local name="${mode}_ncu_${clock}"
    
    echo ""
    echo "--- Experiment: $name ---"
    
    local cmd="ncu"
    
    # clock control
    if [ "$clock" == "base" ]; then
        cmd="$cmd --clock-control base"
    fi
    # none = 默认，不加参数
    
    cmd="$cmd --set basic"
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --force-overwrite"
    # 使用 --ncu-mode 只运行一次 kernel
    cmd="$cmd python $SCRIPT_DIR/timing_kernel_torch.py $mode --ncu-mode"
    
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "$OUTPUT_DIR/${name}_stdout.txt"
}

for mode in short long; do
    for clock in base none; do
        run_ncu_experiment "$mode" "$clock"
    done
done

# ========== Step 3: 结果汇总 ==========
echo ""
echo "=============================================="
echo "Step 3: Extract NCU Duration"
echo "=============================================="

for rep in "$OUTPUT_DIR"/*.ncu-rep; do
    if [ -f "$rep" ]; then
        name=$(basename "$rep" .ncu-rep)
        echo ""
        echo "[$name]"
        ncu --import "$rep" --csv 2>/dev/null | grep -E "Duration|SM Frequency" | head -5
    fi
done

# ========== Step 4: 对比分析 ==========
echo ""
echo "=============================================="
echo "Step 4: Comparison"
echo "=============================================="

python "$SCRIPT_DIR/compare_ncu_torch.py" "$OUTPUT_DIR" 2>&1 || echo "Comparison failed"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
ls -la "$OUTPUT_DIR"
