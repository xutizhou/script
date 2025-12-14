#!/bin/bash
# NCU Timing Experiment V6 (Complete)
# 
# 实验设计：
# 1. Torch Profiler (boost): 默认 boost 频率 (~1.9 GHz)
# 2. Torch Profiler (base): 用 nvidia-smi 锁定到 base 频率 (~1.15 GHz)
# 3. NCU 不同 set：basic, full
# 4. NCU 不同 clock-control：base, none
#
# 验证方法：
# - 如果 Torch Profiler (base) ≈ NCU Duration，则 NCU 计时准确
#
# 注意：nvidia-smi 需要在容器外运行，或者用 --privileged 容器
#
# 用法: ./quick_experiment_v6.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v6_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

echo "=============================================="
echo "NCU Timing Experiment V6 (Complete)"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

# ========== Step 0: 编译 ==========
echo "=============================================="
echo "Step 0: Compile"
echo "=============================================="

echo "Compiling CUDA kernel binary..."
nvcc -O3 -arch=sm_90 "$KERNEL_SRC" -o "$KERNEL_BIN"
echo "Done."

echo ""
echo "Compiling pybind11 module..."
cd "$SCRIPT_DIR"
python setup.py build_ext --inplace 2>&1 | tail -3
cd - > /dev/null
echo "Done."
echo ""

# ========== GPU 频率信息 ==========
echo "=============================================="
echo "GPU Frequency Info"
echo "=============================================="
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,clocks.sm --format=csv 2>/dev/null || echo "Could not query GPU clocks"
echo ""

# ========== Step 1: Torch Profiler (Boost) ==========
echo "=============================================="
echo "Step 1: Torch Profiler at BOOST Frequency"
echo "=============================================="

# 尝试解锁频率（可能会失败）
echo "Attempting to unlock GPU clocks (boost mode)..."
nvidia-smi -rgc 2>/dev/null && echo "GPU clocks unlocked." || echo "Could not unlock GPU clocks (may need root/privileged mode)"
sleep 1

echo ""
echo "Current GPU clocks:"
nvidia-smi --query-gpu=clocks.gr,clocks.sm --format=csv 2>/dev/null || true

for mode in short long; do
    echo ""
    echo "--- $mode kernel (boost) ---"
    python "$SCRIPT_DIR/timing_kernel_torch.py" $mode --runs 10 \
        --output-json "$OUTPUT_DIR/torch_boost_${mode}.json" \
        2>&1 | tee "$OUTPUT_DIR/torch_boost_${mode}.log"
done

# ========== Step 2: Torch Profiler (Base) ==========
echo ""
echo "=============================================="
echo "Step 2: Torch Profiler at BASE Frequency (~1155 MHz)"
echo "=============================================="

# 尝试锁定到 base 频率
echo "Attempting to lock GPU to base frequency (~1155 MHz)..."
nvidia-smi -lgc 1155,1155 2>/dev/null && echo "GPU clocks locked to 1155 MHz." || \
nvidia-smi -lgc 1155 2>/dev/null && echo "GPU clocks locked to 1155 MHz." || \
echo "Could not lock GPU clocks (may need root/privileged mode)"
sleep 1

echo ""
echo "Current GPU clocks:"
nvidia-smi --query-gpu=clocks.gr,clocks.sm --format=csv 2>/dev/null || true

for mode in short long; do
    echo ""
    echo "--- $mode kernel (base freq) ---"
    python "$SCRIPT_DIR/timing_kernel_torch.py" $mode --runs 10 \
        --output-json "$OUTPUT_DIR/torch_base_${mode}.json" \
        2>&1 | tee "$OUTPUT_DIR/torch_base_${mode}.log"
done

# 解锁频率
echo ""
echo "Attempting to unlock GPU clocks..."
nvidia-smi -rgc 2>/dev/null && echo "GPU clocks unlocked." || echo "Could not unlock GPU clocks"

# ========== Step 3: NCU 实验 (不同 set 和 clock-control) ==========
echo ""
echo "=============================================="
echo "Step 3: NCU Profiling (Different Sets & Clock Control)"
echo "=============================================="

run_ncu_experiment() {
    local mode=$1
    local ncu_set=$2
    local clock=$3
    local name="${mode}_${ncu_set}_${clock}"
    
    echo ""
    echo "=== Experiment: $name ==="
    
    local cmd="ncu --set $ncu_set"
    
    if [ "$clock" == "base" ]; then
        cmd="$cmd --clock-control base"
    fi
    
    cmd="$cmd -o $OUTPUT_DIR/$name"
    cmd="$cmd --log-file $OUTPUT_DIR/${name}.log"
    cmd="$cmd --force-overwrite"
    cmd="$cmd $KERNEL_BIN $mode"
    
    echo "CMD: $cmd"
    
    start_time=$(date +%s)
    eval "$cmd" 2>&1 | tee "$OUTPUT_DIR/${name}_stdout.txt"
    end_time=$(date +%s)
    
    total_time=$((end_time - start_time))
    echo "Total profiling time: ${total_time}s" | tee -a "$OUTPUT_DIR/${name}_stdout.txt"
}

for mode in short long; do
    for ncu_set in basic full; do
        for clock in base none; do
            run_ncu_experiment "$mode" "$ncu_set" "$clock"
        done
    done
done

# ========== Step 4: 结果汇总 ==========
echo ""
echo "=============================================="
echo "Step 4: Results Summary"
echo "=============================================="

python3 << PYTHON_SCRIPT
import os
import sys
import glob
import json
import subprocess
import csv
import re
from io import StringIO

results_dir = "$OUTPUT_DIR"

print("=" * 80)
print("COMPLETE NCU TIMING EXPERIMENT RESULTS")
print("=" * 80)
print()

# ========== Torch Profiler Results ==========
print("=" * 80)
print("TORCH PROFILER RESULTS")
print("=" * 80)
print()
print(f"{'Mode':<10} {'Boost (us)':<15} {'Base (us)':<15} {'Ratio':<10}")
print("-" * 50)

torch_results = {}
for mode in ['short', 'long']:
    boost_file = os.path.join(results_dir, f'torch_boost_{mode}.json')
    base_file = os.path.join(results_dir, f'torch_base_{mode}.json')
    
    boost_us = None
    base_us = None
    
    if os.path.exists(boost_file):
        with open(boost_file, 'r') as f:
            boost_us = json.load(f).get('mean_us')
    
    if os.path.exists(base_file):
        with open(base_file, 'r') as f:
            base_us = json.load(f).get('mean_us')
    
    torch_results[mode] = {'boost': boost_us, 'base': base_us}
    
    boost_str = f"{boost_us:.2f}" if boost_us else "N/A"
    base_str = f"{base_us:.2f}" if base_us else "N/A"
    
    if boost_us and base_us and boost_us > 0:
        ratio = base_us / boost_us
        ratio_str = f"{ratio:.2f}x"
    else:
        ratio_str = "N/A"
    
    print(f"{mode:<10} {boost_str:<15} {base_str:<15} {ratio_str:<10}")

# ========== NCU Results ==========
print()
print("=" * 80)
print("NCU DURATION RESULTS")
print("=" * 80)
print()
print(f"{'Experiment':<25} {'Duration (us)':<15} {'SM Freq':<12} {'Total Time':<12}")
print("-" * 70)

ncu_results = {}
for ncu_file in sorted(glob.glob(os.path.join(results_dir, "*.ncu-rep"))):
    name = os.path.basename(ncu_file).replace('.ncu-rep', '')
    
    stdout_file = os.path.join(results_dir, f'{name}_stdout.txt')
    total_time = "N/A"
    if os.path.exists(stdout_file):
        with open(stdout_file, 'r') as f:
            for line in f:
                if 'Total profiling time:' in line:
                    match = re.search(r'(\d+)s', line)
                    if match:
                        total_time = f"{int(match.group(1))}s"
    
    try:
        cmd = ['ncu', '--import', ncu_file, '--csv']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        duration = None
        duration_unit = None
        sm_freq = None
        
        if proc.returncode == 0 and proc.stdout:
            reader = csv.DictReader(StringIO(proc.stdout))
            for row in reader:
                kernel = row.get('Kernel Name', '')
                if 'timed_rw_kernel' not in kernel:
                    continue
                
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')
                metric_unit = row.get('Metric Unit', '')
                
                if metric_name == 'Duration' and metric_value:
                    duration = float(metric_value.replace(',', ''))
                    duration_unit = metric_unit
                elif metric_name == 'SM Frequency' and metric_value:
                    sm_freq = f"{metric_value} {metric_unit}"
        
        if duration:
            if 'ms' in (duration_unit or '').lower():
                dur_us = duration * 1000
            elif 'ns' in (duration_unit or '').lower():
                dur_us = duration / 1000
            else:
                dur_us = duration
            
            ncu_results[name] = dur_us
            print(f"{name:<25} {dur_us:<15.2f} {sm_freq or 'N/A':<12} {total_time:<12}")
        else:
            print(f"{name:<25} {'N/A':<15} {'N/A':<12} {total_time:<12}")
    except Exception as e:
        print(f"{name:<25} Error: {e}")

# ========== 关键对比: Torch Profiler (base) vs NCU ==========
print()
print("=" * 80)
print("KEY COMPARISON: Torch Profiler (base) vs NCU Duration")
print("=" * 80)
print()
print("如果 Torch Profiler (base) ≈ NCU Duration，则 NCU 计时准确。")
print("(两者都在 base 频率 ~1.15 GHz 下运行)")
print()

for mode in ['short', 'long']:
    print(f"--- {mode.upper()} Kernel ---")
    
    torch_base = torch_results.get(mode, {}).get('base')
    torch_boost = torch_results.get(mode, {}).get('boost')
    
    if torch_boost:
        print(f"  Torch Profiler (boost ~1.9 GHz):  {torch_boost:.2f} us")
    if torch_base:
        print(f"  Torch Profiler (base ~1.15 GHz):  {torch_base:.2f} us")
    else:
        print(f"  Torch Profiler (base): N/A (频率锁定失败?)")
    
    print()
    print(f"  {'NCU Config':<20} {'Duration (us)':<15} {'vs Torch(base)':<20} {'Status':<15}")
    print(f"  {'-'*70}")
    
    for name, dur_us in sorted(ncu_results.items()):
        if not name.startswith(mode):
            continue
        
        parts = name.replace(f'{mode}_', '').split('_')
        ncu_set = parts[0] if len(parts) > 0 else '?'
        clock = parts[1] if len(parts) > 1 else '?'
        config = f"{ncu_set}/{clock}"
        
        if torch_base and torch_base > 0:
            diff = dur_us - torch_base
            diff_pct = (diff / torch_base) * 100
            diff_str = f"{diff:+.2f} us ({diff_pct:+.1f}%)"
            
            if abs(diff_pct) < 15:
                status = "✓ 准确"
            else:
                status = "✗ 有差异"
        else:
            diff_str = "N/A"
            status = "N/A"
        
        print(f"  {config:<20} {dur_us:<15.2f} {diff_str:<20} {status:<15}")
    
    print()

# ========== NCU Set Overhead ==========
print()
print("=" * 80)
print("NCU SET OVERHEAD (Total Profiling Time)")
print("=" * 80)
print()

for mode in ['short', 'long']:
    print(f"--- {mode.upper()} Kernel ---")
    print(f"  {'Set':<10} {'Clock':<10} {'Duration (us)':<15} {'Total Time':<12}")
    print(f"  {'-'*50}")
    
    for ncu_set in ['basic', 'full']:
        for clock in ['base', 'none']:
            name = f"{mode}_{ncu_set}_{clock}"
            stdout_file = os.path.join(results_dir, f'{name}_stdout.txt')
            
            total_time = "N/A"
            if os.path.exists(stdout_file):
                with open(stdout_file, 'r') as f:
                    for line in f:
                        if 'Total profiling time:' in line:
                            match = re.search(r'(\d+)s', line)
                            if match:
                                total_time = f"{int(match.group(1))}s"
            
            dur_us = ncu_results.get(name)
            dur_str = f"{dur_us:.2f}" if dur_us else "N/A"
            
            print(f"  {ncu_set:<10} {clock:<10} {dur_str:<15} {total_time:<12}")
    
    print()

# ========== 结论 ==========
print()
print("=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print("""
1. NCU 计时准确性验证：
   - 如果 nvidia-smi 成功锁定频率到 base (~1.15 GHz)
   - 则 Torch Profiler (base) 应该 ≈ NCU Duration
   - 差异 < 15% 表示 NCU 计时准确

2. 如果 nvidia-smi 锁定频率失败：
   - Torch Profiler (boost) 和 (base) 结果相同
   - 此时用频率比验证: NCU/Torch ≈ 1.65x

3. NCU Set Overhead：
   - basic: 收集基本指标，overhead 较小
   - full:  收集完整指标，需要更多 replay，overhead 较大
""")
print("=" * 80)
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
