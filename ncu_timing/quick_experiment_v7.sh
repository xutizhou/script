#!/bin/bash
# NCU Timing Experiment V7 (H20 Optimized)
# 
# H20 GPU 频率信息:
#   - Max (Boost): 1980 MHz
#   - NCU Base:    1830 MHz (NCU --clock-control base 使用的频率)
#   - Min (Idle):  345 MHz
#
# 实验设计：
# 1. Torch Profiler (boost): 默认 boost 频率 (~1980 MHz)
# 2. Torch Profiler (locked 1830): 用 nvidia-smi 锁定到 1830 MHz (匹配 NCU base)
# 3. NCU 不同 set：basic, full
# 4. NCU 不同 clock-control：base, none
#
# 验证方法：
# - 如果 Torch Profiler (locked 1830) ≈ NCU Duration (base)，则 NCU 计时准确
#
# 用法: ./quick_experiment_v7.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"./ncu_results_v7_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/timing_kernel.cu"
KERNEL_BIN="$SCRIPT_DIR/timing_kernel"

# H20 频率设置
NCU_BASE_FREQ=1830  # NCU --clock-control base 使用的频率
BOOST_FREQ=1980     # Max boost 频率
LOW_FREQ=1200       # 手动设置的低频率 (用于对比)

echo "=============================================="
echo "NCU Timing Experiment V7 (H20 Optimized)"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "Boost Frequency: ${BOOST_FREQ} MHz"
echo "NCU Base Frequency: ${NCU_BASE_FREQ} MHz"
echo "Low Frequency: ${LOW_FREQ} MHz"
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

# ========== Step 1: Torch Profiler (Boost ~1980 MHz) ==========
echo "=============================================="
echo "Step 1: Torch Profiler at BOOST Frequency (~${BOOST_FREQ} MHz)"
echo "=============================================="

# 解锁频率
echo "Unlocking GPU clocks (boost mode)..."
nvidia-smi -rgc 2>/dev/null && echo "GPU clocks unlocked." || echo "Could not unlock GPU clocks"
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

# ========== Step 2: Torch Profiler (Locked 1830 MHz - match NCU base) ==========
echo ""
echo "=============================================="
echo "Step 2: Torch Profiler at ${NCU_BASE_FREQ} MHz (NCU Base)"
echo "=============================================="

# 尝试锁定到 NCU base 频率 1830 MHz
echo "Attempting to lock GPU to ${NCU_BASE_FREQ} MHz (NCU base frequency)..."
nvidia-smi -lgc ${NCU_BASE_FREQ},${NCU_BASE_FREQ} 2>/dev/null && echo "GPU clocks locked to ${NCU_BASE_FREQ} MHz." || \
nvidia-smi -lgc ${NCU_BASE_FREQ} 2>/dev/null && echo "GPU clocks locked to ${NCU_BASE_FREQ} MHz." || \
echo "Could not lock GPU clocks (may need root/privileged mode)"
sleep 2

echo ""
echo "Current GPU clocks (after lock attempt):"
nvidia-smi --query-gpu=clocks.gr,clocks.sm --format=csv 2>/dev/null || true

for mode in short long; do
    echo ""
    echo "--- $mode kernel (locked ${NCU_BASE_FREQ} MHz) ---"
    python "$SCRIPT_DIR/timing_kernel_torch.py" $mode --runs 10 \
        --output-json "$OUTPUT_DIR/torch_locked_${mode}.json" \
        2>&1 | tee "$OUTPUT_DIR/torch_locked_${mode}.log"
done

# ========== Step 2.5: Torch Profiler (Locked LOW_FREQ MHz) ==========
echo ""
echo "=============================================="
echo "Step 2.5: Torch Profiler at ${LOW_FREQ} MHz (Low Frequency)"
echo "=============================================="

# 锁定到低频率
echo "Attempting to lock GPU to ${LOW_FREQ} MHz (low frequency)..."
nvidia-smi -lgc ${LOW_FREQ},${LOW_FREQ} 2>/dev/null && echo "GPU clocks locked to ${LOW_FREQ} MHz." || \
nvidia-smi -lgc ${LOW_FREQ} 2>/dev/null && echo "GPU clocks locked to ${LOW_FREQ} MHz." || \
echo "Could not lock GPU clocks (may need root/privileged mode)"
sleep 2

echo ""
echo "Current GPU clocks (after lock attempt):"
nvidia-smi --query-gpu=clocks.gr,clocks.sm --format=csv 2>/dev/null || true

for mode in short long; do
    echo ""
    echo "--- $mode kernel (locked ${LOW_FREQ} MHz) ---"
    python "$SCRIPT_DIR/timing_kernel_torch.py" $mode --runs 10 \
        --output-json "$OUTPUT_DIR/torch_low_${mode}.json" \
        2>&1 | tee "$OUTPUT_DIR/torch_low_${mode}.log"
done

# 解锁频率
echo ""
echo "Unlocking GPU clocks..."
nvidia-smi -rgc 2>/dev/null && echo "GPU clocks unlocked." || echo "Could not unlock GPU clocks"

# ========== Step 3: NCU 实验 ==========
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

# ========== Step 3.5: NCU 低频率实验 ==========
echo ""
echo "=============================================="
echo "Step 3.5: NCU Profiling at Low Frequency (${LOW_FREQ} MHz)"
echo "=============================================="

# 锁定到低频率
echo "Locking GPU to ${LOW_FREQ} MHz for NCU profiling..."
nvidia-smi -lgc ${LOW_FREQ},${LOW_FREQ} 2>/dev/null && echo "GPU clocks locked to ${LOW_FREQ} MHz." || \
echo "Could not lock GPU clocks"
sleep 2

echo ""
echo "Current GPU clocks:"
nvidia-smi --query-gpu=clocks.gr,clocks.sm --format=csv 2>/dev/null || true

# 运行 NCU (clock-control=none 让 NCU 不改变我们设置的频率，频率已锁定到 LOW_FREQ)
for mode in short long; do
    for ncu_set in basic; do
        name="${mode}_${ncu_set}_none_low"
        echo ""
        echo "=== Experiment: $name (${LOW_FREQ} MHz) ==="
        
        cmd="ncu --set $ncu_set --clock-control none"
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
    done
done

# 解锁频率
echo ""
echo "Unlocking GPU clocks..."
nvidia-smi -rgc 2>/dev/null && echo "GPU clocks unlocked." || echo "Could not unlock GPU clocks"

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
ncu_base_freq = ${NCU_BASE_FREQ}
boost_freq = ${BOOST_FREQ}
low_freq = ${LOW_FREQ}

print("=" * 80)
print("NCU TIMING EXPERIMENT V7 RESULTS (H20 Optimized)")
print("=" * 80)
print()
print(f"H20 GPU Frequencies:")
print(f"  Boost:    {boost_freq} MHz")
print(f"  NCU Base: {ncu_base_freq} MHz")
print()

# ========== Torch Profiler Results ==========
print("=" * 80)
print("TORCH PROFILER RESULTS")
print("=" * 80)
print()
print(f"{'Mode':<10} {'Boost (us)':<15} {'Locked {ncu_base_freq}MHz (us)':<20} {'Ratio':<10}")
print("-" * 60)

torch_results = {}
for mode in ['short', 'long']:
    boost_file = os.path.join(results_dir, f'torch_boost_{mode}.json')
    locked_file = os.path.join(results_dir, f'torch_locked_{mode}.json')
    low_file = os.path.join(results_dir, f'torch_low_{mode}.json')
    
    boost_us = locked_us = low_us = None
    
    if os.path.exists(boost_file):
        with open(boost_file, 'r') as f:
            boost_us = json.load(f).get('mean_us')
    
    if os.path.exists(locked_file):
        with open(locked_file, 'r') as f:
            locked_us = json.load(f).get('mean_us')
    
    if os.path.exists(low_file):
        with open(low_file, 'r') as f:
            low_us = json.load(f).get('mean_us')
    
    torch_results[mode] = {'boost': boost_us, 'locked': locked_us, 'low': low_us}

# ========== NCU Results ==========
print()
print("=" * 80)
print("NCU DURATION RESULTS")
print("=" * 80)
print()
print(f"{'Experiment':<25} {'Duration (us)':<15} {'SM Freq (GHz)':<15} {'Total Time':<12}")
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
                    sm_freq = float(metric_value.replace(',', ''))
        
        if duration:
            if 'ms' in (duration_unit or '').lower():
                dur_us = duration * 1000
            elif 'ns' in (duration_unit or '').lower():
                dur_us = duration / 1000
            else:
                dur_us = duration
            
            ncu_results[name] = {'duration': dur_us, 'freq': sm_freq}
            freq_str = f"{sm_freq:.2f}" if sm_freq else "N/A"
            print(f"{name:<25} {dur_us:<15.2f} {freq_str:<15} {total_time:<12}")
        else:
            print(f"{name:<25} {'N/A':<15} {'N/A':<15} {total_time:<12}")
    except Exception as e:
        print(f"{name:<25} Error: {e}")

# ========== 关键对比 ==========
print()
print("=" * 80)
print("KEY COMPARISON: Torch Profiler (locked) vs NCU Duration (base)")
print("=" * 80)
print()
print(f"NCU --clock-control base 锁定 GPU 到 ~{ncu_base_freq} MHz")
print(f"Torch Profiler (locked) 也锁定到 {ncu_base_freq} MHz")
print("两者应该一致，差异 < 15% 表示 NCU 计时准确")
print()

for mode in ['short', 'long']:
    print(f"--- {mode.upper()} Kernel ---")
    
    torch_boost = torch_results.get(mode, {}).get('boost')
    torch_locked = torch_results.get(mode, {}).get('locked')
    
    if torch_boost:
        print(f"  Torch Profiler (boost ~{boost_freq} MHz):   {torch_boost:.2f} us")
    if torch_locked:
        print(f"  Torch Profiler (locked ~{ncu_base_freq} MHz): {torch_locked:.2f} us")
    
    print()
    print(f"  {'NCU Config':<15} {'Duration (us)':<15} {'SM Freq (GHz)':<15} {'vs Torch(locked)':<20} {'Status':<15}")
    print(f"  {'-'*80}")
    
    for name, data in sorted(ncu_results.items()):
        if not name.startswith(mode):
            continue
        
        dur_us = data['duration']
        freq = data.get('freq')
        
        parts = name.replace(f'{mode}_', '').split('_')
        ncu_set = parts[0] if len(parts) > 0 else '?'
        clock = parts[1] if len(parts) > 1 else '?'
        config = f"{ncu_set}/{clock}"
        
        freq_str = f"{freq:.2f}" if freq else "N/A"
        
        if torch_locked and torch_locked > 0:
            diff = dur_us - torch_locked
            diff_pct = (diff / torch_locked) * 100
            diff_str = f"{diff:+.2f} us ({diff_pct:+.1f}%)"
            
            if abs(diff_pct) < 15:
                status = "✓ 准确"
            else:
                status = "✗ 有差异"
        else:
            diff_str = "N/A"
            status = "N/A"
        
        print(f"  {config:<15} {dur_us:<15.2f} {freq_str:<15} {diff_str:<20} {status:<15}")
    
    print()

# ========== 频率验证 ==========
print()
print("=" * 80)
print("FREQUENCY VALIDATION")
print("=" * 80)
print()
print(f"如果 nvidia-smi 频率锁定失败，用频率比验证:")
print(f"  理论比值: {boost_freq}/{ncu_base_freq} = {boost_freq/ncu_base_freq:.3f}")
print()

for mode in ['short', 'long']:
    torch_boost = torch_results.get(mode, {}).get('boost')
    torch_locked = torch_results.get(mode, {}).get('locked')
    
    ncu_base_dur = None
    for name, data in ncu_results.items():
        if name == f'{mode}_basic_base':
            ncu_base_dur = data['duration']
            break
    
    print(f"--- {mode.upper()} Kernel ---")
    if torch_boost and ncu_base_dur:
        actual_ratio = ncu_base_dur / torch_boost
        expected_ratio = boost_freq / ncu_base_freq
        ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio * 100
        
        print(f"  NCU(base) / Torch(boost) = {ncu_base_dur:.2f} / {torch_boost:.2f} = {actual_ratio:.3f}")
        print(f"  Expected ratio: {expected_ratio:.3f}")
        print(f"  Difference: {ratio_diff:.1f}%")
        
        if ratio_diff < 15:
            print(f"  -> ✓ 频率比验证通过")
        else:
            print(f"  -> ✗ 频率比不匹配")
    print()

print("=" * 80)

# ========== LONG Kernel 综合表格 ==========
print()
print("=" * 80)
print("LONG KERNEL COMPREHENSIVE RESULTS TABLE")
print("=" * 80)
print()

# 收集所有 long kernel 数据
long_data = []

# Torch Profiler 结果
torch_boost = torch_results.get('long', {}).get('boost')
torch_locked = torch_results.get('long', {}).get('locked')
torch_low = torch_results.get('long', {}).get('low')

if torch_boost:
    long_data.append({
        'tool': 'Torch Profiler',
        'set': '-',
        'clock': 'boost',
        'freq_mhz': boost_freq,
        'duration_us': torch_boost,
        'duration_ms': torch_boost / 1000,
    })

if torch_locked:
    long_data.append({
        'tool': 'Torch Profiler',
        'set': '-',
        'clock': 'locked',
        'freq_mhz': ncu_base_freq,
        'duration_us': torch_locked,
        'duration_ms': torch_locked / 1000,
    })

if torch_low:
    long_data.append({
        'tool': 'Torch Profiler',
        'set': '-',
        'clock': 'low',
        'freq_mhz': low_freq,
        'duration_us': torch_low,
        'duration_ms': torch_low / 1000,
    })

# NCU 结果 (包含所有 clock 选项: base, none, low)
for name, data in sorted(ncu_results.items()):
    if not name.startswith('long'):
        continue
    
    parts = name.replace('long_', '').split('_')
    ncu_set = parts[0] if len(parts) > 0 else '?'
    clock = parts[1] if len(parts) > 1 else '?'
    
    freq_mhz = int(data.get('freq', 0) * 1000) if data.get('freq') else None
    
    long_data.append({
        'tool': 'NCU',
        'set': ncu_set,
        'clock': clock,
        'freq_mhz': freq_mhz,
        'duration_us': data['duration'],
        'duration_ms': data['duration'] / 1000,
    })

# 打印表格
print(f"{'Tool':<18} {'Set':<10} {'Clock':<10} {'Freq (MHz)':<12} {'Duration (us)':<15} {'Duration (ms)':<15}")
print("-" * 90)

baseline_us = torch_locked if torch_locked else (torch_boost if torch_boost else None)

for row in long_data:
    freq_str = f"{row['freq_mhz']}" if row['freq_mhz'] else "N/A"
    
    diff_str = ""
    if baseline_us and not (row['tool'] == 'Torch Profiler' and row['clock'] == 'locked'):
        diff = row['duration_us'] - baseline_us
        diff_pct = (diff / baseline_us) * 100
        diff_str = f" ({diff_pct:+.1f}%)"
    
    print(f"{row['tool']:<18} {row['set']:<10} {row['clock']:<10} {freq_str:<12} {row['duration_us']:<15.2f} {row['duration_ms']:<15.3f}{diff_str}")

print()
print("注：百分比差异相对于 Torch Profiler (locked) 计算")
print()

# Markdown 格式表格
print("--- Markdown 格式 (方便复制) ---")
print()
print("| Tool | Set | Clock | Freq (MHz) | Duration (us) | Duration (ms) | vs Baseline |")
print("|------|-----|-------|------------|---------------|---------------|-------------|")

for row in long_data:
    freq_str = f"{row['freq_mhz']}" if row['freq_mhz'] else "N/A"
    
    if baseline_us:
        diff = row['duration_us'] - baseline_us
        diff_pct = (diff / baseline_us) * 100
        diff_str = f"{diff_pct:+.1f}%"
    else:
        diff_str = "N/A"
    
    if row['tool'] == 'Torch Profiler' and row['clock'] == 'locked':
        diff_str = "baseline"
    
    print(f"| {row['tool']} | {row['set']} | {row['clock']} | {freq_str} | {row['duration_us']:.2f} | {row['duration_ms']:.3f} | {diff_str} |")

print()
print("=" * 80)
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
