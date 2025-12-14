#!/usr/bin/env python3
"""
对比 NCU Duration 和 Torch Profiler Duration

用法:
    python compare_ncu_torch.py <results_dir>
"""

import os
import sys
import glob
import json
import subprocess
import csv
import re
from io import StringIO


def extract_ncu_duration(ncu_rep_path):
    """从 NCU 报告中提取 Duration 和 SM Frequency"""
    result = {
        'file': os.path.basename(ncu_rep_path),
        'duration_us': None,
        'sm_freq_ghz': None,
        'error': None,
    }
    
    try:
        cmd = ['ncu', '--import', ncu_rep_path, '--csv']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if proc.returncode == 0 and proc.stdout.strip():
            reader = csv.DictReader(StringIO(proc.stdout))
            
            for row in reader:
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')
                metric_unit = row.get('Metric Unit', '')
                
                if not metric_value:
                    continue
                    
                metric_value_clean = metric_value.replace(',', '')
                
                try:
                    if metric_name == 'Duration':
                        val = float(metric_value_clean)
                        if 'us' in metric_unit.lower():
                            result['duration_us'] = val
                        elif 'ns' in metric_unit.lower():
                            result['duration_us'] = val / 1000
                        elif 'ms' in metric_unit.lower():
                            result['duration_us'] = val * 1000
                        else:
                            result['duration_us'] = val  # assume us
                    
                    elif metric_name == 'SM Frequency':
                        val = float(metric_value_clean)
                        if 'ghz' in metric_unit.lower():
                            result['sm_freq_ghz'] = val
                        elif 'mhz' in metric_unit.lower():
                            result['sm_freq_ghz'] = val / 1000
                        else:
                            result['sm_freq_ghz'] = val
                        
                except (ValueError, TypeError):
                    pass
                    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def load_torch_profiler_result(json_path):
    """加载 torch profiler JSON 结果"""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_ncu_torch.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("=" * 80)
    print("NCU vs Torch Profiler Duration Comparison")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()
    
    # 加载 torch profiler 结果
    torch_results = {}
    for mode in ['short', 'long']:
        json_path = os.path.join(results_dir, f'torch_profiler_{mode}.json')
        result = load_torch_profiler_result(json_path)
        if result:
            torch_results[mode] = result
    
    if not torch_results:
        print("WARNING: No torch profiler results found!")
        print()
    
    # 加载 NCU 结果
    ncu_files = sorted(glob.glob(os.path.join(results_dir, "*.ncu-rep")))
    ncu_results = {}
    
    for ncu_file in ncu_files:
        result = extract_ncu_duration(ncu_file)
        name = os.path.basename(ncu_file).replace('.ncu-rep', '')
        ncu_results[name] = result
    
    if not ncu_results:
        print("ERROR: No NCU reports found!")
        sys.exit(1)
    
    # ========== 结果表格 ==========
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    # Torch Profiler 结果
    print()
    print("Torch Profiler (Reference - Boost Frequency ~1.9 GHz):")
    print("-" * 50)
    print(f"  {'Mode':<10} {'Duration (us)':<15} {'Duration (ms)':<15}")
    for mode in ['short', 'long']:
        if mode in torch_results:
            dur_us = torch_results[mode]['mean_us']
            dur_ms = dur_us / 1000
            print(f"  {mode:<10} {dur_us:<15.2f} {dur_ms:<15.4f}")
        else:
            print(f"  {mode:<10} {'N/A':<15} {'N/A':<15}")
    
    # NCU 结果
    print()
    print("NCU Duration (Base Frequency ~1.15 GHz):")
    print("-" * 70)
    print(f"  {'Experiment':<25} {'Duration (us)':<15} {'Duration (ms)':<15} {'SM Freq':<12}")
    for name in sorted(ncu_results.keys()):
        result = ncu_results[name]
        if result['duration_us']:
            dur_us = result['duration_us']
            dur_ms = dur_us / 1000
            freq = f"{result['sm_freq_ghz']:.2f} GHz" if result['sm_freq_ghz'] else "N/A"
            print(f"  {name:<25} {dur_us:<15.2f} {dur_ms:<15.4f} {freq:<12}")
        else:
            print(f"  {name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    # ========== 对比分析 ==========
    print()
    print("=" * 80)
    print("Comparison: NCU vs Torch Profiler")
    print("=" * 80)
    
    for mode in ['short', 'long']:
        print()
        print(f"--- {mode.upper()} Kernel ---")
        
        # Torch profiler 基准
        if mode in torch_results:
            torch_dur = torch_results[mode]['mean_us']
            print(f"  Torch Profiler (boost ~1.9 GHz): {torch_dur:.2f} us")
        else:
            torch_dur = None
            print(f"  Torch Profiler: N/A")
        
        print()
        print(f"  {'NCU Config':<20} {'Duration':<12} {'vs Torch':<25} {'Ratio':<10}")
        print(f"  {'-'*70}")
        
        # NCU 结果
        for name in sorted(ncu_results.keys()):
            if mode not in name:
                continue
            
            result = ncu_results[name]
            
            # 提取 clock 类型
            clock_match = re.search(r'_(base|none)$', name)
            clock = clock_match.group(1) if clock_match else "?"
            config = f"clock={clock}"
            
            if result['duration_us']:
                dur = result['duration_us']
                dur_str = f"{dur:.2f} us"
                
                if torch_dur and torch_dur > 0:
                    diff = dur - torch_dur
                    ratio = dur / torch_dur
                    diff_str = f"{diff:+.2f} us ({(diff/torch_dur)*100:+.1f}%)"
                    ratio_str = f"{ratio:.2f}x"
                else:
                    diff_str = "N/A"
                    ratio_str = "N/A"
            else:
                dur_str = "N/A"
                diff_str = "N/A"
                ratio_str = "N/A"
            
            print(f"  {config:<20} {dur_str:<12} {diff_str:<25} {ratio_str:<10}")
    
    # ========== 结论 ==========
    print()
    print("=" * 80)
    print("Analysis")
    print("=" * 80)
    print("""
Expected Behavior:
  - Torch Profiler runs at BOOST frequency (~1.9 GHz)
  - NCU with --clock-control base runs at BASE frequency (~1.15 GHz)
  - NCU with default (none) may also lock to base frequency
  
Frequency Ratio:
  - boost / base ≈ 1.9 / 1.15 ≈ 1.65x
  - So NCU duration should be ~1.65x longer than Torch Profiler
  
If NCU duration is significantly longer (>2x), it may indicate:
  - Additional profiling overhead
  - Memory bandwidth limited kernels scale differently
  
If NCU duration is close to Torch Profiler:
  - NCU might be using boost frequency (--clock-control none)
""")


if __name__ == '__main__':
    main()
