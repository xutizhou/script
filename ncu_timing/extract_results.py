#!/usr/bin/env python3
"""
提取 NCU 实验结果并汇总

用法:
    python extract_results.py <results_dir>
"""

import os
import sys
import glob
import subprocess
import csv
from io import StringIO


def extract_ncu_metrics(ncu_rep_path):
    """从 NCU 报告中提取 Duration 和 SM Frequency"""
    result = {
        'duration': None,
        'duration_unit': None,
        'sm_freq': None,
        'sm_freq_unit': None,
    }
    
    try:
        # 方法1: 使用 --page raw
        cmd = ['ncu', '--import', ncu_rep_path, '--page', 'raw', '--csv']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if proc.returncode == 0 and proc.stdout:
            for line in proc.stdout.split('\n'):
                if 'timed_rw_kernel' not in line:
                    continue
                    
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                
                # 找 Metric Name 列
                for i, part in enumerate(parts):
                    clean = part.strip().strip('"')
                    if clean == 'gpu__time_duration.avg' and i + 2 < len(parts):
                        result['duration_unit'] = parts[i+1].strip().strip('"')
                        result['duration'] = parts[i+2].strip().strip('"')
                    elif clean == 'sm__frequency.avg' and i + 2 < len(parts):
                        result['sm_freq_unit'] = parts[i+1].strip().strip('"')
                        result['sm_freq'] = parts[i+2].strip().strip('"')
        
        # 方法2: 如果方法1失败，尝试不带 --page
        if result['duration'] is None:
            cmd = ['ncu', '--import', ncu_rep_path, '--csv']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
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
                        result['duration'] = metric_value.replace(',', '')
                        result['duration_unit'] = metric_unit
                    elif metric_name == 'SM Frequency' and metric_value:
                        result['sm_freq'] = metric_value.replace(',', '')
                        result['sm_freq_unit'] = metric_unit
                        
    except Exception as e:
        print(f"Error extracting from {ncu_rep_path}: {e}", file=sys.stderr)
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("=" * 70)
    print("NCU Timing Experiment Results")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()
    
    # ========== cudaEvent Baseline ==========
    print("=== cudaEvent Baseline (Boost Frequency ~1.9 GHz) ===")
    print()
    print(f"{'Mode':<15} {'Duration':<25}")
    print("-" * 40)
    
    for mode in ['short', 'long']:
        log_file = os.path.join(results_dir, f'baseline_{mode}.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Kernel duration:' in line:
                        # 格式: Kernel duration: 0.013 ms (13.120 us)
                        parts = line.split('Kernel duration:')[1].strip()
                        print(f"{mode:<15} {parts}")
                        break
        else:
            print(f"{mode:<15} N/A")
    
    # ========== NCU Duration ==========
    print()
    print("=== NCU Duration ===")
    print()
    print(f"{'Experiment':<20} {'Duration':<20} {'SM Freq':<15}")
    print("-" * 55)
    
    ncu_files = sorted(glob.glob(os.path.join(results_dir, "*.ncu-rep")))
    ncu_results = {}
    
    for ncu_file in ncu_files:
        name = os.path.basename(ncu_file).replace('.ncu-rep', '')
        result = extract_ncu_metrics(ncu_file)
        ncu_results[name] = result
        
        if result['duration']:
            duration_str = f"{result['duration']} {result['duration_unit']}"
        else:
            duration_str = "N/A"
        
        if result['sm_freq']:
            freq_str = f"{result['sm_freq']} {result['sm_freq_unit']}"
        else:
            freq_str = "N/A"
        
        print(f"{name:<20} {duration_str:<20} {freq_str:<15}")
    
    # ========== 对比分析 ==========
    print()
    print("=" * 70)
    print("Comparison Analysis")
    print("=" * 70)
    print()
    
    # 读取 baseline
    baselines = {}
    for mode in ['short', 'long']:
        log_file = os.path.join(results_dir, f'baseline_{mode}.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Kernel duration:' in line:
                        # 提取 us 值
                        if '(' in line and 'us)' in line:
                            us_part = line.split('(')[1].split('us)')[0].strip()
                            try:
                                baselines[mode] = float(us_part)
                            except:
                                pass
                        break
    
    for mode in ['short', 'long']:
        print(f"--- {mode.upper()} Kernel ---")
        
        if mode in baselines:
            baseline_us = baselines[mode]
            print(f"  cudaEvent (boost): {baseline_us:.2f} us")
        else:
            baseline_us = None
            print(f"  cudaEvent (boost): N/A")
        
        print()
        print(f"  {'NCU Config':<15} {'Duration (us)':<15} {'Ratio':<10} {'SM Freq':<15}")
        print(f"  {'-'*55}")
        
        for name, result in sorted(ncu_results.items()):
            if not name.startswith(mode):
                continue
            
            clock = name.replace(f"{mode}_", "")
            
            if result['duration']:
                try:
                    dur_val = float(result['duration'].replace(',', ''))
                    dur_unit = result['duration_unit'] or ''
                    
                    # 转换为 us
                    if 'ms' in dur_unit.lower():
                        dur_us = dur_val * 1000
                    elif 'ns' in dur_unit.lower():
                        dur_us = dur_val / 1000
                    else:
                        dur_us = dur_val
                    
                    dur_str = f"{dur_us:.2f}"
                    
                    if baseline_us and baseline_us > 0:
                        ratio = dur_us / baseline_us
                        ratio_str = f"{ratio:.2f}x"
                    else:
                        ratio_str = "N/A"
                except:
                    dur_str = result['duration']
                    ratio_str = "N/A"
            else:
                dur_str = "N/A"
                ratio_str = "N/A"
            
            freq_str = f"{result['sm_freq']} {result['sm_freq_unit']}" if result['sm_freq'] else "N/A"
            
            print(f"  {clock:<15} {dur_str:<15} {ratio_str:<10} {freq_str:<15}")
        
        print()
    
    print("=" * 70)
    print("Notes:")
    print("  - Expected ratio (boost/base): ~1.65x")
    print("  - If ratio > 2x: additional profiling overhead")
    print("  - NCU always locks to base frequency (~1.15 GHz)")
    print("=" * 70)


if __name__ == '__main__':
    main()

