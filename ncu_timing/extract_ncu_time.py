#!/usr/bin/env python3
"""
从NCU报告(.ncu-rep)中提取真正的kernel duration
基于实际NCU输出格式解析
"""

import os
import sys
import glob
import subprocess
import csv
from io import StringIO


def extract_from_ncu_rep(ncu_rep_path):
    """使用ncu --import从.ncu-rep文件提取kernel信息"""
    result = {
        'file': os.path.basename(ncu_rep_path),
        'kernel_name': None,
        'duration_us': None,
        'duration_ns': None,
        'cycles': None,
        'sm_freq_ghz': None,
        'grid_size': None,
        'block_size': None,
        'error': None,
    }
    
    try:
        # 使用CSV格式提取，最容易解析
        cmd = ['ncu', '--import', ncu_rep_path, '--csv']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if proc.returncode == 0 and proc.stdout.strip():
            reader = csv.DictReader(StringIO(proc.stdout))
            
            for row in reader:
                # 提取kernel名称
                if result['kernel_name'] is None:
                    result['kernel_name'] = row.get('Kernel Name', '')
                
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')
                metric_unit = row.get('Metric Unit', '')
                
                if not metric_value:
                    continue
                    
                # 清理数值中的逗号
                metric_value_clean = metric_value.replace(',', '')
                
                try:
                    if metric_name == 'Duration':
                        val = float(metric_value_clean)
                        if metric_unit == 'us':
                            result['duration_us'] = val
                            result['duration_ns'] = val * 1000
                        elif metric_unit == 'ns':
                            result['duration_ns'] = val
                            result['duration_us'] = val / 1000
                        elif metric_unit == 'ms':
                            result['duration_us'] = val * 1000
                            result['duration_ns'] = val * 1e6
                    
                    elif metric_name == 'SM Frequency':
                        val = float(metric_value_clean)
                        if metric_unit == 'Ghz':
                            result['sm_freq_ghz'] = val
                        elif metric_unit == 'Mhz':
                            result['sm_freq_ghz'] = val / 1000
                    
                    elif metric_name == 'Elapsed Cycles':
                        result['cycles'] = int(float(metric_value_clean))
                    
                    elif metric_name == 'Block Size':
                        result['block_size'] = metric_value
                    
                    elif metric_name == 'Grid Size':
                        result['grid_size'] = metric_value
                        
                except (ValueError, TypeError):
                    pass
                
                # 一旦找到Duration就可以停止（每个kernel只取第一个）
                if result['duration_us'] is not None:
                    break
                    
    except subprocess.TimeoutExpired:
        result['error'] = 'timeout'
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ncu_time.py <results_dir>")
        print("\nThis script extracts kernel duration from NCU .ncu-rep files")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("=" * 90)
    print("NCU Kernel Duration Extraction")
    print("=" * 90)
    print(f"Directory: {results_dir}")
    print()
    
    # 查找所有.ncu-rep文件
    ncu_files = sorted(glob.glob(os.path.join(results_dir, "*.ncu-rep")))
    
    if not ncu_files:
        print("No .ncu-rep files found!")
        sys.exit(1)
    
    print(f"Found {len(ncu_files)} NCU report files")
    print()
    
    # 提取所有结果
    results = []
    for ncu_file in ncu_files:
        print(f"Processing: {os.path.basename(ncu_file)}...", end=" ", flush=True)
        result = extract_from_ncu_rep(ncu_file)
        results.append(result)
        
        if result['duration_us']:
            print(f"Duration: {result['duration_us']:.2f} us")
        elif result['error']:
            print(f"Error: {result['error']}")
        else:
            print("No duration found")
    
    print()
    print("=" * 90)
    print("Results Summary - NCU Reported Kernel Duration")
    print("=" * 90)
    print()
    print(f"{'Report':<30} {'Duration (us)':<15} {'Duration (ns)':<15} {'Cycles':<12} {'SM Freq (GHz)':<12}")
    print("-" * 90)
    
    for r in results:
        name = r['file'].replace('.ncu-rep', '')
        dur_us = f"{r['duration_us']:.2f}" if r['duration_us'] else "N/A"
        dur_ns = f"{r['duration_ns']:.0f}" if r['duration_ns'] else "N/A"
        cycles = f"{r['cycles']:,}" if r['cycles'] else "N/A"
        freq = f"{r['sm_freq_ghz']:.2f}" if r['sm_freq_ghz'] else "N/A"
        print(f"{name:<30} {dur_us:<15} {dur_ns:<15} {cycles:<12} {freq:<12}")
    
    # 按mode分组分析
    print()
    print("=" * 90)
    print("Comparison: NCU Duration vs Baseline")
    print("=" * 90)
    
    # Baseline时间（从之前的实验）
    baselines = {
        'short': 13.856,  # us
        'long': 17718.0,  # us (17.718 ms)
    }
    
    by_mode = {'short': [], 'long': []}
    for r in results:
        name = r['file'].replace('.ncu-rep', '')
        if name.startswith('short'):
            by_mode['short'].append(r)
        elif name.startswith('long'):
            by_mode['long'].append(r)
    
    for mode in ['short', 'long']:
        if by_mode[mode]:
            baseline = baselines.get(mode, 0)
            print(f"\n--- {mode.upper()} Kernel (Baseline: {baseline:.2f} us) ---")
            print(f"{'Config':<25} {'NCU Duration (us)':<18} {'vs Baseline':<20} {'SM Freq':<12}")
            print("-" * 75)
            
            for r in by_mode[mode]:
                name = r['file'].replace('.ncu-rep', '').replace(f'{mode}_', '')
                if r['duration_us']:
                    dur_us = f"{r['duration_us']:.2f}"
                    if baseline > 0:
                        diff = r['duration_us'] - baseline
                        diff_pct = (diff / baseline) * 100
                        diff_str = f"{diff:+.2f} us ({diff_pct:+.1f}%)"
                    else:
                        diff_str = "N/A"
                else:
                    dur_us = "N/A"
                    diff_str = "N/A"
                freq = f"{r['sm_freq_ghz']:.2f} GHz" if r['sm_freq_ghz'] else "N/A"
                print(f"{name:<25} {dur_us:<18} {diff_str:<20} {freq:<12}")
    
    # Clock control对比
    print()
    print("=" * 90)
    print("Clock Control Impact Analysis")
    print("=" * 90)
    
    for mode in ['short', 'long']:
        mode_results = by_mode.get(mode, [])
        if not mode_results:
            continue
            
        # 分组 by set
        by_set = {}
        for r in mode_results:
            name = r['file'].replace('.ncu-rep', '').replace(f'{mode}_', '')
            parts = name.split('_')
            if len(parts) >= 2:
                ncu_set = parts[0]
                clock = parts[1]
                if ncu_set not in by_set:
                    by_set[ncu_set] = {}
                by_set[ncu_set][clock] = r
        
        print(f"\n--- {mode.upper()} Kernel ---")
        for ncu_set, clocks in by_set.items():
            base_r = clocks.get('base')
            none_r = clocks.get('none')
            
            if base_r and none_r and base_r['duration_us'] and none_r['duration_us']:
                base_dur = base_r['duration_us']
                none_dur = none_r['duration_us']
                diff = base_dur - none_dur
                diff_pct = (diff / none_dur) * 100
                
                base_freq = f"{base_r['sm_freq_ghz']:.2f}" if base_r['sm_freq_ghz'] else "?"
                none_freq = f"{none_r['sm_freq_ghz']:.2f}" if none_r['sm_freq_ghz'] else "?"
                
                print(f"  {ncu_set}:")
                print(f"    base:  {base_dur:.2f} us @ {base_freq} GHz")
                print(f"    none:  {none_dur:.2f} us @ {none_freq} GHz")
                print(f"    diff:  {diff:+.2f} us ({diff_pct:+.1f}%)")
    
    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)
    print("""
NOTE: The Duration reported by NCU is the actual kernel execution time,
      NOT the total profiling time (which includes replay overhead).
      
      The baseline duration was measured using cudaEventElapsedTime without NCU.
      Differences between NCU duration and baseline can be due to:
      1. Clock control settings (base frequency vs boost)
      2. NCU instrumentation overhead
      3. Measurement methodology differences
    """)


if __name__ == '__main__':
    main()
