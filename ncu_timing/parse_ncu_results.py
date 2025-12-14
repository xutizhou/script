#!/usr/bin/env python3
"""
NCU Results Parser

解析NCU实验结果，提取:
1. Kernel duration (Time)
2. Memory restore信息 (replay时restore的memory大小)
3. SM Frequency
4. 其他关键指标

用法:
    python parse_ncu_results.py <results_dir>
    python parse_ncu_results.py <results_dir> --export-csv
"""

import os
import re
import sys
import glob
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_log_file(log_path):
    """解析NCU日志文件，提取关键信息"""
    result = {
        'log_file': os.path.basename(log_path),
        'kernel_duration_ns': None,
        'kernel_duration_us': None,
        'kernel_duration_ms': None,
        'cycles': None,
        'sm_frequency_hz': None,
        'memory_restore_bytes': [],
        'replay_count': 0,
        'wall_clock_time': None,
        'gpu_name': None,
        'warnings': [],
        'errors': [],
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        result['errors'].append(f"Failed to read file: {e}")
        return result
    
    for line in lines:
        # 提取kernel duration
        # 格式可能是: "125664 ns", "0.126 ms", "126 us" 等
        duration_match = re.search(r'Duration[:\s]+(\d+(?:\.\d+)?)\s*(ns|us|ms|s)', line, re.IGNORECASE)
        if duration_match:
            value = float(duration_match.group(1))
            unit = duration_match.group(2).lower()
            if unit == 'ns':
                result['kernel_duration_ns'] = value
                result['kernel_duration_us'] = value / 1000
                result['kernel_duration_ms'] = value / 1e6
            elif unit == 'us':
                result['kernel_duration_ns'] = value * 1000
                result['kernel_duration_us'] = value
                result['kernel_duration_ms'] = value / 1000
            elif unit == 'ms':
                result['kernel_duration_ns'] = value * 1e6
                result['kernel_duration_us'] = value * 1000
                result['kernel_duration_ms'] = value
            elif unit == 's':
                result['kernel_duration_ns'] = value * 1e9
                result['kernel_duration_us'] = value * 1e6
                result['kernel_duration_ms'] = value * 1000
        
        # 另一种格式: "Time: 125664 ns" 或表格格式
        time_match = re.search(r'Time[:\s]+(\d+(?:\.\d+)?)\s*(ns|us|ms)', line, re.IGNORECASE)
        if time_match and result['kernel_duration_ns'] is None:
            value = float(time_match.group(1))
            unit = time_match.group(2).lower()
            if unit == 'ns':
                result['kernel_duration_ns'] = value
                result['kernel_duration_us'] = value / 1000
                result['kernel_duration_ms'] = value / 1e6
            elif unit == 'us':
                result['kernel_duration_ns'] = value * 1000
                result['kernel_duration_us'] = value
                result['kernel_duration_ms'] = value / 1000
            elif unit == 'ms':
                result['kernel_duration_ns'] = value * 1e6
                result['kernel_duration_us'] = value * 1000
                result['kernel_duration_ms'] = value
        
        # 提取cycles
        cycles_match = re.search(r'Cycles[:\s]+(\d+(?:,\d+)*)', line, re.IGNORECASE)
        if cycles_match:
            result['cycles'] = int(cycles_match.group(1).replace(',', ''))
        
        # 提取SM频率
        freq_match = re.search(r'SM Frequency[:\s]+(\d+(?:\.\d+)?)\s*(hz|khz|mhz|ghz)', line, re.IGNORECASE)
        if freq_match:
            value = float(freq_match.group(1))
            unit = freq_match.group(2).lower()
            if unit == 'hz':
                result['sm_frequency_hz'] = value
            elif unit == 'khz':
                result['sm_frequency_hz'] = value * 1e3
            elif unit == 'mhz':
                result['sm_frequency_hz'] = value * 1e6
            elif unit == 'ghz':
                result['sm_frequency_hz'] = value * 1e9
        
        # 提取memory restore信息
        # NCU在replay时会打印类似: "Restoring X bytes of memory"
        restore_match = re.search(r'[Rr]estor(?:e|ing)[:\s]+(\d+(?:\.\d+)?)\s*(bytes|KB|MB|GB)', line, re.IGNORECASE)
        if restore_match:
            value = float(restore_match.group(1))
            unit = restore_match.group(2).upper()
            if unit == 'BYTES':
                result['memory_restore_bytes'].append(value)
            elif unit == 'KB':
                result['memory_restore_bytes'].append(value * 1024)
            elif unit == 'MB':
                result['memory_restore_bytes'].append(value * 1024 * 1024)
            elif unit == 'GB':
                result['memory_restore_bytes'].append(value * 1024 * 1024 * 1024)
        
        # 提取replay计数
        replay_match = re.search(r'[Rr]eplay(?:s|ing)?[:\s]+(\d+)', line)
        if replay_match:
            result['replay_count'] = max(result['replay_count'], int(replay_match.group(1)))
        
        # 提取wall clock时间
        wall_match = re.search(r'[Ww]all\s*[Cc]lock\s*[Tt]ime[:\s]+(\d+(?:\.\d+)?)\s*s', line)
        if wall_match:
            result['wall_clock_time'] = float(wall_match.group(1))
        
        # 提取GPU名称
        gpu_match = re.search(r'(NVIDIA\s+\S+)', line)
        if gpu_match and result['gpu_name'] is None:
            result['gpu_name'] = gpu_match.group(1)
        
        # 收集警告和错误
        if 'warning' in line.lower():
            result['warnings'].append(line.strip())
        if 'error' in line.lower() and 'cudaGetErrorString' not in line:
            result['errors'].append(line.strip())
    
    return result


def parse_ncu_report(report_path):
    """使用ncu --import解析ncu-rep文件"""
    result = {
        'report_file': os.path.basename(report_path),
        'kernel_name': None,
        'duration_ns': None,
        'duration_us': None,
        'cycles': None,
        'sm_frequency': None,
        'grid_size': None,
        'block_size': None,
        'registers_per_thread': None,
        'shared_memory': None,
        'achieved_occupancy': None,
        'memory_throughput': None,
    }
    
    try:
        # 尝试使用ncu --import来提取信息
        cmd = ['ncu', '--import', report_path, '--csv', '--page', 'raw']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if proc.returncode == 0:
            lines = proc.stdout.strip().split('\n')
            if len(lines) > 1:
                # 解析CSV输出
                headers = lines[0].split(',')
                values = lines[1].split(',') if len(lines) > 1 else []
                
                for i, header in enumerate(headers):
                    if i < len(values):
                        value = values[i].strip('"')
                        header_lower = header.lower()
                        
                        if 'duration' in header_lower:
                            try:
                                result['duration_ns'] = float(value)
                                result['duration_us'] = float(value) / 1000
                            except:
                                pass
                        elif 'kernel' in header_lower and 'name' in header_lower:
                            result['kernel_name'] = value
                        elif 'cycle' in header_lower:
                            try:
                                result['cycles'] = int(value.replace(',', ''))
                            except:
                                pass
    except subprocess.TimeoutExpired:
        result['error'] = 'NCU import timed out'
    except FileNotFoundError:
        result['error'] = 'NCU not found in PATH'
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_memory_restore(log_dir):
    """分析所有日志中的memory restore信息"""
    print("\n" + "=" * 60)
    print("Memory Restore Analysis")
    print("=" * 60)
    
    restore_info = defaultdict(list)
    
    for log_file in glob.glob(os.path.join(log_dir, "*.log")):
        result = parse_log_file(log_file)
        
        # 从文件名提取实验配置
        basename = os.path.basename(log_file).replace('.log', '')
        parts = basename.split('_')
        
        if result['memory_restore_bytes']:
            restore_info[basename] = result['memory_restore_bytes']
    
    if restore_info:
        print("\nMemory restore sizes by experiment:")
        for exp_name, sizes in sorted(restore_info.items()):
            total = sum(sizes)
            print(f"  {exp_name}:")
            print(f"    Total restores: {len(sizes)}")
            print(f"    Total bytes: {total:,.0f} ({total/1024/1024:.2f} MB)")
            if sizes:
                print(f"    Min: {min(sizes):,.0f} bytes")
                print(f"    Max: {max(sizes):,.0f} bytes")
                print(f"    Avg: {total/len(sizes):,.0f} bytes")
    else:
        print("\nNo memory restore information found in logs.")
        print("Note: Memory restore info might require verbose NCU logging.")
        print("Try adding --log-file with verbose options to capture this data.")
    
    print("\n" + "-" * 60)
    print("Factors affecting memory restore size:")
    print("-" * 60)
    print("""
1. Kernel Memory Footprint:
   - Total allocated GPU memory that the kernel touches
   - Larger data = more memory to save/restore

2. NCU Set/Sections:
   - 'full' set requires more replays → more memory operations
   - Different sections collect different metrics

3. Number of Replays:
   - More metrics = more replays
   - Each replay may need to restore memory state

4. Memory Access Patterns:
   - Kernels with random access may require saving more state
   - Sequential access patterns are more predictable

5. Shared Memory Usage:
   - Kernels using shared memory need it saved/restored

6. Global Memory Writes:
   - NCU must save/restore memory modified by the kernel
""")


def main():
    parser = argparse.ArgumentParser(description='Parse NCU experiment results')
    parser.add_argument('results_dir', help='Directory containing NCU results')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory")
        sys.exit(1)
    
    print("=" * 60)
    print("NCU Results Analysis")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print()
    
    # 收集所有结果
    all_results = []
    
    # 解析日志文件
    log_files = sorted(glob.glob(os.path.join(args.results_dir, "*.log")))
    print(f"Found {len(log_files)} log files")
    print()
    
    # 按实验类型分组结果
    results_by_mode = defaultdict(list)
    results_by_set = defaultdict(list)
    results_by_clock = defaultdict(list)
    
    for log_file in log_files:
        result = parse_log_file(log_file)
        all_results.append(result)
        
        # 从文件名提取配置
        basename = os.path.basename(log_file).replace('.log', '')
        parts = basename.split('_')
        
        # 尝试解析文件名格式: mode_set_clockX.log
        if 'baseline' in basename:
            mode = parts[1] if len(parts) > 1 else 'unknown'
            results_by_mode[mode].append(('baseline', 'none', result))
        else:
            mode = parts[0] if len(parts) > 0 else 'unknown'
            ncu_set = parts[1] if len(parts) > 1 else 'unknown'
            clock = parts[2] if len(parts) > 2 else 'unknown'
            
            results_by_mode[mode].append((ncu_set, clock, result))
            results_by_set[ncu_set].append((mode, clock, result))
            results_by_clock[clock].append((mode, ncu_set, result))
    
    # 打印汇总表格
    print("=" * 80)
    print("Kernel Duration Summary (from logs)")
    print("=" * 80)
    print(f"{'Experiment':<40} {'Duration (us)':<15} {'Duration (ms)':<15} {'Cycles':<15}")
    print("-" * 80)
    
    for result in all_results:
        name = result['log_file'].replace('.log', '')
        dur_us = f"{result['kernel_duration_us']:.3f}" if result['kernel_duration_us'] else "N/A"
        dur_ms = f"{result['kernel_duration_ms']:.6f}" if result['kernel_duration_ms'] else "N/A"
        cycles = f"{result['cycles']:,}" if result['cycles'] else "N/A"
        print(f"{name:<40} {dur_us:<15} {dur_ms:<15} {cycles:<15}")
    
    print()
    
    # 按模式分析
    print("=" * 60)
    print("Analysis by Kernel Mode")
    print("=" * 60)
    
    for mode in ['short', 'long']:
        if mode in results_by_mode:
            print(f"\n--- {mode.upper()} kernel ---")
            mode_results = results_by_mode[mode]
            
            # 找baseline
            baseline_dur = None
            for ncu_set, clock, result in mode_results:
                if ncu_set == 'baseline':
                    baseline_dur = result['kernel_duration_us']
                    print(f"  Baseline: {baseline_dur:.3f} us" if baseline_dur else "  Baseline: N/A")
                    break
            
            print(f"\n  {'Set':<12} {'Clock':<10} {'Duration (us)':<15} {'Diff from baseline':<20}")
            print("  " + "-" * 57)
            
            for ncu_set, clock, result in mode_results:
                if ncu_set == 'baseline':
                    continue
                dur = result['kernel_duration_us']
                if dur and baseline_dur:
                    diff = dur - baseline_dur
                    diff_pct = (diff / baseline_dur) * 100
                    diff_str = f"{diff:+.3f} us ({diff_pct:+.1f}%)"
                else:
                    diff_str = "N/A"
                dur_str = f"{dur:.3f}" if dur else "N/A"
                print(f"  {ncu_set:<12} {clock:<10} {dur_str:<15} {diff_str:<20}")
    
    # 按NCU set分析
    print("\n" + "=" * 60)
    print("Analysis by NCU Set")
    print("=" * 60)
    
    for ncu_set in ['basic', 'detailed', 'full', 'roofline']:
        if ncu_set in results_by_set:
            print(f"\n--- {ncu_set} ---")
            set_results = results_by_set[ncu_set]
            
            for mode, clock, result in set_results:
                dur = result['kernel_duration_us']
                dur_str = f"{dur:.3f} us" if dur else "N/A"
                print(f"  {mode} ({clock}): {dur_str}")
    
    # 按clock分析
    print("\n" + "=" * 60)
    print("Analysis by Clock Control")
    print("=" * 60)
    
    for clock in ['clockbase', 'clocknone']:
        if clock in results_by_clock:
            print(f"\n--- {clock} ---")
            clock_results = results_by_clock[clock]
            
            for mode, ncu_set, result in clock_results:
                dur = result['kernel_duration_us']
                freq = result['sm_frequency_hz']
                dur_str = f"{dur:.3f} us" if dur else "N/A"
                freq_str = f"{freq/1e9:.2f} GHz" if freq else "N/A"
                print(f"  {mode} ({ncu_set}): {dur_str} @ {freq_str}")
    
    # Memory restore分析
    analyze_memory_restore(args.results_dir)
    
    # 导出CSV
    if args.export_csv:
        csv_file = os.path.join(args.results_dir, 'results_summary.csv')
        print(f"\nExporting to CSV: {csv_file}")
        
        with open(csv_file, 'w') as f:
            headers = ['experiment', 'mode', 'ncu_set', 'clock', 'duration_ns', 
                      'duration_us', 'duration_ms', 'cycles', 'sm_frequency_hz', 
                      'wall_clock_time', 'gpu_name']
            f.write(','.join(headers) + '\n')
            
            for result in all_results:
                basename = result['log_file'].replace('.log', '')
                parts = basename.split('_')
                
                if 'baseline' in basename:
                    mode = parts[1] if len(parts) > 1 else ''
                    ncu_set = 'baseline'
                    clock = 'none'
                else:
                    mode = parts[0] if len(parts) > 0 else ''
                    ncu_set = parts[1] if len(parts) > 1 else ''
                    clock = parts[2].replace('clock', '') if len(parts) > 2 else ''
                
                row = [
                    basename,
                    mode,
                    ncu_set,
                    clock,
                    str(result['kernel_duration_ns'] or ''),
                    str(result['kernel_duration_us'] or ''),
                    str(result['kernel_duration_ms'] or ''),
                    str(result['cycles'] or ''),
                    str(result['sm_frequency_hz'] or ''),
                    str(result['wall_clock_time'] or ''),
                    str(result['gpu_name'] or ''),
                ]
                f.write(','.join(row) + '\n')
        
        print(f"CSV exported successfully!")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()

