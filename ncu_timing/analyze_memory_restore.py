#!/usr/bin/env python3
"""
NCU Memory Restore Analysis

分析NCU profiling过程中的memory restore行为：
1. 每次replay restore的memory大小
2. 影响restore memory大小的因素
3. 不同配置下的memory restore模式

用法:
    python analyze_memory_restore.py <log_file_or_dir>
    python analyze_memory_restore.py <log_file_or_dir> --verbose
"""

import os
import re
import sys
import glob
import argparse
from collections import defaultdict


def parse_memory_info_from_log(log_content):
    """从NCU日志中提取详细的memory信息"""
    info = {
        'restores': [],           # 每次restore的字节数
        'replays': [],            # replay信息
        'memory_allocations': [], # GPU内存分配
        'kernel_memory_footprint': None,
        'sections_collected': [],
        'metrics_count': 0,
        'passes': 0,
    }
    
    lines = log_content.split('\n')
    
    for i, line in enumerate(lines):
        # Memory restore模式 - 多种可能的格式
        patterns = [
            r'[Rr]estor(?:e|ing)\s+(\d+(?:\.\d+)?)\s*(B|bytes|KB|MB|GB)',
            r'[Mm]emory\s+[Rr]estor(?:e|ed|ing)[:\s]+(\d+(?:\.\d+)?)\s*(B|bytes|KB|MB|GB)',
            r'[Ss]av(?:e|ing)\s+(\d+(?:\.\d+)?)\s*(B|bytes|KB|MB|GB)\s+.*[Mm]emory',
            r'[Bb]ackup\s+(\d+(?:\.\d+)?)\s*(B|bytes|KB|MB|GB)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).upper()
                if unit in ['B', 'BYTES']:
                    bytes_val = value
                elif unit == 'KB':
                    bytes_val = value * 1024
                elif unit == 'MB':
                    bytes_val = value * 1024 * 1024
                elif unit == 'GB':
                    bytes_val = value * 1024 * 1024 * 1024
                info['restores'].append({
                    'bytes': bytes_val,
                    'line': line.strip(),
                    'line_num': i + 1,
                })
                break
        
        # Replay信息
        replay_match = re.search(r'[Rr]eplay\s*#?(\d+)', line)
        if replay_match:
            info['replays'].append({
                'replay_num': int(replay_match.group(1)),
                'line': line.strip(),
            })
        
        # Pass信息
        pass_match = re.search(r'[Pp]ass\s*(\d+)\s*(?:of|/)\s*(\d+)', line)
        if pass_match:
            info['passes'] = max(info['passes'], int(pass_match.group(2)))
        
        # Section信息
        section_match = re.search(r'[Cc]ollecting\s+(?:section\s+)?["\']?(\w+)', line)
        if section_match:
            section_name = section_match.group(1)
            if section_name not in info['sections_collected']:
                info['sections_collected'].append(section_name)
        
        # Metrics count
        metrics_match = re.search(r'(\d+)\s+[Mm]etrics', line)
        if metrics_match:
            info['metrics_count'] = max(info['metrics_count'], int(metrics_match.group(1)))
        
        # Memory allocation
        alloc_match = re.search(r'[Aa]llocat(?:e|ed|ing)\s+(\d+(?:\.\d+)?)\s*(B|bytes|KB|MB|GB)', line, re.IGNORECASE)
        if alloc_match:
            value = float(alloc_match.group(1))
            unit = alloc_match.group(2).upper()
            if unit in ['B', 'BYTES']:
                bytes_val = value
            elif unit == 'KB':
                bytes_val = value * 1024
            elif unit == 'MB':
                bytes_val = value * 1024 * 1024
            elif unit == 'GB':
                bytes_val = value * 1024 * 1024 * 1024
            info['memory_allocations'].append(bytes_val)
    
    return info


def analyze_single_log(log_path, verbose=False):
    """分析单个日志文件"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(log_path)}")
    print(f"{'='*60}")
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    info = parse_memory_info_from_log(content)
    
    # 打印摘要
    print(f"\nReplay/Pass Information:")
    print(f"  Total replays detected: {len(info['replays'])}")
    print(f"  Total passes: {info['passes']}")
    print(f"  Sections collected: {len(info['sections_collected'])}")
    if info['sections_collected']:
        print(f"    {', '.join(info['sections_collected'][:10])}")
        if len(info['sections_collected']) > 10:
            print(f"    ... and {len(info['sections_collected'])-10} more")
    
    print(f"\nMemory Restore Information:")
    if info['restores']:
        total_restored = sum(r['bytes'] for r in info['restores'])
        print(f"  Number of restore operations: {len(info['restores'])}")
        print(f"  Total bytes restored: {total_restored:,.0f} ({total_restored/1024/1024:.2f} MB)")
        
        if len(info['restores']) > 0:
            bytes_list = [r['bytes'] for r in info['restores']]
            print(f"  Min restore size: {min(bytes_list):,.0f} bytes")
            print(f"  Max restore size: {max(bytes_list):,.0f} bytes")
            print(f"  Avg restore size: {sum(bytes_list)/len(bytes_list):,.0f} bytes")
        
        if verbose:
            print(f"\n  Detailed restore operations:")
            for i, restore in enumerate(info['restores'][:20]):  # 限制显示前20个
                print(f"    [{restore['line_num']}] {restore['bytes']:,.0f} bytes")
            if len(info['restores']) > 20:
                print(f"    ... and {len(info['restores'])-20} more")
    else:
        print("  No memory restore operations detected in log.")
        print("  Note: You may need to enable verbose logging in NCU:")
        print("    ncu --log-file <file> --print-details all ...")
    
    if info['memory_allocations']:
        print(f"\nMemory Allocations:")
        total_alloc = sum(info['memory_allocations'])
        print(f"  Total allocations: {len(info['memory_allocations'])}")
        print(f"  Total bytes allocated: {total_alloc:,.0f} ({total_alloc/1024/1024:.2f} MB)")
    
    return info


def compare_experiments(results_dir):
    """比较不同实验配置的memory restore行为"""
    print("\n" + "=" * 70)
    print("Cross-Experiment Memory Restore Comparison")
    print("=" * 70)
    
    log_files = sorted(glob.glob(os.path.join(results_dir, "*.log")))
    
    if not log_files:
        print("No log files found.")
        return
    
    # 收集所有实验的数据
    experiments = {}
    
    for log_file in log_files:
        basename = os.path.basename(log_file).replace('.log', '')
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            info = parse_memory_info_from_log(content)
            experiments[basename] = info
        except Exception as e:
            print(f"Error processing {basename}: {e}")
    
    # 按不同维度分组分析
    print("\n--- By NCU Set ---")
    by_set = defaultdict(list)
    for name, info in experiments.items():
        parts = name.split('_')
        if len(parts) >= 2 and 'baseline' not in name:
            ncu_set = parts[1]
            by_set[ncu_set].append((name, info))
    
    print(f"\n{'Set':<15} {'Experiments':<5} {'Avg Restores':<15} {'Avg Total MB':<15}")
    print("-" * 50)
    
    for ncu_set in ['basic', 'detailed', 'full', 'roofline']:
        if ncu_set in by_set:
            exps = by_set[ncu_set]
            restore_counts = [len(info['restores']) for _, info in exps]
            total_mbs = [sum(r['bytes'] for r in info['restores'])/1024/1024 for _, info in exps]
            
            avg_restores = sum(restore_counts) / len(restore_counts) if restore_counts else 0
            avg_mb = sum(total_mbs) / len(total_mbs) if total_mbs else 0
            
            print(f"{ncu_set:<15} {len(exps):<5} {avg_restores:<15.1f} {avg_mb:<15.2f}")
    
    # 按kernel模式分组
    print("\n--- By Kernel Mode ---")
    by_mode = defaultdict(list)
    for name, info in experiments.items():
        parts = name.split('_')
        if parts:
            mode = parts[0]
            by_mode[mode].append((name, info))
    
    print(f"\n{'Mode':<15} {'Experiments':<5} {'Avg Restores':<15} {'Avg Total MB':<15}")
    print("-" * 50)
    
    for mode in ['short', 'long', 'baseline']:
        if mode in by_mode:
            exps = by_mode[mode]
            restore_counts = [len(info['restores']) for _, info in exps]
            total_mbs = [sum(r['bytes'] for r in info['restores'])/1024/1024 for _, info in exps]
            
            avg_restores = sum(restore_counts) / len(restore_counts) if restore_counts else 0
            avg_mb = sum(total_mbs) / len(total_mbs) if total_mbs else 0
            
            print(f"{mode:<15} {len(exps):<5} {avg_restores:<15.1f} {avg_mb:<15.2f}")


def print_factors_analysis():
    """打印影响memory restore大小的因素分析"""
    print("\n" + "=" * 70)
    print("Factors Affecting NCU Memory Restore Size")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ 1. KERNEL MEMORY FOOTPRINT                                          │
├─────────────────────────────────────────────────────────────────────┤
│ • Global memory read/written by the kernel                          │
│ • Larger data buffers = more memory to backup/restore               │
│ • Example: 64MB data buffer requires ~64MB restore per replay       │
│                                                                     │
│ Detection: Check cudaMalloc sizes in your code                      │
│ Mitigation: Use smaller test data for profiling                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. NCU SET / SECTIONS SELECTED                                      │
├─────────────────────────────────────────────────────────────────────┤
│ • Different sets collect different metrics                          │
│ • More sections = more replay passes                                │
│                                                                     │
│ Set         Approx Metrics    Replays Needed                        │
│ ─────────   ──────────────    ──────────────                        │
│ basic       ~190              Few (2-3)                             │
│ detailed    ~560              Medium (5-10)                         │
│ full        ~5900             Many (20-50+)                         │
│ roofline    ~5260             Many (requires full memory analysis)  │
│                                                                     │
│ Each replay may need memory restore if kernel modifies memory       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. MEMORY MODIFICATION PATTERN                                      │
├─────────────────────────────────────────────────────────────────────┤
│ • NCU tracks which memory regions are modified                      │
│ • Sparse writes may still require full region backup                │
│ • Page-granularity tracking (typically 4KB or 64KB pages)           │
│                                                                     │
│ Pattern Impact:                                                     │
│ • Sequential writes: Efficient, contiguous backup                   │
│ • Random writes: May touch many pages, larger backup                │
│ • Read-only kernel: Minimal/no restore needed                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. SHARED MEMORY AND LOCAL MEMORY                                   │
├─────────────────────────────────────────────────────────────────────┤
│ • Shared memory must be saved/restored for replay                   │
│ • Local memory (register spills) increases state size               │
│                                                                     │
│ Check with:                                                         │
│ • Shared memory: Look for __shared__ declarations                   │
│ • Local memory: Check NCU "Local Memory" metrics                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 5. TEXTURE AND SURFACE MEMORY                                       │
├─────────────────────────────────────────────────────────────────────┤
│ • Texture memory bindings need to be preserved                      │
│ • Surface writes require backup/restore                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 6. CUDA GRAPH AND KERNEL DEPENDENCIES                               │
├─────────────────────────────────────────────────────────────────────┤
│ • Multi-kernel graphs may have complex dependencies                 │
│ • Inter-kernel data flow requires state preservation                │
└─────────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
TIPS FOR REDUCING MEMORY RESTORE OVERHEAD
════════════════════════════════════════════════════════════════════════

1. Use smaller test inputs when profiling
   - Profile with representative but minimal data sizes
   - Full-scale runs are for performance validation, not profiling

2. Use --set basic for quick profiling
   - Fewer metrics = fewer replays = less memory restore

3. Profile specific sections instead of full set
   - ncu --section SpeedOfLight ./app  (minimal restore)
   - Only collect what you need

4. Consider --replay-mode kernel vs --replay-mode application
   - kernel mode: Faster, but may miss some metrics
   - application mode: Full accuracy, more memory restore

5. Use --target-processes all vs specific
   - Limiting to specific kernels reduces overall memory impact
""")


def main():
    parser = argparse.ArgumentParser(description='Analyze NCU memory restore behavior')
    parser.add_argument('path', help='Log file or directory containing log files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--factors', '-f', action='store_true', help='Print factors analysis')
    args = parser.parse_args()
    
    if args.factors:
        print_factors_analysis()
        return
    
    if os.path.isfile(args.path):
        # 分析单个文件
        analyze_single_log(args.path, verbose=args.verbose)
    elif os.path.isdir(args.path):
        # 分析目录中所有日志
        log_files = sorted(glob.glob(os.path.join(args.path, "*.log")))
        
        if not log_files:
            print(f"No log files found in {args.path}")
            sys.exit(1)
        
        print(f"Found {len(log_files)} log files")
        
        for log_file in log_files:
            analyze_single_log(log_file, verbose=args.verbose)
        
        # 比较实验
        compare_experiments(args.path)
        
        # 打印因素分析
        print_factors_analysis()
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()

