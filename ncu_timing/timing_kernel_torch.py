#!/usr/bin/env python3
"""
使用 torch.profiler 测量 CUDA kernel 时间

通过 pybind11 调用与 timing_kernel.cu 完全相同的 CUDA kernel，
使 torch profiler 可以测量真实的 kernel 执行时间。

配置与 timing_kernel.cu 完全一致：
- short: N=4096, iterations=1, blocks=1, threads=128
- long: N=16M, iterations=100, blocks=128, threads=128

编译模块:
    cd /path/to/ncu_timing
    pip install -e .

用法:
    python timing_kernel_torch.py short
    python timing_kernel_torch.py long
"""

import torch
import torch.cuda
import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Optional

# 尝试导入 pybind 模块
try:
    import timing_kernel_pybind
    HAS_PYBIND = True
except ImportError:
    HAS_PYBIND = False
    print("WARNING: timing_kernel_pybind not found. Please compile it first:")
    print("  cd /path/to/ncu_timing && pip install -e .")


def run_kernel(src: torch.Tensor, dst: torch.Tensor, 
               num_blocks: int, threads_per_block: int, iterations: int):
    """运行 CUDA kernel"""
    if not HAS_PYBIND:
        raise RuntimeError("timing_kernel_pybind not available")
    timing_kernel_pybind.run_timed_rw_kernel(src, dst, num_blocks, threads_per_block, iterations)


def run_kernel_with_profiler(num_elements: int, iterations: int, 
                              num_blocks: int, threads_per_block: int) -> Optional[float]:
    """
    使用 torch.profiler 运行 kernel 并获取时间
    
    Returns:
        float: kernel 时间 (微秒)
    """
    if not HAS_PYBIND:
        return None
    
    # 分配内存
    src = torch.randn(num_elements, dtype=torch.float32, device='cuda')
    dst = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
    
    # 预热
    for _ in range(3):
        run_kernel(src, dst, num_blocks, threads_per_block, iterations)
    torch.cuda.synchronize()
    
    # 使用 profiler 计时
    trace_file = tempfile.mktemp(suffix='.json', prefix='ncu_timing_trace_')
    profiler_time_us = None
    
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            with torch.profiler.record_function("timed_rw_kernel"):
                run_kernel(src, dst, num_blocks, threads_per_block, iterations)
            torch.cuda.synchronize()
        
        # 导出 trace
        profiler.export_chrome_trace(trace_file)
        
        # 解析 trace 获取 kernel 时间
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        # 查找 timed_rw_kernel 的 GPU 时间
        for event in trace_data.get('traceEvents', []):
            name = event.get('name', '')
            cat = event.get('cat', '')
            
            # 查找 CUDA kernel 事件
            if cat == 'kernel' and 'timed_rw_kernel' in name:
                dur = event.get('dur', 0)
                if dur > 0:
                    profiler_time_us = dur
                    break
                    
    finally:
        if os.path.exists(trace_file):
            os.remove(trace_file)
    
    return profiler_time_us


def run_multiple_measurements(num_elements: int, iterations: int,
                               num_blocks: int, threads_per_block: int,
                               num_runs: int = 10) -> Optional[Dict]:
    """运行多次测量并返回统计结果"""
    import numpy as np
    
    profiler_times = []
    
    for i in range(num_runs):
        profiler_time = run_kernel_with_profiler(num_elements, iterations, 
                                                  num_blocks, threads_per_block)
        if profiler_time is not None:
            profiler_times.append(profiler_time)
    
    if not profiler_times:
        return None
        
    profiler_times = np.array(profiler_times)
    
    return {
        'mean_us': float(profiler_times.mean()),
        'std_us': float(profiler_times.std()),
        'min_us': float(profiler_times.min()),
        'max_us': float(profiler_times.max()),
        'num_runs': len(profiler_times),
    }


def main():
    parser = argparse.ArgumentParser(description='Measure CUDA kernel time using torch.profiler')
    parser.add_argument('mode', choices=['short', 'long', 'custom'],
                        help='Kernel mode: short (~10us), long (~10ms), or custom')
    parser.add_argument('--size', type=int, default=1024*1024,
                        help='Number of elements (for custom mode)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations (for custom mode)')
    parser.add_argument('--blocks', type=int, default=128,
                        help='Number of blocks (for custom mode)')
    parser.add_argument('--threads', type=int, default=128,
                        help='Threads per block (for custom mode)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of measurement runs')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--ncu-mode', action='store_true',
                        help='Run once without profiler (for NCU profiling)')
    args = parser.parse_args()
    
    # 与 timing_kernel.cu 完全相同的配置
    if args.mode == 'short':
        num_elements = 4096       # 16KB
        iterations = 1
        num_blocks = 1
        threads_per_block = 128
        print("=== Short Kernel Mode (N=4096, iter=1, blocks=1, threads=128) ===")
    elif args.mode == 'long':
        num_elements = 16 * 1024 * 1024  # 64MB
        iterations = 100
        num_blocks = 128
        threads_per_block = 128
        print("=== Long Kernel Mode (N=16M, iter=100, blocks=128, threads=128) ===")
    else:
        num_elements = args.size
        iterations = args.iterations
        num_blocks = args.blocks
        threads_per_block = args.threads
        print("=== Custom Kernel Mode ===")
    
    data_size = num_elements * 4  # float32
    print(f"Configuration:")
    print(f"  Elements:    {num_elements} ({data_size / 1024 / 1024:.2f} MB)")
    print(f"  Iterations:  {iterations}")
    print(f"  Blocks:      {num_blocks}")
    print(f"  Threads:     {threads_per_block}")
    print()
    
    if not HAS_PYBIND:
        print("ERROR: timing_kernel_pybind module not available!")
        print("Please compile it first:")
        print("  cd /path/to/ncu_timing && pip install -e .")
        sys.exit(1)
    
    # NCU 模式：只运行一次 kernel，不做 profiling
    if args.ncu_mode:
        print("Running kernel once (NCU mode)...")
        src = torch.randn(num_elements, dtype=torch.float32, device='cuda')
        dst = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
        run_kernel(src, dst, num_blocks, threads_per_block, iterations)
        torch.cuda.synchronize()
        print("Done.")
        return
    
    # 正常模式：使用 profiler 测量
    print(f"Running {args.runs} measurements with torch.profiler...")
    results = run_multiple_measurements(num_elements, iterations, 
                                         num_blocks, threads_per_block, args.runs)
    
    if results is None:
        print("ERROR: Failed to get profiler measurements")
        return
    
    # 打印结果
    print()
    print("=" * 60)
    print("Torch Profiler Results (GPU Kernel Duration)")
    print("=" * 60)
    print(f"  Mean:  {results['mean_us']:.3f} us ({results['mean_us']/1000:.3f} ms)")
    print(f"  Std:   {results['std_us']:.3f} us")
    print(f"  Min:   {results['min_us']:.3f} us")
    print(f"  Max:   {results['max_us']:.3f} us")
    print(f"  Runs:  {results['num_runs']}")
    print("=" * 60)
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
