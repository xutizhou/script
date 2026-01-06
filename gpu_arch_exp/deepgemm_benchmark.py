#!/usr/bin/env python3
"""
DeepGEMM Benchmark for NCU Warp Stall Analysis

相比cuBLAS GEMM，DeepGEMM可以看到源码，更方便分析warp stall

Usage:
    python deepgemm_benchmark.py

NCU Profiling:
    ncu --set full -o deepgemm_report python deepgemm_benchmark.py
"""

import torch
import deep_gemm
import time

def warmup_gpu():
    """Warmup GPU"""
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    for _ in range(10):
        torch.mm(x, x)
    torch.cuda.synchronize()

def run_bf16_gemm(m, n, k, repeat=3):
    """Run BF16 GEMM using DeepGEMM
    
    DeepGEMM使用NT layout:
    - A: M×K (row-major)  
    - B: N×K (row-major, transposed)
    - D = A @ B.T
    """
    print(f"=== DeepGEMM BF16 GEMM: M={m}, N={n}, K={k} ===")
    
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16).contiguous()
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16).contiguous()
    d = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(3):
        deep_gemm.bf16_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for i in range(repeat):
        deep_gemm.bf16_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate TFLOPS
    flops = 2 * m * n * k * repeat
    tflops = flops / elapsed / 1e12
    
    print(f"  Time: {elapsed*1000/repeat:.2f} ms/iter, TFLOPS: {tflops:.1f}")
    return d

def run_cublas_gemm(m, n, k, repeat=3):
    """Run cuBLAS GEMM for comparison (无法看到源码)"""
    print(f"=== cuBLAS GEMM: M={m}, N={n}, K={k} ===")
    
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16).contiguous()
    b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16).contiguous()
    
    # Warmup
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    # Benchmark  
    start = time.time()
    for i in range(repeat):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    flops = 2 * m * n * k * repeat
    tflops = flops / elapsed / 1e12
    
    print(f"  Time: {elapsed*1000/repeat:.2f} ms/iter, TFLOPS: {tflops:.1f}")
    return c

def main():
    print("DeepGEMM vs cuBLAS Warp Stall Benchmark")
    print("=" * 60)
    print("DeepGEMM: 可以看到源码，方便分析warp stall")
    print("cuBLAS:   闭源，只能看到kernel名称")
    print("=" * 60)
    
    device = torch.cuda.get_device_properties(0)
    print(f"Device: {device.name}")
    print(f"SM: {device.major}.{device.minor}")
    print()
    
    warmup_gpu()
    
    # GEMM sizes
    sizes = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    repeat = 5
    
    print("\n" + "=" * 60)
    print("DeepGEMM BF16 GEMM (有源码)")
    print("=" * 60)
    for m, n, k in sizes:
        run_bf16_gemm(m, n, k, repeat=repeat)
    
    print("\n" + "=" * 60)
    print("cuBLAS GEMM (无源码)")
    print("=" * 60)
    for m, n, k in sizes:
        run_cublas_gemm(m, n, k, repeat=repeat)
    
    print("\n" + "=" * 60)
    print("Completed! Use NCU to profile:")
    print("  ncu --set full -o deepgemm_report python deepgemm_benchmark.py")

if __name__ == "__main__":
    main()
