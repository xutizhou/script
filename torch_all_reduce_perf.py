# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
import time
import argparse
import os

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PyTorch All-Reduce Bandwidth Test")
    parser.add_argument("--start_size", type=str, default="8B", help="起始数据大小 (e.g., 8B, 1K, 128M)")
    parser.add_argument("--end_size", type=str, default="128M", help="结束数据大小 (e.g., 64K, 256M, 1G)")
    parser.add_argument("--step_factor", type=int, default=2, help="每次测试数据大小的递增倍数")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="测试使用的数据类型")
    parser.add_argument("--warmup", type=int, default=20, help="每个尺寸的预热迭代次数")
    parser.add_argument("--trials", type=int, default=100, help="每个尺寸的正式测试迭代次数")
    parser.add_argument("--profile_size", type=str, default=None, help="对指定尺寸进行性能分析，并生成跟踪文件 (e.g., '64M')。此选项将禁用尺寸扫描。")
    parser.add_argument("--profile_out", type=str, default="all_reduce_trace.json", help="性能分析跟踪文件的输出路径。")
    return parser.parse_known_args()

def parse_size(size_str: str) -> int:
    """将带有单位的尺寸字符串转换为字节数"""
    size_str = size_str.upper()
    if size_str.endswith('K'):
        return int(size_str[:-1]) * 1024
    if size_str.endswith('M'):
        return int(size_str[:-1]) * 1024**2
    if size_str.endswith('G'):
        return int(size_str[:-1]) * 1024**3
    if size_str.endswith('B'):
        return int(size_str[:-1])
    return int(size_str)

def main():
    args, _ = parse_args()

    # --- 1. 初始化分布式环境 ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # --- 2. 解析参数 ---
    start_bytes = parse_size(args.start_size)
    end_bytes = parse_size(args.end_size)
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    dtype_size = torch.finfo(dtype).bits // 8

    # --- 模式选择: 分析或基准测试 ---
    if args.profile_size:
        # ########## 分析模式 ##########
        profile_size_bytes = parse_size(args.profile_size)
        num_elements = profile_size_bytes // dtype_size
        if num_elements == 0:
            if rank == 0:
                print(f"错误: 分析尺寸 {args.profile_size} 对于数据类型 {args.dtype} 来说太小。")
            return

        tensor = torch.randn(num_elements, dtype=dtype, device=f"cuda:{local_rank}")

        if rank == 0:
            print("=" * 70)
            print(f"分析模式: PyTorch All-Reduce")
            print(f"World Size: {world_size} GPUs | Data Type: {args.dtype} | Size: {args.profile_size}")
            print(f"跟踪文件将被保存到: {args.profile_out}")
            print("=" * 70)

        # 预热
        for _ in range(args.warmup):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # 使用 profiler 包裹测试循环
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(args.trials):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            print("分析完成。正在保存跟踪文件...")
            prof.export_chrome_trace(args.profile_out)
            print(f"跟踪文件已保存到 {args.profile_out}")
            print("您可以在 chrome://tracing 或 https://ui.perfetto.dev/ 中打开此文件进行分析。")

    else:
        # ########## 基准测试模式 (原始逻辑) ##########
        if rank == 0:
            print("=" * 70)
            print(f"PyTorch All-Reduce Bandwidth Test")
            print(f"World Size: {world_size} GPUs")
            print(f"Data Type: {args.dtype}")
            print(f"Size Range: {args.start_size} to {args.end_size}")
            print(f"{'Size':>10s}{'Time (us)':>15s}{'Bus Bandwidth (GB/s)':>25s}")
            print("-" * 70)

        # --- 3. 循环测试不同大小的数据 ---
        size = start_bytes
        while size <= end_bytes:
            if size == 0:
                size = 1 # 避免除以0
            
            num_elements = size // dtype_size
            if num_elements == 0:
                size *= args.step_factor
                continue
            
            tensor = torch.randn(num_elements, dtype=dtype, device=f"cuda:{local_rank}")
            
            # 预热
            for _ in range(args.warmup):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            # 正式测试
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(args.trials):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / args.trials
            
            # --- 4. 计算并打印带宽 ---
            # 带宽计算公式参考 nccl-tests: 2 * (N-1)/N * size / time
            # 这反映了每个GPU在ring-allreduce中需要发送和接收的数据量
            bus_bandwidth = (2 * (world_size - 1) / world_size) * size / avg_time
            bus_bandwidth_gbps = bus_bandwidth / 1e9

            if rank == 0:
                size_str = f"{size // 1024**2}M" if size >= 1024**2 else f"{size // 1024}K" if size >= 1024 else f"{size}B"
                print(f"{size_str:>10s}{avg_time * 1e6:>15.2f}{bus_bandwidth_gbps:>25.2f}")

            if size == end_bytes:
                break
            size = int(size * args.step_factor)
            if size > end_bytes:
                size = end_bytes


if __name__ == "__main__":
    main() 