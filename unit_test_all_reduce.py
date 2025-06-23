# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist
import time
import argparse
import os

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Unit test for torch.distributed.all_reduce with a specific group")
    parser.add_argument("--size", type=str, default="64M", help="Tensor size for the test (e.g., 1K, 64M)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="测试使用的数据类型")
    parser.add_argument("--warmup", type=int, default=10, help="预热迭代次数")
    parser.add_argument("--trials", type=int, default=50, help="正式测试迭代次数")
    # 使用 parse_known_args 来处理启动器传入的 --local-rank
    args, _ = parser.parse_known_args()
    return args

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
    args = parse_args()

    # 1. 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 2. 创建一个进程组来模拟 `self.device_group`
    #    这里我们创建一个包含所有可用 GPU 的组
    device_group = dist.new_group(ranks=list(range(world_size)))

    # 3. 准备测试张量
    size_bytes = parse_size(args.size)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    itemsize = torch.finfo(dtype).bits // 8
    
    if itemsize == 0: # 避免除以0
        itemsize = 1
        
    num_elements = size_bytes // itemsize
    if num_elements == 0:
        if rank == 0:
            print(f"错误: 尺寸 '{args.size}' 对于数据类型 '{args.dtype}' 来说太小。")
        return

    input_tensor = torch.randn(num_elements, dtype=dtype, device=f"cuda:{local_rank}")

    # 4. 预热
    for _ in range(args.warmup):
        # 在指定的 group 上执行 all_reduce
        dist.all_reduce(input_tensor, group=device_group)
    torch.cuda.synchronize()

    # 5. 精确计时
    start_time = time.time()
    for _ in range(args.trials):
        # 这是我们要单测的代码行
        dist.all_reduce(input_tensor, group=device_group)
    torch.cuda.synchronize()
    end_time = time.time()

    # 6. 报告结果
    if rank == 0:
        avg_time_s = ((end_time - start_time) / args.trials)
        avg_time_ms = avg_time_s * 1000

        # 带宽计算公式参考 nccl-tests: 2 * (N-1)/N * size / time
        bus_bandwidth = (2 * (world_size - 1) / world_size) * size_bytes / avg_time_s
        bus_bandwidth_gbps = bus_bandwidth / 1e9

        print("=" * 60)
        print("Unit Test: torch.distributed.all_reduce(input_, group=...)")
        print(f"  World Size: {world_size} GPUs")
        print(f"  Group Size: {dist.get_world_size(group=device_group)} GPUs")
        print(f"  Tensor Size: {args.size}")
        print(f"  DType: {args.dtype}")
        print("-" * 60)
        print(f"  平均执行时间: {avg_time_ms:.4f} ms")
        print(f"  等效总线带宽: {bus_bandwidth_gbps:.2f} GB/s")
        print("=" * 60)

if __name__ == "__main__":
    main() 