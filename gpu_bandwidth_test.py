# -*- coding: utf-8 -*-
import torch
import torch.utils.benchmark as benchmark

def benchmark_write(size_gb=1.0, device='cuda'):
    """
    场景1: 使用 fill_ 操作测试纯写带宽 (HBM Write)。
    对应文档中一个方向的带宽测试。
    """
    print(f"--- 场景1: 测试纯写带宽 (fill_), 写入数据量: {size_gb:.2f} GB ---")
    num_elements = int(size_gb * (1024**3) / 4)
    data = torch.empty(num_elements, dtype=torch.float32, device=device)
    
    timer = benchmark.Timer(
        stmt='data.fill_(1.0)',
        globals={'data': data},
        label='HBM Bandwidth',
        description='纯写 (Fill)'
    )
    
    m = timer.timeit(100)
    
    # 写入 size_gb
    bandwidth_gb_s = size_gb / m.mean
    
    print(m)
    print(f"有效写入带宽 (Effective Write Bandwidth): {bandwidth_gb_s:.2f} GB/s\n")
    return bandwidth_gb_s

def benchmark_copy(size_gb=1.0, device='cuda'):
    """
    场景2: 使用 copy_ 操作测试合并读写带宽 (HBM Read + Write)。
    这是最基础的带宽测试，访问模式是完全合并的。
    """
    print(f"--- 场景2: 测试合并读写带宽 (copy_), 总数据量: {2 * size_gb:.2f} GB ---")
    num_elements = int(size_gb * (1024**3) / 4)
    input_data = torch.randn(num_elements, dtype=torch.float32, device=device)
    output_data = torch.empty_like(input_data)
    
    timer = benchmark.Timer(
        stmt='output_data.copy_(input_data)',
        globals={'input_data': input_data, 'output_data': output_data},
        label='HBM Bandwidth',
        description='复制 (Copy)'
    )
    
    m = timer.timeit(100)
    
    # 读取 size_gb 并写入 size_gb
    total_data_gb = 2 * size_gb
    bandwidth_gb_s = total_data_gb / m.mean
    
    print(m)
    print(f"有效读写带宽 (Effective R+W Bandwidth): {bandwidth_gb_s:.2f} GB/s\n")
    return bandwidth_gb_s

def benchmark_triad(size_gb=1.0, device='cuda'):
    """
    场景3: 使用三元操作 (C = A + B) 来测试高负载下的综合带宽。
    这种 2次读 + 1次写 的模式通常被用来压榨内存带宽的极限。
    """
    print(f"--- 场景3: 测试三元操作带宽 (Triad), 总数据量: {3 * size_gb:.2f} GB ---")
    num_elements = int(size_gb * (1024**3) / 4)
    a = torch.randn(num_elements, dtype=torch.float32, device=device)
    b = torch.randn(num_elements, dtype=torch.float32, device=device)
    c = torch.empty_like(a)

    timer = benchmark.Timer(
        stmt='torch.add(a, b, out=c)',
        globals={'a': a, 'b': b, 'c': c},
        label='HBM Bandwidth',
        description='三元操作 (Triad: C=A+B)'
    )

    m = timer.timeit(100)
    
    # 读取 2 * size_gb, 写入 1 * size_gb
    total_data_gb = 3 * size_gb
    bandwidth_gb_s = total_data_gb / m.mean

    print(m)
    print(f"有效三元操作带宽 (Effective Triad Bandwidth): {bandwidth_gb_s:.2f} GB/s\n")
    return bandwidth_gb_s

def benchmark_read(size_gb=1.0, device='cuda'):
    """
    场景4: 使用 sum 归约操作来测试纯读带宽。
    读取大量数据，只写入一个标量，以此来近似纯读性能。
    """
    print(f"--- 场景4: 测试纯读带宽 (Sum Reduction), 读取数据量: {size_gb:.2f} GB ---")
    num_elements = int(size_gb * (1024**3) / 4)
    input_data = torch.randn(num_elements, dtype=torch.float32, device=device)
    # 输出只是一个标量
    output_data = torch.empty(1, dtype=torch.float32, device=device)

    # 使用 dim=0 和 keepdim=True 来确保写入到预分配的 out 张量
    timer = benchmark.Timer(
        stmt='torch.sum(input_data, dim=0, keepdim=True, out=output_data)',
        globals={'input_data': input_data, 'output_data': output_data},
        label='HBM Bandwidth',
        description='纯读 (Sum Reduction)'
    )

    m = timer.timeit(100)
    
    # 写入的数据量 (4字节) 相对于读取量可以忽略不计
    read_data_gb = size_gb
    bandwidth_gb_s = read_data_gb / m.mean

    print(m)
    print(f"有效读取带宽 (Effective Read Bandwidth): {bandwidth_gb_s:.2f} GB/s\n")
    return bandwidth_gb_s

def benchmark_stride_copy(size_gb=1.0, stride=2, device='cuda'):
    """
    场景5: 使用带步长的 copy 专门测试非合并访问 (Uncoalesced Access)。
    根据 CUDA Best Practices Guide, 这种访问模式会极大地降低带宽。
    """
    print(f"--- 场景5: 测试非合并读写带宽 (Stride Copy, stride={stride}), 总数据量: {2 * size_gb / stride:.2f} GB ---")
    num_elements = int(size_gb * (1024**3) / 4)
    
    # 创建足够大的张量以容纳跨步访问
    input_data = torch.randn(num_elements, dtype=torch.float32, device=device)
    output_data = torch.empty_like(input_data)

    # 创建索引来实现跨步复制
    indices = torch.arange(0, num_elements // stride, device=device) * stride
    
    # 定义一个简单的kernel-like函数来进行跨步复制
    def stride_copy_func():
        output_data.index_copy_(0, indices, input_data.index_select(0, indices))

    timer = benchmark.Timer(
        stmt='stride_copy_func()',
        globals=locals(),
        label='HBM Bandwidth',
        description=f'跨步复制 (Stride Copy, stride={stride})'
    )

    m = timer.timeit(100)
    
    # 每次操作读写 (size_gb / stride) 的数据
    total_data_gb = 2 * (size_gb / stride)
    bandwidth_gb_s = total_data_gb / m.mean

    print(m)
    print(f"有效跨步带宽 (Effective Strided Bandwidth, stride={stride}): {bandwidth_gb_s:.2f} GB/s\n")
    return bandwidth_gb_s


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用。该脚本需要 GPU 环境。")
        return
    
    device = 'cuda'
    # 使用足够大的数据量来确保我们衡量的是HBM带宽，而不是缓存
    # 同时避免太大导致OOM
    data_size_gb = 4.0
    
    print("="*60)
    print(f"开始 HBM 带宽测试 (基于 CUDA Best Practices Guide 原则)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"测试使用的数据块大小: {data_size_gb} GB")
    print("="*60 + "\n")
    
    # 测试1: 纯写
    write_bw = benchmark_write(data_size_gb, device=device)
    
    # 测试2: 合并复制
    copy_bw = benchmark_copy(data_size_gb, device=device)
    
    # 测试3: 三元操作
    triad_bw = benchmark_triad(data_size_gb, device=device)

    # 测试4: 纯读
    read_bw = benchmark_read(data_size_gb, device=device)

    # 测试5: 跨步复制
    stride_bw_2 = benchmark_stride_copy(data_size_gb, stride=2, device=device)
    stride_bw_32 = benchmark_stride_copy(data_size_gb, stride=32, device=device)
    
    print("--- 结论 (基于 CUDA Best Practices Guide) ---")
    print("有效带宽计算公式: (读取字节数 + 写入字节数) / 时间\n")
    print(f"场景1 - 纯写 (Fill) 带宽: {write_bw:.2f} GB/s")
    print(f"  -衡量了单向写入HBM的性能。\n")
    print(f"场景2 - 合并复制 (Copy) 带宽: {copy_bw:.2f} GB/s")
    print(f"  -这是理想情况下的均衡读写性能，内存访问是完全合并的。\n")
    print(f"场景3 - 三元操作 (Triad) 带宽: {triad_bw:.2f} GB/s")
    print(f"  -由于其高的内存/计算流量比(2读1写)，通常被认为是衡量可用峰值HBM带宽的有效方法。\n")
    print(f"场景4 - 纯读 (Sum) 带宽: {read_bw:.2f} GB/s")
    print(f"  -通过归约操作近似纯读性能。其带宽 ({read_bw:.2f} GB/s) 与纯写带宽 ({write_bw:.2f} GB/s) 共同展示了单向数据传输的能力。\n")
    print(f"场景5 - 跨步复制 (Stride=2) 带宽: {stride_bw_2:.2f} GB/s")
    print(f"场景5 - 跨步复制 (Stride=32) 带宽: {stride_bw_32:.2f} GB/s")
    print(f"  -通过非合并访问，带宽显著下降 (对比场景2: {copy_bw:.2f} GB/s)。")
    print(f"  -这清晰地展示了遵循'合并访问'原则对性能至关重要。\n")
    print("总结: '三元操作' 的结果最接近硬件的理论峰值带宽。而'跨步复制'的低性能反向证明了内存访问模式的重要性。")

if __name__ == "__main__":
    main() 