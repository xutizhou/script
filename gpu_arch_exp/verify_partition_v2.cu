/**
 * 验证 A/B 分区交替访问 V2
 * 
 * 核心思路：
 * L2 cache partition 通常根据地址的某些位来决定映射
 * 通过在同一个大缓冲区中使用不同的偏移，
 * 我们可以更精确地控制访问哪个 partition
 * 
 * H20 的 L2 cache = 60MB，假设有 N 个 partition
 * 每个 partition 的大小 = 60MB / N
 * 
 * 地址映射通常基于 cache line (128B) 和 partition 数量
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

#define CACHE_LINE_SIZE 128
#define WARMUP_ITERS 5
#define TEST_ITERS 20

// ============================================================
// Pointer chasing kernel - 最精确的延迟测量
// ============================================================
__global__ void single_pointer_chase(int* data, int start_idx, int chain_length, 
                                      uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        
        // Warmup
        for (int i = 0; i < chain_length; i++) {
            idx = data[idx];
        }
        
        idx = start_idx;
        uint64_t start = clock64();
        
        for (int i = 0; i < chain_length; i++) {
            idx = data[idx];
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) data[0] = idx;
        *result = end - start;
    }
}

__global__ void alternate_pointer_chase(int* data, 
                                         int start_A, int start_B,
                                         int chain_length, 
                                         uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx_A = start_A;
        int idx_B = start_B;
        
        // Warmup
        for (int i = 0; i < chain_length / 2; i++) {
            idx_A = data[idx_A];
            idx_B = data[idx_B];
        }
        
        idx_A = start_A;
        idx_B = start_B;
        uint64_t start = clock64();
        
        // 交替访问 A 和 B 区域
        for (int i = 0; i < chain_length / 2; i++) {
            idx_A = data[idx_A];  // 访问 A 区域
            idx_B = data[idx_B];  // 访问 B 区域
        }
        
        uint64_t end = clock64();
        
        if (idx_A == -999999) data[0] = idx_A;
        if (idx_B == -999999) data[0] = idx_B;
        *result = end - start;
    }
}

// 在缓冲区的指定区域创建 pointer chain
// 区域 A: [region_start_A, region_start_A + region_size)
void create_chain_in_region(int* h_data, size_t total_elements,
                            size_t region_start, size_t region_size,
                            int chain_length, int stride_elements) {
    // 在指定区域内创建链
    size_t region_end = region_start + region_size;
    
    int idx = region_start;
    for (int i = 0; i < chain_length - 1; i++) {
        int next_idx = region_start + ((idx - region_start + stride_elements) % region_size);
        h_data[idx] = next_idx;
        idx = next_idx;
    }
    h_data[idx] = region_start;  // 回到起点
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("验证 A/B 分区交替访问 V2 - 使用地址偏移\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    print_separator();
    
    // 分配一个大缓冲区，包含多个逻辑区域
    // 使用 2x L2 cache 大小确保覆盖所有 partition
    size_t buffer_size = 128 * 1024 * 1024;  // 128 MB
    size_t buffer_elements = buffer_size / sizeof(int);
    
    int* h_data = (int*)malloc(buffer_size);
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    
    uint64_t *d_result, h_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    
    printf("\n缓冲区信息:\n");
    printf("  总大小: %zu MB\n", buffer_size / (1024 * 1024));
    printf("  设备指针: %p\n", d_data);
    
    int chain_length = 5000;
    int stride_elements = 32;  // 128 bytes (cache line)
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1: 相邻区域 vs 远距离区域\n");
    print_separator();
    printf("测试不同偏移量对延迟的影响\n\n");
    
    // 测试不同的偏移量
    std::vector<size_t> offsets_mb = {0, 1, 2, 4, 8, 16, 30, 32, 60, 64};
    size_t region_size = 8 * 1024 * 1024 / sizeof(int);  // 8 MB per region
    
    printf("偏移(MB)   A区域延迟     B区域延迟     A-B交替延迟   交替开销\n");
    printf("------------------------------------------------------------------------\n");
    
    for (size_t offset_mb : offsets_mb) {
        size_t offset_bytes = offset_mb * 1024 * 1024;
        size_t offset_elements = offset_bytes / sizeof(int);
        
        if (offset_elements + region_size * 2 > buffer_elements) {
            printf("%4zu       (超出范围)\n", offset_mb);
            continue;
        }
        
        // 区域 A: 从 0 开始
        size_t region_A_start = 0;
        // 区域 B: 从 offset 开始
        size_t region_B_start = offset_elements;
        
        // 清空并创建链
        memset(h_data, 0, buffer_size);
        create_chain_in_region(h_data, buffer_elements, 
                               region_A_start, region_size, 
                               chain_length, stride_elements);
        create_chain_in_region(h_data, buffer_elements, 
                               region_B_start, region_size, 
                               chain_length, stride_elements);
        
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        
        // 测试 A 区域
        for (int w = 0; w < WARMUP_ITERS; w++) {
            single_pointer_chase<<<1, 32>>>(d_data, region_A_start, chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            single_pointer_chase<<<1, 32>>>(d_data, region_A_start, chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += h_result;
        }
        double latency_A = (double)total / TEST_ITERS / chain_length;
        
        // 测试 B 区域
        for (int w = 0; w < WARMUP_ITERS; w++) {
            single_pointer_chase<<<1, 32>>>(d_data, region_B_start, chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            single_pointer_chase<<<1, 32>>>(d_data, region_B_start, chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += h_result;
        }
        double latency_B = (double)total / TEST_ITERS / chain_length;
        
        // 测试 A-B 交替
        for (int w = 0; w < WARMUP_ITERS; w++) {
            alternate_pointer_chase<<<1, 32>>>(d_data, region_A_start, region_B_start, 
                                                chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            alternate_pointer_chase<<<1, 32>>>(d_data, region_A_start, region_B_start, 
                                                chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += h_result;
        }
        double latency_AB = (double)total / TEST_ITERS / (chain_length / 2);
        
        double expected = (latency_A + latency_B) / 2;
        double overhead = (latency_AB - expected) / expected * 100;
        
        printf("%4zu       %7.2f       %7.2f       %7.2f       %+.1f%%\n",
               offset_mb, latency_A, latency_B, latency_AB, overhead);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验2: 基于 cache line stride 的分区探测\n");
    print_separator();
    printf("使用大 stride 访问可能落在不同 partition 的 cache line\n\n");
    
    // L2 partition 通常使用地址的中间位来选择
    // 对于 60MB L2 with N partitions:
    // - 每个 partition ~= 60MB/N
    // - partition 选择位通常在 cache line 对齐后的低位
    
    // 测试不同的 cache line stride
    std::vector<int> cache_line_strides = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    
    region_size = 32 * 1024 * 1024 / sizeof(int);  // 32 MB
    chain_length = 3000;
    
    printf("CL Stride    区域延迟(cycles)    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    for (int cl_stride : cache_line_strides) {
        int stride_elements = cl_stride * CACHE_LINE_SIZE / sizeof(int);
        
        memset(h_data, 0, buffer_size);
        create_chain_in_region(h_data, buffer_elements, 
                               0, region_size, 
                               chain_length, stride_elements);
        
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++) {
            single_pointer_chase<<<1, 32>>>(d_data, 0, chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            single_pointer_chase<<<1, 32>>>(d_data, 0, chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += h_result;
        }
        double latency = (double)total / TEST_ITERS / chain_length;
        
        size_t data_footprint = (size_t)cl_stride * CACHE_LINE_SIZE * chain_length;
        const char* note = "";
        if (data_footprint > (size_t)prop.l2CacheSize) {
            note = "→ 超出 L2, DRAM 访问";
        } else if (cl_stride >= 8) {
            note = "→ 可能跨多个 partition";
        }
        
        printf("%4d CL      %7.2f cycles       %s\n", 
               cl_stride, latency, note);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: L2 partition hash 探测\n");
    print_separator();
    printf("通过访问特定地址模式来推断 partition mapping\n\n");
    
    // 假设 L2 有 8 个 partition (60MB / 8 = 7.5MB each)
    // 或者 16 个 partition (60MB / 16 = 3.75MB each)
    
    std::vector<int> num_partitions_guess = {2, 4, 8, 16, 32};
    
    printf("假设分区数   测试 stride(bytes)   延迟(cycles)   推测\n");
    printf("------------------------------------------------------------------------\n");
    
    size_t l2_size = prop.l2CacheSize;
    chain_length = 2000;
    
    for (int n_part : num_partitions_guess) {
        // 每个 partition 的大小
        size_t partition_size = l2_size / n_part;
        
        // 使用 partition 大小作为 stride
        // 这样每次访问都应该落在不同的 partition
        size_t stride_bytes = partition_size;
        size_t stride_elements = stride_bytes / sizeof(int);
        
        // 限制区域大小
        size_t test_region_size = std::min(buffer_elements, 
                                           (size_t)chain_length * stride_elements);
        if (test_region_size > buffer_elements) {
            printf("%6d       %10zu bytes   (数据不足)\n", n_part, stride_bytes);
            continue;
        }
        
        memset(h_data, 0, buffer_size);
        
        // 创建跨 partition 的链
        int idx = 0;
        for (int i = 0; i < chain_length - 1; i++) {
            size_t next_idx = (idx + stride_elements) % buffer_elements;
            h_data[idx] = next_idx;
            idx = next_idx;
        }
        h_data[idx] = 0;
        
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++) {
            single_pointer_chase<<<1, 32>>>(d_data, 0, chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            single_pointer_chase<<<1, 32>>>(d_data, 0, chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += h_result;
        }
        double latency = (double)total / TEST_ITERS / chain_length;
        
        const char* note = "";
        if (latency > 350) {
            note = "→ 高延迟, 可能是真正的跨分区";
        } else if (latency > 300) {
            note = "→ 中等延迟";
        } else {
            note = "→ 低延迟, 可能都在同一分区";
        }
        
        printf("%6d       %10zu bytes   %7.2f        %s\n", 
               n_part, stride_bytes, latency, note);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("结论\n");
    print_separator();
    printf("\n确认跨分区访问的条件:\n");
    printf("1. 如果实验1中大偏移量(≥L2/2)的交替访问延迟明显更高 → 跨分区\n");
    printf("2. 如果实验2中大 stride 的延迟明显更高 → 跨分区\n");
    printf("3. 如果实验3中某个假设的分区数导致高延迟 → 可能是真实分区数\n");
    print_separator();
    
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    
    return 0;
}

