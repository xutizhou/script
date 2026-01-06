/**
 * L2 Cache Partition Benchmark V2 - 改进版
 * 
 * 改进点：
 * 1. 使用 pointer chasing 精确测量延迟
 * 2. 控制访问次数相同，只改变地址分布
 * 3. 更准确的跨分区 vs 本地分区对比
 * 4. L2 分区哈希探测
 * 
 * H20 规格：
 * - L2 Cache: 60 MB (61440 KB)
 * - 推测有 2 个 uGPU 分区，每个 30 MB
 * - L2 cache line: 128 bytes
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

#define WARMUP_ITERS 10
#define TEST_ITERS 50
#define CHAIN_LENGTH 10000

// ============================================================
// 核心工具：Pointer Chasing
// 精确测量内存访问延迟，避免预取和乱序执行的影响
// ============================================================
__global__ void pointer_chase_latency_kernel(int* chain, int start_idx, 
                                              int chase_count, uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        
        // 预热 cache
        for (int i = 0; i < chase_count; i++) {
            idx = chain[idx];
        }
        
        idx = start_idx;
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < chase_count; i++) {
            idx = chain[idx];
        }
        
        uint64_t end = clock64();
        
        // 防止优化
        if (idx == -999999) chain[0] = idx;
        
        *result = end - start;
    }
}

// ============================================================
// 实验1: 数据足迹 vs L2 命中率
// 精确测量不同数据大小时的访问延迟
// ============================================================
void create_sequential_chain(int* h_chain, int array_elements, int chain_length) {
    // 创建顺序访问链，每次跳 32 个 int (128 bytes = 1 cache line)
    int stride = 32;  // 32 * 4 = 128 bytes
    
    for (int i = 0; i < array_elements; i++) {
        h_chain[i] = 0;
    }
    
    int idx = 0;
    for (int i = 0; i < chain_length - 1; i++) {
        int next_idx = (idx + stride) % array_elements;
        h_chain[idx] = next_idx;
        idx = next_idx;
    }
    h_chain[idx] = 0;  // 回到起点
}

void create_random_chain(int* h_chain, int array_elements, int chain_length) {
    // 创建随机访问链，覆盖整个数组
    std::vector<int> indices;
    int stride = 32;  // 每 128 bytes 一个访问点
    
    for (int i = 0; i < array_elements; i += stride) {
        indices.push_back(i);
    }
    
    // 随机打乱
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // 只取 chain_length 个
    if (indices.size() > (size_t)chain_length) {
        indices.resize(chain_length);
    }
    
    for (int i = 0; i < array_elements; i++) {
        h_chain[i] = 0;
    }
    
    // 构建链
    for (size_t i = 0; i < indices.size() - 1; i++) {
        h_chain[indices[i]] = indices[i + 1];
    }
    h_chain[indices.back()] = indices[0];  // 回到起点
}

// ============================================================
// 实验2: 地址分布测试
// 对比连续地址 vs 分散地址的延迟差异
// ============================================================
void create_strided_chain(int* h_chain, size_t array_elements, int chain_length, size_t stride_bytes) {
    size_t stride = stride_bytes / sizeof(int);
    
    // 初始化为 0
    memset(h_chain, 0, array_elements * sizeof(int));
    
    size_t idx = 0;
    for (int i = 0; i < chain_length - 1; i++) {
        size_t next_idx = (idx + stride) % array_elements;
        h_chain[idx] = (int)next_idx;  // 存储下一个索引
        idx = next_idx;
    }
    h_chain[idx] = 0;  // 回到起点
}

// ============================================================
// 实验3: L2 分区哈希探测
// 尝试发现 L2 分区的地址映射规律
// ============================================================
void create_offset_chain(int* h_chain, int base_offset, int array_elements, 
                         int chain_length, int stride_bytes) {
    int stride = stride_bytes / sizeof(int);
    
    for (int i = 0; i < array_elements; i++) {
        h_chain[i] = 0;
    }
    
    int idx = base_offset % array_elements;
    int start_idx = idx;
    
    for (int i = 0; i < chain_length - 1; i++) {
        int next_idx = (idx + stride) % array_elements;
        if (next_idx == start_idx) {
            // 避免过早回到起点
            next_idx = (next_idx + stride) % array_elements;
        }
        h_chain[idx] = next_idx;
        idx = next_idx;
    }
    h_chain[idx] = start_idx;
}

// ============================================================
// 实验4: 多线程并发访问延迟
// 测量多个 warp 同时访问不同/相同分区的延迟
// ============================================================
__global__ void multi_warp_chase_kernel(int* chain, int* start_indices, 
                                         int chase_count, uint64_t* results) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        int idx = start_indices[warp_id];
        
        // Warmup
        for (int i = 0; i < chase_count / 10; i++) {
            idx = chain[idx];
        }
        
        idx = start_indices[warp_id];
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < chase_count; i++) {
            idx = chain[idx];
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) chain[0] = idx;
        results[warp_id] = end - start;
    }
}

// ============================================================
// 实验5: 带宽测试 - 本地 vs 跨分区
// ============================================================
__global__ void bandwidth_test_kernel(float* data, int elements_per_thread,
                                       int stride, float* output, uint64_t* time_result,
                                       int buffer_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int i = 0; i < elements_per_thread; i++) {
        // 使用模运算确保不越界
        int idx = ((long long)tid * stride + (long long)i * stride) % buffer_elements;
        sum += data[idx];
    }
    
    output[tid] = sum;
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        if (blockIdx.x == 0) {
            *time_result = end_time - start_time;
        }
    }
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("L2 Cache Partition Benchmark V2 (改进版)\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d KB (%.1f MB)\n", prop.l2CacheSize / 1024, prop.l2CacheSize / 1024.0 / 1024.0);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t half_l2 = l2_size / 2;
    size_t quarter_l2 = l2_size / 4;
    
    printf("\n关键阈值 (假设 2 分区):\n");
    printf("  C (L2 总容量):    %.1f MB\n", l2_size / 1024.0 / 1024.0);
    printf("  C/2 (分区阈值):   %.1f MB\n", half_l2 / 1024.0 / 1024.0);
    printf("  C/4 (安全区域):   %.1f MB\n", quarter_l2 / 1024.0 / 1024.0);
    print_separator();
    
    uint64_t *d_result, h_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1: Pointer Chasing 延迟测量\n");
    printf("精确测量不同数据足迹的访问延迟\n");
    printf("(每次访问 1 个 cache line，共 %d 次访问)\n", CHAIN_LENGTH);
    print_separator();
    
    std::vector<size_t> test_sizes_mb = {4, 8, 12, 16, 20, 24, 28, 30, 32, 36, 40, 50, 60, 80, 100};
    
    printf("\n数据足迹      平均延迟(cycles)    相对C/2    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    double baseline_latency = 0;
    
    for (size_t size_mb : test_sizes_mb) {
        size_t size = size_mb * 1024 * 1024;
        int array_elements = size / sizeof(int);
        
        int* h_chain = (int*)malloc(size);
        int* d_chain;
        CUDA_CHECK(cudaMalloc(&d_chain, size));
        
        // 创建随机访问链
        create_random_chain(h_chain, array_elements, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_chain, h_chain, size, cudaMemcpyHostToDevice));
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Test
        uint64_t total_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_time += h_result;
        }
        
        double avg_latency = (double)total_time / TEST_ITERS / CHAIN_LENGTH;
        double relative_to_half = (double)size / half_l2;
        
        if (size_mb == 4) baseline_latency = avg_latency;
        
        const char* note = "";
        if (relative_to_half >= 0.95 && relative_to_half <= 1.05) note = "← C/2 阈值";
        else if (relative_to_half >= 1.95 && relative_to_half <= 2.05) note = "← C 阈值";
        else if (avg_latency > baseline_latency * 1.3 && avg_latency < baseline_latency * 1.5) note = "(stitch?)";
        else if (avg_latency > baseline_latency * 1.8) note = "(DRAM)";
        
        printf("%5zu MB       %8.2f cycles      %.2fx      %s\n",
               size_mb, avg_latency, relative_to_half, note);
        
        free(h_chain);
        CUDA_CHECK(cudaFree(d_chain));
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验2: 顺序 vs 随机访问对比\n");
    printf("相同数据足迹，不同访问模式\n");
    print_separator();
    
    std::vector<size_t> compare_sizes_mb = {16, 32, 48, 64};
    
    printf("\n数据足迹    顺序访问(cycles)    随机访问(cycles)    随机/顺序\n");
    printf("------------------------------------------------------------------------\n");
    
    for (size_t size_mb : compare_sizes_mb) {
        size_t size = size_mb * 1024 * 1024;
        int array_elements = size / sizeof(int);
        
        int* h_chain = (int*)malloc(size);
        int* d_chain;
        CUDA_CHECK(cudaMalloc(&d_chain, size));
        
        // 顺序访问
        create_sequential_chain(h_chain, array_elements, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_chain, h_chain, size, cudaMemcpyHostToDevice));
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t seq_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            seq_time += h_result;
        }
        double seq_avg = (double)seq_time / TEST_ITERS / CHAIN_LENGTH;
        
        // 随机访问
        create_random_chain(h_chain, array_elements, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_chain, h_chain, size, cudaMemcpyHostToDevice));
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t rand_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_chain, 0, CHAIN_LENGTH, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            rand_time += h_result;
        }
        double rand_avg = (double)rand_time / TEST_ITERS / CHAIN_LENGTH;
        
        printf("%5zu MB     %8.2f cycles      %8.2f cycles      %.2fx\n",
               size_mb, seq_avg, rand_avg, rand_avg / seq_avg);
        
        free(h_chain);
        CUDA_CHECK(cudaFree(d_chain));
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: 地址偏移测试 - 探测分区哈希\n");
    printf("相同访问模式，不同起始地址偏移\n");
    print_separator();
    
    size_t probe_size = 32 * 1024 * 1024;  // 32 MB
    int probe_elements = probe_size / sizeof(int);
    
    int* h_probe = (int*)malloc(probe_size);
    int* d_probe;
    CUDA_CHECK(cudaMalloc(&d_probe, probe_size));
    
    printf("\n偏移量 (MB)    延迟(cycles)    相对baseline\n");
    printf("------------------------------------------------\n");
    
    double offset_baseline = 0;
    std::vector<size_t> offsets_mb = {0, 1, 2, 4, 8, 16, 24, 30, 32, 48, 64};
    
    for (size_t offset_mb : offsets_mb) {
        size_t offset_bytes = offset_mb * 1024 * 1024;
        int offset_elements = offset_bytes / sizeof(int);
        
        create_offset_chain(h_probe, offset_elements, probe_elements, CHAIN_LENGTH, 128);
        CUDA_CHECK(cudaMemcpy(d_probe, h_probe, probe_size, cudaMemcpyHostToDevice));
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_probe, offset_elements % probe_elements, 
                                                     CHAIN_LENGTH, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t offset_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_probe, offset_elements % probe_elements, 
                                                     CHAIN_LENGTH, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            offset_time += h_result;
        }
        double offset_avg = (double)offset_time / TEST_ITERS / CHAIN_LENGTH;
        
        if (offset_mb == 0) offset_baseline = offset_avg;
        
        printf("%6zu MB       %8.2f cycles    %.2fx\n",
               offset_mb, offset_avg, offset_avg / offset_baseline);
    }
    
    free(h_probe);
    CUDA_CHECK(cudaFree(d_probe));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验4: 多 Warp 并发访问\n");
    printf("测量多个 warp 同时访问时的延迟\n");
    print_separator();
    
    size_t multi_size = 40 * 1024 * 1024;  // 40 MB > C/2
    int multi_elements = multi_size / sizeof(int);
    
    int* h_multi = (int*)malloc(multi_size);
    int* d_multi;
    CUDA_CHECK(cudaMalloc(&d_multi, multi_size));
    
    // 创建链
    create_random_chain(h_multi, multi_elements, CHAIN_LENGTH);
    CUDA_CHECK(cudaMemcpy(d_multi, h_multi, multi_size, cudaMemcpyHostToDevice));
    
    uint64_t* d_results;
    uint64_t h_results[256];
    int* d_starts;
    int h_starts[256];
    
    CUDA_CHECK(cudaMalloc(&d_results, 256 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_starts, 256 * sizeof(int)));
    
    printf("\nWarp数量    平均延迟(cycles)    最大延迟(cycles)    最大/平均\n");
    printf("------------------------------------------------------------------------\n");
    
    for (int num_warps : {1, 2, 4, 8, 16, 32, 64}) {
        // 设置起始位置 - 分散在不同位置
        for (int i = 0; i < num_warps; i++) {
            h_starts[i] = (i * multi_elements / num_warps) % multi_elements;
            // 确保是有效的链起点
            h_starts[i] = (h_starts[i] / 32) * 32;  // 对齐到 cache line
        }
        CUDA_CHECK(cudaMemcpy(d_starts, h_starts, num_warps * sizeof(int), cudaMemcpyHostToDevice));
        
        int threads_per_block = 128;
        int num_blocks = (num_warps * 32 + threads_per_block - 1) / threads_per_block;
        if (num_blocks == 0) num_blocks = 1;
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            multi_warp_chase_kernel<<<num_blocks, threads_per_block>>>(
                d_multi, d_starts, CHAIN_LENGTH, d_results);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double total_avg = 0, total_max = 0;
        for (int iter = 0; iter < TEST_ITERS; iter++) {
            multi_warp_chase_kernel<<<num_blocks, threads_per_block>>>(
                d_multi, d_starts, CHAIN_LENGTH, d_results);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, num_warps * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            uint64_t sum = 0, max_val = 0;
            for (int i = 0; i < num_warps; i++) {
                sum += h_results[i];
                if (h_results[i] > max_val) max_val = h_results[i];
            }
            total_avg += (double)sum / num_warps / CHAIN_LENGTH;
            total_max += (double)max_val / CHAIN_LENGTH;
        }
        
        double avg_latency = total_avg / TEST_ITERS;
        double max_latency = total_max / TEST_ITERS;
        
        printf("%4d        %8.2f cycles      %8.2f cycles      %.2fx\n",
               num_warps, avg_latency, max_latency, max_latency / avg_latency);
    }
    
    free(h_multi);
    CUDA_CHECK(cudaFree(d_multi));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_starts));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验5: Stride 大小对延迟的影响\n");
    printf("相同数据足迹 (32 MB)，不同 stride\n");
    print_separator();
    
    size_t stride_size = 32 * 1024 * 1024;
    int stride_elements = stride_size / sizeof(int);
    
    int* h_stride = (int*)malloc(stride_size);
    int* d_stride;
    CUDA_CHECK(cudaMalloc(&d_stride, stride_size));
    
    printf("\nStride (bytes)    延迟(cycles)    说明\n");
    printf("------------------------------------------------\n");
    
    // 不同 stride 测试
    std::vector<int> strides = {128, 256, 512, 1024, 2048, 4096, 8192, 
                                16384, 32768, 65536, 131072, 262144};
    
    for (int stride : strides) {
        create_strided_chain(h_stride, stride_elements, CHAIN_LENGTH, stride);
        CUDA_CHECK(cudaMemcpy(d_stride, h_stride, stride_size, cudaMemcpyHostToDevice));
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_stride, 0, CHAIN_LENGTH, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t stride_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_stride, 0, CHAIN_LENGTH, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            stride_time += h_result;
        }
        double stride_avg = (double)stride_time / TEST_ITERS / CHAIN_LENGTH;
        
        const char* note = "";
        if (stride == 128) note = "(1 cache line)";
        else if (stride == 4096) note = "(1 page)";
        else if (stride >= 32768) note = "(可能跨分区)";
        
        printf("%8d bytes     %8.2f cycles    %s\n", stride, stride_avg, note);
    }
    
    free(h_stride);
    CUDA_CHECK(cudaFree(d_stride));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验6: 跨 Partition 访问延迟测试\n");
    printf("使用不同 stride 模式触发不同 L2 slice 访问\n");
    print_separator();
    
    // 分配超大缓冲区用于测试大 stride
    size_t cross_buffer_size = 512 * 1024 * 1024;  // 512 MB - 支持更大的 stride
    
    // 使用 pointer chasing 精确测量跨分区延迟
    // 创建不同 stride 的链来访问不同 L2 slice
    printf("\nStride             Stride说明              延迟(cycles/访问)    相对baseline\n");
    printf("------------------------------------------------------------------------------------\n");
    
    struct StrideTest {
        size_t stride_bytes;
        const char* description;
    };
    
    std::vector<StrideTest> cross_tests = {
        {128, "1 cache line"},
        {4096, "1 page (4KB)"},
        {65536, "64 KB"},
        {262144, "256 KB"},
        {1048576, "1 MB"},
        {2097152, "2 MB"},
        {4194304, "4 MB"},
        {8388608, "8 MB"},
        {16777216, "16 MB"},
        {33554432, "32 MB (约C/2)"},
        {67108864, "64 MB (约C)"},
        {134217728, "128 MB (2x C)"},
        {268435456, "256 MB (4x C)"},
    };
    
    double cross_baseline = 0;
    int cross_chain_length = 2000;  // 减少 chain 长度以适应大 stride
    
    // 重新分配 int 数组用于 pointer chasing
    int* h_cross_chain = (int*)malloc(cross_buffer_size);
    int* d_cross_chain;
    CUDA_CHECK(cudaMalloc(&d_cross_chain, cross_buffer_size));
    
    size_t cross_elements = cross_buffer_size / sizeof(int);
    
    for (const auto& test : cross_tests) {
        // 创建跨分区访问的链
        create_strided_chain(h_cross_chain, cross_elements, cross_chain_length, test.stride_bytes);
        CUDA_CHECK(cudaMemcpy(d_cross_chain, h_cross_chain, cross_buffer_size, cudaMemcpyHostToDevice));
        
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_cross_chain, 0, cross_chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t total_time = 0;
        for (int iter = 0; iter < TEST_ITERS; iter++) {
            pointer_chase_latency_kernel<<<1, 32>>>(d_cross_chain, 0, cross_chain_length, d_result);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_time += h_result;
        }
        
        double avg_latency = (double)total_time / TEST_ITERS / cross_chain_length;
        
        if (test.stride_bytes == 128) cross_baseline = avg_latency;
        
        const char* indicator = "";
        if (avg_latency > 200) indicator = " (DRAM?)";
        else if (avg_latency > 50) indicator = " (L2 miss?)";
        
        // 格式化 stride 显示
        char stride_str[32];
        if (test.stride_bytes >= 1024 * 1024) {
            snprintf(stride_str, sizeof(stride_str), "%zu MB", test.stride_bytes / (1024 * 1024));
        } else if (test.stride_bytes >= 1024) {
            snprintf(stride_str, sizeof(stride_str), "%zu KB", test.stride_bytes / 1024);
        } else {
            snprintf(stride_str, sizeof(stride_str), "%zu B", test.stride_bytes);
        }
        
        printf("%10s        %-20s  %8.2f cycles        %.2fx%s\n",
               stride_str, test.description, avg_latency,
               cross_baseline > 0 ? avg_latency / cross_baseline : 1.0, indicator);
    }
    
    free(h_cross_chain);
    CUDA_CHECK(cudaFree(d_cross_chain));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("总结与结论\n");
    print_separator();
    printf("\n分区特性验证:\n");
    printf("1. C/2 阈值 (30 MB): 超过此阈值后延迟增加表明存在分区效应\n");
    printf("2. 随机访问比顺序访问延迟更高（更多 cache miss）\n");
    printf("3. 多 warp 并发访问时延迟增加表明分区带宽有限\n");
    printf("4. 大 stride 可能导致跨分区访问，延迟增加\n");
    printf("\n代码优化建议:\n");
    printf("- 保持工作集 < %.1f MB (C/2)\n", half_l2 / 1024.0 / 1024.0);
    printf("- 使用顺序/coalesced 访问模式\n");
    printf("- 避免大 stride 导致的跨分区访问\n");
    printf("- 考虑数据布局以利用 L2 分区局部性\n");
    print_separator();
    
    CUDA_CHECK(cudaFree(d_result));
    
    return 0;
}

