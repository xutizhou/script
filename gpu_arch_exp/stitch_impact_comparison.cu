/**
 * Stitch Traffic 影响对比实验
 * 
 * 目标：通过对比实验清晰展示 Stitch 对 GPU 性能的影响
 * 
 * 实验设计原则：
 * 1. 控制变量：每组实验只改变一个因素
 * 2. 实际场景：模拟真实的内存访问模式
 * 3. 量化影响：用时间和带宽量化 Stitch 开销
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// 对比实验1: 本地访问 vs 跨 Partition 访问
// ============================================================================
// 场景：相同数据量，不同的地址分布

// 本地访问：所有线程访问连续的内存区域（映射到同一 L2 partition）
__global__ void local_access_kernel(float* __restrict__ data, 
                                     float* __restrict__ output,
                                     size_t elements,
                                     int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    // 每个线程访问一小块连续内存
    size_t chunk_size = elements / total_threads;
    size_t start = tid * chunk_size;
    size_t end = min(start + chunk_size, elements);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = start; i < end; i++) {
            sum += __ldcg(&data[i]);  // 绕过 L1，直接访问 L2
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 跨 Partition 访问：每个线程访问分散的内存位置
__global__ void cross_partition_kernel(float* __restrict__ data,
                                        float* __restrict__ output,
                                        size_t elements,
                                        int iterations,
                                        size_t stride) {  // stride 控制跨 partition 程度
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    size_t accesses_per_thread = elements / total_threads;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < accesses_per_thread; i++) {
            // 使用大步长访问，跨越多个 L2 partition
            size_t idx = (tid + i * stride) % elements;
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// ============================================================================
// 对比实验2: 单 Buffer vs 多 Buffer（模拟 Attention QKV）
// ============================================================================

// 单 Buffer 访问
__global__ void single_buffer_kernel(float* __restrict__ data,
                                      float* __restrict__ output,
                                      size_t elements,
                                      int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (tid * 32 + i) % elements;
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 多 Buffer 交替访问（模拟 Q, K, V 访问模式）
__global__ void multi_buffer_kernel(float* __restrict__ Q,
                                     float* __restrict__ K,
                                     float* __restrict__ V,
                                     float* __restrict__ output,
                                     size_t elements,
                                     int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (tid * 32 + i) % elements;
            // 交替访问 Q, K, V - 模拟 Attention 计算
            sum += __ldcg(&Q[idx]);
            sum += __ldcg(&K[idx]);
            sum += __ldcg(&V[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 单 Buffer 但3倍访问量（公平对比）
__global__ void single_buffer_3x_kernel(float* __restrict__ data,
                                         float* __restrict__ output,
                                         size_t elements,
                                         int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (tid * 32 + i) % elements;
            // 同一 buffer 访问3次
            sum += __ldcg(&data[idx]);
            sum += __ldcg(&data[idx]);
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// ============================================================================
// 对比实验3: 数据局部性优化前后对比
// ============================================================================

// 未优化：随机访问模式（常见于稀疏计算）
__global__ void unoptimized_access_kernel(float* __restrict__ data,
                                           int* __restrict__ indices,
                                           float* __restrict__ output,
                                           size_t num_accesses,
                                           int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    size_t accesses_per_thread = num_accesses / total_threads;
    size_t start = tid * accesses_per_thread;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < accesses_per_thread; i++) {
            int idx = indices[start + i];
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 优化后：排序索引，提高局部性
// （使用相同的索引但排序后）

// ============================================================================
// 对比实验4: 工作集大小对 Stitch 的影响
// ============================================================================

__global__ void working_set_kernel(float* __restrict__ data,
                                    float* __restrict__ output,
                                    size_t working_set_size,
                                    int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    // 限制访问范围在 working_set_size 内
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 64; i++) {
            size_t idx = ((size_t)tid * 64 + i) % working_set_size;
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// ============================================================================
// 对比实验5: 带宽压力测试
// ============================================================================

// 低压力：少量 blocks
// 高压力：大量 blocks 同时访问

__global__ void bandwidth_stress_kernel(float* __restrict__ data,
                                         float* __restrict__ output,
                                         size_t elements,
                                         int accesses_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    for (int i = 0; i < accesses_per_thread; i++) {
        // 大步长访问，确保跨 partition
        size_t idx = ((size_t)tid * 128 + i * total_threads * 128) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) output[0] = sum;
}

// ============================================================================
// 辅助函数
// ============================================================================

void print_separator(const char* title) {
    printf("\n");
    printf("================================================================\n");
    printf("%s\n", title);
    printf("================================================================\n");
}

void print_comparison(const char* test1, float time1, 
                     const char* test2, float time2) {
    float diff = (time2 - time1) / time1 * 100.0f;
    printf("\n┌────────────────────────────────────────────────────────────┐\n");
    printf("│ %-25s: %8.3f ms                     │\n", test1, time1);
    printf("│ %-25s: %8.3f ms                     │\n", test2, time2);
    printf("│ ─────────────────────────────────────────────────────────  │\n");
    if (diff > 0) {
        printf("│ Stitch 导致的性能损失: %+.1f%% (%.3f ms)                   │\n", 
               diff, time2 - time1);
    } else {
        printf("│ 差异: %+.1f%%                                              │\n", diff);
    }
    printf("└────────────────────────────────────────────────────────────┘\n");
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    size_t l2_size = prop.l2CacheSize;
    size_t l2_half = l2_size / 2;
    
    printf("================================================================\n");
    printf("       Stitch Traffic 影响对比实验\n");
    printf("================================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", l2_size / 1024.0 / 1024.0);
    printf("L2/2 (Stitch 阈值): %.1f MB\n", l2_half / 1024.0 / 1024.0);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    
    // 分配内存
    size_t buffer_size = 64 * 1024 * 1024;  // 64 MB
    size_t elements = buffer_size / sizeof(float);
    
    float *d_data, *d_Q, *d_K, *d_V, *d_output;
    int *d_indices;
    
    CHECK_CUDA(cudaMalloc(&d_data, buffer_size));
    CHECK_CUDA(cudaMalloc(&d_Q, buffer_size));
    CHECK_CUDA(cudaMalloc(&d_K, buffer_size));
    CHECK_CUDA(cudaMalloc(&d_V, buffer_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, elements * sizeof(int)));
    
    // 初始化数据
    std::vector<float> h_data(elements);
    std::vector<int> h_indices(elements);
    for (size_t i = 0; i < elements; i++) {
        h_data[i] = 1.0f;
        h_indices[i] = i;
    }
    
    // 随机打乱索引
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> h_random_indices = h_indices;
    std::shuffle(h_random_indices.begin(), h_random_indices.end(), g);
    
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Q, h_data.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_data.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_data.data(), buffer_size, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float ms;
    int warmup = 3;
    int repeat = 10;
    
    // ========================================================================
    // 对比实验1: 本地访问 vs 跨 Partition 访问
    // ========================================================================
    print_separator("对比实验1: 本地访问 vs 跨 Partition 访问");
    printf("场景：相同数据量 (32 MB)，不同的访问模式\n");
    printf("- 本地访问：连续内存块，高 L2 命中\n");
    printf("- 跨 Partition：大步长访问，频繁触发 Stitch\n");
    
    {
        int blocks = 256;
        int threads = 256;
        size_t test_elements = 8 * 1024 * 1024;  // 32 MB
        int iterations = 10;
        
        // 预热
        for (int i = 0; i < warmup; i++) {
            local_access_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 测试本地访问
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            local_access_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float local_time = ms / repeat;
        
        // 预热
        for (int i = 0; i < warmup; i++) {
            cross_partition_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations, 1024*1024);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 测试跨 Partition 访问（stride = 4MB = 1M floats）
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            cross_partition_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations, 1024*1024);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float cross_time = ms / repeat;
        
        print_comparison("本地访问 (连续)", local_time, 
                        "跨 Partition (stride=4MB)", cross_time);
    }
    
    // ========================================================================
    // 对比实验2: 单 Buffer vs 多 Buffer (Attention QKV 场景)
    // ========================================================================
    print_separator("对比实验2: 单 Buffer vs 多 Buffer (Attention 场景)");
    printf("场景：模拟 Attention 中 Q, K, V 的访问模式\n");
    printf("- 单 Buffer 3x：同一内存区域访问3次（L2 命中率高）\n");
    printf("- 多 Buffer QKV：3个独立 buffer 各访问1次（3倍唯一地址）\n");
    
    {
        int blocks = 128;
        int threads = 256;
        size_t test_elements = 4 * 1024 * 1024;  // 每个 buffer 16 MB
        int iterations = 20;
        
        // 预热
        for (int i = 0; i < warmup; i++) {
            single_buffer_3x_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 测试单 Buffer 3x 访问
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            single_buffer_3x_kernel<<<blocks, threads>>>(d_data, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float single_time = ms / repeat;
        
        // 预热
        for (int i = 0; i < warmup; i++) {
            multi_buffer_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 测试多 Buffer 访问
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            multi_buffer_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_output, test_elements, iterations);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float multi_time = ms / repeat;
        
        print_comparison("单 Buffer (3x访问)", single_time,
                        "多 Buffer (Q,K,V)", multi_time);
        
        printf("\n解释: 虽然访问次数相同，但多 Buffer 产生 3 倍唯一地址，\n");
        printf("      导致 L2 Fabric Miss 增加，触发更多 Stitch Traffic\n");
    }
    
    // ========================================================================
    // 对比实验3: 工作集大小对性能的影响
    // ========================================================================
    print_separator("对比实验3: 工作集大小 vs 性能");
    printf("场景：固定访问次数，改变工作集大小\n");
    printf("预期：工作集 > L2/2 (%.0f MB) 后性能下降\n", l2_half / 1024.0 / 1024.0);
    
    {
        int blocks = 256;
        int threads = 256;
        int iterations = 50;
        
        std::vector<std::pair<size_t, const char*>> working_sets = {
            {4 * 1024 * 1024, "16 MB (< L2/4)"},
            {8 * 1024 * 1024, "32 MB (≈ L2/2)"},
            {12 * 1024 * 1024, "48 MB (> L2/2)"},
            {16 * 1024 * 1024, "64 MB (> L2)"},
        };
        
        float base_time = 0;
        
        printf("\n工作集大小          时间(ms)      相对基准      推断\n");
        printf("────────────────────────────────────────────────────────────\n");
        
        for (auto& ws : working_sets) {
            // 预热
            for (int i = 0; i < warmup; i++) {
                working_set_kernel<<<blocks, threads>>>(d_data, d_output, ws.first, iterations);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < repeat; i++) {
                working_set_kernel<<<blocks, threads>>>(d_data, d_output, ws.first, iterations);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            float time = ms / repeat;
            
            if (base_time == 0) base_time = time;
            
            const char* inference = (ws.first * sizeof(float) <= l2_half) ? 
                "L2 本地命中" : "跨 Partition / DRAM";
            
            printf("%-20s %8.3f ms    %5.2fx        %s\n",
                   ws.second, time, time / base_time, inference);
        }
    }
    
    // ========================================================================
    // 对比实验4: 并发度对 Stitch 带宽的影响
    // ========================================================================
    print_separator("对比实验4: 并发度对 Stitch 带宽的影响");
    printf("场景：固定总访问量，改变并发 block 数\n");
    printf("预期：高并发时 Stitch 带宽成为瓶颈\n");
    
    {
        int threads = 256;
        size_t total_accesses = 256 * 256 * 100;  // 固定总访问量
        
        std::vector<int> block_counts = {8, 32, 128, 256, 512};
        
        printf("\nBlock数   每线程访问   时间(ms)      有效带宽(GB/s)   推断\n");
        printf("────────────────────────────────────────────────────────────────\n");
        
        for (int blocks : block_counts) {
            int accesses_per_thread = total_accesses / (blocks * threads);
            
            // 预热
            for (int i = 0; i < warmup; i++) {
                bandwidth_stress_kernel<<<blocks, threads>>>(d_data, d_output, elements, accesses_per_thread);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < repeat; i++) {
                bandwidth_stress_kernel<<<blocks, threads>>>(d_data, d_output, elements, accesses_per_thread);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            float time = ms / repeat;
            
            float bandwidth = (total_accesses * sizeof(float)) / (time / 1000.0f) / 1e9;
            const char* inference = (blocks <= 64) ? "低竞争" : 
                                   (blocks <= 256) ? "中等竞争" : "高竞争";
            
            printf("%5d     %6d       %8.3f ms    %8.1f         %s\n",
                   blocks, accesses_per_thread, time, bandwidth, inference);
        }
    }
    
    // ========================================================================
    // 对比实验5: 索引排序优化效果
    // ========================================================================
    print_separator("对比实验5: 数据局部性优化 (索引排序)");
    printf("场景：稀疏访问，对比随机索引 vs 排序索引\n");
    printf("优化原理：排序后相邻访问更可能在同一 L2 partition\n");
    
    {
        int blocks = 128;
        int threads = 256;
        size_t num_accesses = blocks * threads * 64;
        int iterations = 10;
        
        // 准备随机索引和排序索引
        std::vector<int> random_indices(num_accesses);
        for (size_t i = 0; i < num_accesses; i++) {
            random_indices[i] = rand() % elements;
        }
        
        std::vector<int> sorted_indices = random_indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        
        int *d_random_idx, *d_sorted_idx;
        CHECK_CUDA(cudaMalloc(&d_random_idx, num_accesses * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_sorted_idx, num_accesses * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_random_idx, random_indices.data(), 
                             num_accesses * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sorted_idx, sorted_indices.data(),
                             num_accesses * sizeof(int), cudaMemcpyHostToDevice));
        
        // 测试随机索引
        for (int i = 0; i < warmup; i++) {
            unoptimized_access_kernel<<<blocks, threads>>>(d_data, d_random_idx, d_output, num_accesses, iterations);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            unoptimized_access_kernel<<<blocks, threads>>>(d_data, d_random_idx, d_output, num_accesses, iterations);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float random_time = ms / repeat;
        
        // 测试排序索引
        for (int i = 0; i < warmup; i++) {
            unoptimized_access_kernel<<<blocks, threads>>>(d_data, d_sorted_idx, d_output, num_accesses, iterations);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            unoptimized_access_kernel<<<blocks, threads>>>(d_data, d_sorted_idx, d_output, num_accesses, iterations);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float sorted_time = ms / repeat;
        
        print_comparison("随机索引 (未优化)", random_time,
                        "排序索引 (优化后)", sorted_time);
        
        float speedup = random_time / sorted_time;
        printf("\n优化效果: %.2fx 加速\n", speedup);
        printf("原理: 排序后访问模式更连续，减少跨 Partition 访问\n");
        
        CHECK_CUDA(cudaFree(d_random_idx));
        CHECK_CUDA(cudaFree(d_sorted_idx));
    }
    
    // ========================================================================
    // 总结
    // ========================================================================
    print_separator("总结: Stitch Traffic 的影响和优化建议");
    
    printf("\n");
    printf("┌────────────────────────────────────────────────────────────────┐\n");
    printf("│                    Stitch 影响因素                             │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│ 1. 访问模式: 跨 Partition 访问比本地访问慢                     │\n");
    printf("│ 2. Buffer 数量: 多 Buffer 增加唯一地址，触发更多 Stitch        │\n");
    printf("│ 3. 工作集大小: > L2/2 后性能显著下降                           │\n");
    printf("│ 4. 并发度: 高并发时 Stitch 带宽成为瓶颈                        │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│                    优化建议                                    │\n");
    printf("├────────────────────────────────────────────────────────────────┤\n");
    printf("│ 1. 保持工作集 < L2/2 (%.0f MB)                                 │\n", l2_half / 1024.0 / 1024.0);
    printf("│ 2. 尽量复用 Buffer，减少独立 Buffer 数量                       │\n");
    printf("│ 3. 排序/重排索引，提高访问局部性                               │\n");
    printf("│ 4. 分批处理，避免超大工作集                                    │\n");
    printf("│ 5. 使用 cudaAccessPolicyWindow 控制缓存策略                    │\n");
    printf("└────────────────────────────────────────────────────────────────┘\n");
    
    // 清理
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}

