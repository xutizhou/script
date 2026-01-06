/**
 * L2 Cache Partition Benchmark
 * 
 * 实验目的：验证 Hopper 架构的 L2 Cache 分区特性
 * 
 * 关键概念：
 * - POC (Point of Coherence): 每个地址只在一个 L2 分区有权威副本
 * - LCN (Local Cache Node): 本地 L2 可以缓存远程数据的副本
 * - Stitch Traffic: 跨 uGPU 分区的 L2 访问流量
 * 
 * 验证的关键阈值：
 * - X ≤ C/2: 数据可复制到两个分区，无 stitch traffic
 * - C/2 < X ≤ C: LCN miss 产生 stitch traffic，无 DRAM traffic
 * - X > C: POC miss，产生 DRAM traffic
 * 
 * H20 L2 Cache: 60 MB (61440 KB)
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

#define WARMUP_ITERS 5
#define TEST_ITERS 20

// ============================================================
// 实验1: L2 容量探测 - 验证 C/2 阈值
// 逐渐增加数据足迹，观察延迟变化
// ============================================================
__global__ void sequential_access_kernel(float* data, int num_elements, 
                                          uint64_t* result, int iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        
        uint64_t start = clock64();
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < num_elements; i += 32) {  // 每 128 bytes 访问一次
                sum += data[i];
            }
        }
        
        uint64_t end = clock64();
        
        // 防止优化掉
        if (sum == -999999.0f) data[0] = sum;
        
        *result = end - start;
    }
}

// ============================================================
// 实验2: Stride 访问测试 - 验证分区映射
// 不同 stride 可能落在不同 L2 分区
// ============================================================
__global__ void stride_access_kernel(float* data, int num_elements, int stride,
                                      uint64_t* result, int iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        
        uint64_t start = clock64();
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < num_elements; i += stride) {
                sum += data[i];
            }
        }
        
        uint64_t end = clock64();
        
        if (sum == -999999.0f) data[0] = sum;
        *result = end - start;
    }
}

// ============================================================
// 实验3: 多 SM 并发访问 - 验证 Stitch 竞争
// 多个 SM 访问分布在不同分区的数据
// ============================================================
__global__ void multi_sm_distributed_access_kernel(float* data, int elements_per_sm,
                                                    int total_elements, int stride,
                                                    uint64_t* results, int iterations) {
    __shared__ float smem_sum;
    
    if (threadIdx.x == 0) {
        smem_sum = 0.0f;
    }
    __syncthreads();
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 每个 block 访问分散的地址（模拟跨分区访问）
    int base_idx = (bid * stride) % total_elements;
    
    float local_sum = 0.0f;
    
    uint64_t start = clock64();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < elements_per_sm; i += blockDim.x) {
            int idx = (base_idx + i * stride) % total_elements;
            local_sum += data[idx];
        }
    }
    
    uint64_t end = clock64();
    
    atomicAdd(&smem_sum, local_sum);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        results[blockIdx.x] = end - start;
        // 防止优化
        if (smem_sum == -999999.0f) data[0] = smem_sum;
    }
}

// ============================================================
// 实验4: 本地化 vs 分散访问
// 对比 coalesced 访问 vs 跨分区随机访问
// ============================================================
__global__ void localized_access_kernel(float* data, int num_elements,
                                         uint64_t* results, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    // 本地化访问：每个 warp 访问连续的数据
    int elements_per_thread = num_elements / total_threads;
    int start_idx = tid * elements_per_thread;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < elements_per_thread; i++) {
            sum += data[start_idx + i];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[blockIdx.x] = end_time - start_time;
    }
    
    // 防止优化
    if (sum == -999999.0f) data[tid] = sum;
}

__global__ void scattered_access_kernel(float* data, int num_elements,
                                         uint64_t* results, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    // 分散访问：使用大 stride 来跨分区
    int large_stride = num_elements / 64;  // 确保跨越多个分区
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 1024; i++) {
            int idx = (tid + i * large_stride) % num_elements;
            sum += data[idx];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[blockIdx.x] = end_time - start_time;
    }
    
    if (sum == -999999.0f) data[tid] = sum;
}

// ============================================================
// 实验5: 写操作测试 - 验证 Coherency Traffic
// 写操作需要 invalidate 远程 LCN 副本
// ============================================================
__global__ void write_coherency_kernel(float* data, int num_elements,
                                        int stride, uint64_t* results, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    // 多个 SM 写入分散的地址，触发 coherency traffic
    for (int iter = 0; iter < iterations; iter++) {
        int idx = (tid * stride) % num_elements;
        data[idx] += 1.0f;
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[blockIdx.x] = end_time - start_time;
    }
}

// ============================================================
// 实验6: Pointer Chasing - 精确测量跨分区延迟
// ============================================================
__global__ void pointer_chase_kernel(int* chain, int chain_length,
                                      uint64_t* result, int iterations) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = 0;
        
        uint64_t start = clock64();
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < chain_length; i++) {
                idx = chain[idx];
            }
        }
        
        uint64_t end = clock64();
        
        // 防止优化
        if (idx == -999999) chain[0] = idx;
        
        *result = end - start;
    }
}

void print_separator() {
    printf("================================================================\n");
}

void setup_pointer_chain(int* h_chain, int* d_chain, int chain_length, int stride, int array_size) {
    // 创建 pointer chasing chain
    for (int i = 0; i < array_size; i++) {
        h_chain[i] = 0;
    }
    
    int idx = 0;
    for (int i = 0; i < chain_length - 1; i++) {
        int next_idx = (idx + stride) % array_size;
        h_chain[idx] = next_idx;
        idx = next_idx;
    }
    h_chain[idx] = 0;  // 回到起点
    
    cudaMemcpy(d_chain, h_chain, array_size * sizeof(int), cudaMemcpyHostToDevice);
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("L2 Cache Partition Benchmark\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d KB (%.1f MB)\n", prop.l2CacheSize / 1024, prop.l2CacheSize / 1024.0 / 1024.0);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t half_l2 = l2_size / 2;
    
    printf("\n关键阈值:\n");
    printf("  C (L2 总容量): %.1f MB\n", l2_size / 1024.0 / 1024.0);
    printf("  C/2 (分区阈值): %.1f MB\n", half_l2 / 1024.0 / 1024.0);
    print_separator();
    
    // 分配内存
    uint64_t *d_result, h_result;
    uint64_t *d_results, h_results[256];
    
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_results, 256 * sizeof(uint64_t)));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1: L2 容量探测 - 验证 C/2 阈值\n");
    printf("逐渐增加数据足迹，观察延迟变化\n");
    print_separator();
    
    // 测试不同大小的数据
    std::vector<size_t> test_sizes = {
        1 * 1024 * 1024,      // 1 MB
        5 * 1024 * 1024,      // 5 MB
        10 * 1024 * 1024,     // 10 MB
        15 * 1024 * 1024,     // 15 MB (约 C/4)
        20 * 1024 * 1024,     // 20 MB
        25 * 1024 * 1024,     // 25 MB
        30 * 1024 * 1024,     // 30 MB (约 C/2) ← 关键阈值
        35 * 1024 * 1024,     // 35 MB
        40 * 1024 * 1024,     // 40 MB
        50 * 1024 * 1024,     // 50 MB
        60 * 1024 * 1024,     // 60 MB (约 C)
        80 * 1024 * 1024,     // 80 MB (> C)
        100 * 1024 * 1024,    // 100 MB
    };
    
    printf("\n数据足迹      访问次数    总时间(cycles)  每次访问(cycles)  相对C/2\n");
    printf("------------------------------------------------------------------------\n");
    
    double baseline_latency = 0;
    
    for (size_t size : test_sizes) {
        float* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size));
        CUDA_CHECK(cudaMemset(d_data, 0, size));
        
        int num_elements = size / sizeof(float);
        int num_accesses = num_elements / 32;
        int iterations = 10;
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            sequential_access_kernel<<<1, 32>>>(d_data, num_elements, d_result, iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Test
        uint64_t total_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            sequential_access_kernel<<<1, 32>>>(d_data, num_elements, d_result, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_time += h_result;
        }
        
        double avg_time = (double)total_time / TEST_ITERS;
        double per_access = avg_time / (num_accesses * iterations);
        double relative_to_half = (double)size / half_l2;
        
        if (size == 1 * 1024 * 1024) {
            baseline_latency = per_access;
        }
        
        const char* marker = "";
        if (relative_to_half >= 0.95 && relative_to_half <= 1.05) marker = " ← C/2";
        else if (relative_to_half >= 1.95 && relative_to_half <= 2.05) marker = " ← C";
        
        printf("%6.1f MB     %8d    %12.0f    %8.2f cycles    %.2fx%s\n",
               size / 1024.0 / 1024.0, num_accesses * iterations, avg_time, 
               per_access, relative_to_half, marker);
        
        CUDA_CHECK(cudaFree(d_data));
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验2: Stride 访问 - 验证分区哈希映射\n");
    printf("不同 stride 测试地址如何映射到 L2 分区\n");
    print_separator();
    
    size_t stride_test_size = 32 * 1024 * 1024;  // 32 MB
    float* d_stride_data;
    CUDA_CHECK(cudaMalloc(&d_stride_data, stride_test_size));
    CUDA_CHECK(cudaMemset(d_stride_data, 0, stride_test_size));
    
    int stride_elements = stride_test_size / sizeof(float);
    
    printf("\nStride (bytes)   Stride (ints)   访问时间(cycles)   相对baseline\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<int> strides = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    double stride_baseline = 0;
    
    for (int stride : strides) {
        int iterations = 100;
        int num_accesses = stride_elements / stride;
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            stride_access_kernel<<<1, 32>>>(d_stride_data, stride_elements, stride, d_result, iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Test
        uint64_t total_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            stride_access_kernel<<<1, 32>>>(d_stride_data, stride_elements, stride, d_result, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total_time += h_result;
        }
        
        double avg_time = (double)total_time / TEST_ITERS;
        double per_access = avg_time / (num_accesses * iterations);
        
        if (stride == 32) stride_baseline = per_access;
        
        printf("%6d bytes     %8d        %8.2f cycles      %.2fx\n",
               stride * 4, stride, per_access, per_access / stride_baseline);
    }
    
    CUDA_CHECK(cudaFree(d_stride_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: 多 SM 并发访问 - Stitch 竞争\n");
    printf("多个 SM 同时访问分散的数据，观察竞争影响\n");
    print_separator();
    
    size_t multi_sm_size = 40 * 1024 * 1024;  // 40 MB (> C/2)
    float* d_multi_sm_data;
    CUDA_CHECK(cudaMalloc(&d_multi_sm_data, multi_sm_size));
    CUDA_CHECK(cudaMemset(d_multi_sm_data, 0, multi_sm_size));
    
    int multi_elements = multi_sm_size / sizeof(float);
    int elements_per_sm = 1024;
    int iterations = 100;
    
    printf("\nSM数量   大Stride访问时间(cycles)   小Stride访问时间(cycles)   比值\n");
    printf("------------------------------------------------------------------------\n");
    
    for (int num_blocks : {1, 2, 4, 8, 16, 32, 64}) {
        // 大 stride (跨分区)
        int large_stride = multi_elements / 64;
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            multi_sm_distributed_access_kernel<<<num_blocks, 128>>>(
                d_multi_sm_data, elements_per_sm, multi_elements, large_stride, d_results, iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t large_stride_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            multi_sm_distributed_access_kernel<<<num_blocks, 128>>>(
                d_multi_sm_data, elements_per_sm, multi_elements, large_stride, d_results, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            uint64_t max_time = 0;
            for (int j = 0; j < num_blocks; j++) {
                if (h_results[j] > max_time) max_time = h_results[j];
            }
            large_stride_time += max_time;
        }
        
        // 小 stride (本地化)
        int small_stride = 1;
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            multi_sm_distributed_access_kernel<<<num_blocks, 128>>>(
                d_multi_sm_data, elements_per_sm, multi_elements, small_stride, d_results, iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t small_stride_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            multi_sm_distributed_access_kernel<<<num_blocks, 128>>>(
                d_multi_sm_data, elements_per_sm, multi_elements, small_stride, d_results, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            uint64_t max_time = 0;
            for (int j = 0; j < num_blocks; j++) {
                if (h_results[j] > max_time) max_time = h_results[j];
            }
            small_stride_time += max_time;
        }
        
        double large_avg = (double)large_stride_time / TEST_ITERS;
        double small_avg = (double)small_stride_time / TEST_ITERS;
        
        printf("%4d     %16.0f cycles          %16.0f cycles          %.2fx\n",
               num_blocks, large_avg, small_avg, large_avg / small_avg);
    }
    
    CUDA_CHECK(cudaFree(d_multi_sm_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验4: 本地化 vs 分散访问\n");
    printf("对比 coalesced 访问 vs 跨分区随机访问\n");
    print_separator();
    
    size_t locality_size = 32 * 1024 * 1024;  // 32 MB
    float* d_locality_data;
    CUDA_CHECK(cudaMalloc(&d_locality_data, locality_size));
    CUDA_CHECK(cudaMemset(d_locality_data, 0, locality_size));
    
    int locality_elements = locality_size / sizeof(float);
    int num_blocks = 32;
    int block_size = 256;
    iterations = 10;
    
    // 本地化访问
    for (int i = 0; i < WARMUP_ITERS; i++) {
        localized_access_kernel<<<num_blocks, block_size>>>(d_locality_data, locality_elements, d_results, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint64_t localized_time = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        localized_access_kernel<<<num_blocks, block_size>>>(d_locality_data, locality_elements, d_results, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        uint64_t max_time = 0;
        for (int j = 0; j < num_blocks; j++) {
            if (h_results[j] > max_time) max_time = h_results[j];
        }
        localized_time += max_time;
    }
    
    // 分散访问
    for (int i = 0; i < WARMUP_ITERS; i++) {
        scattered_access_kernel<<<num_blocks, block_size>>>(d_locality_data, locality_elements, d_results, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint64_t scattered_time = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        scattered_access_kernel<<<num_blocks, block_size>>>(d_locality_data, locality_elements, d_results, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        uint64_t max_time = 0;
        for (int j = 0; j < num_blocks; j++) {
            if (h_results[j] > max_time) max_time = h_results[j];
        }
        scattered_time += max_time;
    }
    
    double local_avg = (double)localized_time / TEST_ITERS;
    double scatter_avg = (double)scattered_time / TEST_ITERS;
    
    printf("\n本地化访问 (coalesced):  %.0f cycles\n", local_avg);
    printf("分散访问 (scattered):    %.0f cycles\n", scatter_avg);
    printf("分散/本地化 比值:        %.2fx\n", scatter_avg / local_avg);
    
    CUDA_CHECK(cudaFree(d_locality_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验5: 写操作 Coherency Traffic\n");
    printf("多 SM 写入分散地址，触发 coherency traffic\n");
    print_separator();
    
    size_t write_size = 32 * 1024 * 1024;
    float* d_write_data;
    CUDA_CHECK(cudaMalloc(&d_write_data, write_size));
    CUDA_CHECK(cudaMemset(d_write_data, 0, write_size));
    
    int write_elements = write_size / sizeof(float);
    iterations = 100;
    
    printf("\nStride (元素)   写入时间(cycles)\n");
    printf("--------------------------------\n");
    
    for (int stride : {1, 16, 64, 256, 1024, 4096, 16384}) {
        CUDA_CHECK(cudaMemset(d_write_data, 0, write_size));
        
        for (int i = 0; i < WARMUP_ITERS; i++) {
            write_coherency_kernel<<<32, 256>>>(d_write_data, write_elements, stride, d_results, iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint64_t write_time = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            write_coherency_kernel<<<32, 256>>>(d_write_data, write_elements, stride, d_results, iterations);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            uint64_t max_time = 0;
            for (int j = 0; j < 32; j++) {
                if (h_results[j] > max_time) max_time = h_results[j];
            }
            write_time += max_time;
        }
        
        printf("%8d        %.0f cycles\n", stride, (double)write_time / TEST_ITERS);
    }
    
    CUDA_CHECK(cudaFree(d_write_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("结论与代码优化建议\n");
    print_separator();
    printf("1. 数据足迹控制:\n");
    printf("   - 尽量保持工作集 < C/2 (%.1f MB) 避免 stitch traffic\n", half_l2 / 1024.0 / 1024.0);
    printf("   - 超过 C (%.1f MB) 会产生 DRAM traffic\n", l2_size / 1024.0 / 1024.0);
    printf("\n");
    printf("2. 内存访问模式:\n");
    printf("   - 使用 coalesced 访问，保持数据局部性\n");
    printf("   - 避免大 stride 导致的跨分区访问\n");
    printf("\n");
    printf("3. uGPU Localization (如果可用):\n");
    printf("   - 使用 TPC affinity 将计算限制在单个 uGPU\n");
    printf("   - 使用 localized memory allocation\n");
    printf("\n");
    printf("4. L2 Cache 控制:\n");
    printf("   - 使用 cudaAccessPolicyWindow 设置 evict_last\n");
    printf("   - 对流式数据使用 evict_first 避免污染 cache\n");
    print_separator();
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_results));
    
    return 0;
}

