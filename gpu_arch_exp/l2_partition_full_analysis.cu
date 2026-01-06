/**
 * L2 Partition 完整分析实验
 * 
 * 实验框架:
 * Step 1: 确定 L1/L2/DRAM 容量边界
 * Step 2: 测试 partition 特性
 * Step 3: 测量 Stitch Traffic
 * Step 4: 验证优化效果 (evict_last vs evict_normal)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
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

#define WARMUP_ITERS 3
#define TEST_ITERS 10
#define CHAIN_LENGTH 10000

// ============================================================
// Step 1: Pointer Chasing 延迟测量 - 确定容量边界
// ============================================================
__global__ void pointer_chase_latency(int* data, int start_idx, int chain_length,
                                       uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        
        // Warmup
        for (int i = 0; i < chain_length; i++) {
            idx = __ldcg(&data[idx]);  // 绕过 L1, 从 L2 读取
        }
        
        idx = start_idx;
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < chain_length; i++) {
            idx = __ldcg(&data[idx]);
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) data[0] = idx;
        *result = end - start;
    }
}

// L1 cache 测试版本 (不使用 __ldcg)
__global__ void pointer_chase_l1(int* data, int start_idx, int chain_length,
                                  uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        
        // Warmup
        for (int i = 0; i < chain_length; i++) {
            idx = data[idx];  // 可能命中 L1
        }
        
        idx = start_idx;
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < chain_length; i++) {
            idx = data[idx];
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) data[0] = idx;
        *result = end - start;
    }
}

// ============================================================
// Step 2: Partition 特性测试
// ============================================================

// 单 partition 访问 (数据足迹 < L2/2)
__global__ __noinline__ void single_partition_access(float* data, size_t region_size, 
                                                      float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < 500; i++) {
        size_t idx = (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 双 partition 访问 (数据足迹 > L2/2)
__global__ __noinline__ void dual_partition_access(float* data, size_t region_size,
                                                    float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问两倍大小的区域
    for (int i = 0; i < 500; i++) {
        size_t idx = (tid * 32 + i) % (region_size * 2);
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 交替 partition 访问
__global__ __noinline__ void alternating_partition_access(float* data, 
                                                           size_t region_size,
                                                           size_t offset,
                                                           float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < 500; i++) {
        size_t base = (i % 2 == 0) ? 0 : offset;
        size_t idx = base + (tid * 32 + i / 2) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// Step 3: Stitch Traffic 测量
// ============================================================

// Coalesced 访问模式
__global__ __noinline__ void coalesced_access(float* data, size_t elements, 
                                               float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    // 连续访问，warp 内 coalesced
    for (int i = 0; i < 500; i++) {
        size_t idx = (tid + i * total_threads) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// Non-coalesced (strided) 访问模式
__global__ __noinline__ void strided_access(float* data, size_t elements,
                                             size_t stride, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 大步长访问，非 coalesced
    for (int i = 0; i < 500; i++) {
        size_t idx = ((size_t)tid * stride + i * stride) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 随机访问模式
__global__ __noinline__ void random_access(float* data, size_t elements,
                                            unsigned int seed, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    unsigned int state = seed + tid;
    for (int i = 0; i < 500; i++) {
        state = state * 1103515245 + 12345;
        size_t idx = (state >> 16) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// Step 4: evict_last vs evict_normal
// ============================================================

// 使用 streaming 模式访问 (evict_first behavior)
__global__ __noinline__ void streaming_access(float* data, size_t elements, 
                                               float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < 500; i++) {
        size_t idx = (tid * 32 + i) % elements;
        // __ldcs: streaming load (evict first)
        sum += __ldcs(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 使用 cached 模式访问 (evict_last behavior)
__global__ __noinline__ void cached_access(float* data, size_t elements, 
                                            float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < 500; i++) {
        size_t idx = (tid * 32 + i) % elements;
        // __ldcg: cached global (evict normal/last)
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 辅助函数
// ============================================================

void create_chain(int* h_data, size_t elements, int chain_length, int stride) {
    memset(h_data, 0, elements * sizeof(int));
    
    int idx = 0;
    for (int i = 0; i < chain_length - 1; i++) {
        int next = (idx + stride) % elements;
        h_data[idx] = next;
        idx = next;
    }
    h_data[idx] = 0;
}

void print_separator() {
    printf("================================================================\n");
}

double run_pointer_chase(int* d_data, int chain_length, uint64_t* d_result, bool use_l1) {
    uint64_t h_result;
    
    for (int w = 0; w < WARMUP_ITERS; w++) {
        if (use_l1) {
            pointer_chase_l1<<<1, 32>>>(d_data, 0, chain_length, d_result);
        } else {
            pointer_chase_latency<<<1, 32>>>(d_data, 0, chain_length, d_result);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint64_t total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        if (use_l1) {
            pointer_chase_l1<<<1, 32>>>(d_data, 0, chain_length, d_result);
        } else {
            pointer_chase_latency<<<1, 32>>>(d_data, 0, chain_length, d_result);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    
    return (double)total / TEST_ITERS / chain_length;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("L2 Partition 完整分析实验\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L1 Cache: %zu KB (per SM)\n", prop.l2CacheSize / 1024);  // Note: l1 not directly available
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    printf("Shared Memory: %zu KB (per SM)\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t max_buffer_size = 2 * l2_size;
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("Step 1: 确定 L1/L2/DRAM 容量边界 (Pointer Chasing)\n");
    print_separator();
    
    std::vector<size_t> sizes_kb = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 
                                     4096, 8192, 16384, 30720, 32768, 61440, 65536, 
                                     81920, 98304, 122880};
    
    size_t max_size_bytes = 128 * 1024 * 1024;
    int* h_data = (int*)malloc(max_size_bytes);
    int* d_data;
    uint64_t* d_result;
    
    CUDA_CHECK(cudaMalloc(&d_data, max_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    
    printf("\n大小(KB)   L1延迟(cycles)   L2延迟(cycles)   L2/L1比    层级推断\n");
    printf("------------------------------------------------------------------------\n");
    
    double prev_l2_latency = 0;
    
    for (size_t size_kb : sizes_kb) {
        size_t size_bytes = size_kb * 1024;
        size_t elements = size_bytes / sizeof(int);
        
        if (size_bytes > max_size_bytes) break;
        
        int stride = 32;  // 128 bytes
        create_chain(h_data, elements, CHAIN_LENGTH, stride);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size_bytes, cudaMemcpyHostToDevice));
        
        double l1_latency = run_pointer_chase(d_data, CHAIN_LENGTH, d_result, true);
        double l2_latency = run_pointer_chase(d_data, CHAIN_LENGTH, d_result, false);
        
        const char* level = "";
        if (l1_latency < 50) {
            level = "L1 Cache";
        } else if (l2_latency < 350 && l2_latency > 200) {
            level = "L2 Cache";
        } else if (l2_latency > 350) {
            level = "DRAM";
        } else {
            level = "L1/L2 边界";
        }
        
        // 检测跳变
        if (prev_l2_latency > 0 && l2_latency > prev_l2_latency * 1.3) {
            level = "*** 跳变点 ***";
        }
        
        printf("%8zu   %10.2f       %10.2f       %.2fx      %s\n",
               size_kb, l1_latency, l2_latency, l2_latency / l1_latency, level);
        
        prev_l2_latency = l2_latency;
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("Step 2: Partition 特性测试\n");
    print_separator();
    
    float *d_float_data, *d_output;
    CUDA_CHECK(cudaMalloc(&d_float_data, max_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_float_data, 1, max_buffer_size));
    
    int num_blocks = 128;
    int block_size = 256;
    size_t l2_half = l2_size / 2;
    
    printf("\n数据配置: L2 = %.0f MB, L2/2 = %.0f MB\n", 
           l2_size / 1024.0 / 1024.0, l2_half / 1024.0 / 1024.0);
    
    printf("\n测试模式          数据足迹     相对基准    推断\n");
    printf("------------------------------------------------------------------------\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 单 partition (15 MB < L2/2)
    size_t single_region = 15 * 1024 * 1024 / sizeof(float);
    
    // Warmup
    single_partition_access<<<num_blocks, block_size>>>(d_float_data, single_region, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        single_partition_access<<<num_blocks, block_size>>>(d_float_data, single_region, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float single_time;
    cudaEventElapsedTime(&single_time, start, stop);
    printf("single_partition    15 MB        1.00x       单 partition 内\n");
    
    // 双 partition (30 MB = L2/2)
    size_t dual_region = 15 * 1024 * 1024 / sizeof(float);
    
    dual_partition_access<<<num_blocks, block_size>>>(d_float_data, dual_region, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        dual_partition_access<<<num_blocks, block_size>>>(d_float_data, dual_region, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dual_time;
    cudaEventElapsedTime(&dual_time, start, stop);
    printf("dual_partition      30 MB        %.2fx       可能跨 partition\n", dual_time / single_time);
    
    // 交替 partition 访问
    size_t alt_region = 15 * 1024 * 1024 / sizeof(float);
    size_t alt_offset = 32 * 1024 * 1024 / sizeof(float);
    
    alternating_partition_access<<<num_blocks, block_size>>>(d_float_data, alt_region, alt_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        alternating_partition_access<<<num_blocks, block_size>>>(d_float_data, alt_region, alt_offset, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float alt_time;
    cudaEventElapsedTime(&alt_time, start, stop);
    printf("alternating         15+15 MB     %.2fx       交替访问两区域\n", alt_time / single_time);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("Step 3: Stitch Traffic 测量 (访问模式对比)\n");
    print_separator();
    printf("使用 NCU 分析时关注: lts__t_requests_srcunit_ltcfabric.sum\n\n");
    
    size_t full_elements = max_buffer_size / sizeof(float);
    
    printf("访问模式          时间(ms)     相对coalesced    预期 Stitch\n");
    printf("------------------------------------------------------------------------\n");
    
    // Coalesced
    coalesced_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        coalesced_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float coalesced_time;
    cudaEventElapsedTime(&coalesced_time, start, stop);
    printf("coalesced           %.3f        1.00x            低\n", coalesced_time);
    
    // Strided (stride = L2/8)
    size_t stride = l2_size / 8 / sizeof(float);
    strided_access<<<num_blocks, block_size>>>(d_float_data, full_elements, stride, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        strided_access<<<num_blocks, block_size>>>(d_float_data, full_elements, stride, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float strided_time;
    cudaEventElapsedTime(&strided_time, start, stop);
    printf("strided (L2/8)      %.3f        %.2fx            中等\n", strided_time, strided_time / coalesced_time);
    
    // Random
    random_access<<<num_blocks, block_size>>>(d_float_data, full_elements, 12345, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        random_access<<<num_blocks, block_size>>>(d_float_data, full_elements, 12345, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float random_time;
    cudaEventElapsedTime(&random_time, start, stop);
    printf("random              %.3f        %.2fx            高\n", random_time, random_time / coalesced_time);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("Step 4: evict_last vs evict_normal (Cache 策略对比)\n");
    print_separator();
    printf("__ldcs = streaming (evict first), __ldcg = cached (evict normal)\n\n");
    
    printf("Cache 策略         时间(ms)     相对 streaming    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    // Streaming access
    streaming_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        streaming_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float streaming_time;
    cudaEventElapsedTime(&streaming_time, start, stop);
    printf("streaming (__ldcs)  %.3f        1.00x             不占用 L2\n", streaming_time);
    
    // Cached access
    cached_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cached_access<<<num_blocks, block_size>>>(d_float_data, full_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cached_time;
    cudaEventElapsedTime(&cached_time, start, stop);
    printf("cached (__ldcg)     %.3f        %.2fx             占用 L2\n", cached_time, cached_time / streaming_time);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("NCU 分析命令\n");
    print_separator();
    printf("\n# 分析 Stitch Traffic:\n");
    printf("ncu --metrics lts__t_requests_srcunit_ltcfabric.sum,\\\n");
    printf("              lts__ltcfabric2lts_cycles_active.sum,\\\n");
    printf("              lts__t_sectors.sum,\\\n");
    printf("              dram__bytes_read.sum,\\\n");
    printf("              lts__t_request_hit_rate \\\n");
    printf("    ./l2_partition_full_analysis\n");
    print_separator();
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_float_data));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

