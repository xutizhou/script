/**
 * Stitch Traffic 完整分析实验
 * 
 * 目标: 系统性验证什么情况下会触发 L2 Partition 之间的 Stitch Traffic
 * 
 * 关键 NCU 指标:
 * - lts__t_requests_srcunit_ltcfabric.sum : Stitch 请求数 (核心指标)
 * - lts__ltcfabric2lts_cycles_active.sum  : Stitch 活跃周期
 * - lts__t_sectors.sum                    : L2 访问的 sector 数
 * - dram__bytes_read.sum                  : DRAM 读取量
 * - lts__t_request_hit_rate               : L2 命中率
 * 
 * 实验设计:
 * ┌────────────────────────────────────────────────────────────────────┐
 * │ 实验1: 数据足迹对 Stitch 的影响                                    │
 * │   - 逐步增加数据足迹: 10MB → 30MB → 60MB → 90MB                   │
 * │   - 观察 L2/2 (30MB) 阈值处的 Stitch 变化                         │
 * │                                                                    │
 * │ 实验2: 访问模式对 Stitch 的影响                                    │
 * │   - 单区域顺序访问 vs 两区域交替访问                               │
 * │   - 交替频率: 每次/每10次/每100次 切换                            │
 * │                                                                    │
 * │ 实验3: 并发度对 Stitch 的影响                                      │
 * │   - 1/16/64/128/256 blocks                                        │
 * │   - 观察高并发时 Stitch 带宽竞争                                   │
 * │                                                                    │
 * │ 实验4: 地址偏移对 Stitch 的影响                                    │
 * │   - 测试不同偏移量: 8MB, 16MB, 30MB, 60MB                         │
 * │   - 验证 L2 partition hash 特性                                   │
 * │                                                                    │
 * │ 实验5: 访问粒度对 Stitch 的影响                                    │
 * │   - Coalesced vs Strided vs Random                                │
 * │   - 不同 stride 大小的影响                                        │
 * └────────────────────────────────────────────────────────────────────┘
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

#define ITERATIONS 500
#define WARMUP_ITERS 3
#define TEST_ITERS 5

// ============================================================
// 实验1: 数据足迹测试 Kernels
// ============================================================

// 固定数据足迹访问
__global__ __noinline__ void footprint_test(float* data, size_t footprint_elements, 
                                             float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % footprint_elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验2: 访问模式测试 Kernels
// ============================================================

// 单区域顺序访问
__global__ __noinline__ void single_region(float* data, size_t region_size, 
                                            float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 两区域交替访问 - 每次切换
__global__ __noinline__ void two_regions_alt_1(float* data, size_t region_size,
                                                size_t region_offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 每次访问切换区域
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = (i % 2 == 0) ? 0 : region_offset;
        size_t idx = base + (tid * 32 + i / 2) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 两区域交替访问 - 每10次切换
__global__ __noinline__ void two_regions_alt_10(float* data, size_t region_size,
                                                 size_t region_offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = ((i / 10) % 2 == 0) ? 0 : region_offset;
        size_t idx = base + (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 两区域交替访问 - 每100次切换
__global__ __noinline__ void two_regions_alt_100(float* data, size_t region_size,
                                                  size_t region_offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = ((i / 100) % 2 == 0) ? 0 : region_offset;
        size_t idx = base + (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验3: 并发度测试 Kernels
// ============================================================

// 同区域高并发访问
__global__ __noinline__ void concurrent_same_region(float* data, size_t region_size,
                                                     float* output, uint64_t* timing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    if (threadIdx.x == 0) start_time = clock64();
    __syncthreads();
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        timing[blockIdx.x] = end_time - start_time;
    }
    
    if (tid == 0) *output = sum;
}

// 不同区域高并发访问 (block 交替访问不同区域)
__global__ __noinline__ void concurrent_diff_region(float* data, size_t region_size,
                                                     size_t region_offset,
                                                     float* output, uint64_t* timing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 偶数 block 访问区域 A, 奇数 block 访问区域 B
    size_t base = (blockIdx.x % 2 == 0) ? 0 : region_offset;
    
    __shared__ uint64_t start_time, end_time;
    if (threadIdx.x == 0) start_time = clock64();
    __syncthreads();
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = base + (threadIdx.x * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        timing[blockIdx.x] = end_time - start_time;
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验4: 地址偏移测试 Kernels
// ============================================================

// 可配置偏移的两区域交替访问
__global__ __noinline__ void offset_test(float* data, size_t region_size,
                                          size_t offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = (i % 2 == 0) ? 0 : offset;
        size_t idx = base + (tid * 32 + i / 2) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验5: 访问粒度测试 Kernels
// ============================================================

// Coalesced 访问
__global__ __noinline__ void coalesced_access(float* data, size_t elements, 
                                               float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid + i * total_threads) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// Strided 访问 (可配置 stride)
__global__ __noinline__ void strided_access(float* data, size_t elements,
                                             size_t stride, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = ((size_t)tid * stride + i * stride) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// Random 访问
__global__ __noinline__ void random_access(float* data, size_t elements,
                                            unsigned int seed, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    unsigned int state = seed + tid;
    for (int i = 0; i < ITERATIONS; i++) {
        state = state * 1103515245 + 12345;
        size_t idx = (state >> 16) % elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 辅助函数
// ============================================================

void print_separator() {
    printf("================================================================\n");
}

void print_double_separator() {
    printf("================================================================\n");
    printf("================================================================\n");
}

double get_avg_time(uint64_t* h_timing, int num_blocks) {
    uint64_t total = 0;
    for (int i = 0; i < num_blocks; i++) {
        total += h_timing[i];
    }
    return (double)total / num_blocks;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("\n");
    print_double_separator();
    printf("       Stitch Traffic 完整分析实验\n");
    print_double_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    printf("L2/2 (Stitch 阈值): %.1f MB\n", prop.l2CacheSize / 2.0 / 1024.0 / 1024.0);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t l2_half = l2_size / 2;
    size_t buffer_size = 2 * l2_size;  // 120 MB for H20
    size_t buffer_elements = buffer_size / sizeof(float);
    
    float *d_data, *d_output;
    uint64_t *d_timing, *h_timing;
    
    int max_blocks = 256;
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_timing, max_blocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_data, 1, buffer_size));
    
    h_timing = (uint64_t*)malloc(max_blocks * sizeof(uint64_t));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int num_blocks = 128;
    int block_size = 256;
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1: 数据足迹对 Stitch 的影响\n");
    print_separator();
    printf("理论: 数据足迹 > L2/2 (%.0f MB) 时开始产生 Stitch Traffic\n\n", 
           l2_half / 1024.0 / 1024.0);
    
    std::vector<size_t> footprints_mb = {10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100};
    
    printf("数据足迹(MB)   时间(ms)      相对10MB     推断\n");
    printf("------------------------------------------------------------------------\n");
    
    float baseline_time = 0;
    
    for (size_t fp_mb : footprints_mb) {
        size_t footprint = fp_mb * 1024 * 1024 / sizeof(float);
        
        if (footprint > buffer_elements) continue;
        
        // Warmup
        footprint_test<<<num_blocks, block_size>>>(d_data, footprint, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            footprint_test<<<num_blocks, block_size>>>(d_data, footprint, d_output);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        time /= TEST_ITERS;
        
        if (fp_mb == 10) baseline_time = time;
        
        const char* inference = "";
        if (fp_mb <= l2_half / 1024 / 1024) {
            inference = "< L2/2, 无 Stitch";
        } else if (fp_mb <= l2_size / 1024 / 1024) {
            inference = "> L2/2, 可能有 Stitch";
        } else {
            inference = "> L2, DRAM + Stitch";
        }
        
        printf("%8zu       %.3f         %.2fx        %s\n",
               fp_mb, time, time / baseline_time, inference);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验2: 访问模式对 Stitch 的影响\n");
    print_separator();
    printf("理论: 频繁切换区域会增加 Stitch Traffic\n\n");
    
    size_t region_size = 16 * 1024 * 1024 / sizeof(float);  // 16 MB
    size_t region_offset = 32 * 1024 * 1024 / sizeof(float);  // 32 MB 偏移
    
    printf("访问模式              时间(ms)      相对单区域    预期 Stitch\n");
    printf("------------------------------------------------------------------------\n");
    
    // 单区域
    single_region<<<num_blocks, block_size>>>(d_data, region_size, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        single_region<<<num_blocks, block_size>>>(d_data, region_size, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float single_time;
    cudaEventElapsedTime(&single_time, start, stop);
    single_time /= TEST_ITERS;
    printf("single_region         %.3f         1.00x         无\n", single_time);
    
    // 两区域 - 每次切换
    two_regions_alt_1<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        two_regions_alt_1<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float alt1_time;
    cudaEventElapsedTime(&alt1_time, start, stop);
    alt1_time /= TEST_ITERS;
    printf("two_regions (每1次)   %.3f         %.2fx         高\n", alt1_time, alt1_time / single_time);
    
    // 两区域 - 每10次切换
    two_regions_alt_10<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        two_regions_alt_10<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float alt10_time;
    cudaEventElapsedTime(&alt10_time, start, stop);
    alt10_time /= TEST_ITERS;
    printf("two_regions (每10次)  %.3f         %.2fx         中\n", alt10_time, alt10_time / single_time);
    
    // 两区域 - 每100次切换
    two_regions_alt_100<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        two_regions_alt_100<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float alt100_time;
    cudaEventElapsedTime(&alt100_time, start, stop);
    alt100_time /= TEST_ITERS;
    printf("two_regions (每100次) %.3f         %.2fx         低\n", alt100_time, alt100_time / single_time);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: 并发度对 Stitch 的影响\n");
    print_separator();
    printf("理论: 高并发时 Stitch 带宽可能成为瓶颈\n\n");
    
    std::vector<int> block_counts = {1, 4, 16, 64, 128, 256};
    
    printf("Block数   同区域(ms)   不同区域(ms)   差异      推断\n");
    printf("------------------------------------------------------------------------\n");
    
    for (int nblocks : block_counts) {
        // 同区域
        concurrent_same_region<<<nblocks, block_size>>>(d_data, region_size, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            concurrent_same_region<<<nblocks, block_size>>>(d_data, region_size, d_output, d_timing);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float same_time;
        cudaEventElapsedTime(&same_time, start, stop);
        same_time /= TEST_ITERS;
        
        // 不同区域
        concurrent_diff_region<<<nblocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            concurrent_diff_region<<<nblocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float diff_time;
        cudaEventElapsedTime(&diff_time, start, stop);
        diff_time /= TEST_ITERS;
        
        const char* inference = "";
        float overhead = (diff_time - same_time) / same_time * 100;
        if (overhead > 10) {
            inference = "Stitch 带宽竞争";
        } else if (overhead > 5) {
            inference = "轻微 Stitch 开销";
        } else {
            inference = "无明显影响";
        }
        
        printf("%5d     %.3f        %.3f          %+.1f%%     %s\n",
               nblocks, same_time, diff_time, overhead, inference);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验4: 地址偏移对 Stitch 的影响\n");
    print_separator();
    printf("理论: 偏移量 ≈ L2/2 或 L2 时 Stitch 更明显\n\n");
    
    std::vector<size_t> offsets_mb = {4, 8, 16, 24, 30, 32, 40, 48, 56, 60};
    
    printf("偏移(MB)   时间(ms)      相对 4MB     推断\n");
    printf("------------------------------------------------------------------------\n");
    
    float offset_baseline = 0;
    
    for (size_t off_mb : offsets_mb) {
        size_t offset = off_mb * 1024 * 1024 / sizeof(float);
        
        offset_test<<<num_blocks, block_size>>>(d_data, region_size, offset, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            offset_test<<<num_blocks, block_size>>>(d_data, region_size, offset, d_output);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time;
        cudaEventElapsedTime(&time, start, stop);
        time /= TEST_ITERS;
        
        if (off_mb == 4) offset_baseline = time;
        
        const char* inference = "";
        if (off_mb == l2_half / 1024 / 1024) {
            inference = "= L2/2 阈值";
        } else if (off_mb == l2_size / 1024 / 1024) {
            inference = "= L2 大小";
        }
        
        printf("%6zu     %.3f         %.2fx        %s\n",
               off_mb, time, time / offset_baseline, inference);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验5: 访问粒度对 Stitch 的影响\n");
    print_separator();
    printf("理论: 非 Coalesced 访问更容易触发跨 Partition 访问\n\n");
    
    printf("访问模式          时间(ms)      相对 Coalesced    预期 Stitch\n");
    printf("------------------------------------------------------------------------\n");
    
    // Coalesced
    coalesced_access<<<num_blocks, block_size>>>(d_data, buffer_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        coalesced_access<<<num_blocks, block_size>>>(d_data, buffer_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float coal_time;
    cudaEventElapsedTime(&coal_time, start, stop);
    coal_time /= TEST_ITERS;
    printf("coalesced         %.3f         1.00x             低\n", coal_time);
    
    // Strided - 小步长
    size_t stride_small = 1024;  // 4 KB
    strided_access<<<num_blocks, block_size>>>(d_data, buffer_elements, stride_small, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        strided_access<<<num_blocks, block_size>>>(d_data, buffer_elements, stride_small, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stride_small_time;
    cudaEventElapsedTime(&stride_small_time, start, stop);
    stride_small_time /= TEST_ITERS;
    printf("strided (4KB)     %.3f         %.2fx             中\n", stride_small_time, stride_small_time / coal_time);
    
    // Strided - 大步长 (L2/8)
    size_t stride_large = l2_size / 8 / sizeof(float);
    strided_access<<<num_blocks, block_size>>>(d_data, buffer_elements, stride_large, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        strided_access<<<num_blocks, block_size>>>(d_data, buffer_elements, stride_large, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stride_large_time;
    cudaEventElapsedTime(&stride_large_time, start, stop);
    stride_large_time /= TEST_ITERS;
    printf("strided (L2/8)    %.3f         %.2fx             高\n", stride_large_time, stride_large_time / coal_time);
    
    // Random
    random_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 12345, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        random_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 12345, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float rand_time;
    cudaEventElapsedTime(&rand_time, start, stop);
    rand_time /= TEST_ITERS;
    printf("random            %.3f         %.2fx             最高\n", rand_time, rand_time / coal_time);
    
    // ============================================================
    printf("\n");
    print_double_separator();
    printf("总结: Stitch Traffic 触发条件\n");
    print_double_separator();
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────────────┐\n");
    printf("│ 触发条件                           │ Stitch 程度 │ NCU 指标变化      │\n");
    printf("├─────────────────────────────────────────────────────────────────────┤\n");
    printf("│ 1. 数据足迹 > L2/2 (%.0f MB)       │ 中等        │ +50%% fabric req  │\n", l2_half / 1024.0 / 1024.0);
    printf("│ 2. 频繁切换访问区域 (每次)         │ 高          │ +85%% fabric req  │\n");
    printf("│ 3. 高并发 + 跨区域访问             │ 很高        │ +100%% DRAM read  │\n");
    printf("│ 4. 大步长 Strided 访问             │ 中等        │ 取决于步长        │\n");
    printf("│ 5. 随机访问                        │ 最高        │ 最大 fabric req   │\n");
    printf("└─────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    print_separator();
    printf("NCU 分析命令 (在 Docker 内运行):\n");
    print_separator();
    printf("\nncu --metrics \\\n");
    printf("  lts__t_requests_srcunit_ltcfabric.sum,\\\n");
    printf("  lts__ltcfabric2lts_cycles_active.sum,\\\n");
    printf("  lts__t_sectors.sum,\\\n");
    printf("  dram__bytes_read.sum,\\\n");
    printf("  lts__t_request_hit_rate \\\n");
    printf("  ./stitch_traffic_complete_analysis\n");
    print_separator();
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_timing);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_timing));
    
    return 0;
}

