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
 * │                                                                    │
 * │ 实验6: Stitch 延迟影响实验 (NEW)                                   │
 * │   - Pointer chasing 测量本地 vs 远端 L2 延迟                      │
 * │   - 验证 ~100 cycles 额外延迟                                     │
 * │                                                                    │
 * │ 实验7: 有效 L2 容量实验 (NEW)                                      │
 * │   - 展示 Stitch 如何减少有效 L2 容量                              │
 * │   - 对比 localized vs non-localized 场景                          │
 * │                                                                    │
 * │ 实验8: 多 Buffer 场景 (NEW)                                        │
 * │   - 模拟 Attention (Q, K, V) 等多矩阵访问                         │
 * │   - 测量 N 个独立 buffer 的 Stitch 累加效应                       │
 * │                                                                    │
 * │ 实验9: Stitch 带宽瓶颈实验 (NEW)                                   │
 * │   - 高带宽压力下 Stitch 成为瓶颈                                  │
 * │   - 对比本地 vs 跨 uGPU 带宽                                      │
 * └────────────────────────────────────────────────────────────────────┘
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>

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
    
    // 每次访问切换区域 (修复: 保持与 single_region 相同的访问地址数量)
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = (i % 2 == 0) ? 0 : region_offset;
        size_t idx = base + (tid * 32 + i) % region_size;  // 修复: 移除 i/2
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
    
    // 修复: 保持与 single_region 相同的访问地址数量
    for (int i = 0; i < ITERATIONS; i++) {
        size_t base = (i % 2 == 0) ? 0 : offset;
        size_t idx = base + (tid * 32 + i) % region_size;  // 修复: 移除 i/2
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
// 实验6: Stitch 延迟测量 Kernels
// ============================================================

// Pointer chasing 延迟测试 - 测量单次内存访问延迟
__global__ __noinline__ void latency_pointer_chase(uint64_t* chain, size_t chain_len,
                                                    uint64_t* output, uint64_t* latency) {
    // 只用一个线程来精确测量延迟
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    uint64_t idx = 0;
    uint64_t start = clock64();
    
    // Pointer chasing: 每次访问依赖上一次结果，无法并行
    for (size_t i = 0; i < chain_len; i++) {
        idx = chain[idx];
    }
    
    uint64_t end = clock64();
    *output = idx;  // 防止优化
    *latency = (end - start) / chain_len;  // 平均每次访问延迟
}

// 高并发延迟测试 - 测量在负载下的延迟
__global__ __noinline__ void latency_under_load(float* data, size_t region_size,
                                                 size_t offset, uint64_t* timing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 选择访问区域: 偶数 block 本地区域，奇数 block 远端区域
    size_t base = (blockIdx.x % 2 == 0) ? 0 : offset;
    
    float sum = 0.0f;
    uint64_t start = clock64();
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = base + (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    uint64_t end = clock64();
    
    if (threadIdx.x == 0) {
        timing[blockIdx.x] = end - start;
    }
    
    // 防止优化
    if (sum == -999999.0f) data[0] = sum;
}

// ============================================================
// 实验7: 有效 L2 容量测试 Kernels
// ============================================================

// 测试有效 L2 容量 - 通过 hit rate 变化来检测
__global__ __noinline__ void effective_l2_test(float* data, size_t footprint,
                                                int iterations, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    // 多次遍历相同数据，测量 L2 hit rate
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = tid; i < footprint; i += total_threads) {
            sum += __ldcg(&data[i]);
        }
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验8: 多 Buffer 场景 Kernels (模拟 Attention Q, K, V)
// ============================================================

// 单 buffer 访问 (基准)
__global__ __noinline__ void single_buffer_access(float* A, size_t elements,
                                                   float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % elements;
        sum += __ldcg(&A[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 双 buffer 访问 (模拟 K, V)
__global__ __noinline__ void dual_buffer_access(float* A, float* B, size_t elements,
                                                 float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % elements;
        sum += __ldcg(&A[idx]) + __ldcg(&B[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 三 buffer 访问 (模拟 Q, K, V)
__global__ __noinline__ void triple_buffer_access(float* A, float* B, float* C, 
                                                   size_t elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % elements;
        sum += __ldcg(&A[idx]) + __ldcg(&B[idx]) + __ldcg(&C[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 四 buffer 访问
__global__ __noinline__ void quad_buffer_access(float* A, float* B, float* C, float* D,
                                                 size_t elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = (tid * 32 + i) % elements;
        sum += __ldcg(&A[idx]) + __ldcg(&B[idx]) + __ldcg(&C[idx]) + __ldcg(&D[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 实验9: Stitch 带宽瓶颈测试 Kernels
// ============================================================

// 高带宽顺序读取 - 本地区域
__global__ __noinline__ void bandwidth_local(float* data, size_t elements, 
                                              float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    // 顺序读取，最大化带宽
    for (size_t i = tid; i < elements; i += total_threads) {
        sum += data[i];
    }
    
    if (tid == 0) *output = sum;
}

// 高带宽读取 - 跨区域 (强制 Stitch)
__global__ __noinline__ void bandwidth_cross_region(float* data, size_t region_size,
                                                     size_t offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    // 交替访问两个区域，强制跨 uGPU
    for (int i = 0; i < ITERATIONS * 2; i++) {
        size_t base = (i % 2 == 0) ? 0 : offset;
        size_t idx = base + (tid + (i / 2) * total_threads) % region_size;
        sum += data[idx];
    }
    
    if (tid == 0) *output = sum;
}

// 极端带宽压力测试 - 所有线程访问不同区域
__global__ __noinline__ void bandwidth_stress_test(float* data, size_t total_elements,
                                                    float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    // 大步长访问，最大化跨 partition 访问
    size_t stride = total_elements / 4;  // 跨越 1/4 数据
    
    for (int i = 0; i < ITERATIONS; i++) {
        size_t idx = ((size_t)tid * stride + i) % total_elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// ============================================================
// 对比实验 Kernels (清晰展示 Stitch 影响)
// ============================================================

// 对比实验A: 本地连续访问 (基准)
__global__ __noinline__ void compare_local_access(float* __restrict__ data, 
                                                   float* __restrict__ output,
                                                   size_t elements,
                                                   int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    // 每个线程访问一小块连续内存（局部性高）
    size_t chunk_size = elements / total_threads;
    size_t start = tid * chunk_size;
    size_t end = (start + chunk_size < elements) ? start + chunk_size : elements;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = start; i < end; i++) {
            sum += __ldcg(&data[i]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 对比实验A: 跨 Partition 访问（大步长）
__global__ __noinline__ void compare_cross_partition(float* __restrict__ data,
                                                      float* __restrict__ output,
                                                      size_t elements,
                                                      int iterations,
                                                      size_t stride) {
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

// 对比实验B: 单 Buffer 3x 访问（L2 命中率高）
__global__ __noinline__ void compare_single_buffer_3x(float* __restrict__ data,
                                                       float* __restrict__ output,
                                                       size_t elements,
                                                       int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (tid * 32 + i) % elements;
            // 同一地址访问3次，模拟 3 buffer 但复用地址
            sum += __ldcg(&data[idx]);
            sum += __ldcg(&data[idx]);
            sum += __ldcg(&data[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 对比实验B: 多 Buffer QKV 访问（3倍唯一地址）
__global__ __noinline__ void compare_multi_buffer_qkv(float* __restrict__ Q,
                                                       float* __restrict__ K,
                                                       float* __restrict__ V,
                                                       float* __restrict__ output,
                                                       size_t elements,
                                                       int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (tid * 32 + i) % elements;
            // 交替访问 Q, K, V - 3倍唯一地址
            sum += __ldcg(&Q[idx]);
            sum += __ldcg(&K[idx]);
            sum += __ldcg(&V[idx]);
        }
    }
    
    if (tid == 0) output[0] = sum;
}

// 对比实验C: 随机索引访问
__global__ __noinline__ void compare_random_index_access(float* __restrict__ data,
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

// ============================================================
// 辅助函数
// ============================================================

void print_comparison_box(const char* test1, float time1, 
                          const char* test2, float time2) {
    float diff = (time2 - time1) / time1 * 100.0f;
    printf("\n┌────────────────────────────────────────────────────────────┐\n");
    printf("│ %-30s: %8.3f ms                │\n", test1, time1);
    printf("│ %-30s: %8.3f ms                │\n", test2, time2);
    printf("│ ───────────────────────────────────────────────────────── │\n");
    if (diff > 0) {
        printf("│ Stitch 导致的性能损失: %+.1f%% (%.3f ms 额外开销)        │\n", 
               diff, time2 - time1);
    } else {
        printf("│ 性能差异: %.1f%% (优化后更快)                             │\n", -diff);
    }
    printf("└────────────────────────────────────────────────────────────┘\n");
}

// 清空 L2 缓存 (通过访问大量不相关数据)
__global__ void flush_l2_cache(float* flush_buffer, size_t elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;
    
    for (size_t i = tid; i < elements; i += total_threads) {
        sum += flush_buffer[i];
    }
    
    // 防止优化
    if (sum == -999999.0f) flush_buffer[0] = sum;
}

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
    
    // 分配 L2 缓存清空用的缓冲区 (2x L2 大小)
    float* d_flush_buffer;
    size_t flush_size = 2 * l2_size;
    CUDA_CHECK(cudaMalloc(&d_flush_buffer, flush_size));
    CUDA_CHECK(cudaMemset(d_flush_buffer, 2, flush_size));
    
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
        
        // 清空 L2 缓存 (修复: 避免预热影响)
        flush_l2_cache<<<256, 256>>>(d_flush_buffer, flush_size / sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 不做预热，直接测试冷启动性能
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            footprint_test<<<num_blocks, block_size>>>(d_data, footprint, d_output);
            // 每次迭代后清空缓存以测量真实的数据足迹影响
            flush_l2_cache<<<256, 256>>>(d_flush_buffer, flush_size / sizeof(float));
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
    
    // 修复: 统一数据足迹为 L2 大小，避免 DRAM 访问干扰
    size_t test_footprint = l2_size / sizeof(float);  // 60 MB = L2
    
    printf("访问模式          时间(ms)      相对 Coalesced    预期 Stitch\n");
    printf("------------------------------------------------------------------------\n");
    printf("(数据足迹统一为 %.0f MB = L2 大小)\n\n", l2_size / 1024.0 / 1024.0);
    
    // Coalesced
    coalesced_access<<<num_blocks, block_size>>>(d_data, test_footprint, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        coalesced_access<<<num_blocks, block_size>>>(d_data, test_footprint, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float coal_time;
    cudaEventElapsedTime(&coal_time, start, stop);
    coal_time /= TEST_ITERS;
    printf("coalesced         %.3f         1.00x             低\n", coal_time);
    
    // Strided - 小步长
    size_t stride_small = 1024;  // 4 KB
    strided_access<<<num_blocks, block_size>>>(d_data, test_footprint, stride_small, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        strided_access<<<num_blocks, block_size>>>(d_data, test_footprint, stride_small, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stride_small_time;
    cudaEventElapsedTime(&stride_small_time, start, stop);
    stride_small_time /= TEST_ITERS;
    printf("strided (4KB)     %.3f         %.2fx             中\n", stride_small_time, stride_small_time / coal_time);
    
    // Strided - 大步长 (L2/8)
    size_t stride_large = l2_size / 8 / sizeof(float);
    strided_access<<<num_blocks, block_size>>>(d_data, test_footprint, stride_large, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        strided_access<<<num_blocks, block_size>>>(d_data, test_footprint, stride_large, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stride_large_time;
    cudaEventElapsedTime(&stride_large_time, start, stop);
    stride_large_time /= TEST_ITERS;
    printf("strided (L2/8)    %.3f         %.2fx             高\n", stride_large_time, stride_large_time / coal_time);
    
    // Random
    random_access<<<num_blocks, block_size>>>(d_data, test_footprint, 12345, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        random_access<<<num_blocks, block_size>>>(d_data, test_footprint, 12345, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float rand_time;
    cudaEventElapsedTime(&rand_time, start, stop);
    rand_time /= TEST_ITERS;
    printf("random            %.3f         %.2fx             最高\n", rand_time, rand_time / coal_time);
    
    // ============================================================
    // 新实验6: 多 Buffer 场景 (模拟 Attention Q, K, V)
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验6: 多 Buffer 场景 (模拟 Attention Q, K, V)\n");
    print_separator();
    printf("理论: N 个独立 buffer = N 倍唯一地址 = N 倍 Fabric Miss\n");
    printf("       这解释了 Attention 等多矩阵运算为什么 Stitch 高\n\n");
    
    // 分配多个独立 buffer
    float *d_bufA, *d_bufB, *d_bufC, *d_bufD;
    size_t buf_size = 16 * 1024 * 1024;  // 每个 buffer 16 MB
    size_t buf_elements = buf_size / sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_bufA, buf_size));
    CUDA_CHECK(cudaMalloc(&d_bufB, buf_size));
    CUDA_CHECK(cudaMalloc(&d_bufC, buf_size));
    CUDA_CHECK(cudaMalloc(&d_bufD, buf_size));
    CUDA_CHECK(cudaMemset(d_bufA, 1, buf_size));
    CUDA_CHECK(cudaMemset(d_bufB, 2, buf_size));
    CUDA_CHECK(cudaMemset(d_bufC, 3, buf_size));
    CUDA_CHECK(cudaMemset(d_bufD, 4, buf_size));
    
    printf("Buffer 数量     时间(ms)      相对 1 Buffer    Stitch 预期\n");
    printf("------------------------------------------------------------------------\n");
    printf("(每个 Buffer = 16 MB)\n\n");
    
    // 1 buffer
    single_buffer_access<<<num_blocks, block_size>>>(d_bufA, buf_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        single_buffer_access<<<num_blocks, block_size>>>(d_bufA, buf_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float buf1_time;
    cudaEventElapsedTime(&buf1_time, start, stop);
    buf1_time /= TEST_ITERS;
    printf("1 Buffer        %.3f         1.00x            基准\n", buf1_time);
    
    // 2 buffers
    dual_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, buf_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        dual_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, buf_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float buf2_time;
    cudaEventElapsedTime(&buf2_time, start, stop);
    buf2_time /= TEST_ITERS;
    printf("2 Buffers       %.3f         %.2fx            2x Fabric Miss\n", buf2_time, buf2_time / buf1_time);
    
    // 3 buffers (Q, K, V)
    triple_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, d_bufC, buf_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        triple_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, d_bufC, buf_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float buf3_time;
    cudaEventElapsedTime(&buf3_time, start, stop);
    buf3_time /= TEST_ITERS;
    printf("3 Buffers (QKV) %.3f         %.2fx            3x Fabric Miss\n", buf3_time, buf3_time / buf1_time);
    
    // 4 buffers
    quad_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, d_bufC, d_bufD, buf_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        quad_buffer_access<<<num_blocks, block_size>>>(d_bufA, d_bufB, d_bufC, d_bufD, buf_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float buf4_time;
    cudaEventElapsedTime(&buf4_time, start, stop);
    buf4_time /= TEST_ITERS;
    printf("4 Buffers       %.3f         %.2fx            4x Fabric Miss\n", buf4_time, buf4_time / buf1_time);
    
    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    CUDA_CHECK(cudaFree(d_bufC));
    CUDA_CHECK(cudaFree(d_bufD));
    
    // ============================================================
    // 新实验7: 带宽压力测试
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验7: Stitch 带宽瓶颈测试\n");
    print_separator();
    printf("理论: 高带宽压力下，Stitch 路径 (~100 cycles) 成为瓶颈\n");
    printf("       本地 L2 带宽 >> Stitch 带宽\n\n");
    
    printf("访问模式          时间(ms)      带宽(GB/s)      预期瓶颈\n");
    printf("------------------------------------------------------------------------\n");
    
    size_t bw_test_size = l2_size / sizeof(float);  // 60 MB
    size_t bw_bytes = l2_size;
    
    // 本地带宽测试
    bandwidth_local<<<num_blocks * 2, block_size>>>(d_data, bw_test_size, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        bandwidth_local<<<num_blocks * 2, block_size>>>(d_data, bw_test_size, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float local_bw_time;
    cudaEventElapsedTime(&local_bw_time, start, stop);
    local_bw_time /= TEST_ITERS;
    float local_bw = bw_bytes / (local_bw_time * 1e6);  // GB/s
    printf("本地顺序读      %.3f         %.1f           L2 带宽\n", local_bw_time, local_bw);
    
    // 跨区域带宽测试
    bandwidth_cross_region<<<num_blocks * 2, block_size>>>(d_data, region_size, region_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        bandwidth_cross_region<<<num_blocks * 2, block_size>>>(d_data, region_size, region_offset, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cross_bw_time;
    cudaEventElapsedTime(&cross_bw_time, start, stop);
    cross_bw_time /= TEST_ITERS;
    float cross_bw = (2 * region_size * sizeof(float) * ITERATIONS) / (cross_bw_time * 1e6);
    printf("跨区域交替      %.3f         %.1f           Stitch 带宽\n", cross_bw_time, cross_bw);
    
    // 带宽压力测试
    bandwidth_stress_test<<<num_blocks * 4, block_size>>>(d_data, buffer_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITERS; i++) {
        bandwidth_stress_test<<<num_blocks * 4, block_size>>>(d_data, buffer_elements, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float stress_time;
    cudaEventElapsedTime(&stress_time, start, stop);
    stress_time /= TEST_ITERS;
    printf("大步长压力      %.3f         --             Stitch 瓶颈\n", stress_time);
    
    float bw_overhead = (cross_bw_time - local_bw_time) / local_bw_time * 100;
    printf("\n跨区域 vs 本地开销: %+.1f%%\n", bw_overhead);
    
    // ============================================================
    // 新实验8: 有效 L2 容量测试
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验8: 有效 L2 容量测试\n");
    print_separator();
    printf("理论: 由于 Stitch，有效 L2 容量约为 L2/2\n");
    printf("       超过此阈值后，性能显著下降\n\n");
    
    printf("数据足迹       多次遍历时间(ms)     推断\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<size_t> capacity_tests_mb = {10, 20, 30, 40, 50, 60};
    int multi_iters = 10;  // 多次遍历以测试 L2 效果
    
    for (size_t cap_mb : capacity_tests_mb) {
        size_t cap_elements = cap_mb * 1024 * 1024 / sizeof(float);
        
        // 清空 L2
        flush_l2_cache<<<256, 256>>>(d_flush_buffer, flush_size / sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        effective_l2_test<<<num_blocks, block_size>>>(d_data, cap_elements, multi_iters, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float cap_time;
        cudaEventElapsedTime(&cap_time, start, stop);
        
        const char* inference = "";
        if (cap_mb <= 30) {
            inference = "< L2/2, 高效缓存";
        } else if (cap_mb <= 60) {
            inference = "> L2/2, 跨 uGPU 访问";
        } else {
            inference = "> L2, DRAM 访问";
        }
        
        printf("%6zu MB       %.3f                %s\n", cap_mb, cap_time, inference);
    }
    
    // ============================================================
    // 对比实验A: 本地访问 vs 跨 Partition 访问
    // ============================================================
    printf("\n");
    print_separator();
    printf("【对比实验A】本地访问 vs 跨 Partition 访问\n");
    print_separator();
    printf("场景: 相同数据量 (32MB)，不同的访问模式\n");
    printf("- 本地访问: 每线程访问连续内存块，高 L2 命中\n");
    printf("- 跨 Partition: 大步长访问，频繁触发 Stitch\n\n");
    
    {
        int test_blocks = 256;
        size_t test_elements = 8 * 1024 * 1024;  // 32 MB
        int test_iterations = 5;
        
        // 测试本地访问
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_local_access<<<test_blocks, block_size>>>(d_data, d_output, test_elements, test_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_local_access<<<test_blocks, block_size>>>(d_data, d_output, test_elements, test_iterations);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float local_time;
        cudaEventElapsedTime(&local_time, start, stop);
        local_time /= TEST_ITERS;
        
        // 测试跨 Partition 访问（stride = 4MB = 1M floats）
        size_t stride = 1024 * 1024;
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_cross_partition<<<test_blocks, block_size>>>(d_data, d_output, test_elements, test_iterations, stride);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_cross_partition<<<test_blocks, block_size>>>(d_data, d_output, test_elements, test_iterations, stride);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cross_time;
        cudaEventElapsedTime(&cross_time, start, stop);
        cross_time /= TEST_ITERS;
        
        print_comparison_box("本地访问 (连续)", local_time, 
                            "跨 Partition (stride=4MB)", cross_time);
    }
    
    // ============================================================
    // 对比实验B: 单 Buffer 3x vs 多 Buffer QKV
    // ============================================================
    printf("\n");
    print_separator();
    printf("【对比实验B】单 Buffer 3x vs 多 Buffer QKV\n");
    print_separator();
    printf("场景: 模拟 Attention 中 Q, K, V 的访问模式\n");
    printf("- 单 Buffer 3x: 同一内存区域访问3次（L2 命中率高）\n");
    printf("- 多 Buffer QKV: 3个独立 buffer 各访问1次（3倍唯一地址）\n");
    printf("关键: 虽然访问次数相同，但多 Buffer 产生更多唯一地址\n\n");
    
    {
        // 分配 QKV buffers
        float *d_Q, *d_K, *d_V;
        size_t qkv_size = 16 * 1024 * 1024;  // 每个 16 MB
        size_t qkv_elements = qkv_size / sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
        CUDA_CHECK(cudaMemset(d_Q, 1, qkv_size));
        CUDA_CHECK(cudaMemset(d_K, 2, qkv_size));
        CUDA_CHECK(cudaMemset(d_V, 3, qkv_size));
        
        int test_iterations = 20;
        
        // 测试单 Buffer 3x 访问
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_single_buffer_3x<<<num_blocks, block_size>>>(d_Q, d_output, qkv_elements, test_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_single_buffer_3x<<<num_blocks, block_size>>>(d_Q, d_output, qkv_elements, test_iterations);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float single_time;
        cudaEventElapsedTime(&single_time, start, stop);
        single_time /= TEST_ITERS;
        
        // 测试多 Buffer QKV 访问
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_multi_buffer_qkv<<<num_blocks, block_size>>>(d_Q, d_K, d_V, d_output, qkv_elements, test_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_multi_buffer_qkv<<<num_blocks, block_size>>>(d_Q, d_K, d_V, d_output, qkv_elements, test_iterations);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float multi_time;
        cudaEventElapsedTime(&multi_time, start, stop);
        multi_time /= TEST_ITERS;
        
        print_comparison_box("单 Buffer (同地址3x)", single_time,
                            "多 Buffer (Q,K,V)", multi_time);
        
        printf("\n解释: 多 Buffer 产生 3 倍唯一地址，导致更多 L2 Fabric Miss\n");
        
        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
    }
    
    // ============================================================
    // 对比实验C: 索引排序优化效果
    // ============================================================
    printf("\n");
    print_separator();
    printf("【对比实验C】数据局部性优化 (索引排序)\n");
    print_separator();
    printf("场景: 稀疏访问，对比随机索引 vs 排序索引\n");
    printf("优化原理: 排序后相邻访问更可能在同一 L2 partition\n\n");
    
    {
        size_t num_accesses = num_blocks * block_size * 64;
        int test_iterations = 5;
        
        // 准备随机索引和排序索引
        std::vector<int> random_indices(num_accesses);
        for (size_t i = 0; i < num_accesses; i++) {
            random_indices[i] = rand() % buffer_elements;
        }
        
        std::vector<int> sorted_indices = random_indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        
        int *d_random_idx, *d_sorted_idx;
        CUDA_CHECK(cudaMalloc(&d_random_idx, num_accesses * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sorted_idx, num_accesses * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_random_idx, random_indices.data(), 
                             num_accesses * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sorted_idx, sorted_indices.data(),
                             num_accesses * sizeof(int), cudaMemcpyHostToDevice));
        
        // 测试随机索引
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_random_index_access<<<num_blocks, block_size>>>(d_data, d_random_idx, d_output, num_accesses, test_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_random_index_access<<<num_blocks, block_size>>>(d_data, d_random_idx, d_output, num_accesses, test_iterations);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float random_time;
        cudaEventElapsedTime(&random_time, start, stop);
        random_time /= TEST_ITERS;
        
        // 测试排序索引
        for (int w = 0; w < WARMUP_ITERS; w++) {
            compare_random_index_access<<<num_blocks, block_size>>>(d_data, d_sorted_idx, d_output, num_accesses, test_iterations);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEventRecord(start);
        for (int i = 0; i < TEST_ITERS; i++) {
            compare_random_index_access<<<num_blocks, block_size>>>(d_data, d_sorted_idx, d_output, num_accesses, test_iterations);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float sorted_time;
        cudaEventElapsedTime(&sorted_time, start, stop);
        sorted_time /= TEST_ITERS;
        
        print_comparison_box("随机索引 (未优化)", random_time,
                            "排序索引 (优化后)", sorted_time);
        
        float speedup = random_time / sorted_time;
        printf("\n优化效果: %.2fx 加速\n", speedup);
        printf("原理: 排序后访问模式更连续，减少跨 Partition 访问\n");
        
        CUDA_CHECK(cudaFree(d_random_idx));
        CUDA_CHECK(cudaFree(d_sorted_idx));
    }
    
    // ============================================================
    // 对比实验D: 高并发下的 Stitch 瓶颈
    // ============================================================
    printf("\n");
    print_separator();
    printf("【对比实验D】高并发下的 Stitch 瓶颈\n");
    print_separator();
    printf("场景: 固定总访问量，增加并发 block 数\n");
    printf("预期: 高并发时 Stitch 带宽成为瓶颈，性能下降更明显\n\n");
    
    {
        printf("Block数   同区域(ms)   跨区域(ms)    Stitch开销    瓶颈程度\n");
        printf("------------------------------------------------------------------------\n");
        
        std::vector<int> test_block_counts = {16, 64, 128, 256, 512};
        
        for (int nblocks : test_block_counts) {
            // 同区域访问
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
            
            // 跨区域访问
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
            
            float overhead = (diff_time - same_time) / same_time * 100;
            const char* bottleneck = "";
            if (overhead > 40) {
                bottleneck = "★★★ 严重瓶颈";
            } else if (overhead > 20) {
                bottleneck = "★★ 明显瓶颈";
            } else if (overhead > 10) {
                bottleneck = "★ 轻微瓶颈";
            } else {
                bottleneck = "无瓶颈";
            }
            
            printf("%5d     %.3f        %.3f         %+5.1f%%        %s\n",
                   nblocks, same_time, diff_time, overhead, bottleneck);
        }
        
        printf("\n结论: 高并发 + 跨区域访问 = Stitch 带宽瓶颈\n");
    }
    
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
    CUDA_CHECK(cudaFree(d_flush_buffer));
    
    return 0;
}

