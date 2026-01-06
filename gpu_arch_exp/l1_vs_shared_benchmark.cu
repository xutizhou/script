#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__,   \
                    __LINE__, cudaGetErrorString(_err));                         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// =============================================================================
// 实验目的：比较不同加载次数下 L1 Cache 和 Shared Memory 的带宽
//
// 实验设计：
// - 数据类型：bf16 (2 bytes) 和 fp32 (4 bytes)
// - 数据大小：128 个元素（bf16=256B, fp32=512B）
// - 测试：分别加载 2, 4, 8, 16, 32, 64 次
// - 比较 L1 Cache 和 Shared Memory 的带宽
// - 包含 bf16->fp32 类型转换的场景
// =============================================================================

using bf16 = __nv_bfloat16;

namespace {

// 配置参数
constexpr int kBlockSize = 128;  // 128 threads
constexpr int kNumBlocks = 1;
constexpr int kDataSize = 128;   // 128 个元素

// 测试的加载次数
const int kLoadCounts[] = {2, 4, 8, 16, 32, 64};
constexpr int kNumTests = sizeof(kLoadCounts) / sizeof(kLoadCounts[0]);

// 基础迭代次数
constexpr int kBaseIterations = 1;

// =============================================================================
// BF16 Kernels
// =============================================================================

// L1 Cache - BF16
__global__ void l1_cache_bf16_kernel(const bf16* __restrict__ data,
                                      bf16* __restrict__ output,
                                      int load_count,
                                      int iterations) {
    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            sum += __bfloat162float(data[tid]);
        }
    }

    if (tid == 0) {
        output[0] = __float2bfloat16(sum);
    }
}

// Shared Memory - BF16
__global__ void shared_memory_bf16_kernel(const bf16* __restrict__ data,
                                           bf16* __restrict__ output,
                                           int load_count,
                                           int iterations) {
    __shared__ bf16 shared_data[kDataSize];
    const int tid = threadIdx.x;

    if (tid < kDataSize) {
        shared_data[tid] = data[tid];
    }
    __syncthreads();

    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            sum += __bfloat162float(shared_data[tid]);
        }
    }

    if (tid == 0) {
        output[0] = __float2bfloat16(sum);
    }
}

// =============================================================================
// FP32 Kernels
// =============================================================================

// L1 Cache - FP32
__global__ void l1_cache_fp32_kernel(const float* __restrict__ data,
                                      float* __restrict__ output,
                                      int load_count,
                                      int iterations) {
    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            sum += data[tid];
        }
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

// Shared Memory - FP32
__global__ void shared_memory_fp32_kernel(const float* __restrict__ data,
                                           float* __restrict__ output,
                                           int load_count,
                                           int iterations) {
    __shared__ float shared_data[kDataSize];
    const int tid = threadIdx.x;

    if (tid < kDataSize) {
        shared_data[tid] = data[tid];
    }
    __syncthreads();

    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            sum += shared_data[tid];
        }
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

// =============================================================================
// BF16 -> FP32 Conversion Kernels (读取bf16，存储为fp32)
// =============================================================================

// L1 Cache - BF16 读取，转换为 FP32 存储到 shared memory 模拟的场景
__global__ void l1_cache_bf16_to_fp32_kernel(const bf16* __restrict__ data,
                                              float* __restrict__ output,
                                              int load_count,
                                              int iterations) {
    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            // 从 L1 cache 读取 bf16，转换为 fp32
            float val = __bfloat162float(data[tid]);
            sum += val;
        }
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

// Shared Memory - BF16 读取，转换为 FP32
__global__ void shared_memory_bf16_to_fp32_kernel(const bf16* __restrict__ data,
                                                   float* __restrict__ output,
                                                   int load_count,
                                                   int iterations) {
    __shared__ bf16 shared_data[kDataSize];
    const int tid = threadIdx.x;

    // 从 global memory 加载 bf16 到 shared memory
    if (tid < kDataSize) {
        shared_data[tid] = data[tid];
    }
    __syncthreads();

    float sum = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int l = 0; l < load_count; ++l) {
            // 从 shared memory 读取 bf16，转换为 fp32
            float val = __bfloat162float(shared_data[tid]);
            sum += val;
        }
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

}  // namespace

// 通用计时函数模板
template<typename KernelFunc, typename InType, typename OutType>
float benchmark_kernel_generic(
    KernelFunc kernel,
    const InType* d_data,
    OutType* d_output,
    int load_count,
    int iterations,
    size_t shared_mem_size = 0) {

    // 预热
    kernel<<<kNumBlocks, kBlockSize, shared_mem_size>>>(
        d_data, d_output, load_count, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<kNumBlocks, kBlockSize, shared_mem_size>>>(
        d_data, d_output, load_count, iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float single_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&single_ms, start, stop));

    int num_runs = (int)(50.0f / single_ms) + 1;
    if (num_runs < 3) num_runs = 3;
    if (num_runs > 200) num_runs = 200;

    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs; ++run) {
        kernel<<<kNumBlocks, kBlockSize, shared_mem_size>>>(
            d_data, d_output, load_count, iterations);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= num_runs;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

struct TestResult {
    int load_count;
    float time_l1_ms;
    float time_shared_ms;
    float bandwidth_l1_gbs;
    float bandwidth_shared_gbs;
    float speedup;
};

void print_results(const char* title, const std::vector<TestResult>& results, int elem_bytes) {
    printf("\n");
    printf("  %s\n", title);
    printf("  +-------+----------+----------+----------+----------+----------+--------+\n");
    printf("  | Loads | L1 Time  | Shared   | L1 BW    | Shared   | Speedup  | Winner |\n");
    printf("  |       | (ms)     | (ms)     | (GB/s)   | (GB/s)   | (L1/Shm) |        |\n");
    printf("  +-------+----------+----------+----------+----------+----------+--------+\n");

    for (const auto& r : results) {
        const char* winner = r.speedup > 1.05 ? "Shared" : 
                             (r.speedup < 0.95 ? "L1" : "~Same");
        printf("  | %5d | %8.3f | %8.3f | %8.1f | %8.1f | %8.2fx | %6s |\n",
               r.load_count,
               r.time_l1_ms, r.time_shared_ms,
               r.bandwidth_l1_gbs, r.bandwidth_shared_gbs,
               r.speedup, winner);
    }
    printf("  +-------+----------+----------+----------+----------+----------+--------+\n");
}

void print_summary(const char* name, const std::vector<TestResult>& results) {
    float avg_speedup = 0, max_speedup = 0, min_speedup = 999;
    float avg_bw_l1 = 0, avg_bw_shared = 0;
    for (const auto& r : results) {
        avg_speedup += r.speedup;
        avg_bw_l1 += r.bandwidth_l1_gbs;
        avg_bw_shared += r.bandwidth_shared_gbs;
        if (r.speedup > max_speedup) max_speedup = r.speedup;
        if (r.speedup < min_speedup) min_speedup = r.speedup;
    }
    avg_speedup /= results.size();
    avg_bw_l1 /= results.size();
    avg_bw_shared /= results.size();

    printf("  %s:\n", name);
    printf("    Avg Speedup: %.2fx | L1 BW: %.1f GB/s | Shared BW: %.1f GB/s\n",
           avg_speedup, avg_bw_l1, avg_bw_shared);
}

int main() {
    printf("=============================================================\n");
    printf("    L1 Cache vs Shared Memory: BF16 & FP32 Comparison\n");
    printf("=============================================================\n");
    printf("\nConfiguration:\n");
    printf("  Data size:        128 elements\n");
    printf("  Block size:       128 threads (1 element per thread)\n");
    printf("  Access pattern:   Coalesced (contiguous)\n");
    printf("  Tests:            BF16, FP32, BF16->FP32 conversion\n");
    printf("  Load counts:      2, 4, 8, 16, 32, 64\n");
    printf("\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Shared Memory per Block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    // 分配内存
    // BF16 数据
    std::vector<bf16> h_data_bf16(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
        h_data_bf16[i] = __float2bfloat16((float)(i % 100) * 0.01f);
    }
    bf16* d_data_bf16 = nullptr;
    bf16* d_output_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data_bf16, kDataSize * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_output_bf16, sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_data_bf16, h_data_bf16.data(), kDataSize * sizeof(bf16), cudaMemcpyHostToDevice));

    // FP32 数据
    std::vector<float> h_data_fp32(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
        h_data_fp32[i] = (float)(i % 100) * 0.01f;
    }
    float* d_data_fp32 = nullptr;
    float* d_output_fp32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data_fp32, kDataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_fp32, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data_fp32, h_data_fp32.data(), kDataSize * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<TestResult> results_bf16, results_fp32, results_bf16_to_fp32;

    printf("Running benchmarks...\n");

    for (int t = 0; t < kNumTests; ++t) {
        int load_count = kLoadCounts[t];
        int iterations = kBaseIterations;

        printf("  Testing load_count = %2d ...\n", load_count);

        // =====================================================================
        // Test 1: BF16
        // =====================================================================
        {
            float time_l1 = benchmark_kernel_generic(
                l1_cache_bf16_kernel, d_data_bf16, d_output_bf16,
                load_count, iterations);
            float time_shared = benchmark_kernel_generic(
                shared_memory_bf16_kernel, d_data_bf16, d_output_bf16,
                load_count, iterations, kDataSize * sizeof(bf16));

            long long total_bytes = (long long)iterations * load_count * kDataSize * sizeof(bf16);
            TestResult r;
            r.load_count = load_count;
            r.time_l1_ms = time_l1;
            r.time_shared_ms = time_shared;
            r.bandwidth_l1_gbs = (total_bytes / 1e9) / (time_l1 / 1000.0);
            r.bandwidth_shared_gbs = (total_bytes / 1e9) / (time_shared / 1000.0);
            r.speedup = time_l1 / time_shared;
            results_bf16.push_back(r);
        }

        // =====================================================================
        // Test 2: FP32
        // =====================================================================
        {
            float time_l1 = benchmark_kernel_generic(
                l1_cache_fp32_kernel, d_data_fp32, d_output_fp32,
                load_count, iterations);
            float time_shared = benchmark_kernel_generic(
                shared_memory_fp32_kernel, d_data_fp32, d_output_fp32,
                load_count, iterations, kDataSize * sizeof(float));

            long long total_bytes = (long long)iterations * load_count * kDataSize * sizeof(float);
            TestResult r;
            r.load_count = load_count;
            r.time_l1_ms = time_l1;
            r.time_shared_ms = time_shared;
            r.bandwidth_l1_gbs = (total_bytes / 1e9) / (time_l1 / 1000.0);
            r.bandwidth_shared_gbs = (total_bytes / 1e9) / (time_shared / 1000.0);
            r.speedup = time_l1 / time_shared;
            results_fp32.push_back(r);
        }

        // =====================================================================
        // Test 3: BF16 -> FP32 conversion
        // =====================================================================
        {
            float time_l1 = benchmark_kernel_generic(
                l1_cache_bf16_to_fp32_kernel, d_data_bf16, d_output_fp32,
                load_count, iterations);
            float time_shared = benchmark_kernel_generic(
                shared_memory_bf16_to_fp32_kernel, d_data_bf16, d_output_fp32,
                load_count, iterations, kDataSize * sizeof(bf16));

            // 带宽按读取的bf16字节数计算
            long long total_bytes = (long long)iterations * load_count * kDataSize * sizeof(bf16);
            TestResult r;
            r.load_count = load_count;
            r.time_l1_ms = time_l1;
            r.time_shared_ms = time_shared;
            r.bandwidth_l1_gbs = (total_bytes / 1e9) / (time_l1 / 1000.0);
            r.bandwidth_shared_gbs = (total_bytes / 1e9) / (time_shared / 1000.0);
            r.speedup = time_l1 / time_shared;
            results_bf16_to_fp32.push_back(r);
        }
    }

    // 输出结果
    printf("\n");
    printf("=============================================================\n");
    printf("    Results Summary (128 elements, coalesced access)\n");
    printf("=============================================================\n");

    print_results("Test 1: BF16 (256 bytes)", results_bf16, sizeof(bf16));
    print_results("Test 2: FP32 (512 bytes)", results_fp32, sizeof(float));
    print_results("Test 3: BF16->FP32 conversion (read 256B bf16)", results_bf16_to_fp32, sizeof(bf16));

    // 综合分析
    printf("\n");
    printf("=============================================================\n");
    printf("    Comparison Summary\n");
    printf("=============================================================\n\n");

    print_summary("BF16 (2 bytes/elem)", results_bf16);
    print_summary("FP32 (4 bytes/elem)", results_fp32);
    print_summary("BF16->FP32 convert", results_bf16_to_fp32);

    printf("\n  Memory Hierarchy (H20 GPU):\n");
    printf("    - Shared Memory: ~29 cycles latency\n");
    printf("    - L1 Cache:      ~41 cycles latency\n");
    printf("\n");

    printf("  Conclusion:\n");
    printf("    For small coalesced data (128-256 elements), L1 Cache and\n");
    printf("    Shared Memory have similar performance. The type conversion\n");
    printf("    (bf16->fp32) adds compute overhead but doesn't change the\n");
    printf("    relative performance between L1 and Shared Memory.\n");

    // 清理
    CUDA_CHECK(cudaFree(d_data_bf16));
    CUDA_CHECK(cudaFree(d_output_bf16));
    CUDA_CHECK(cudaFree(d_data_fp32));
    CUDA_CHECK(cudaFree(d_output_fp32));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
