#include <cuda_runtime.h>

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

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr int kNumWarps = 1024;
constexpr int kTotalThreads = kThreadsPerWarp * kNumWarps;
constexpr int kBlockSize = 256;

// 64 bytes = 2 sectors (每个sector 32 bytes)
constexpr int kStrideBytes = 64;
constexpr int kStrideElements = kStrideBytes / sizeof(float);  // 16 floats

// 32 bytes = 1 sector = 8 floats
constexpr int kSectorBytes = 32;
constexpr int kSectorElements = kSectorBytes / sizeof(float);  // 8 floats

// =============================================================================
// 实验1: 单轮访问，验证DRAM load粒度是64字节
// 每个线程加载4字节，stride为64字节
// 预期：DRAM加载量 = kTotalThreads * 64 bytes
// =============================================================================
__global__ void single_pass_stride64_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }
    // 每个线程访问4字节，stride为64字节
    // thread 0 -> offset 0
    // thread 1 -> offset 64
    // thread 2 -> offset 128
    // ...
    output[tid] = input[static_cast<size_t>(tid) * kStrideElements];
}

// =============================================================================
// 实验2: 两轮访问，验证L2 cache是否缓存整个64字节块
// 第一轮：访问每个64字节块的第一个sector（offset 0）
// 第二轮：访问每个64字节块的第二个sector（offset 32字节）
// 预期：如果L2缓存整个64字节，第二轮应该全部L2 hit，DRAM无额外加载
// =============================================================================
__global__ void two_pass_stride64_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }

    // 计算每个线程对应的64字节块的起始位置
    const size_t base_offset = static_cast<size_t>(tid) * kStrideElements;

    // 第一轮：访问64字节块的第一个sector（offset 0）
    // 这会触发DRAM加载整个64字节块到L2 cache
    float val1 = input[base_offset];

    // 第二轮：访问64字节块的第二个sector（offset 32字节 = 8 floats）
    // 如果L2 cache已经缓存了整个64字节块，这次访问应该是L2 hit
    float val2 = input[base_offset + kSectorElements];

    output[tid] = val1 + val2;
}

// =============================================================================
// 对照实验: 只访问第二个sector，验证它确实需要DRAM加载
// =============================================================================
__global__ void single_pass_sector2_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }
    // 只访问每个64字节块的第二个sector
    const size_t base_offset = static_cast<size_t>(tid) * kStrideElements;
    output[tid] = input[base_offset + kSectorElements];
}

// =============================================================================
// 实验3: 两轮访问 + block内同步
// 在两轮访问之间加入__syncthreads()，减少block内的cache竞争
// =============================================================================
__global__ void two_pass_with_block_sync_kernel(const float* __restrict__ input,
                                                 float* __restrict__ output,
                                                 int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }

    const size_t base_offset = static_cast<size_t>(tid) * kStrideElements;

    // 第一轮：访问64字节块的第一个sector
    float val1 = input[base_offset];

    // Block内同步：确保block内所有线程完成第一轮
    __syncthreads();

    // 第二轮：访问64字节块的第二个sector
    float val2 = input[base_offset + kSectorElements];

    output[tid] = val1 + val2;
}

// =============================================================================
// 实验4a: 只访问第一个sector（用于两个kernel分离实验的第一轮）
// =============================================================================
__global__ void first_pass_sector1_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }
    const size_t base_offset = static_cast<size_t>(tid) * kStrideElements;
    output[tid] = input[base_offset];
}

// =============================================================================
// 实验4b: 只访问第二个sector（用于两个kernel分离实验的第二轮）
// 注意：这个kernel在第一个kernel之后立即运行，不flush L2 cache
// =============================================================================
__global__ void second_pass_sector2_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }
    const size_t base_offset = static_cast<size_t>(tid) * kStrideElements;
    // 读取第二个sector，加到之前的值上
    output[tid] += input[base_offset + kSectorElements];
}

}  // namespace

void run_experiment(const char* name,
                    void (*kernel)(const float*, float*, int),
                    const float* d_input,
                    float* d_output,
                    std::vector<float>& h_output) {
    printf("\n========================================\n");
    printf("Running: %s\n", name);
    printf("========================================\n");

    const dim3 block_dim(kBlockSize);
    const dim3 grid_dim((kTotalThreads + block_dim.x - 1) / block_dim.x);

    // 清空output
    CUDA_CHECK(cudaMemset(d_output, 0, kTotalThreads * sizeof(float)));

    // 运行kernel
    kernel<<<grid_dim, block_dim>>>(d_input, d_output, kTotalThreads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 计算checksum
    double checksum = 0.0;
    for (float value : h_output) {
        checksum += static_cast<double>(value);
    }

    printf("Threads: %d, Stride: %d bytes, Checksum: %.3f\n",
           kTotalThreads, kStrideBytes, checksum);
}

int main() {
    printf("=== DRAM Sector and L2 Cache Experiment ===\n");
    printf("Configuration:\n");
    printf("  Total threads: %d\n", kTotalThreads);
    printf("  Stride: %d bytes (%d floats)\n", kStrideBytes, kStrideElements);
    printf("  Sector size: %d bytes (%d floats)\n", kSectorBytes, kSectorElements);
    printf("\n");

    // 分配足够大的input buffer
    // 每个线程访问一个64字节块，需要 kTotalThreads * 64 bytes
    const size_t input_elements =
        static_cast<size_t>(kTotalThreads) * kStrideElements;
    const size_t input_bytes = input_elements * sizeof(float);

    printf("Input buffer: %zu elements, %zu bytes (%.2f MB)\n",
           input_elements, input_bytes, input_bytes / (1024.0 * 1024.0));

    // 初始化host数据
    std::vector<float> h_input(input_elements);
    for (size_t i = 0; i < input_elements; ++i) {
        h_input[i] = static_cast<float>(i % 1024) * 0.5f;
    }
    std::vector<float> h_output(kTotalThreads, 0.0f);

    // 分配device内存
    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, kTotalThreads * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes,
                          cudaMemcpyHostToDevice));

    // 运行实验1: 单轮访问，stride 64字节
    // 用ncu观察：dram__bytes_read.sum 应该等于 kTotalThreads * 64
    run_experiment("Experiment 1: Single pass, stride 64 bytes",
                   single_pass_stride64_kernel, d_input, d_output, h_output);

    // 清空L2 cache（通过访问大量不相关数据）
    // 分配一个大buffer来flush L2 cache
    const size_t flush_size = 64 * 1024 * 1024;  // 64MB，大于L2 cache
    float* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_size));
    CUDA_CHECK(cudaMemset(d_flush, 0, flush_size));
    CUDA_CHECK(cudaFree(d_flush));

    // 运行实验2: 两轮访问
    // 用ncu观察：
    //   - dram__bytes_read.sum 应该仍然约等于 kTotalThreads * 64（因为第二轮L2 hit）
    //   - l2_cache_hit_rate 应该约为 50%（第一轮全miss，第二轮全hit）
    run_experiment("Experiment 2: Two passes (sector1 then sector2)",
                   two_pass_stride64_kernel, d_input, d_output, h_output);

    // 清空L2 cache
    CUDA_CHECK(cudaMalloc(&d_flush, flush_size));
    CUDA_CHECK(cudaMemset(d_flush, 0, flush_size));
    CUDA_CHECK(cudaFree(d_flush));

    // 对照实验: 只访问第二个sector
    // 用ncu观察：dram__bytes_read.sum 应该等于 kTotalThreads * 64
    run_experiment("Control: Single pass, only sector2",
                   single_pass_sector2_kernel, d_input, d_output, h_output);

    // 清空L2 cache
    CUDA_CHECK(cudaMalloc(&d_flush, flush_size));
    CUDA_CHECK(cudaMemset(d_flush, 0, flush_size));
    CUDA_CHECK(cudaFree(d_flush));

    // 实验3: 两轮访问 + block内同步
    // 用ncu观察：L2 hit rate应该比实验2更高（更接近50%）
    run_experiment("Experiment 3: Two passes with __syncthreads()",
                   two_pass_with_block_sync_kernel, d_input, d_output, h_output);

    // 清空L2 cache
    CUDA_CHECK(cudaMalloc(&d_flush, flush_size));
    CUDA_CHECK(cudaMemset(d_flush, 0, flush_size));
    CUDA_CHECK(cudaFree(d_flush));

    // 实验4: 两个独立kernel（全局同步）
    // 这是最接近理论预期的实验，第二个kernel应该接近100% L2 hit
    printf("\n========================================\n");
    printf("Running: Experiment 4: Two separate kernels (global sync)\n");
    printf("========================================\n");

    const dim3 block_dim(kBlockSize);
    const dim3 grid_dim((kTotalThreads + block_dim.x - 1) / block_dim.x);

    CUDA_CHECK(cudaMemset(d_output, 0, kTotalThreads * sizeof(float)));

    // 第一个kernel: 访问sector1
    first_pass_sector1_kernel<<<grid_dim, block_dim>>>(d_input, d_output, kTotalThreads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // 全局同步！

    // 第二个kernel: 访问sector2（不flush L2 cache）
    second_pass_sector2_kernel<<<grid_dim, block_dim>>>(d_input, d_output, kTotalThreads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float value : h_output) {
        checksum += static_cast<double>(value);
    }
    printf("Threads: %d, Stride: %d bytes, Checksum: %.3f\n",
           kTotalThreads, kStrideBytes, checksum);

    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaDeviceReset());

    printf("\n=== Experiment Complete ===\n");
    printf("\nTo profile with NCU, run:\n");
    printf("  ncu --set full ./dram_sector_cache_benchmark\n");
    printf("\nExpected results:\n");
    printf("  Exp1: L2 hit ~18%% (baseline from output writes)\n");
    printf("  Exp2: L2 hit ~40%% (some second-pass hits)\n");
    printf("  Exp3: L2 hit higher than Exp2 (block sync helps)\n");
    printf("  Exp4: second_pass_sector2_kernel should have ~100%% L2 hit!\n");
    printf("        (because first kernel already loaded all 64-byte blocks)\n");

    return EXIT_SUCCESS;
}
