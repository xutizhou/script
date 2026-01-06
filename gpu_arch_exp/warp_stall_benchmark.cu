/**
 * Warp Stall Analysis Benchmark
 * 
 * 用于NCU Warp State Statistics分析的各种kernel
 * 
 * 编译: nvcc -O3 -arch=sm_80 warp_stall_benchmark.cu -o warp_stall_benchmark -lcublas
 * 
 * NCU运行 (收集warp stall统计):
 * ncu --set full -o warp_stall_report ./warp_stall_benchmark
 * 
 * 或者只收集warp state相关指标:
 * ncu --metrics smsp__warp_issue_stalled_barrier_per_issue_active.pct,\
 * smsp__warp_issue_stalled_drain_per_issue_active.pct,\
 * smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct,\
 * smsp__warp_issue_stalled_math_pipe_throttle_per_issue_active.pct,\
 * smsp__warp_issue_stalled_mio_throttle_per_issue_active.pct,\
 * smsp__warp_issue_stalled_short_scoreboard_per_issue_active.pct,\
 * smsp__warp_issue_stalled_lg_throttle_per_issue_active.pct,\
 * smsp__warp_issue_stalled_no_instruction_per_issue_active.pct,\
 * smsp__warp_issue_stalled_not_selected_per_issue_active.pct,\
 * smsp__warp_issue_stalled_membar_per_issue_active.pct,\
 * smsp__warp_issue_stalled_wait_per_issue_active.pct,\
 * smsp__warp_issue_stalled_sleeping_per_issue_active.pct,\
 * smsp__warp_issue_stalled_tex_throttle_per_issue_active.pct,\
 * smsp__warp_issue_stalled_imc_miss_per_issue_active.pct,\
 * smsp__warp_issue_stalled_misc_per_issue_active.pct \
 * -o warp_stall_report ./warp_stall_benchmark
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Kernel 1: 纯读 (Pure Read) - 应该看到 long_scoreboard stall (等L1TEX)
// ============================================================================
__global__ void kernel_pure_read(const float* __restrict__ input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }
    
    // 只写一次避免写的影响
    if (idx == 0) {
        output[0] = sum;
    }
}

// ============================================================================
// Kernel 2: 纯写 (Pure Write) - 可能看到 mio_throttle 或 lg_throttle
// ============================================================================
__global__ void kernel_pure_write(float* output, int N, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        output[i] = value;
    }
}

// ============================================================================
// Kernel 3: Vector Add (读+写平衡) - 典型的memory bound kernel
// ============================================================================
__global__ void kernel_vector_add(const float* __restrict__ a, 
                                   const float* __restrict__ b, 
                                   float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// Kernel 4: 纯计算 (Pure Compute) - 应该看到 math_pipe_throttle 或 not_selected
// ============================================================================
__global__ void kernel_pure_compute(float* output, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (float)idx * 0.1f;
    
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        val = val * 1.001f + 0.001f;
        val = sqrtf(val * val + 1.0f);
        val = sinf(val) * cosf(val);
    }
    
    if (idx < N) {
        output[idx] = val;
    }
}

// ============================================================================
// Kernel 5: 带Barrier的kernel - 应该看到 barrier stall
// ============================================================================
__global__ void kernel_with_barrier(float* data, int N) {
    __shared__ float smem[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load to shared memory
    if (idx < N) {
        smem[tid] = data[idx];
    }
    __syncthreads();  // Barrier 1
    
    // Some computation
    if (tid > 0 && tid < 255 && idx < N) {
        smem[tid] = smem[tid-1] + smem[tid] + smem[tid+1];
    }
    __syncthreads();  // Barrier 2
    
    // Write back
    if (idx < N) {
        data[idx] = smem[tid];
    }
}

// ============================================================================
// Kernel 6: 随机访问 (Random Access) - 可能看到更多 long_scoreboard
// ============================================================================
__global__ void kernel_random_access(const float* __restrict__ input,
                                      const int* __restrict__ indices,
                                      float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int random_idx = indices[idx];
        output[idx] = input[random_idx];
    }
}

// ============================================================================
// Kernel 7: 原子操作 (Atomic) - 可能看到 wait stall
// ============================================================================
__global__ void kernel_atomic_add(int* counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        atomicAdd(counter, 1);
    }
}

// ============================================================================
// Kernel 8: Shared Memory Intensive - 可能看到 short_scoreboard
// ============================================================================
__global__ void kernel_smem_intensive(float* output, int N) {
    __shared__ float smem[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Fill shared memory
    smem[tid] = (float)tid;
    smem[tid + 256] = (float)(tid + 256);
    smem[tid + 512] = (float)(tid + 512);
    smem[tid + 768] = (float)(tid + 768);
    __syncthreads();
    
    // Intensive shared memory access
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 1024; i += 4) {
        sum += smem[i] + smem[i+1] + smem[i+2] + smem[i+3];
    }
    
    if (idx < N) {
        output[idx] = sum;
    }
}

// ============================================================================
// Kernel 9: 混合 (Mixed) - 计算+访存交替
// ============================================================================
__global__ void kernel_mixed(const float* __restrict__ input, 
                              float* output, int N, int compute_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        
        // Compute phase
        #pragma unroll 1
        for (int i = 0; i < compute_iters; i++) {
            val = val * 1.001f + 0.001f;
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// 工具函数
// ============================================================================
void init_random(float* data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

void init_random_indices(int* indices, int N, int max_val) {
    for (int i = 0; i < N; i++) {
        indices[i] = rand() % max_val;
    }
}

void run_benchmark(const char* name, void (*func)(), int warmup, int repeat) {
    printf("\n=== %s ===\n", name);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Time: %.3f ms (avg: %.3f ms)\n", ms, ms / repeat);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // 配置
    const int N = 64 * 1024 * 1024;  // 64M elements
    const int GEMM_M = 4096;
    const int GEMM_N = 4096;
    const int GEMM_K = 4096;
    
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    printf("Warp Stall Analysis Benchmark\n");
    printf("N = %d, Grid = %d, Block = %d\n", N, grid_size, block_size);
    printf("GEMM: M=%d, N=%d, K=%d\n", GEMM_M, GEMM_N, GEMM_K);
    
    // 设备信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);
    
    // 分配内存
    float *h_a, *h_b;
    float *d_a, *d_b, *d_c;
    int *h_indices, *d_indices;
    int *d_counter;
    
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_indices = (int*)malloc(N * sizeof(int));
    
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    
    // 初始化数据
    init_random(h_a, N);
    init_random(h_b, N);
    init_random_indices(h_indices, N, N);
    
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    
    // GEMM矩阵
    float *d_gemm_a, *d_gemm_b, *d_gemm_c;
    CHECK_CUDA(cudaMalloc(&d_gemm_a, GEMM_M * GEMM_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gemm_b, GEMM_K * GEMM_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gemm_c, GEMM_M * GEMM_N * sizeof(float)));
    
    // 用随机数据初始化GEMM矩阵
    float* h_gemm = (float*)malloc(GEMM_M * GEMM_K * sizeof(float));
    init_random(h_gemm, GEMM_M * GEMM_K);
    CHECK_CUDA(cudaMemcpy(d_gemm_a, h_gemm, GEMM_M * GEMM_K * sizeof(float), cudaMemcpyHostToDevice));
    init_random(h_gemm, GEMM_K * GEMM_N);
    CHECK_CUDA(cudaMemcpy(d_gemm_b, h_gemm, GEMM_K * GEMM_N * sizeof(float), cudaMemcpyHostToDevice));
    free(h_gemm);
    
    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    printf("Running kernels for NCU profiling...\n");
    printf("Use: ncu --set full -o warp_stall_report ./warp_stall_benchmark\n\n");
    
    // ========================================================================
    // 运行各个kernel
    // ========================================================================
    
    // 1. Pure Read
    printf("1. Kernel: Pure Read (expect long_scoreboard stall)\n");
    for (int i = 0; i < 3; i++) {
        kernel_pure_read<<<grid_size, block_size>>>(d_a, d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 2. Pure Write
    printf("2. Kernel: Pure Write (expect mio_throttle/lg_throttle)\n");
    for (int i = 0; i < 3; i++) {
        kernel_pure_write<<<grid_size, block_size>>>(d_c, N, 1.0f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 3. Vector Add
    printf("3. Kernel: Vector Add (memory bound, balanced R/W)\n");
    for (int i = 0; i < 3; i++) {
        kernel_vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 4. Pure Compute
    printf("4. Kernel: Pure Compute (expect math_pipe_throttle)\n");
    for (int i = 0; i < 3; i++) {
        kernel_pure_compute<<<grid_size, block_size>>>(d_c, N, 1000);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. With Barrier
    printf("5. Kernel: With Barrier (expect barrier stall)\n");
    for (int i = 0; i < 3; i++) {
        kernel_with_barrier<<<grid_size, block_size>>>(d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 6. Random Access
    printf("6. Kernel: Random Access (expect long_scoreboard, cache misses)\n");
    for (int i = 0; i < 3; i++) {
        kernel_random_access<<<grid_size, block_size>>>(d_a, d_indices, d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 7. Atomic
    printf("7. Kernel: Atomic Add (expect wait stall)\n");
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));
    for (int i = 0; i < 3; i++) {
        kernel_atomic_add<<<1024, 256>>>(d_counter, 1024 * 256);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 8. Shared Memory Intensive
    printf("8. Kernel: Shared Memory Intensive (expect short_scoreboard)\n");
    for (int i = 0; i < 3; i++) {
        kernel_smem_intensive<<<grid_size, block_size>>>(d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 9. Mixed
    printf("9. Kernel: Mixed Compute+Memory\n");
    for (int i = 0; i < 3; i++) {
        kernel_mixed<<<grid_size, block_size>>>(d_a, d_c, N, 100);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 10. Large GEMM (cuBLAS)
    printf("10. Kernel: Large GEMM %dx%dx%d (cuBLAS)\n", GEMM_M, GEMM_N, GEMM_K);
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  GEMM_M, GEMM_N, GEMM_K,
                                  &alpha,
                                  d_gemm_a, GEMM_M,
                                  d_gemm_b, GEMM_K,
                                  &beta,
                                  d_gemm_c, GEMM_M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 11. Very Large GEMM
    printf("11. Kernel: Very Large GEMM 8192x8192x8192 (cuBLAS)\n");
    const int LARGE_M = 8192, LARGE_N = 8192, LARGE_K = 8192;
    float *d_large_a, *d_large_b, *d_large_c;
    CHECK_CUDA(cudaMalloc(&d_large_a, (size_t)LARGE_M * LARGE_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_large_b, (size_t)LARGE_K * LARGE_N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_large_c, (size_t)LARGE_M * LARGE_N * sizeof(float)));
    
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  LARGE_M, LARGE_N, LARGE_K,
                                  &alpha,
                                  d_large_a, LARGE_M,
                                  d_large_b, LARGE_K,
                                  &beta,
                                  d_large_c, LARGE_M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("\nAll kernels completed.\n");
    printf("\nTo analyze warp stalls, run:\n");
    printf("  ncu --set full -o warp_stall_report ./warp_stall_benchmark\n");
    printf("  ncu-ui warp_stall_report.ncu-rep\n");
    
    // 清理
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_indices);
    cudaFree(d_counter);
    cudaFree(d_gemm_a);
    cudaFree(d_gemm_b);
    cudaFree(d_gemm_c);
    cudaFree(d_large_a);
    cudaFree(d_large_b);
    cudaFree(d_large_c);
    free(h_a);
    free(h_b);
    free(h_indices);
    
    return 0;
}

