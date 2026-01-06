/**
 * Warp Stall Analysis Benchmark V2
 * 
 * 更多实验探究不同类型的warp stall
 * 
 * 编译: nvcc -O3 -arch=sm_90 warp_stall_benchmark_v2.cu -o warp_stall_benchmark_v2 -lcublas
 * 
 * NCU运行:
 * ncu --set full -o warp_stall_report_v2 ./warp_stall_benchmark_v2
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

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
// 实验1: LG Throttle - 大量outstanding内存请求
// ============================================================================
__global__ void kernel_lg_throttle_many_requests(const float* __restrict__ input, 
                                                   float* output, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 发起大量非连续的内存请求
    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 32; i++) {
        int offset = (idx + i * stride) % N;
        sum += input[offset];
    }
    
    if (idx < N) {
        output[idx] = sum;
    }
}

// ============================================================================
// 实验2: MIO Throttle - 密集的shared memory访问
// ============================================================================
__global__ void kernel_mio_throttle_smem_intensive(float* output, int N) {
    __shared__ float smem[4096];  // 大的shared memory
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 初始化shared memory
    for (int i = tid; i < 4096; i += blockDim.x) {
        smem[i] = (float)i;
    }
    __syncthreads();
    
    // 密集的shared memory访问 (可能有bank冲突)
    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        // 故意造成bank冲突: 32个线程访问同一个bank
        int bank_conflict_idx = (tid / 32) * 32 + i % 128;
        sum += smem[bank_conflict_idx];
    }
    
    if (idx < N) {
        output[idx] = sum;
    }
}

// ============================================================================
// 实验3: Branch Divergence - 分支发散
// ============================================================================
__global__ void kernel_branch_divergence(const float* __restrict__ input,
                                          float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    float val = 0.0f;
    if (idx < N) {
        val = input[idx];
        
        // 每个lane走不同的分支 - 最大化warp发散
        switch (lane % 8) {
            case 0: val = sqrtf(val); break;
            case 1: val = sinf(val); break;
            case 2: val = cosf(val); break;
            case 3: val = expf(val * 0.1f); break;
            case 4: val = logf(val + 1.0f); break;
            case 5: val = tanf(val); break;
            case 6: val = val * val; break;
            case 7: val = 1.0f / (val + 0.001f); break;
        }
        
        // 更多分支
        if (lane < 16) {
            val = val * 2.0f;
        } else {
            val = val * 0.5f;
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// 实验4: Memory Barrier (Membar) - __threadfence操作
// ============================================================================
__global__ void kernel_membar_heavy(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        data[idx] = (float)idx;
        __threadfence();  // 全局内存fence
        
        data[idx] += 1.0f;
        __threadfence();
        
        data[idx] += 1.0f;
        __threadfence();
        
        data[idx] += 1.0f;
        __threadfence();
    }
}

// ============================================================================
// 实验5: Instruction Cache Miss - 大量代码跳转
// ============================================================================
__device__ __noinline__ float compute_path_0(float x) {
    return sqrtf(x) * sinf(x) * cosf(x);
}
__device__ __noinline__ float compute_path_1(float x) {
    return expf(x * 0.01f) * logf(x + 1.0f);
}
__device__ __noinline__ float compute_path_2(float x) {
    return tanf(x) * atanf(x);
}
__device__ __noinline__ float compute_path_3(float x) {
    return powf(x, 0.5f) * cbrtf(x);
}
__device__ __noinline__ float compute_path_4(float x) {
    return sinhf(x * 0.1f) * coshf(x * 0.1f);
}
__device__ __noinline__ float compute_path_5(float x) {
    return erff(x) * erfcf(x);
}
__device__ __noinline__ float compute_path_6(float x) {
    return tgammaf(x * 0.1f + 1.0f);
}
__device__ __noinline__ float compute_path_7(float x) {
    return j0f(x) * j1f(x);
}

__global__ void kernel_instruction_cache_miss(const float* __restrict__ input,
                                               float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        int path = idx % 8;
        
        // 调用不同的函数，造成代码跳转
        #pragma unroll 1
        for (int i = 0; i < 10; i++) {
            switch ((path + i) % 8) {
                case 0: val = compute_path_0(val); break;
                case 1: val = compute_path_1(val); break;
                case 2: val = compute_path_2(val); break;
                case 3: val = compute_path_3(val); break;
                case 4: val = compute_path_4(val); break;
                case 5: val = compute_path_5(val); break;
                case 6: val = compute_path_6(val); break;
                case 7: val = compute_path_7(val); break;
            }
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// 实验6: Strided Access - 不同stride的内存访问
// ============================================================================
__global__ void kernel_strided_access(const float* __restrict__ input,
                                       float* output, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N / stride) {
        output[idx] = input[idx * stride];
    }
}

// ============================================================================
// 实验7: Coalesced vs Non-coalesced对比
// ============================================================================
// Coalesced访问
__global__ void kernel_coalesced_access(const float4* __restrict__ input,
                                         float4* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N / 4) {
        output[idx] = input[idx];
    }
}

// Non-coalesced (AoS pattern)
struct Particle {
    float x, y, z, w;
    float vx, vy, vz, vw;
};

__global__ void kernel_non_coalesced_aos(const Particle* __restrict__ particles,
                                          float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // 访问结构体中的不连续字段
        output[idx] = particles[idx].x + particles[idx].vx;
    }
}

// ============================================================================
// 实验8: High Occupancy vs Low Occupancy
// ============================================================================
// 高occupancy: 少量寄存器和shared memory
__global__ void kernel_high_occupancy(const float* __restrict__ input,
                                       float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        val = val * 2.0f + 1.0f;
        output[idx] = val;
    }
}

// 低occupancy: 大量寄存器
__global__ void kernel_low_occupancy(const float* __restrict__ input,
                                      float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用大量寄存器
    float r0 = input[idx % N];
    float r1 = r0 * 1.001f, r2 = r0 * 1.002f, r3 = r0 * 1.003f, r4 = r0 * 1.004f;
    float r5 = r0 * 1.005f, r6 = r0 * 1.006f, r7 = r0 * 1.007f, r8 = r0 * 1.008f;
    float r9 = r0 * 1.009f, r10 = r0 * 1.010f, r11 = r0 * 1.011f, r12 = r0 * 1.012f;
    float r13 = r0 * 1.013f, r14 = r0 * 1.014f, r15 = r0 * 1.015f, r16 = r0 * 1.016f;
    float r17 = r0 * 1.017f, r18 = r0 * 1.018f, r19 = r0 * 1.019f, r20 = r0 * 1.020f;
    float r21 = r0 * 1.021f, r22 = r0 * 1.022f, r23 = r0 * 1.023f, r24 = r0 * 1.024f;
    float r25 = r0 * 1.025f, r26 = r0 * 1.026f, r27 = r0 * 1.027f, r28 = r0 * 1.028f;
    float r29 = r0 * 1.029f, r30 = r0 * 1.030f, r31 = r0 * 1.031f, r32 = r0 * 1.032f;
    
    // 使用所有寄存器
    float result = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8;
    result += r9 + r10 + r11 + r12 + r13 + r14 + r15 + r16;
    result += r17 + r18 + r19 + r20 + r21 + r22 + r23 + r24;
    result += r25 + r26 + r27 + r28 + r29 + r30 + r31 + r32;
    
    if (idx < N) {
        output[idx] = result;
    }
}

// ============================================================================
// 实验9: Drain - 大量写操作后的drain
// ============================================================================
__global__ void kernel_drain_heavy_write(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 大量连续写操作
    for (int i = idx; i < N; i += stride) {
        output[i] = (float)i;
    }
    // kernel结束时会有drain等待所有写完成
}

// ============================================================================
// 实验10: Wait Stall - 原子操作到不同位置
// ============================================================================
// 单一位置原子操作 (高竞争)
__global__ void kernel_atomic_single_location(int* counter) {
    atomicAdd(counter, 1);
}

// 分散位置原子操作 (低竞争)
__global__ void kernel_atomic_distributed(int* counters, int num_counters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int target = idx % num_counters;
    atomicAdd(&counters[target], 1);
}

// Warp-level原子 (shared memory)
__global__ void kernel_atomic_warp_level(int* output) {
    __shared__ int smem_counter;
    
    if (threadIdx.x == 0) smem_counter = 0;
    __syncthreads();
    
    atomicAdd(&smem_counter, 1);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(output, smem_counter);
    }
}

// ============================================================================
// 实验11: Texture访问 (tex_throttle) - 使用新的texture object API
// ============================================================================
__global__ void kernel_texture_heavy(cudaTextureObject_t tex, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 16; i++) {
            sum += tex1Dfetch<float>(tex, (idx + i * 1000) % N);
        }
        output[idx] = sum;
    }
}

// ============================================================================
// 实验12: 不同block size对比
// ============================================================================
template<int BLOCK_SIZE>
__global__ void kernel_block_size_test(const float* __restrict__ input,
                                        float* output, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        val = val * 2.0f + 1.0f;
        output[idx] = val;
    }
}

// ============================================================================
// 实验13: 深度依赖链 (long scoreboard的另一种情况)
// ============================================================================
__global__ void kernel_dependency_chain(const float* __restrict__ input,
                                         float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        
        // 深度依赖链: 每个操作都依赖上一个
        #pragma unroll 1
        for (int i = 0; i < 100; i++) {
            val = sqrtf(val + 1.0f);
        }
        
        output[idx] = val;
    }
}

// ============================================================================
// 实验14: 独立操作 (可以隐藏延迟)
// ============================================================================
__global__ void kernel_independent_ops(const float* __restrict__ input,
                                        float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float v0 = input[idx];
        float v1 = input[(idx + N/4) % N];
        float v2 = input[(idx + N/2) % N];
        float v3 = input[(idx + 3*N/4) % N];
        
        // 独立的计算 - 可以并行执行
        v0 = sqrtf(v0 + 1.0f);
        v1 = sqrtf(v1 + 1.0f);
        v2 = sqrtf(v2 + 1.0f);
        v3 = sqrtf(v3 + 1.0f);
        
        output[idx] = v0 + v1 + v2 + v3;
    }
}

// ============================================================================
// Main
// ============================================================================
void init_random(float* data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] = (float)rand() / RAND_MAX + 0.1f;
    }
}

int main(int argc, char** argv) {
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    printf("Warp Stall Analysis Benchmark V2\n");
    printf("N = %d, Grid = %d, Block = %d\n", N, grid_size, block_size);
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, SM: %d.%d\n\n", prop.name, prop.major, prop.minor);
    
    // 分配内存
    float *h_input = (float*)malloc(N * sizeof(float));
    float *d_input, *d_output;
    int *d_counters;
    Particle *d_particles;
    
    init_random(h_input, N);
    
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_counters, 1024 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_counters, 0, 1024 * sizeof(int)));
    
    // Create texture object
    cudaTextureObject_t tex = 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = N * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    
    CHECK_CUDA(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    
    printf("Running kernels for NCU profiling...\n\n");
    
    // ========================================================================
    // 运行实验
    // ========================================================================
    
    printf("=== Experiment 1: LG Throttle (many outstanding requests) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_lg_throttle_many_requests<<<grid_size, block_size>>>(d_input, d_output, N, 1024);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 2: MIO Throttle (shared memory bank conflicts) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_mio_throttle_smem_intensive<<<1024, 256>>>(d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 3: Branch Divergence ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_branch_divergence<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 4: Memory Barrier (threadfence) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_membar_heavy<<<grid_size, block_size>>>(d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 5: Instruction Cache Miss ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_instruction_cache_miss<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 6: Strided Access (stride=1,2,4,8,16,32) ===\n");
    for (int stride = 1; stride <= 32; stride *= 2) {
        kernel_strided_access<<<grid_size, block_size>>>(d_input, d_output, N, stride);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 7a: Coalesced Access ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_coalesced_access<<<grid_size, block_size>>>((float4*)d_input, (float4*)d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 7b: Non-coalesced AoS Access ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_non_coalesced_aos<<<grid_size, block_size>>>(d_particles, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 8a: High Occupancy ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_high_occupancy<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 8b: Low Occupancy (many registers) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_low_occupancy<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 9: Drain (heavy write) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_drain_heavy_write<<<grid_size, block_size>>>(d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 10a: Atomic Single Location (high contention) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_atomic_single_location<<<1024, 256>>>(d_counters);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 10b: Atomic Distributed (low contention) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_atomic_distributed<<<1024, 256>>>(d_counters, 1024);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 10c: Atomic Warp Level (shared memory) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_atomic_warp_level<<<1024, 256>>>(d_counters);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 11: Texture Access ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_texture_heavy<<<grid_size, block_size>>>(tex, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 12: Block Size Comparison ===\n");
    kernel_block_size_test<64><<<N/64, 64>>>(d_input, d_output, N);
    kernel_block_size_test<128><<<N/128, 128>>>(d_input, d_output, N);
    kernel_block_size_test<256><<<N/256, 256>>>(d_input, d_output, N);
    kernel_block_size_test<512><<<N/512, 512>>>(d_input, d_output, N);
    kernel_block_size_test<1024><<<N/1024, 1024>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 13: Dependency Chain (serial) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_dependency_chain<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("=== Experiment 14: Independent Operations (parallel) ===\n");
    for (int i = 0; i < 3; i++) {
        kernel_independent_ops<<<grid_size, block_size>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("\nAll experiments completed.\n");
    printf("\nTo analyze warp stalls:\n");
    printf("  ncu --set full -o warp_stall_report_v2 ./warp_stall_benchmark_v2\n");
    printf("  ncu-ui warp_stall_report_v2.ncu-rep\n");
    
    // 清理
    cudaDestroyTextureObject(tex);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_counters);
    cudaFree(d_particles);
    free(h_input);
    
    return 0;
}

