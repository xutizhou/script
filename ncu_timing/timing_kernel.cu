/**
 * NCU Timing Accuracy Experiment Kernel
 * 
 * 这个kernel通过控制读写数据量来控制运行时间，用于验证NCU计时准确性
 * 
 * 编译: nvcc -O3 -arch=sm_90 timing_kernel.cu -o timing_kernel
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 简单的读写kernel，通过iterations参数控制运行时间
__global__ void timed_rw_kernel(float* __restrict__ src, 
                                 float* __restrict__ dst, 
                                 size_t num_elements,
                                 int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    // 多次迭代读写来控制运行时间
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = idx; i < num_elements; i += stride) {
            sum += src[i];
        }
    }
    
    // 写回结果，防止编译器优化掉读操作
    for (size_t i = idx; i < num_elements; i += stride) {
        dst[i] = sum / (float)(iterations * num_elements);
    }
}

// 向量化版本，使用float4提高带宽利用率
__global__ void timed_rw_kernel_vectorized(float4* __restrict__ src, 
                                           float4* __restrict__ dst, 
                                           size_t num_vec_elements,
                                           int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = idx; i < num_vec_elements; i += stride) {
            float4 val = src[i];
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
            sum.w += val.w;
        }
    }
    
    float scale = 1.0f / (float)(iterations * num_vec_elements);
    for (size_t i = idx; i < num_vec_elements; i += stride) {
        dst[i] = make_float4(sum.x * scale, sum.y * scale, sum.z * scale, sum.w * scale);
    }
}

// 纯 Memory Bound Kernel - Stream Copy (无计算，只有内存读写)
// 类似 STREAM benchmark 的 Copy 操作：dst[i] = src[i]
__global__ void membound_copy_kernel(float4* __restrict__ src, 
                                      float4* __restrict__ dst, 
                                      size_t num_vec_elements,
                                      int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 多次迭代来控制运行时间
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = idx; i < num_vec_elements; i += stride) {
            dst[i] = src[i];  // 纯 copy，无计算
        }
    }
}

// 打印使用说明
void print_usage(const char* prog_name) {
    printf("Usage: %s <mode> [options]\n", prog_name);
    printf("\nModes:\n");
    printf("  short    - Run short kernel (~10us target)\n");
    printf("  long     - Run long kernel (~10ms target)\n");
    printf("  membound - Run pure memory-bound kernel (~30ms, stream copy)\n");
    printf("  custom   - Run with custom parameters\n");
    printf("\nOptions for custom mode:\n");
    printf("  --size <N>        - Number of elements (default: 1M)\n");
    printf("  --iterations <N>  - Number of iterations (default: 1)\n");
    printf("  --blocks <N>      - Number of blocks (default: 128)\n");
    printf("  --threads <N>     - Threads per block (default: 128)\n");
    printf("\nExamples:\n");
    printf("  %s short\n", prog_name);
    printf("  %s long\n", prog_name);
    printf("  %s membound\n", prog_name);
    printf("  %s custom --size 1000000 --iterations 10\n", prog_name);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 默认参数
    size_t num_elements = 1024 * 1024;  // 1M elements = 4MB
    int iterations = 1;
    int num_blocks = 128;
    int threads_per_block = 128;
    
    const char* mode = argv[1];
    
    if (strcmp(mode, "short") == 0) {
        // 短时间kernel目标 ~10us
        // B200上，小数据量+少量迭代
        num_elements = 4096;        // 16KB
        iterations = 1;
        num_blocks = 1;
        threads_per_block = 128;
        printf("=== Short Kernel Mode (target ~10us) ===\n");
    } else if (strcmp(mode, "long") == 0) {
        // 长时间kernel目标 ~10ms
        // 大数据量+多次迭代
        num_elements = 16 * 1024 * 1024;  // 64MB
        iterations = 100;
        num_blocks = 128;
        threads_per_block = 128;
        printf("=== Long Kernel Mode (target ~10ms) ===\n");
    } else if (strcmp(mode, "membound") == 0) {
        // 纯 Memory Bound Kernel (~30ms)
        // 大数据量 stream copy，无计算
        num_elements = 256 * 1024 * 1024;  // 1GB (256M float4 = 4GB total r+w)
        iterations = 10;
        num_blocks = 512;
        threads_per_block = 256;
        printf("=== Memory Bound Kernel Mode (pure stream copy, target ~30ms) ===\n");
    } else if (strcmp(mode, "custom") == 0) {
        printf("=== Custom Kernel Mode ===\n");
        // 解析自定义参数
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
                num_elements = atol(argv[++i]);
            } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
                num_blocks = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                threads_per_block = atoi(argv[++i]);
            }
        }
    } else {
        printf("Unknown mode: %s\n", mode);
        print_usage(argv[0]);
        return 1;
    }
    
    // 打印配置
    size_t data_size = num_elements * sizeof(float);
    printf("Configuration:\n");
    printf("  Elements:    %zu (%.2f MB)\n", num_elements, data_size / (1024.0 * 1024.0));
    printf("  Iterations:  %d\n", iterations);
    printf("  Blocks:      %d\n", num_blocks);
    printf("  Threads:     %d\n", threads_per_block);
    printf("  Total data:  %.2f MB (read) + %.2f MB (write)\n", 
           data_size / (1024.0 * 1024.0), data_size / (1024.0 * 1024.0));
    
    // 分配GPU内存
    float *d_src, *d_dst;
    cudaError_t err;
    
    err = cudaMalloc(&d_src, data_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_src failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_dst, data_size);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_dst failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_src);
        return 1;
    }
    
    // 初始化数据
    float* h_src = (float*)malloc(data_size);
    for (size_t i = 0; i < num_elements; i++) {
        h_src[i] = (float)(i % 1000) / 1000.0f;
    }
    cudaMemcpy(d_src, h_src, data_size, cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, data_size);
    
    // 同步确保初始化完成
    cudaDeviceSynchronize();
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    printf("\nWarming up...\n");
    timed_rw_kernel<<<num_blocks, threads_per_block>>>(d_src, d_dst, num_elements, iterations);
    cudaDeviceSynchronize();
    
    // 运行并计时
    printf("Running kernel...\n");
    cudaEventRecord(start);
    timed_rw_kernel<<<num_blocks, threads_per_block>>>(d_src, d_dst, num_elements, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    printf("\n=== Results ===\n");
    printf("Kernel duration: %.3f ms (%.3f us)\n", elapsed_ms, elapsed_ms * 1000.0f);
    
    // 计算带宽
    double total_bytes = (double)data_size * iterations + data_size;  // reads + writes
    double bandwidth_gbps = (total_bytes / (elapsed_ms * 1e-3)) / 1e9;
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    
    printf("\nKernel completed successfully.\n");
    return 0;
}

