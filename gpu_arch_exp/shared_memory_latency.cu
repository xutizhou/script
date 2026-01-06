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

// =============================================================================
// Shared Memory Latency Benchmark
// 使用 pointer chasing 测量 shared memory 的访问延迟
// =============================================================================

// 获取 SM 时钟周期
__device__ __forceinline__ unsigned int get_smclock() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(ret));
    return ret;
}

// Pointer chasing kernel for shared memory latency
__global__ void shared_memory_latency_kernel(int64_t* results, int iterations) {
    // 使用 shared memory
    __shared__ int64_t shared_buf[1024];  // 8KB shared memory
    
    const int tid = threadIdx.x;
    const int num_elements = 1024;
    
    // 只让 thread 0 执行 pointer chasing
    if (tid == 0) {
        // 初始化 pointer chain（随机化以避免预取）
        for (int i = 0; i < num_elements - 1; i++) {
            shared_buf[i] = (i + 17) % num_elements;  // 简单跳跃模式
        }
        shared_buf[num_elements - 1] = 0;
    }
    __syncthreads();
    
    if (tid == 0) {
        int64_t idx = 0;
        
        // 预热
        for (int i = 0; i < 1000; i++) {
            idx = shared_buf[idx];
        }
        
        // 开始计时
        unsigned int start = get_smclock();
        
        // Pointer chasing
        #pragma unroll 1
        for (int i = 0; i < iterations; i++) {
            idx = shared_buf[idx];
        }
        
        unsigned int end = get_smclock();
        
        // 防止编译器优化掉循环
        results[0] = idx;
        results[1] = (end - start);
        results[2] = iterations;
    }
}

// Global memory (L1 cache) latency for comparison
__global__ void l1_cache_latency_kernel(int64_t* buf, int64_t* results, 
                                         int num_elements, int iterations) {
    const int tid = threadIdx.x;
    
    if (tid == 0) {
        int64_t idx = 0;
        
        // 预热 - 让数据进入 L1 cache
        for (int i = 0; i < 1000; i++) {
            idx = buf[idx];
        }
        
        // 开始计时
        unsigned int start = get_smclock();
        
        // Pointer chasing
        #pragma unroll 1
        for (int i = 0; i < iterations; i++) {
            idx = buf[idx];
        }
        
        unsigned int end = get_smclock();
        
        results[0] = idx;
        results[1] = (end - start);
        results[2] = iterations;
    }
}

// Register latency (baseline)
__global__ void register_latency_kernel(int64_t* results, int iterations) {
    const int tid = threadIdx.x;
    
    if (tid == 0) {
        int64_t val = 1;
        
        // 预热
        for (int i = 0; i < 1000; i++) {
            val = val + 1;
        }
        
        // 开始计时
        unsigned int start = get_smclock();
        
        // Simple register operations
        #pragma unroll 1
        for (int i = 0; i < iterations; i++) {
            val = val + 1;
        }
        
        unsigned int end = get_smclock();
        
        results[0] = val;
        results[1] = (end - start);
        results[2] = iterations;
    }
}

int main() {
    printf("=============================================================\n");
    printf("    Memory Latency Benchmark: Registers vs Shared vs L1\n");
    printf("=============================================================\n\n");
    
    // 获取 GPU 时钟频率
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_ghz = prop.clockRate / 1e6;  // Convert kHz to GHz
    printf("GPU: %s\n", prop.name);
    printf("SM Clock: %.2f GHz (%.0f MHz)\n\n", clock_ghz, clock_ghz * 1000);
    
    const int iterations = 100000;
    
    // 分配结果缓冲区
    int64_t* d_results;
    int64_t h_results[3];
    CUDA_CHECK(cudaMalloc(&d_results, 3 * sizeof(int64_t)));
    
    // =========================================================================
    // Test 1: Register latency (baseline)
    // =========================================================================
    printf("Testing Register latency (baseline)...\n");
    register_latency_kernel<<<1, 32>>>(d_results, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(int64_t), 
                          cudaMemcpyDeviceToHost));
    
    double reg_cycles = (double)h_results[1] / h_results[2];
    double reg_ns = reg_cycles / clock_ghz;
    printf("  Register: %.2f cycles (%.2f ns)\n\n", reg_cycles, reg_ns);
    
    // =========================================================================
    // Test 2: Shared Memory latency
    // =========================================================================
    printf("Testing Shared Memory latency...\n");
    shared_memory_latency_kernel<<<1, 32>>>(d_results, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(int64_t), 
                          cudaMemcpyDeviceToHost));
    
    double shared_cycles = (double)h_results[1] / h_results[2];
    double shared_ns = shared_cycles / clock_ghz;
    printf("  Shared Memory: %.2f cycles (%.2f ns)\n\n", shared_cycles, shared_ns);
    
    // =========================================================================
    // Test 3: L1 Cache latency (small buffer that fits in L1)
    // =========================================================================
    printf("Testing L1 Cache latency...\n");
    
    const int l1_elements = 1024;  // 8KB, fits in L1
    int64_t* d_l1_buf;
    CUDA_CHECK(cudaMalloc(&d_l1_buf, l1_elements * sizeof(int64_t)));
    
    // 初始化 pointer chain
    std::vector<int64_t> h_l1_buf(l1_elements);
    for (int i = 0; i < l1_elements - 1; i++) {
        h_l1_buf[i] = (i + 17) % l1_elements;
    }
    h_l1_buf[l1_elements - 1] = 0;
    CUDA_CHECK(cudaMemcpy(d_l1_buf, h_l1_buf.data(), 
                          l1_elements * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    l1_cache_latency_kernel<<<1, 32>>>(d_l1_buf, d_results, l1_elements, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(int64_t), 
                          cudaMemcpyDeviceToHost));
    
    double l1_cycles = (double)h_results[1] / h_results[2];
    double l1_ns = l1_cycles / clock_ghz;
    printf("  L1 Cache: %.2f cycles (%.2f ns)\n\n", l1_cycles, l1_ns);
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("=============================================================\n");
    printf("    Summary: Memory Latency Comparison\n");
    printf("=============================================================\n\n");
    
    printf("  +-----------------+------------+------------+\n");
    printf("  | Memory Type     | Latency    | Latency    |\n");
    printf("  |                 | (cycles)   | (ns)       |\n");
    printf("  +-----------------+------------+------------+\n");
    printf("  | Register        | %10.2f | %10.2f |\n", reg_cycles, reg_ns);
    printf("  | Shared Memory   | %10.2f | %10.2f |\n", shared_cycles, shared_ns);
    printf("  | L1 Cache        | %10.2f | %10.2f |\n", l1_cycles, l1_ns);
    printf("  +-----------------+------------+------------+\n\n");
    
    printf("  Shared Memory vs L1 Cache speedup: %.2fx\n", l1_cycles / shared_cycles);
    printf("  Shared Memory vs Register overhead: %.2fx\n", shared_cycles / reg_cycles);
    
    printf("\n  Note: These are pointer-chasing latencies (serial access).\n");
    printf("        Throughput tests may show different relative performance.\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_l1_buf));
    
    return 0;
}




