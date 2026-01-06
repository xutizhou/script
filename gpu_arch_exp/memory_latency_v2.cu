#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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
// 更准确的内存延迟测量
// 使用大量展开的 pointer chasing 来消除循环开销
// =============================================================================

__device__ __forceinline__ unsigned int get_clock() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(ret));
    return ret;
}

// Shared Memory Latency - 使用展开的 pointer chasing
__global__ void shared_latency_unrolled(int64_t* results) {
    __shared__ int64_t buf[256];
    
    if (threadIdx.x == 0) {
        // 初始化简单的 pointer chain: 0->1->2->...->255->0
        for (int i = 0; i < 255; i++) {
            buf[i] = i + 1;
        }
        buf[255] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        int64_t idx = 0;
        
        // 预热
        for (int i = 0; i < 256; i++) idx = buf[idx];
        
        unsigned int start = get_clock();
        
        // 256次展开的 pointer chasing（无循环开销）
        #define CHASE8 idx=buf[idx];idx=buf[idx];idx=buf[idx];idx=buf[idx];\
                       idx=buf[idx];idx=buf[idx];idx=buf[idx];idx=buf[idx];
        #define CHASE64 CHASE8 CHASE8 CHASE8 CHASE8 CHASE8 CHASE8 CHASE8 CHASE8
        #define CHASE256 CHASE64 CHASE64 CHASE64 CHASE64
        
        CHASE256  // 256 次访问
        
        unsigned int end = get_clock();
        
        results[0] = idx;
        results[1] = end - start;
    }
}

// L1 Cache Latency - 小数据在 L1 中
__global__ void l1_latency_unrolled(int64_t* buf, int64_t* results) {
    if (threadIdx.x == 0) {
        int64_t idx = 0;
        
        // 预热 - 确保数据在 L1
        for (int i = 0; i < 256; i++) idx = buf[idx];
        
        unsigned int start = get_clock();
        
        // 256次展开
        #define CHASE_G8 idx=buf[idx];idx=buf[idx];idx=buf[idx];idx=buf[idx];\
                         idx=buf[idx];idx=buf[idx];idx=buf[idx];idx=buf[idx];
        #define CHASE_G64 CHASE_G8 CHASE_G8 CHASE_G8 CHASE_G8 CHASE_G8 CHASE_G8 CHASE_G8 CHASE_G8
        #define CHASE_G256 CHASE_G64 CHASE_G64 CHASE_G64 CHASE_G64
        
        CHASE_G256
        
        unsigned int end = get_clock();
        
        results[0] = idx;
        results[1] = end - start;
    }
}

// L2 Cache Latency - 中等数据（超过L1但在L2内）
__global__ void l2_latency_unrolled(int64_t* buf, int64_t* results, int stride) {
    if (threadIdx.x == 0) {
        int64_t idx = 0;
        
        // 预热
        for (int i = 0; i < 256; i++) idx = buf[idx];
        
        unsigned int start = get_clock();
        
        CHASE_G256
        
        unsigned int end = get_clock();
        
        results[0] = idx;
        results[1] = end - start;
    }
}

int main() {
    printf("=============================================================\n");
    printf("    Accurate Memory Latency Benchmark (Unrolled)\n");
    printf("=============================================================\n\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_ghz = prop.clockRate / 1e6;
    printf("GPU: %s\n", prop.name);
    printf("SM Clock: %.2f GHz\n\n", clock_ghz);
    
    int64_t* d_results;
    int64_t h_results[2];
    CUDA_CHECK(cudaMalloc(&d_results, 2 * sizeof(int64_t)));
    
    const int accesses = 256;
    
    // =========================================================================
    // Test 1: Shared Memory
    // =========================================================================
    printf("1. Shared Memory Latency:\n");
    shared_latency_unrolled<<<1, 32>>>(d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    
    double shared_total = (double)h_results[1];
    double shared_per_access = shared_total / accesses;
    printf("   Total cycles for %d accesses: %.0f\n", accesses, shared_total);
    printf("   Per-access latency: %.1f cycles (%.1f ns)\n\n", 
           shared_per_access, shared_per_access / clock_ghz);
    
    // =========================================================================
    // Test 2: L1 Cache (8KB buffer)
    // =========================================================================
    printf("2. L1 Cache Latency (8KB buffer):\n");
    
    const int l1_elements = 256;  // 2KB
    int64_t* d_l1_buf;
    CUDA_CHECK(cudaMalloc(&d_l1_buf, l1_elements * sizeof(int64_t)));
    
    // 初始化
    int64_t* h_buf = new int64_t[l1_elements];
    for (int i = 0; i < l1_elements - 1; i++) h_buf[i] = i + 1;
    h_buf[l1_elements - 1] = 0;
    CUDA_CHECK(cudaMemcpy(d_l1_buf, h_buf, l1_elements * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    l1_latency_unrolled<<<1, 32>>>(d_l1_buf, d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    
    double l1_total = (double)h_results[1];
    double l1_per_access = l1_total / accesses;
    printf("   Total cycles for %d accesses: %.0f\n", accesses, l1_total);
    printf("   Per-access latency: %.1f cycles (%.1f ns)\n\n",
           l1_per_access, l1_per_access / clock_ghz);
    
    // =========================================================================
    // Test 3: L2 Cache (1MB buffer - exceeds L1)
    // =========================================================================
    printf("3. L2 Cache Latency (1MB buffer):\n");
    
    const int l2_elements = 128 * 1024;  // 1MB
    int64_t* d_l2_buf;
    CUDA_CHECK(cudaMalloc(&d_l2_buf, l2_elements * sizeof(int64_t)));
    
    // 使用大步长来超过 L1 但留在 L2
    delete[] h_buf;
    h_buf = new int64_t[l2_elements];
    int stride = 512;  // 每次跳 4KB
    for (int i = 0; i < 256; i++) {
        int cur = (i * stride) % l2_elements;
        int next = ((i + 1) * stride) % l2_elements;
        h_buf[cur] = next;
    }
    CUDA_CHECK(cudaMemcpy(d_l2_buf, h_buf, l2_elements * sizeof(int64_t), cudaMemcpyHostToDevice));
    
    l2_latency_unrolled<<<1, 32>>>(d_l2_buf, d_results, stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost));
    
    double l2_total = (double)h_results[1];
    double l2_per_access = l2_total / accesses;
    printf("   Total cycles for %d accesses: %.0f\n", accesses, l2_total);
    printf("   Per-access latency: %.1f cycles (%.1f ns)\n\n",
           l2_per_access, l2_per_access / clock_ghz);
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("=============================================================\n");
    printf("    Summary\n");
    printf("=============================================================\n\n");
    
    printf("  +------------------+----------+----------+----------+\n");
    printf("  | Memory Type      | Latency  | Latency  | vs Shared|\n");
    printf("  |                  | (cycles) | (ns)     |          |\n");
    printf("  +------------------+----------+----------+----------+\n");
    printf("  | Shared Memory    | %8.1f | %8.1f | 1.00x    |\n", 
           shared_per_access, shared_per_access / clock_ghz);
    printf("  | L1 Cache         | %8.1f | %8.1f | %.2fx    |\n", 
           l1_per_access, l1_per_access / clock_ghz, l1_per_access / shared_per_access);
    printf("  | L2 Cache         | %8.1f | %8.1f | %.2fx    |\n", 
           l2_per_access, l2_per_access / clock_ghz, l2_per_access / shared_per_access);
    printf("  +------------------+----------+----------+----------+\n\n");
    
    printf("  Note: Register access latency is effectively 0 cycles\n");
    printf("        (fused with instructions, no separate memory access)\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_l1_buf));
    CUDA_CHECK(cudaFree(d_l2_buf));
    delete[] h_buf;
    
    return 0;
}




