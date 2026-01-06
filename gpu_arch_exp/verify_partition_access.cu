/**
 * 验证 A/B 分区交替访问
 * 
 * 验证方法：
 * 1. 分配两个独立的缓冲区 A 和 B，各 32MB
 * 2. 确保它们的物理地址分布在不同的 L2 分区
 * 3. 对比访问模式：A-only, B-only, A-B交替
 * 4. 使用 NCU 验证 L2 行为
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

#define WARMUP_ITERS 5
#define TEST_ITERS 20

// ============================================================
// 方案1: 两个独立缓冲区
// ============================================================
__global__ void access_buffer_A_only(float* buffer_A, float* buffer_B, 
                                      size_t elements, uint64_t* result, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 256; i++) {
            size_t idx = (tid * 32 + i) % elements;
            sum += buffer_A[idx];  // 只访问 A
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        end_time = clock64();
        *result = end_time - start_time;
    }
    
    if (sum == -999999.0f) buffer_A[tid] = sum;
}

__global__ void access_buffer_B_only(float* buffer_A, float* buffer_B, 
                                      size_t elements, uint64_t* result, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 256; i++) {
            size_t idx = (tid * 32 + i) % elements;
            sum += buffer_B[idx];  // 只访问 B
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        end_time = clock64();
        *result = end_time - start_time;
    }
    
    if (sum == -999999.0f) buffer_B[tid] = sum;
}

__global__ void access_A_then_B(float* buffer_A, float* buffer_B, 
                                 size_t elements, uint64_t* result, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // 先访问 A，再访问 B（顺序访问）
    for (int iter = 0; iter < iterations; iter++) {
        // 先 A
        for (int i = 0; i < 128; i++) {
            size_t idx = (tid * 32 + i) % elements;
            sum += buffer_A[idx];
        }
        // 再 B
        for (int i = 0; i < 128; i++) {
            size_t idx = (tid * 32 + i) % elements;
            sum += buffer_B[idx];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        end_time = clock64();
        *result = end_time - start_time;
    }
    
    if (sum == -999999.0f) buffer_A[tid] = sum;
}

__global__ void access_A_B_interleaved(float* buffer_A, float* buffer_B, 
                                        size_t elements, uint64_t* result, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // A-B 交替访问（每次切换）
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 256; i++) {
            size_t idx = (tid * 32 + i) % elements;
            if (i % 2 == 0) {
                sum += buffer_A[idx];  // 偶数访问 A
            } else {
                sum += buffer_B[idx];  // 奇数访问 B
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        end_time = clock64();
        *result = end_time - start_time;
    }
    
    if (sum == -999999.0f) buffer_A[tid] = sum;
}

// ============================================================
// 方案2: 显示缓冲区地址，验证物理分布
// ============================================================
void print_buffer_info(const char* name, void* ptr, size_t size) {
    printf("  %s: device ptr = %p, size = %zu MB\n", name, ptr, size / (1024 * 1024));
}

// ============================================================
// 方案3: Pointer chasing 精确测量
// ============================================================
__global__ void pointer_chase_A_only(int* chain_A, int* chain_B, 
                                      int chain_length, uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = 0;
        
        // Warmup
        for (int i = 0; i < chain_length; i++) {
            idx = chain_A[idx];
        }
        
        idx = 0;
        uint64_t start = clock64();
        
        for (int i = 0; i < chain_length; i++) {
            idx = chain_A[idx];  // 只访问 A
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) chain_A[0] = idx;
        *result = end - start;
    }
}

__global__ void pointer_chase_B_only(int* chain_A, int* chain_B, 
                                      int chain_length, uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = 0;
        
        // Warmup
        for (int i = 0; i < chain_length; i++) {
            idx = chain_B[idx];
        }
        
        idx = 0;
        uint64_t start = clock64();
        
        for (int i = 0; i < chain_length; i++) {
            idx = chain_B[idx];  // 只访问 B
        }
        
        uint64_t end = clock64();
        
        if (idx == -999999) chain_B[0] = idx;
        *result = end - start;
    }
}

__global__ void pointer_chase_A_B_alternate(int* chain_A, int* chain_B, 
                                             int chain_length, uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx_A = 0, idx_B = 0;
        
        // Warmup
        for (int i = 0; i < chain_length / 2; i++) {
            idx_A = chain_A[idx_A];
            idx_B = chain_B[idx_B];
        }
        
        idx_A = 0;
        idx_B = 0;
        uint64_t start = clock64();
        
        // 交替访问 A 和 B
        for (int i = 0; i < chain_length / 2; i++) {
            idx_A = chain_A[idx_A];  // 访问 A
            idx_B = chain_B[idx_B];  // 访问 B
        }
        
        uint64_t end = clock64();
        
        if (idx_A == -999999) chain_A[0] = idx_A;
        if (idx_B == -999999) chain_B[0] = idx_B;
        *result = end - start;
    }
}

void create_random_chain(int* h_chain, int array_elements, int chain_length) {
    for (int i = 0; i < array_elements; i++) {
        h_chain[i] = 0;
    }
    
    int stride = 32;  // 128 bytes
    int idx = 0;
    for (int i = 0; i < chain_length - 1; i++) {
        int next_idx = (idx + stride) % array_elements;
        h_chain[idx] = next_idx;
        idx = next_idx;
    }
    h_chain[idx] = 0;
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("验证 A/B 分区交替访问\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    print_separator();
    
    // ============================================================
    printf("\n方案1: 两个独立缓冲区 (各 32 MB)\n");
    print_separator();
    
    size_t buffer_size = 32 * 1024 * 1024;  // 32 MB each
    size_t buffer_elements = buffer_size / sizeof(float);
    
    float *d_buffer_A, *d_buffer_B;
    CUDA_CHECK(cudaMalloc(&d_buffer_A, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_buffer_B, buffer_size));
    CUDA_CHECK(cudaMemset(d_buffer_A, 1, buffer_size));
    CUDA_CHECK(cudaMemset(d_buffer_B, 2, buffer_size));
    
    printf("\n缓冲区地址信息:\n");
    print_buffer_info("Buffer A", d_buffer_A, buffer_size);
    print_buffer_info("Buffer B", d_buffer_B, buffer_size);
    
    // 计算地址差
    size_t addr_diff = (size_t)d_buffer_B - (size_t)d_buffer_A;
    printf("  地址差: %zu MB (%.1f × L2)\n", addr_diff / (1024 * 1024), 
           (double)addr_diff / prop.l2CacheSize);
    
    uint64_t *d_result, h_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    
    int num_blocks = 32;
    int block_size = 256;
    int iterations = 50;
    
    printf("\n访问模式对比:\n");
    printf("模式                 时间(cycles)      相对A-only\n");
    printf("------------------------------------------------------------------------\n");
    
    // A-only
    for (int w = 0; w < WARMUP_ITERS; w++) {
        access_buffer_A_only<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    uint64_t total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        access_buffer_A_only<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double a_only = (double)total / TEST_ITERS;
    printf("A-only               %12.0f      1.00x\n", a_only);
    
    // B-only
    for (int w = 0; w < WARMUP_ITERS; w++) {
        access_buffer_B_only<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        access_buffer_B_only<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double b_only = (double)total / TEST_ITERS;
    printf("B-only               %12.0f      %.2fx\n", b_only, b_only / a_only);
    
    // A then B (顺序)
    for (int w = 0; w < WARMUP_ITERS; w++) {
        access_A_then_B<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        access_A_then_B<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double a_then_b = (double)total / TEST_ITERS;
    printf("A-then-B (顺序)      %12.0f      %.2fx\n", a_then_b, a_then_b / a_only);
    
    // A-B interleaved (交替)
    for (int w = 0; w < WARMUP_ITERS; w++) {
        access_A_B_interleaved<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        access_A_B_interleaved<<<num_blocks, block_size>>>(d_buffer_A, d_buffer_B, buffer_elements, d_result, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double a_b_interleaved = (double)total / TEST_ITERS;
    printf("A-B-interleaved      %12.0f      %.2fx\n", a_b_interleaved, a_b_interleaved / a_only);
    
    CUDA_CHECK(cudaFree(d_buffer_A));
    CUDA_CHECK(cudaFree(d_buffer_B));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("方案2: Pointer Chasing 精确测量\n");
    print_separator();
    
    size_t chain_buffer_size = 32 * 1024 * 1024;  // 32 MB
    int chain_elements = chain_buffer_size / sizeof(int);
    int chain_length = 5000;
    
    int* h_chain_A = (int*)malloc(chain_buffer_size);
    int* h_chain_B = (int*)malloc(chain_buffer_size);
    int *d_chain_A, *d_chain_B;
    
    CUDA_CHECK(cudaMalloc(&d_chain_A, chain_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_chain_B, chain_buffer_size));
    
    create_random_chain(h_chain_A, chain_elements, chain_length);
    create_random_chain(h_chain_B, chain_elements, chain_length);
    
    CUDA_CHECK(cudaMemcpy(d_chain_A, h_chain_A, chain_buffer_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chain_B, h_chain_B, chain_buffer_size, cudaMemcpyHostToDevice));
    
    printf("\n缓冲区地址信息:\n");
    print_buffer_info("Chain A", d_chain_A, chain_buffer_size);
    print_buffer_info("Chain B", d_chain_B, chain_buffer_size);
    
    printf("\n访问模式对比 (pointer chasing, %d 次访问):\n", chain_length);
    printf("模式                 延迟(cycles/访问)    相对A-only\n");
    printf("------------------------------------------------------------------------\n");
    
    // A-only pointer chase
    for (int w = 0; w < WARMUP_ITERS; w++) {
        pointer_chase_A_only<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        pointer_chase_A_only<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double chase_a = (double)total / TEST_ITERS / chain_length;
    printf("Chase A-only         %12.2f cycles    1.00x\n", chase_a);
    
    // B-only pointer chase
    for (int w = 0; w < WARMUP_ITERS; w++) {
        pointer_chase_B_only<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        pointer_chase_B_only<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double chase_b = (double)total / TEST_ITERS / chain_length;
    printf("Chase B-only         %12.2f cycles    %.2fx\n", chase_b, chase_b / chase_a);
    
    // A-B alternate pointer chase
    for (int w = 0; w < WARMUP_ITERS; w++) {
        pointer_chase_A_B_alternate<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        pointer_chase_A_B_alternate<<<1, 32>>>(d_chain_A, d_chain_B, chain_length, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    double chase_ab = (double)total / TEST_ITERS / (chain_length / 2);
    printf("Chase A-B-alternate  %12.2f cycles    %.2fx\n", chase_ab, chase_ab / chase_a);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("验证结论\n");
    print_separator();
    printf("\n如果 A-B 交替访问的延迟显著高于单独访问 A 或 B:\n");
    printf("  → 证明确实触发了跨分区访问 (Stitch Traffic)\n");
    printf("\n如果 A-only 和 B-only 延迟相近:\n");
    printf("  → 说明两个缓冲区都能被高效访问\n");
    printf("\n如果交替访问延迟 ≈ (A + B) / 2:\n");
    printf("  → 说明没有额外的跨分区开销\n");
    printf("如果交替访问延迟 > (A + B) / 2:\n");
    printf("  → 说明存在跨分区切换开销 (Stitch)\n");
    print_separator();
    
    free(h_chain_A);
    free(h_chain_B);
    CUDA_CHECK(cudaFree(d_chain_A));
    CUDA_CHECK(cudaFree(d_chain_B));
    CUDA_CHECK(cudaFree(d_result));
    
    return 0;
}

