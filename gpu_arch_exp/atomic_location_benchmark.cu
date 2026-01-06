/**
 * Atomic Operation Location Benchmark
 * 
 * 实验目的：验证以下假设
 * 1. Shared Memory 原子操作在 Shared Memory 硬件上完成
 * 2. Global Memory 原子操作在 L2 Cache 上完成
 * 
 * 实验设计：
 * 1. 延迟测试：测量单线程连续原子操作的延迟
 * 2. 吞吐量测试：测量多线程并发原子操作的吞吐量
 * 3. 跨SM竞争测试：多个SM对同一global memory位置的原子操作
 * 4. L1 bypass验证：如果global atomic在L2完成，即使数据在L1也需要访问L2
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define WARMUP_ITERS 100
#define TEST_ITERS 1000
#define NUM_ATOMIC_OPS 1000

// ============================================================
// 实验1: Shared Memory Atomic 延迟测试
// 单线程连续原子操作，测量延迟
// ============================================================
__global__ void shared_atomic_latency_kernel(uint64_t* result, int num_ops) {
    __shared__ int shared_counter;
    
    if (threadIdx.x == 0) {
        shared_counter = 0;
        
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&shared_counter, 1);
        }
        
        uint64_t end = clock64();
        *result = end - start;
    }
}

// ============================================================
// 实验2: Global Memory Atomic 延迟测试 (单SM)
// 单线程连续原子操作，测量延迟
// ============================================================
__global__ void global_atomic_latency_kernel(int* global_counter, uint64_t* result, int num_ops) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *global_counter = 0;
        
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(global_counter, 1);
        }
        
        uint64_t end = clock64();
        *result = end - start;
    }
}

// ============================================================
// 实验3: Shared Memory 普通读写延迟 (baseline)
// ============================================================
__global__ void shared_normal_rw_latency_kernel(uint64_t* result, int num_ops) {
    __shared__ volatile int shared_counter;
    
    if (threadIdx.x == 0) {
        shared_counter = 0;
        
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            int val = shared_counter;
            shared_counter = val + 1;
        }
        
        uint64_t end = clock64();
        *result = end - start;
    }
}

// ============================================================
// 实验4: 多SM竞争Global Atomic
// 如果global atomic在L2完成，多SM竞争会导致性能下降
// ============================================================
__global__ void multi_sm_global_atomic_kernel(int* global_counter, uint64_t* results, int num_ops) {
    // 每个block的thread 0执行
    if (threadIdx.x == 0) {
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(global_counter, 1);
        }
        
        uint64_t end = clock64();
        results[blockIdx.x] = end - start;
    }
}

// ============================================================
// 实验5: 多线程同warp内Shared Atomic
// 测试warp内线程对同一shared memory位置的原子操作
// ============================================================
__global__ void warp_shared_atomic_kernel(uint64_t* result, int num_ops) {
    __shared__ int shared_counter;
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        shared_counter = 0;
        __threadfence_block();
    }
    __syncthreads();
    
    // 只用一个warp (32线程)
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            start_time = clock64();
        }
        __syncwarp();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&shared_counter, 1);
        }
        
        __syncwarp();
        if (threadIdx.x == 0) {
            end_time = clock64();
            *result = end_time - start_time;
        }
    }
}

// ============================================================
// 实验6: 多线程同warp内Global Atomic
// 对比warp内线程对global memory的原子操作
// ============================================================
__global__ void warp_global_atomic_kernel(int* global_counter, uint64_t* result, int num_ops) {
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *global_counter = 0;
    }
    __syncthreads();
    
    // 只用一个warp (32线程)，只在block 0
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            start_time = clock64();
        }
        __syncwarp();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(global_counter, 1);
        }
        
        __syncwarp();
        if (threadIdx.x == 0) {
            end_time = clock64();
            *result = end_time - start_time;
        }
    }
}

// ============================================================
// 实验7: 每个线程操作不同位置 - Shared Memory
// 无冲突情况下的shared atomic吞吐量
// ============================================================
__global__ void shared_atomic_no_conflict_kernel(uint64_t* result, int num_ops) {
    __shared__ int shared_array[256];
    __shared__ uint64_t start_time, end_time;
    
    int tid = threadIdx.x;
    if (tid < 256) {
        shared_array[tid] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            start_time = clock64();
        }
        __syncwarp();
        
        // 每个线程操作自己的位置 (无冲突)
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&shared_array[threadIdx.x], 1);
        }
        
        __syncwarp();
        if (threadIdx.x == 0) {
            end_time = clock64();
            *result = end_time - start_time;
        }
    }
}

// ============================================================
// 实验8: 每个线程操作不同位置 - Global Memory
// 无冲突情况下的global atomic吞吐量
// ============================================================
__global__ void global_atomic_no_conflict_kernel(int* global_array, uint64_t* result, int num_ops) {
    __shared__ uint64_t start_time, end_time;
    
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            start_time = clock64();
        }
        __syncwarp();
        
        // 每个线程操作自己的位置 (无冲突)
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&global_array[threadIdx.x], 1);
        }
        
        __syncwarp();
        if (threadIdx.x == 0) {
            end_time = clock64();
            *result = end_time - start_time;
        }
    }
}

// ============================================================
// 实验9: Bank Conflict 测试 - Shared Memory
// 如果shared atomic在shared memory硬件上完成，
// 不同bank的原子操作应该可以并行
// ============================================================
__global__ void shared_atomic_bank_test_kernel(uint64_t* result, int num_ops, int stride) {
    // 32 banks, 每个bank 4 bytes
    __shared__ int shared_array[256];
    __shared__ uint64_t start_time, end_time;
    
    int tid = threadIdx.x;
    if (tid < 256) {
        shared_array[tid] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            start_time = clock64();
        }
        __syncwarp();
        
        // stride=1: 无bank conflict (每个线程访问不同bank)
        // stride=32: 全部bank conflict (所有线程访问同一bank)
        int idx = (threadIdx.x * stride) % 256;
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&shared_array[idx], 1);
        }
        
        __syncwarp();
        if (threadIdx.x == 0) {
            end_time = clock64();
            *result = end_time - start_time;
        }
    }
}

// ============================================================
// 实验10: L2 分区测试 - Global Memory
// 测试不同L2分区的global atomic性能
// ============================================================
__global__ void global_atomic_l2_partition_test_kernel(int* global_array, uint64_t* results, 
                                                        int num_ops, int spacing) {
    // spacing控制地址间隔，不同间隔可能落在不同L2分区
    if (threadIdx.x == 0) {
        int idx = blockIdx.x * spacing;
        
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < num_ops; i++) {
            atomicAdd(&global_array[idx], 1);
        }
        
        uint64_t end = clock64();
        results[blockIdx.x] = end - start;
    }
}

void print_separator() {
    printf("================================================================\n");
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("Atomic Operation Location Benchmark\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    print_separator();
    
    // Allocate memory
    uint64_t *d_result, h_result;
    uint64_t *d_results, h_results[128];
    int *d_counter;
    int *d_array;
    
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMalloc(&d_results, 128 * sizeof(uint64_t));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_array, 1024 * 1024 * sizeof(int));  // 4MB array
    cudaMemset(d_array, 0, 1024 * 1024 * sizeof(int));
    
    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < WARMUP_ITERS; i++) {
        shared_atomic_latency_kernel<<<1, 32>>>(d_result, 100);
        global_atomic_latency_kernel<<<1, 32>>>(d_counter, d_result, 100);
    }
    cudaDeviceSynchronize();
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1 & 2: 单线程原子操作延迟对比\n");
    printf("验证：如果global atomic在L2完成，其延迟应该更高\n");
    print_separator();
    
    // Shared atomic latency
    uint64_t shared_atomic_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        shared_atomic_latency_kernel<<<1, 32>>>(d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        shared_atomic_total += h_result;
    }
    double shared_atomic_avg = (double)shared_atomic_total / TEST_ITERS / NUM_ATOMIC_OPS;
    
    // Global atomic latency
    uint64_t global_atomic_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        global_atomic_latency_kernel<<<1, 32>>>(d_counter, d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        global_atomic_total += h_result;
    }
    double global_atomic_avg = (double)global_atomic_total / TEST_ITERS / NUM_ATOMIC_OPS;
    
    // Shared normal R/W latency (baseline)
    uint64_t shared_rw_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        shared_normal_rw_latency_kernel<<<1, 32>>>(d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        shared_rw_total += h_result;
    }
    double shared_rw_avg = (double)shared_rw_total / TEST_ITERS / NUM_ATOMIC_OPS;
    
    printf("Shared Memory 普通读写延迟:     %.2f cycles/op\n", shared_rw_avg);
    printf("Shared Memory Atomic延迟:       %.2f cycles/op\n", shared_atomic_avg);
    printf("Global Memory Atomic延迟:       %.2f cycles/op\n", global_atomic_avg);
    printf("Global/Shared Atomic比值:       %.2fx\n", global_atomic_avg / shared_atomic_avg);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: 多SM竞争Global Atomic\n");
    printf("验证：如果global atomic在L2完成，多SM竞争会增加延迟\n");
    print_separator();
    
    for (int num_blocks : {1, 2, 4, 8, 16, 32, 64}) {
        cudaMemset(d_counter, 0, sizeof(int));
        
        uint64_t total_time = 0;
        for (int iter = 0; iter < TEST_ITERS / 10; iter++) {
            multi_sm_global_atomic_kernel<<<num_blocks, 32>>>(d_counter, d_results, NUM_ATOMIC_OPS);
            cudaDeviceSynchronize();
            cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            uint64_t max_time = 0;
            for (int i = 0; i < num_blocks; i++) {
                if (h_results[i] > max_time) max_time = h_results[i];
            }
            total_time += max_time;
        }
        double avg_time = (double)total_time / (TEST_ITERS / 10) / NUM_ATOMIC_OPS;
        printf("SM数量=%2d: %.2f cycles/op (%.2fx vs 1 SM)\n", 
               num_blocks, avg_time, avg_time / global_atomic_avg);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验4: Warp内线程并发Atomic (32线程竞争同一位置)\n");
    printf("验证：Shared atomic可能在shared memory硬件上串行化\n");
    print_separator();
    
    // Warp shared atomic
    uint64_t warp_shared_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        warp_shared_atomic_kernel<<<1, 64>>>(d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        warp_shared_total += h_result;
    }
    double warp_shared_avg = (double)warp_shared_total / TEST_ITERS / NUM_ATOMIC_OPS / 32;
    
    // Warp global atomic
    uint64_t warp_global_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        warp_global_atomic_kernel<<<1, 64>>>(d_counter, d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        warp_global_total += h_result;
    }
    double warp_global_avg = (double)warp_global_total / TEST_ITERS / NUM_ATOMIC_OPS / 32;
    
    printf("Shared Atomic (32线程竞争): %.2f cycles/op/thread\n", warp_shared_avg);
    printf("Global Atomic (32线程竞争): %.2f cycles/op/thread\n", warp_global_avg);
    printf("预期：如果shared在SM内完成，延迟应该较低\n");
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验5: 无冲突Atomic (每线程操作不同位置)\n");
    printf("验证：无冲突时的吞吐量差异\n");
    print_separator();
    
    // No conflict shared atomic
    uint64_t no_conflict_shared_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        shared_atomic_no_conflict_kernel<<<1, 64>>>(d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        no_conflict_shared_total += h_result;
    }
    double no_conflict_shared_avg = (double)no_conflict_shared_total / TEST_ITERS / NUM_ATOMIC_OPS;
    
    // No conflict global atomic
    uint64_t no_conflict_global_total = 0;
    for (int i = 0; i < TEST_ITERS; i++) {
        global_atomic_no_conflict_kernel<<<1, 64>>>(d_array, d_result, NUM_ATOMIC_OPS);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        no_conflict_global_total += h_result;
    }
    double no_conflict_global_avg = (double)no_conflict_global_total / TEST_ITERS / NUM_ATOMIC_OPS;
    
    printf("Shared Atomic (无冲突, 32位置): %.2f cycles/op (total for 32 threads)\n", 
           no_conflict_shared_avg);
    printf("Global Atomic (无冲突, 32位置): %.2f cycles/op (total for 32 threads)\n", 
           no_conflict_global_avg);
    printf("Global/Shared比值: %.2fx\n", no_conflict_global_avg / no_conflict_shared_avg);
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验6: Shared Memory Bank Conflict 测试\n");
    printf("验证：如果shared atomic在shared memory硬件完成，bank conflict会影响性能\n");
    print_separator();
    
    for (int stride : {1, 2, 4, 8, 16, 32}) {
        uint64_t bank_test_total = 0;
        for (int i = 0; i < TEST_ITERS; i++) {
            shared_atomic_bank_test_kernel<<<1, 64>>>(d_result, NUM_ATOMIC_OPS, stride);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            bank_test_total += h_result;
        }
        double bank_test_avg = (double)bank_test_total / TEST_ITERS / NUM_ATOMIC_OPS;
        
        int num_banks_accessed = (stride == 32) ? 1 : 32;  // stride=32 means all same bank
        printf("Stride=%2d (访问%2d个bank): %.2f cycles/op\n", 
               stride, num_banks_accessed, bank_test_avg);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验7: Global Atomic L2分区测试\n");
    printf("验证：不同L2分区的global atomic性能\n");
    print_separator();
    
    for (int spacing : {1, 32, 128, 512, 2048, 8192, 32768}) {
        cudaMemset(d_array, 0, 1024 * 1024 * sizeof(int));
        
        int num_blocks = 16;
        uint64_t total_time = 0;
        for (int iter = 0; iter < TEST_ITERS / 10; iter++) {
            global_atomic_l2_partition_test_kernel<<<num_blocks, 32>>>(d_array, d_results, 
                                                                        NUM_ATOMIC_OPS, spacing);
            cudaDeviceSynchronize();
            cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            uint64_t sum_time = 0;
            for (int i = 0; i < num_blocks; i++) {
                sum_time += h_results[i];
            }
            total_time += sum_time / num_blocks;
        }
        double avg_time = (double)total_time / (TEST_ITERS / 10) / NUM_ATOMIC_OPS;
        printf("地址间隔=%5d ints (%6d bytes): %.2f cycles/op\n", 
               spacing, spacing * 4, avg_time);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("结论分析\n");
    print_separator();
    printf("1. 如果 Global Atomic 延迟 >> Shared Atomic 延迟:\n");
    printf("   -> 支持 Global Atomic 在 L2 完成 (L2 延迟 > Shared Memory 延迟)\n");
    printf("\n");
    printf("2. 如果多SM竞争Global Atomic时延迟显著增加:\n");
    printf("   -> 支持 Global Atomic 在 L2 完成 (L2是共享资源，有竞争)\n");
    printf("\n");
    printf("3. 如果 Shared Atomic 受 bank conflict 影响:\n");
    printf("   -> 支持 Shared Atomic 在 Shared Memory 硬件完成\n");
    printf("\n");
    printf("4. 如果无冲突时 Global Atomic 吞吐量仍明显低于 Shared:\n");
    printf("   -> 支持两者使用不同硬件单元\n");
    print_separator();
    
    // Cleanup
    cudaFree(d_result);
    cudaFree(d_results);
    cudaFree(d_counter);
    cudaFree(d_array);
    
    return 0;
}

