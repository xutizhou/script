/**
 * Stitch Traffic Benchmark
 * 
 * 目标：触发和测量 L2 partition 之间的 stitch traffic
 * 
 * Stitch Traffic 发生条件：
 * 1. 数据足迹 > C/2 (数据无法完全复制到两个分区)
 * 2. CTA 在一个 uGPU 运行，但访问另一个 uGPU 分区的数据
 * 3. LCN miss 时需要跨分区访问 POC
 * 
 * H20 架构：
 * - L2 Cache: 60 MB
 * - 假设 2 个 uGPU 分区，每个 30 MB
 * - SM 分布在两个 uGPU 上 (78 SMs / 2 = 39 SMs per uGPU)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string.h>

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
// 实验1: 强制跨分区访问
// 使用大数据块，让前半部分 SM 访问后半部分数据
// ============================================================
__global__ void cross_partition_access_kernel(float* data, size_t data_elements,
                                               int access_pattern, // 0=local, 1=remote
                                               uint64_t* results, int iterations) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;
    
    __shared__ uint64_t start_time, end_time;
    
    float sum = 0.0f;
    
    // 计算访问区域
    size_t region_size = data_elements / 2;
    size_t base_offset;
    
    if (access_pattern == 0) {
        // Local: 前半 blocks 访问前半数据，后半 blocks 访问后半数据
        base_offset = (bid < num_blocks / 2) ? 0 : region_size;
    } else {
        // Remote: 前半 blocks 访问后半数据，后半 blocks 访问前半数据
        base_offset = (bid < num_blocks / 2) ? region_size : 0;
    }
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    // 访问数据
    size_t elements_per_block = region_size / (num_blocks / 2);
    size_t block_offset = (bid % (num_blocks / 2)) * elements_per_block;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < elements_per_block && i < 1024; i += blockDim.x) {
            size_t idx = base_offset + block_offset + i;
            if (idx < data_elements) {
                sum += data[idx];
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[bid] = end_time - start_time;
    }
    
    // 防止优化
    if (sum == -999999.0f && tid == 0) data[bid] = sum;
}

// ============================================================
// 实验2: 交替访问模式
// 每个 warp 交替访问两个分区的数据
// ============================================================
__global__ void alternating_partition_kernel(float* data, size_t data_elements,
                                              int alternate_frequency, // 每多少次访问切换分区
                                              uint64_t* results, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    size_t half = data_elements / 2;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 256; i++) {
            size_t idx;
            if (alternate_frequency == 0) {
                // 只访问前半部分
                idx = (tid * 32 + i) % half;
            } else {
                // 交替访问
                int partition = ((i / alternate_frequency) % 2);
                idx = partition * half + ((tid * 32 + i) % half);
            }
            sum += data[idx];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[blockIdx.x] = end_time - start_time;
    }
    
    if (sum == -999999.0f) data[tid] = sum;
}

// ============================================================
// 实验3: 写操作触发 Coherency Traffic
// 多个 SM 写入需要 invalidate 远程 LCN 副本
// ============================================================
__global__ void coherency_write_kernel(float* data, size_t data_elements,
                                        int write_pattern, // 0=local, 1=remote, 2=mixed
                                        uint64_t* results, int iterations) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;
    
    __shared__ uint64_t start_time, end_time;
    
    size_t region_size = data_elements / 2;
    size_t base_offset;
    
    if (write_pattern == 0) {
        base_offset = (bid < num_blocks / 2) ? 0 : region_size;
    } else if (write_pattern == 1) {
        base_offset = (bid < num_blocks / 2) ? region_size : 0;
    } else {
        // Mixed: 所有 blocks 都写入同一区域，触发竞争
        base_offset = 0;
    }
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    size_t elements_per_block = region_size / num_blocks;
    size_t block_offset = (bid % (num_blocks / 2)) * elements_per_block;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < elements_per_block && i < 256; i += blockDim.x) {
            size_t idx = base_offset + block_offset + i;
            if (idx < data_elements) {
                data[idx] += 1.0f;
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[bid] = end_time - start_time;
    }
}

// ============================================================
// 实验4: 数据足迹递增测试
// 逐渐增加数据足迹，观察 stitch traffic 的触发点
// ============================================================
__global__ void footprint_scaling_kernel(float* data, size_t footprint_elements,
                                          uint64_t* results, int iterations) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int global_tid = bid * blockDim.x + tid;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    // 所有线程共同访问整个数据足迹
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = global_tid; i < footprint_elements; i += total_threads) {
            sum += data[i];
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[bid] = end_time - start_time;
    }
    
    if (sum == -999999.0f) data[global_tid % footprint_elements] = sum;
}

// ============================================================
// 实验5: 双缓冲区 Ping-Pong 测试
// 模拟实际应用中的跨分区数据交换
// ============================================================
__global__ void pingpong_kernel(float* buffer_a, float* buffer_b, 
                                 size_t buffer_elements, int direction,
                                 uint64_t* results, int iterations) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    size_t elements_per_block = buffer_elements / num_blocks;
    size_t block_offset = bid * elements_per_block;
    
    float sum = 0.0f;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < elements_per_block && i < 256; i += blockDim.x) {
            size_t idx = block_offset + i;
            if (direction == 0) {
                // A -> B
                sum += buffer_a[idx];
                buffer_b[idx] = sum;
            } else {
                // B -> A
                sum += buffer_b[idx];
                buffer_a[idx] = sum;
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[bid] = end_time - start_time;
    }
}

// ============================================================
// 实验6: SM 亲和性测试
// 限制使用一半的 SM，观察分区行为
// ============================================================
__global__ void sm_affinity_kernel(float* data, size_t data_elements,
                                    int target_region, // 0=前半, 1=后半
                                    uint64_t* results, int iterations) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ uint64_t start_time, end_time;
    
    if (threadIdx.x == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    float sum = 0.0f;
    size_t region_size = data_elements / 2;
    size_t base_offset = target_region * region_size;
    
    size_t elements_per_block = region_size / gridDim.x;
    size_t block_offset = bid * elements_per_block;
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < elements_per_block && i < 512; i += blockDim.x) {
            size_t idx = base_offset + block_offset + i;
            if (idx < data_elements) {
                sum += data[idx];
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        end_time = clock64();
        results[bid] = end_time - start_time;
    }
    
    if (sum == -999999.0f) data[tid] = sum;
}

void print_separator() {
    printf("================================================================\n");
}

double get_max_time(uint64_t* h_results, int num_blocks) {
    uint64_t max_time = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (h_results[i] > max_time) max_time = h_results[i];
    }
    return (double)max_time;
}

double get_avg_time(uint64_t* h_results, int num_blocks) {
    uint64_t sum = 0;
    for (int i = 0; i < num_blocks; i++) {
        sum += h_results[i];
    }
    return (double)sum / num_blocks;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("Stitch Traffic Benchmark\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d KB (%.1f MB)\n", prop.l2CacheSize / 1024, prop.l2CacheSize / 1024.0 / 1024.0);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t half_l2 = l2_size / 2;
    int num_sms = prop.multiProcessorCount;
    
    printf("\n假设配置:\n");
    printf("  uGPU 数量: 2\n");
    printf("  每个 uGPU 的 SM: %d\n", num_sms / 2);
    printf("  每个 uGPU 的 L2: %.1f MB\n", half_l2 / 1024.0 / 1024.0);
    print_separator();
    
    // 分配内存
    uint64_t *d_results;
    uint64_t h_results[256];
    CUDA_CHECK(cudaMalloc(&d_results, 256 * sizeof(uint64_t)));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验1: Local vs Remote 分区访问\n");
    printf("对比本地分区访问和跨分区访问的延迟\n");
    print_separator();
    
    // 使用大于 C/2 的数据，确保分布在两个分区
    size_t exp1_size = 64 * 1024 * 1024;  // 64 MB > L2
    size_t exp1_elements = exp1_size / sizeof(float);
    
    float* d_exp1_data;
    CUDA_CHECK(cudaMalloc(&d_exp1_data, exp1_size));
    CUDA_CHECK(cudaMemset(d_exp1_data, 0, exp1_size));
    
    int num_blocks = 64;
    int block_size = 256;
    int iterations = 100;
    
    printf("\n模式           平均时间(cycles)    最大时间(cycles)    最大/平均\n");
    printf("------------------------------------------------------------------------\n");
    
    // Local 访问
    for (int w = 0; w < WARMUP_ITERS; w++) {
        cross_partition_access_kernel<<<num_blocks, block_size>>>(
            d_exp1_data, exp1_elements, 0, d_results, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double local_avg = 0, local_max = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        cross_partition_access_kernel<<<num_blocks, block_size>>>(
            d_exp1_data, exp1_elements, 0, d_results, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        local_avg += get_avg_time(h_results, num_blocks);
        local_max += get_max_time(h_results, num_blocks);
    }
    local_avg /= TEST_ITERS;
    local_max /= TEST_ITERS;
    printf("Local          %12.0f        %12.0f        %.2fx\n", local_avg, local_max, local_max / local_avg);
    
    // Remote 访问
    for (int w = 0; w < WARMUP_ITERS; w++) {
        cross_partition_access_kernel<<<num_blocks, block_size>>>(
            d_exp1_data, exp1_elements, 1, d_results, iterations);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double remote_avg = 0, remote_max = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        cross_partition_access_kernel<<<num_blocks, block_size>>>(
            d_exp1_data, exp1_elements, 1, d_results, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_results, d_results, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        remote_avg += get_avg_time(h_results, num_blocks);
        remote_max += get_max_time(h_results, num_blocks);
    }
    remote_avg /= TEST_ITERS;
    remote_max /= TEST_ITERS;
    printf("Remote         %12.0f        %12.0f        %.2fx\n", remote_avg, remote_max, remote_max / remote_avg);
    printf("\nRemote/Local 比值: %.2fx (平均), %.2fx (最大)\n", remote_avg / local_avg, remote_max / local_max);
    
    CUDA_CHECK(cudaFree(d_exp1_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验2: 交替访问频率测试\n");
    printf("观察访问模式切换频率对延迟的影响\n");
    print_separator();
    
    size_t exp2_size = 64 * 1024 * 1024;
    size_t exp2_elements = exp2_size / sizeof(float);
    
    float* d_exp2_data;
    CUDA_CHECK(cudaMalloc(&d_exp2_data, exp2_size));
    CUDA_CHECK(cudaMemset(d_exp2_data, 0, exp2_size));
    
    printf("\n交替频率      平均时间(cycles)    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<int> frequencies = {0, 1, 2, 4, 8, 16, 32, 64, 128};
    
    for (int freq : frequencies) {
        for (int w = 0; w < WARMUP_ITERS; w++) {
            alternating_partition_kernel<<<32, 256>>>(
                d_exp2_data, exp2_elements, freq, d_results, 50);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double avg = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            alternating_partition_kernel<<<32, 256>>>(
                d_exp2_data, exp2_elements, freq, d_results, 50);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            avg += get_avg_time(h_results, 32);
        }
        avg /= TEST_ITERS;
        
        const char* desc = "";
        if (freq == 0) desc = "(单分区访问)";
        else if (freq == 1) desc = "(每次切换)";
        else if (freq >= 64) desc = "(很少切换)";
        
        printf("%5d         %12.0f        %s\n", freq, avg, desc);
    }
    
    CUDA_CHECK(cudaFree(d_exp2_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验3: 写操作 Coherency Traffic\n");
    printf("测试写操作对分区一致性的影响\n");
    print_separator();
    
    size_t exp3_size = 64 * 1024 * 1024;
    size_t exp3_elements = exp3_size / sizeof(float);
    
    float* d_exp3_data;
    CUDA_CHECK(cudaMalloc(&d_exp3_data, exp3_size));
    
    printf("\n写模式        平均时间(cycles)    最大时间(cycles)    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    const char* write_modes[] = {"Local", "Remote", "Mixed"};
    
    for (int mode = 0; mode < 3; mode++) {
        CUDA_CHECK(cudaMemset(d_exp3_data, 0, exp3_size));
        
        for (int w = 0; w < WARMUP_ITERS; w++) {
            coherency_write_kernel<<<64, 256>>>(
                d_exp3_data, exp3_elements, mode, d_results, 50);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double avg = 0, max_t = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            coherency_write_kernel<<<64, 256>>>(
                d_exp3_data, exp3_elements, mode, d_results, 50);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            avg += get_avg_time(h_results, 64);
            max_t += get_max_time(h_results, 64);
        }
        avg /= TEST_ITERS;
        max_t /= TEST_ITERS;
        
        const char* desc = "";
        if (mode == 0) desc = "(本地写)";
        else if (mode == 1) desc = "(跨分区写)";
        else desc = "(竞争写)";
        
        printf("%-10s    %12.0f        %12.0f        %s\n", write_modes[mode], avg, max_t, desc);
    }
    
    CUDA_CHECK(cudaFree(d_exp3_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验4: 数据足迹递增 - Stitch 触发点\n");
    printf("观察数据足迹增加时 stitch traffic 的触发\n");
    print_separator();
    
    size_t max_size = 128 * 1024 * 1024;  // 128 MB
    float* d_exp4_data;
    CUDA_CHECK(cudaMalloc(&d_exp4_data, max_size));
    CUDA_CHECK(cudaMemset(d_exp4_data, 0, max_size));
    
    printf("\n数据足迹      平均时间(cycles)    相对C/2    说明\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<size_t> footprints_mb = {8, 16, 24, 28, 30, 32, 36, 40, 48, 56, 60, 64, 80, 100};
    double baseline = 0;
    
    for (size_t fp_mb : footprints_mb) {
        size_t footprint = fp_mb * 1024 * 1024;
        size_t footprint_elements = footprint / sizeof(float);
        
        for (int w = 0; w < WARMUP_ITERS; w++) {
            footprint_scaling_kernel<<<64, 256>>>(
                d_exp4_data, footprint_elements, d_results, 20);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double avg = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            footprint_scaling_kernel<<<64, 256>>>(
                d_exp4_data, footprint_elements, d_results, 20);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            avg += get_avg_time(h_results, 64);
        }
        avg /= TEST_ITERS;
        
        if (fp_mb == 8) baseline = avg;
        
        double relative = (double)footprint / half_l2;
        const char* note = "";
        if (relative >= 0.95 && relative <= 1.05) note = "← C/2";
        else if (relative >= 1.95 && relative <= 2.05) note = "← C";
        
        printf("%5zu MB       %12.0f        %.2fx      %s\n", fp_mb, avg, relative, note);
    }
    
    CUDA_CHECK(cudaFree(d_exp4_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("实验5: SM 数量缩放\n");
    printf("观察不同 SM 数量时的分区行为\n");
    print_separator();
    
    size_t exp5_size = 64 * 1024 * 1024;
    size_t exp5_elements = exp5_size / sizeof(float);
    
    float* d_exp5_data;
    CUDA_CHECK(cudaMalloc(&d_exp5_data, exp5_size));
    CUDA_CHECK(cudaMemset(d_exp5_data, 0, exp5_size));
    
    printf("\nSM数量(blocks)  本地访问(cycles)    远程访问(cycles)    远程/本地\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<int> block_counts = {2, 4, 8, 16, 32, 64, 78};
    
    for (int blocks : block_counts) {
        // Local
        for (int w = 0; w < WARMUP_ITERS; w++) {
            sm_affinity_kernel<<<blocks, 256>>>(d_exp5_data, exp5_elements, 0, d_results, 100);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double local_t = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            sm_affinity_kernel<<<blocks, 256>>>(d_exp5_data, exp5_elements, 0, d_results, 100);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            local_t += get_avg_time(h_results, blocks);
        }
        local_t /= TEST_ITERS;
        
        // Remote
        for (int w = 0; w < WARMUP_ITERS; w++) {
            sm_affinity_kernel<<<blocks, 256>>>(d_exp5_data, exp5_elements, 1, d_results, 100);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double remote_t = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            sm_affinity_kernel<<<blocks, 256>>>(d_exp5_data, exp5_elements, 1, d_results, 100);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_results, d_results, blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            remote_t += get_avg_time(h_results, blocks);
        }
        remote_t /= TEST_ITERS;
        
        printf("%8d        %12.0f        %12.0f        %.2fx\n", 
               blocks, local_t, remote_t, remote_t / local_t);
    }
    
    CUDA_CHECK(cudaFree(d_exp5_data));
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("结论\n");
    print_separator();
    printf("\nStitch Traffic 触发条件:\n");
    printf("1. 数据足迹 > C/2 (%.1f MB)\n", half_l2 / 1024.0 / 1024.0);
    printf("2. SM 访问非本地分区的数据\n");
    printf("3. 写操作需要 invalidate 远程 LCN 副本\n");
    printf("4. 频繁的跨分区访问模式切换\n");
    printf("\n优化建议:\n");
    printf("- 尽量保持数据在本地分区\n");
    printf("- 减少跨分区访问的频率\n");
    printf("- 对于写密集操作，使用 shared memory 聚合\n");
    printf("- 考虑使用 uGPU localization 绑定计算和数据\n");
    print_separator();
    
    CUDA_CHECK(cudaFree(d_results));
    
    return 0;
}

