/**
 * Stitch Bandwidth 竞争分析
 * 
 * 当多个 SM 同时访问不同 L2 partition 时，
 * stitch 带宽可能成为瓶颈
 * 
 * 实验设计:
 * 1. 单 SM 访问单一区域 (基准)
 * 2. 多 SM 访问同一区域 (L2 带宽竞争)
 * 3. 多 SM 访问不同区域 (stitch 带宽竞争)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

#define ITERATIONS 1000
#define WARMUP_ITERS 5
#define TEST_ITERS 10

// ============================================================
// 内存带宽测试 kernel
// ============================================================
__global__ void bandwidth_test(float* data, size_t elements_per_block,
                               size_t block_offset_elements,
                               float* output, uint64_t* timing) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 每个 block 访问自己的区域
    size_t base = bid * block_offset_elements;
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (tid == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    // 访问数据
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = base + (tid * 32 + i) % elements_per_block;
            sum += __ldcg(&data[idx]);
        }
    }
    
    __syncthreads();
    if (tid == 0) {
        end_time = clock64();
        timing[bid] = end_time - start_time;
    }
    
    // 防止优化
    if (sum == -999999.0f) output[bid] = sum;
}

// ============================================================
// 同一区域访问 (所有 block 访问相同数据)
// ============================================================
__global__ void same_region_test(float* data, size_t region_size,
                                  float* output, uint64_t* timing) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 所有 block 访问同一区域 [0, region_size)
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (tid == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = (bid * 256 + tid * 32 + i) % region_size;
            sum += __ldcg(&data[idx]);
        }
    }
    
    __syncthreads();
    if (tid == 0) {
        end_time = clock64();
        timing[bid] = end_time - start_time;
    }
    
    if (sum == -999999.0f) output[bid] = sum;
}

// ============================================================
// 不同区域访问 (每个 block 访问不同区域，跨 partition)
// ============================================================
__global__ void different_region_test(float* data, size_t region_size,
                                       size_t region_offset,
                                       float* output, uint64_t* timing) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 每个 block 访问不同区域
    // Block 0: [0, region_size)
    // Block 1: [region_offset, region_offset + region_size)
    // Block 2: [2*region_offset, 2*region_offset + region_size)
    // ...
    size_t base = (bid % 2) * region_offset;  // 交替在两个区域
    
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (tid == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 32; i++) {
            size_t idx = base + (tid * 32 + i) % region_size;
            sum += __ldcg(&data[idx]);
        }
    }
    
    __syncthreads();
    if (tid == 0) {
        end_time = clock64();
        timing[bid] = end_time - start_time;
    }
    
    if (sum == -999999.0f) output[bid] = sum;
}

// ============================================================
// 激进交替访问 (每个线程在两个区域间交替)
// ============================================================
__global__ void aggressive_alternating_test(float* data, size_t region_size,
                                             size_t region_offset,
                                             float* output, uint64_t* timing) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    float sum = 0.0f;
    
    __shared__ uint64_t start_time, end_time;
    
    if (tid == 0) {
        start_time = clock64();
    }
    __syncthreads();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 32; i++) {
            // 每次访问切换区域
            size_t base = (i % 2) * region_offset;
            size_t idx = base + (tid * 16 + i / 2) % region_size;
            sum += __ldcg(&data[idx]);
        }
    }
    
    __syncthreads();
    if (tid == 0) {
        end_time = clock64();
        timing[bid] = end_time - start_time;
    }
    
    if (sum == -999999.0f) output[bid] = sum;
}

void print_separator() {
    printf("================================================================\n");
}

double get_avg_time(uint64_t* h_timing, int num_blocks) {
    uint64_t total = 0;
    for (int i = 0; i < num_blocks; i++) {
        total += h_timing[i];
    }
    return (double)total / num_blocks;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("Stitch Bandwidth 竞争分析\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    
    // 分配 2x L2 大小的缓冲区
    size_t buffer_size = 2 * l2_size;
    size_t buffer_elements = buffer_size / sizeof(float);
    
    float *d_data, *d_output;
    uint64_t *d_timing, *h_timing;
    
    int max_blocks = 256;
    
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_output, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_timing, max_blocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_data, 1, buffer_size));
    
    h_timing = (uint64_t*)malloc(max_blocks * sizeof(uint64_t));
    
    int block_size = 256;
    size_t region_size = 16 * 1024 * 1024 / sizeof(float);  // 16 MB
    size_t region_offset = 32 * 1024 * 1024 / sizeof(float);  // 32 MB 偏移
    
    // ============================================================
    printf("\n实验1: 并发度 vs 延迟 (同一区域)\n");
    print_separator();
    printf("Block数   平均延迟(cycles)   相对1 block\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<int> block_counts = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    double baseline_same = 0;
    
    for (int num_blocks : block_counts) {
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++) {
            same_region_test<<<num_blocks, block_size>>>(d_data, region_size, d_output, d_timing);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            same_region_test<<<num_blocks, block_size>>>(d_data, region_size, d_output, d_timing);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += get_avg_time(h_timing, num_blocks);
        }
        double avg_time = total / TEST_ITERS;
        
        if (num_blocks == 1) baseline_same = avg_time;
        
        printf("%5d     %12.0f       %.2fx\n", num_blocks, avg_time, avg_time / baseline_same);
    }
    
    // ============================================================
    printf("\n实验2: 并发度 vs 延迟 (不同区域, 交替访问)\n");
    print_separator();
    printf("Block数   平均延迟(cycles)   相对1 block   vs 同区域\n");
    printf("------------------------------------------------------------------------\n");
    
    double baseline_diff = 0;
    
    for (int num_blocks : block_counts) {
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++) {
            different_region_test<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            different_region_test<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += get_avg_time(h_timing, num_blocks);
        }
        double avg_time = total / TEST_ITERS;
        
        if (num_blocks == 1) baseline_diff = avg_time;
        
        // 获取同区域对照
        same_region_test<<<num_blocks, block_size>>>(d_data, region_size, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double same_time = get_avg_time(h_timing, num_blocks);
        
        printf("%5d     %12.0f       %.2fx         %.2fx\n", 
               num_blocks, avg_time, avg_time / baseline_diff, avg_time / same_time);
    }
    
    // ============================================================
    printf("\n实验3: 激进交替访问 (每次访问切换区域)\n");
    print_separator();
    printf("Block数   平均延迟(cycles)   vs 同区域    vs 温和交替\n");
    printf("------------------------------------------------------------------------\n");
    
    for (int num_blocks : block_counts) {
        // 同区域对照
        same_region_test<<<num_blocks, block_size>>>(d_data, region_size, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double same_time = get_avg_time(h_timing, num_blocks);
        
        // 温和交替
        different_region_test<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double mild_time = get_avg_time(h_timing, num_blocks);
        
        // 激进交替
        for (int w = 0; w < WARMUP_ITERS; w++) {
            aggressive_alternating_test<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double total = 0;
        for (int t = 0; t < TEST_ITERS; t++) {
            aggressive_alternating_test<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output, d_timing);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            total += get_avg_time(h_timing, num_blocks);
        }
        double aggressive_time = total / TEST_ITERS;
        
        printf("%5d     %12.0f       %.2fx        %.2fx\n", 
               num_blocks, aggressive_time, aggressive_time / same_time, aggressive_time / mild_time);
    }
    
    // ============================================================
    printf("\n实验4: 不同区域偏移量的影响\n");
    print_separator();
    printf("偏移(MB)   同区域延迟   交替延迟     差异      推断\n");
    printf("------------------------------------------------------------------------\n");
    
    int num_blocks = 64;
    std::vector<size_t> offsets_mb = {8, 16, 24, 30, 32, 40, 48, 56, 60};
    
    for (size_t offset_mb : offsets_mb) {
        size_t offset = offset_mb * 1024 * 1024 / sizeof(float);
        
        // 同区域
        same_region_test<<<num_blocks, block_size>>>(d_data, region_size, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double same_time = get_avg_time(h_timing, num_blocks);
        
        // 交替
        different_region_test<<<num_blocks, block_size>>>(d_data, region_size, offset, d_output, d_timing);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_timing, d_timing, num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double diff_time = get_avg_time(h_timing, num_blocks);
        
        double delta = diff_time - same_time;
        const char* inference = "";
        if (delta > 100000) {
            inference = "高 stitch 开销";
        } else if (delta > 10000) {
            inference = "中等 stitch";
        } else if (delta > 0) {
            inference = "轻微 stitch";
        } else {
            inference = "无 stitch";
        }
        
        printf("%6zu     %10.0f   %10.0f   %+8.0f    %s\n",
               offset_mb, same_time, diff_time, delta, inference);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("分析结论\n");
    print_separator();
    printf("\n如果实验2/3的交替访问延迟明显高于同区域访问:\n");
    printf("  → 存在 Stitch 带宽竞争\n");
    printf("\n如果实验4的某些偏移量导致更高延迟:\n");
    printf("  → 这些偏移量跨越了 L2 partition 边界\n");
    print_separator();
    
    free(h_timing);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_timing));
    
    return 0;
}

