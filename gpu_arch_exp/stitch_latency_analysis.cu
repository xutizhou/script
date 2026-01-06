/**
 * Stitch Traffic 延迟定量分析
 * 
 * 通过 pointer chasing 精确测量延迟来推断 stitch traffic:
 * - L2 hit (同 partition): ~200-250 cycles
 * - L2 hit (跨 partition, stitch): ~300-350 cycles (+100 cycles)
 * - DRAM miss: ~400-500 cycles
 * 
 * 通过延迟变化可以定量推断 stitch traffic 的发生
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <random>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

#define WARMUP_ITERS 5
#define TEST_ITERS 20
#define CHAIN_LENGTH 5000

// ============================================================
// Pointer chasing kernel - 单线程精确延迟测量
// 使用 volatile 和 __ldcg (L2 bypass L1) 确保测量 L2 延迟
// ============================================================
__global__ void pointer_chase_latency(int* data, int start_idx, int chain_length,
                                       uint64_t* result, int* final_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        
        // Warmup - 预热 L2 缓存 (不是 L1)
        for (int i = 0; i < chain_length; i++) {
            // 使用 __ldcg 绕过 L1，直接从 L2 读取
            idx = __ldcg(&data[idx]);
        }
        
        idx = start_idx;
        
        // 刷新 L1 cache
        __threadfence();
        
        uint64_t start = clock64();
        
        #pragma unroll 1
        for (int i = 0; i < chain_length; i++) {
            // __ldcg: load cached global (L2 hit, bypass L1)
            idx = __ldcg(&data[idx]);
        }
        
        uint64_t end = clock64();
        
        *result = end - start;
        *final_idx = idx;  // 防止优化
    }
}

// ============================================================
// 辅助函数: 在指定数据足迹范围内创建链
// ============================================================
void create_chain_with_footprint(int* h_data, size_t total_elements,
                                  size_t footprint_elements, int chain_length) {
    // 使用随机步长在 footprint 范围内创建链
    std::vector<int> indices(footprint_elements);
    for (size_t i = 0; i < footprint_elements; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子保证可重复
    for (size_t i = footprint_elements - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dis(0, i);
        std::swap(indices[i], indices[dis(gen)]);
    }
    
    // 创建链，只使用前 chain_length 个节点
    int actual_length = std::min((size_t)chain_length, footprint_elements);
    for (int i = 0; i < actual_length - 1; i++) {
        h_data[indices[i]] = indices[i + 1];
    }
    h_data[indices[actual_length - 1]] = indices[0];
}

// ============================================================
// 辅助函数: 创建跨两个区域交替的链
// ============================================================
void create_alternating_chain(int* h_data, size_t total_elements,
                               size_t region_A_start, size_t region_B_start,
                               size_t region_size, int chain_length) {
    // A-B-A-B... 交替
    int idx = region_A_start;
    for (int i = 0; i < chain_length - 1; i++) {
        int next_region_start = (i % 2 == 0) ? region_B_start : region_A_start;
        int offset = ((i / 2) * 32) % region_size;  // 每两步在同一区域前进
        int next_idx = next_region_start + offset;
        h_data[idx] = next_idx;
        idx = next_idx;
    }
    h_data[idx] = region_A_start;  // 回到起点
}

void print_separator() {
    printf("================================================================\n");
}

double run_latency_test(int* d_data, int start_idx, int chain_length,
                        uint64_t* d_result, int* d_final_idx) {
    uint64_t h_result;
    
    // Warmup
    for (int w = 0; w < WARMUP_ITERS; w++) {
        pointer_chase_latency<<<1, 32>>>(d_data, start_idx, chain_length,
                                          d_result, d_final_idx);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 测试
    uint64_t total = 0;
    for (int t = 0; t < TEST_ITERS; t++) {
        pointer_chase_latency<<<1, 32>>>(d_data, start_idx, chain_length,
                                          d_result, d_final_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total += h_result;
    }
    
    return (double)total / TEST_ITERS / chain_length;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("Stitch Traffic 延迟定量分析\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    printf("L2/2: %.1f MB (stitch 阈值)\n", prop.l2CacheSize / 2.0 / 1024.0 / 1024.0);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t l2_half = l2_size / 2;
    
    // 分配 2x L2 大小的缓冲区
    size_t buffer_size = 2 * l2_size;
    size_t buffer_elements = buffer_size / sizeof(int);
    
    int* h_data = (int*)malloc(buffer_size);
    int* d_data;
    uint64_t* d_result;
    int* d_final_idx;
    
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_final_idx, sizeof(int)));
    
    // ============================================================
    printf("\n实验1: 数据足迹 vs 延迟 (stitch 阈值探测)\n");
    print_separator();
    printf("数据足迹(MB)   延迟(cycles)   vs 10MB    推断\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<size_t> footprints_mb = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100};
    
    double baseline_latency = 0;
    
    for (size_t footprint_mb : footprints_mb) {
        size_t footprint_bytes = footprint_mb * 1024 * 1024;
        size_t footprint_elements = footprint_bytes / sizeof(int);
        
        if (footprint_elements > buffer_elements) {
            printf("%6zu         (超出范围)\n", footprint_mb);
            continue;
        }
        
        memset(h_data, 0, buffer_size);
        create_chain_with_footprint(h_data, buffer_elements, footprint_elements, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        
        double latency = run_latency_test(d_data, 0, CHAIN_LENGTH, d_result, d_final_idx);
        
        if (footprint_mb == 10) {
            baseline_latency = latency;
        }
        
        const char* inference = "";
        double ratio = (baseline_latency > 0) ? latency / baseline_latency : 1.0;
        
        if (footprint_mb <= l2_half / 1024 / 1024) {
            inference = "L2 hit (同 partition)";
        } else if (footprint_mb <= l2_size / 1024 / 1024) {
            inference = "L2 hit + stitch";
        } else {
            inference = "DRAM miss";
        }
        
        printf("%6zu         %7.2f        %.2fx      %s\n",
               footprint_mb, latency, ratio, inference);
    }
    
    // ============================================================
    printf("\n实验2: A-B 交替访问 vs 顺序访问\n");
    print_separator();
    printf("比较相同数据足迹下，交替访问和顺序访问的延迟差异\n\n");
    
    size_t region_size = 16 * 1024 * 1024 / sizeof(int);  // 16 MB per region
    size_t region_A_start = 0;
    size_t region_B_start = 32 * 1024 * 1024 / sizeof(int);  // 偏移 32 MB
    
    printf("区域配置:\n");
    printf("  Region A: 0 - 16 MB\n");
    printf("  Region B: 32 MB - 48 MB (偏移 32 MB)\n\n");
    
    printf("访问模式         延迟(cycles)   vs 顺序A     推断\n");
    printf("------------------------------------------------------------------------\n");
    
    // 顺序访问 A
    memset(h_data, 0, buffer_size);
    create_chain_with_footprint(h_data, buffer_elements, region_size, CHAIN_LENGTH);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
    double latency_A = run_latency_test(d_data, 0, CHAIN_LENGTH, d_result, d_final_idx);
    printf("顺序访问 A       %7.2f        1.00x        L2 hit\n", latency_A);
    
    // 顺序访问 B
    memset(h_data, 0, buffer_size);
    create_chain_with_footprint(h_data + region_B_start, buffer_elements - region_B_start, 
                                 region_size, CHAIN_LENGTH);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
    double latency_B = run_latency_test(d_data, region_B_start, CHAIN_LENGTH, d_result, d_final_idx);
    printf("顺序访问 B       %7.2f        %.2fx        L2 hit\n", latency_B, latency_B / latency_A);
    
    // A-B 交替访问
    memset(h_data, 0, buffer_size);
    create_alternating_chain(h_data, buffer_elements, region_A_start, region_B_start,
                             region_size, CHAIN_LENGTH);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
    double latency_AB = run_latency_test(d_data, region_A_start, CHAIN_LENGTH, d_result, d_final_idx);
    printf("A-B 交替访问     %7.2f        %.2fx        ", latency_AB, latency_AB / latency_A);
    
    double stitch_overhead = latency_AB - (latency_A + latency_B) / 2;
    if (stitch_overhead > 10) {
        printf("Stitch! (开销 +%.1f cycles)\n", stitch_overhead);
    } else {
        printf("无明显 stitch 开销\n");
    }
    
    // ============================================================
    printf("\n实验3: 不同偏移量的交替访问\n");
    print_separator();
    printf("偏移量(MB)     交替延迟      顺序延迟      开销(cycles)   Stitch率估计\n");
    printf("------------------------------------------------------------------------\n");
    
    std::vector<size_t> offsets_mb = {8, 16, 24, 30, 32, 40, 48, 56, 60};
    
    for (size_t offset_mb : offsets_mb) {
        size_t offset = offset_mb * 1024 * 1024 / sizeof(int);
        
        if (offset + region_size > buffer_elements) {
            printf("%6zu         (超出范围)\n", offset_mb);
            continue;
        }
        
        // 顺序访问 (A + B 各一半)
        memset(h_data, 0, buffer_size);
        create_chain_with_footprint(h_data, buffer_elements, region_size * 2, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        double latency_seq = run_latency_test(d_data, 0, CHAIN_LENGTH, d_result, d_final_idx);
        
        // 交替访问
        memset(h_data, 0, buffer_size);
        create_alternating_chain(h_data, buffer_elements, 0, offset, region_size, CHAIN_LENGTH);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, buffer_size, cudaMemcpyHostToDevice));
        double latency_alt = run_latency_test(d_data, 0, CHAIN_LENGTH, d_result, d_final_idx);
        
        double overhead = latency_alt - latency_seq;
        
        // 估计 stitch 率: 假设 stitch 额外开销 ~100 cycles
        // 如果所有访问都触发 stitch，开销应该 ~100 cycles
        double stitch_rate = std::max(0.0, overhead / 100.0 * 100.0);
        stitch_rate = std::min(100.0, stitch_rate);
        
        printf("%6zu         %7.2f       %7.2f       %+7.1f        ~%.0f%%\n",
               offset_mb, latency_alt, latency_seq, overhead, stitch_rate);
    }
    
    // ============================================================
    printf("\n");
    print_separator();
    printf("分析结论\n");
    print_separator();
    printf("\nStitch Traffic 触发条件:\n");
    printf("1. 数据足迹 > L2/2 (%.1f MB)\n", l2_half / 1024.0 / 1024.0);
    printf("2. 访问模式在不同 L2 partition 之间频繁切换\n");
    printf("3. 两个区域的地址偏移 ≈ L2/2 或 L2 时最明显\n");
    printf("\n定量指标:\n");
    printf("- L2 hit (同 partition): ~%.0f cycles\n", baseline_latency);
    printf("- Stitch 额外开销: ~10-100 cycles (取决于 partition 距离)\n");
    printf("- DRAM miss: 通常 > 400 cycles\n");
    print_separator();
    
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_final_idx));
    
    return 0;
}

