/**
 * NCU Stitch Traffic 定量分析
 * 
 * 关键 NCU 指标:
 * - lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum    : L2 命中
 * - lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum   : L2 未命中
 * - lts__t_sectors_srcunit_tex_op_read_lookup_stitch.sum : Stitch 请求
 * 
 * 实验设计:
 * 1. local_access  - 数据足迹 < L2/2, 无 stitch
 * 2. cross_access  - 数据足迹 > L2/2, 触发 stitch  
 * 3. full_l2       - 数据足迹 = L2, 最大 stitch
 * 4. exceed_l2     - 数据足迹 > L2, stitch + DRAM miss
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

// ============================================================
// 测试1: local_access - 数据足迹 < L2/2 (应该无 stitch)
// ============================================================
__global__ __noinline__ void local_access(float* data, size_t elements, 
                                           size_t footprint_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问限制在 footprint_elements 范围内
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 128 + i * 32) % footprint_elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试2: cross_access - 数据足迹 > L2/2 (触发 stitch)
// ============================================================
__global__ __noinline__ void cross_access(float* data, size_t elements, 
                                           size_t footprint_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问跨越 footprint_elements 范围 (> L2/2)
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 128 + i * 32) % footprint_elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试3: full_l2_access - 数据足迹 = L2 (最大 stitch)
// ============================================================
__global__ __noinline__ void full_l2_access(float* data, size_t elements, 
                                             size_t footprint_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问整个 L2 范围
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 128 + i * 32) % footprint_elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试4: exceed_l2_access - 数据足迹 > L2 (stitch + DRAM)
// ============================================================
__global__ __noinline__ void exceed_l2_access(float* data, size_t elements, 
                                               size_t footprint_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问超过 L2 的范围
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 128 + i * 32) % footprint_elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试5: strided_access - 大步长访问 (跨 partition)
// ============================================================
__global__ __noinline__ void strided_access(float* data, size_t elements, 
                                             size_t stride_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 使用大步长访问，可能跨多个 partition
    for (int i = 0; i < 1000; i++) {
        size_t idx = ((size_t)tid * stride_elements + i * stride_elements) % elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试6: alternating_partition - 交替访问两个区域
// ============================================================
__global__ __noinline__ void alternating_partition(float* data, size_t elements,
                                                    size_t region_size, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 区域 A: [0, region_size)
    // 区域 B: [region_size, 2*region_size)
    // 交替访问
    for (int i = 0; i < 1000; i++) {
        size_t base = (i % 2 == 0) ? 0 : region_size;
        size_t idx = base + (tid * 32 + i * 16) % region_size;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
    }
}

// ============================================================
// 测试7: random_access - 随机访问模式
// ============================================================
__global__ __noinline__ void random_access(float* data, size_t elements, 
                                            unsigned int seed, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    unsigned int rand_state = seed + tid;
    
    for (int i = 0; i < 1000; i++) {
        // 简单的 LCG 随机数
        rand_state = rand_state * 1103515245 + 12345;
        size_t idx = (rand_state >> 16) % elements;
        sum += data[idx];
    }
    
    if (tid == 0) {
        *output = sum;
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
    
    printf("NCU Stitch Traffic 定量分析\n");
    print_separator();
    printf("GPU: %s\n", prop.name);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    print_separator();
    
    size_t l2_size = prop.l2CacheSize;
    size_t l2_half = l2_size / 2;
    
    // 分配 2x L2 大小的缓冲区
    size_t buffer_size = 2 * l2_size;
    size_t buffer_elements = buffer_size / sizeof(float);
    
    float *d_data, *d_output;
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 1, buffer_size));
    
    int num_blocks = 128;
    int block_size = 256;
    
    printf("\n运行测试 kernel (每个 kernel 独立分析)...\n\n");
    
    // Warmup
    local_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                              l2_half / sizeof(float) / 2, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ============================================================
    // 测试 1: local_access (20 MB < L2/2 = 30 MB)
    // ============================================================
    printf("测试1: local_access (20 MB 数据足迹, < L2/2)\n");
    size_t footprint_1 = 20 * 1024 * 1024 / sizeof(float);
    local_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                              footprint_1, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 低 stitch, 高 hit\n\n");
    
    // ============================================================
    // 测试 2: cross_access (40 MB > L2/2 = 30 MB)
    // ============================================================
    printf("测试2: cross_access (40 MB 数据足迹, > L2/2)\n");
    size_t footprint_2 = 40 * 1024 * 1024 / sizeof(float);
    cross_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                              footprint_2, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 中等 stitch\n\n");
    
    // ============================================================
    // 测试 3: full_l2_access (60 MB = L2)
    // ============================================================
    printf("测试3: full_l2_access (60 MB 数据足迹, = L2)\n");
    size_t footprint_3 = 60 * 1024 * 1024 / sizeof(float);
    full_l2_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                                footprint_3, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 高 stitch\n\n");
    
    // ============================================================
    // 测试 4: exceed_l2_access (90 MB > L2)
    // ============================================================
    printf("测试4: exceed_l2_access (90 MB 数据足迹, > L2)\n");
    size_t footprint_4 = 90 * 1024 * 1024 / sizeof(float);
    exceed_l2_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                                  footprint_4, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: stitch + DRAM miss\n\n");
    
    // ============================================================
    // 测试 5: strided_access (步长 = L2/8)
    // ============================================================
    printf("测试5: strided_access (步长 = L2/8 = 7.5 MB)\n");
    size_t stride_5 = l2_size / 8 / sizeof(float);
    strided_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                                stride_5, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 跨 partition 访问\n\n");
    
    // ============================================================
    // 测试 6: alternating_partition (两个 30 MB 区域交替)
    // ============================================================
    printf("测试6: alternating_partition (两个 30 MB 区域交替)\n");
    size_t region_size = l2_half / sizeof(float);
    alternating_partition<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                                       region_size, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 高 stitch (频繁切换)\n\n");
    
    // ============================================================
    // 测试 7: random_access (全范围随机)
    // ============================================================
    printf("测试7: random_access (120 MB 随机访问)\n");
    random_access<<<num_blocks, block_size>>>(d_data, buffer_elements, 
                                               12345, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  预期: 最高 stitch + miss\n\n");
    
    print_separator();
    printf("运行完成！\n\n");
    printf("使用 NCU 分析命令:\n");
    printf("ncu --metrics \\\n");
    printf("  lts__t_sectors_srcunit_tex_op_read.sum,\\\n");
    printf("  lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\\\n");
    printf("  lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\\\n");
    printf("  l2_hit_rate,\\\n");
    printf("  dram__bytes_read.sum \\\n");
    printf("  ./ncu_stitch_analysis\n");
    print_separator();
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

