/**
 * NCU Stitch Traffic 简化测试
 * 专门用于 NCU 分析的简化版本
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

// 测试1: 同一区域访问 (无 stitch)
__global__ __noinline__ void same_region(float* data, size_t region_size, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 所有线程访问同一 16MB 区域
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 32 + i) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 测试2: 两个区域交替访问 (触发 stitch)
__global__ __noinline__ void two_regions(float* data, size_t region_size, 
                                          size_t region_offset, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 交替访问区域 A [0, region_size) 和区域 B [region_offset, region_offset+region_size)
    for (int i = 0; i < 1000; i++) {
        size_t base = (i % 2 == 0) ? 0 : region_offset;
        size_t idx = base + (tid * 32 + i / 2) % region_size;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

// 测试3: 全范围访问 (最大数据足迹)
__global__ __noinline__ void full_range(float* data, size_t total_elements, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // 访问整个 120MB 范围
    for (int i = 0; i < 1000; i++) {
        size_t idx = (tid * 32 + i * 1024) % total_elements;
        sum += __ldcg(&data[idx]);
    }
    
    if (tid == 0) *output = sum;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("NCU Stitch Traffic 简化测试\n");
    printf("GPU: %s, L2: %.1f MB\n", prop.name, prop.l2CacheSize / 1024.0 / 1024.0);
    
    size_t l2_size = prop.l2CacheSize;
    size_t buffer_size = 2 * l2_size;  // 120 MB
    size_t buffer_elements = buffer_size / sizeof(float);
    
    float *d_data, *d_output;
    CUDA_CHECK(cudaMalloc(&d_data, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 1, buffer_size));
    
    int num_blocks = 128;
    int block_size = 256;
    size_t region_size = 16 * 1024 * 1024 / sizeof(float);  // 16 MB
    size_t region_offset = 32 * 1024 * 1024 / sizeof(float);  // 32 MB 偏移
    
    printf("\n配置: %d blocks x %d threads\n", num_blocks, block_size);
    printf("区域大小: 16 MB, 区域偏移: 32 MB\n\n");
    
    // Warmup
    same_region<<<num_blocks, block_size>>>(d_data, region_size, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 测试1: same_region
    printf("运行 same_region...\n");
    same_region<<<num_blocks, block_size>>>(d_data, region_size, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 测试2: two_regions
    printf("运行 two_regions...\n");
    two_regions<<<num_blocks, block_size>>>(d_data, region_size, region_offset, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 测试3: full_range
    printf("运行 full_range...\n");
    full_range<<<num_blocks, block_size>>>(d_data, buffer_elements, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n完成! 使用以下命令分析:\n");
    printf("ncu --metrics lts__t_requests_srcunit_ltcfabric.sum,lts__ltcfabric2lts_cycles_active.sum,dram__bytes_read.sum,lts__t_sectors.sum ./ncu_stitch_test\n");
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

