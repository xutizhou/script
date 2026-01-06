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

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr int kNumWarps = 1024;
constexpr int kTotalThreads = kThreadsPerWarp * kNumWarps;
constexpr int kBlockSize = 256;

__global__ void shared_stride_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int stride_elements,
                                     int total_threads) {
    extern __shared__ float shmem[];
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= total_threads) {
        return;
    }

    const size_t shared_index =
        static_cast<size_t>(threadIdx.x) * stride_elements;

    shmem[shared_index] = input[global_tid];
    __syncthreads();

    const float value = shmem[shared_index];
    output[global_tid] = value * 1.0001f;  // prevent compiler elimination
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <stride_in_bytes>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int stride_bytes = std::atoi(argv[1]);
    if (stride_bytes <= 0 || stride_bytes % static_cast<int>(sizeof(float)) != 0) {
        fprintf(stderr,
                "Stride must be a positive multiple of %zu bytes. Got %d.\n",
                sizeof(float), stride_bytes);
        return EXIT_FAILURE;
    }

    const int stride_elements = stride_bytes / static_cast<int>(sizeof(float));
    const size_t element_count = kTotalThreads;

    std::vector<float> h_input(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        h_input[i] = static_cast<float>(i % 1024) * 0.25f;
    }

    std::vector<float> h_output(element_count, 0.0f);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, element_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, element_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          element_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block_dim(kBlockSize);
    const dim3 grid_dim((kTotalThreads + block_dim.x - 1) / block_dim.x);
    const size_t shared_mem_bytes =
        static_cast<size_t>(block_dim.x) * stride_elements * sizeof(float);

    shared_stride_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
        d_input, d_output, stride_elements, kTotalThreads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          element_count * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float value : h_output) {
        checksum += static_cast<double>(value);
    }

    printf("Shared stride experiment complete: stride_bytes=%d stride_elements=%d "
           "threads=%d checksum=%.3f\n",
           stride_bytes, stride_elements, kTotalThreads, checksum);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

