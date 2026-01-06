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

__global__ void stride_load_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int stride_elements,
                                   int total_threads) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }
    output[tid] = input[static_cast<size_t>(tid) * stride_elements];
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
    const size_t input_elements =
        static_cast<size_t>(kTotalThreads) * stride_elements;

    std::vector<float> h_input(input_elements);
    for (size_t i = 0; i < input_elements; ++i) {
        h_input[i] = static_cast<float>(i % 1024) * 0.5f;
    }

    std::vector<float> h_output(kTotalThreads, 0.0f);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block_dim(kBlockSize);
    const dim3 grid_dim((kTotalThreads + block_dim.x - 1) / block_dim.x);

    stride_load_kernel<<<grid_dim, block_dim>>>(d_input, d_output,
                                                stride_elements, kTotalThreads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float value : h_output) {
        checksum += static_cast<double>(value);
    }

    printf("Completed stride experiment: stride_bytes=%d stride_elements=%d "
           "threads=%d checksum=%.3f\n",
           stride_bytes, stride_elements, kTotalThreads, checksum);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

