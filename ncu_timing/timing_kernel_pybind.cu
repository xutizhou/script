/**
 * PyBind11 wrapper for timing_kernel
 * 
 * 编译:
 *   pip install pybind11
 *   nvcc -O3 -arch=sm_90 -shared -Xcompiler -fPIC \
 *        $(python -m pybind11 --includes) \
 *        timing_kernel_pybind.cu \
 *        -o timing_kernel_pybind$(python3-config --extension-suffix)
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 与 timing_kernel.cu 完全相同的 kernel
__global__ void timed_rw_kernel(float* __restrict__ src, 
                                 float* __restrict__ dst, 
                                 size_t num_elements,
                                 int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    // 多次迭代读写来控制运行时间
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = idx; i < num_elements; i += stride) {
            sum += src[i];
        }
    }
    
    // 写回结果，防止编译器优化掉读操作
    for (size_t i = idx; i < num_elements; i += stride) {
        dst[i] = sum / (float)(iterations * num_elements);
    }
}

// PyTorch wrapper 函数
void run_timed_rw_kernel(torch::Tensor src, torch::Tensor dst, 
                          int num_blocks, int threads_per_block, int iterations) {
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32, "src must be float32");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst must be float32");
    
    size_t num_elements = src.numel();
    
    timed_rw_kernel<<<num_blocks, threads_per_block>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        num_elements,
        iterations
    );
    
    // 检查 kernel 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_timed_rw_kernel", &run_timed_rw_kernel, 
          "Run timed read/write kernel",
          py::arg("src"), py::arg("dst"), 
          py::arg("num_blocks"), py::arg("threads_per_block"), py::arg("iterations"));
}

