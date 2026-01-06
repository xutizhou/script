// CUDA Transpose Benchmark: naive vs shared-memory tiled
// Build: nvcc -O3 transpose_benchmark.cu -o transpose_benchmark
// Run:   ./transpose_benchmark --m 8192 --n 8192 --iters 200

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

#ifndef CHECK_CUDA
#define CHECK_CUDA(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)
#endif

// Kernel 1: naive transpose (global memory only). Reads are coalesced along X, writes are strided (uncoalesced)
__global__ void transpose_naive(float* __restrict__ out, const float* __restrict__ in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index in input
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index in input

    if (x < width && y < height) {
        int in_idx = y * width + x;              // row-major index into input
        int out_idx = x * height + y;            // row-major index into output (transposed dims)
        out[out_idx] = in[in_idx];
    }
}

// Kernel 2: shared-memory tiled transpose with coalesced reads AND writes and bank-conflict avoidance
// Uses a 32x32 tile and writes with padding (+1) to avoid shared memory bank conflicts
#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

#ifndef WCTILE_DIM
#define WCTILE_DIM 16
#endif

#ifndef WCBLOCK_ROWS
#define WCBLOCK_ROWS 8
#endif

template<int UNROLL, int BLOCK_X, int BLOCK_Y>
__global__ void transpose_kernel_optimized(float *input, float *output, int M, int N) {
    constexpr int TILE_M = BLOCK_X;
    constexpr int TILE_N = UNROLL * BLOCK_Y;
    int offset_n = TILE_N * blockIdx.x + threadIdx.y;
    int offset_m = TILE_M * blockIdx.y + threadIdx.x;
    float elements[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        elements[i] = input[offset_m * N + offset_n + i * BLOCK_Y]; // input[offset_m][offset_n + i * BLOCK_X] -> output[offset_n + i * BLOCK_X][offset_m]
    }
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        output[(offset_n + i * BLOCK_Y) * M + offset_m] = elements[i];
    }
}

__global__ void transpose_tiled(float* __restrict__ out, const float* __restrict__ in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 prevents bank conflicts on transpose

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // global x for load
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // global y for load

    // Coalesced loads from input into shared memory tile
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((x < width) && (y + i < height)) {
            tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * width + x];
        }
    }

    __syncthreads();

    // Transposed coordinates for store
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x; // output column index (original row)
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y; // output row index (original column)

    // Coalesced stores from shared memory to output
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((transposed_x < height) && (transposed_y + i < width)) {
            out[(transposed_y + i) * height + transposed_x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// L2 prefetch helper (SM80+). Safe no-op if compiled for lower arch as guarded by inline asm use.
__device__ __forceinline__ void prefetch_l2(const float* addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("prefetch.global.L2 [%0];" :: "l"(addr));
#endif
}

// Enable strong read-only caching for input
#ifndef LDG
#define LDG(x) __ldg(x)
#endif

// Kernel 3: write-coalesced, read non-coalesced (naive). Threads map to output coordinates
// so that stores are coalesced; loads gather a column from input (strided). Relies on L2 cache
// to mitigate read inefficiency, but does not explicitly tile.
__global__ void transpose_write_coalesced_naive(float* __restrict__ out, const float* __restrict__ in, int width, int height) {
    // Output has dimensions [width, height] after transpose. We index output directly.
    int out_x = blockIdx.x * blockDim.x + threadIdx.x; // in [0, height)
    int out_y = blockIdx.y * blockDim.y + threadIdx.y; // in [0, width)

    if (out_x < height && out_y < width) {
        // Corresponding input coordinates (column-major gather)
        int in_x = out_y;   // column in input
        int in_y = out_x;   // row in input
        int in_idx = in_y * width + in_x;      // strided across warp (non-coalesced)
        int out_idx = out_y * height + out_x;  // contiguous across warp (coalesced)
        // Use __ldg on architectures that route via read-only cache; falls back to normal load otherwise
        float v = __ldg(in + in_idx);
        out[out_idx] = v;
    }
}

// Kernel 4: write-coalesced with intra-block blocking across output rows to improve L2 hit-rate.
// Each block covers a TILE_DIM x TILE_DIM tile in output space. Warps (different threadIdx.y) read
// adjacent columns for the same input row nearly back-to-back, increasing chances that row cachelines
// brought by one warp are reused by others from L2. No shared memory is used.
__global__ void transpose_write_coalesced_blocked(float* __restrict__ out, const float* __restrict__ in, int width, int height) {
    int out_x = blockIdx.x * WCTILE_DIM + threadIdx.x; // [0, height)
    int out_y_base = blockIdx.y * WCTILE_DIM + threadIdx.y; // [0, width)

    // Iterate over a WCTILE_DIM stripe of output rows with step WCBLOCK_ROWS. For a fixed thread (fixed out_x),
    // accesses traverse contiguous columns on the same input row across iterations, improving L2 locality.
    for (int j = 0; j < WCTILE_DIM; j += WCBLOCK_ROWS) {
        int out_y = out_y_base + j;
        if (out_x < height && out_y < width) {
            int in_x = out_y;   // column in input
            int in_y = out_x;   // row in input
            int in_idx = in_y * width + in_x;      // strided across warp
            int out_idx = out_y * height + out_x;  // contiguous across warp
            float v = __ldg(in + in_idx);
            out[out_idx] = v;
        }
    }
}

// Kernel 5: write-coalesced blocked with L2 prefetching and unrolling. Aims to boost L2 hit rate by
// prefetching next columns for this block and unrolling the loop to keep multiple requests in flight.
__global__ void transpose_write_coalesced_blocked_opt(float* __restrict__ out, const float* __restrict__ in, int width, int height) {
    const int out_x = blockIdx.x * WCTILE_DIM + threadIdx.x;
    const int out_y0 = blockIdx.y * WCTILE_DIM + threadIdx.y;

    // Prefetch distance across the BLOCK_ROWS sweep
    constexpr int PREFETCH_DIST = WCBLOCK_ROWS * 4;

#pragma unroll
    for (int j = 0; j < WCTILE_DIM; j += WCBLOCK_ROWS) {
        const int out_y = out_y0 + j;
        if (out_x < height && out_y < width) {
            const int in_x = out_y;   // column of input
            const int in_y = out_x;   // row of input
            const int in_idx = in_y * width + in_x;

            // Prefetch a future column for the same row, if in-bounds
            const int pf_x = out_y0 + j + PREFETCH_DIST;
            if (pf_x < width) {
                prefetch_l2(in + in_y * width + pf_x);
            }

            float v = LDG(in + in_idx);
            out[out_y * height + out_x] = v;
        }
    }
}

// Texture-based variant to leverage 2D locality in texture cache for the strided column reads
__global__ void transpose_write_coalesced_blocked_tex(float* __restrict__ out, cudaTextureObject_t tex, int width, int height) {
    const int out_x = blockIdx.x * WCTILE_DIM + threadIdx.x;
    const int out_y0 = blockIdx.y * WCTILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < WCTILE_DIM; j += WCBLOCK_ROWS) {
        const int out_y = out_y0 + j;
        if (out_x < height && out_y < width) {
            const int in_x = out_y;   // column
            const int in_y = out_x;   // row
            // For unnormalized coordinates, tex2D expects coordinates at texel centers (x+0.5, y+0.5)
            float v = tex2D<float>(tex, static_cast<float>(in_x) + 0.5f, static_cast<float>(in_y) + 0.5f);
            out[out_y * height + out_x] = v;
        }
    }
}

static void parse_int_arg(const char* name, int& value, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
            value = std::atoi(argv[i + 1]);
        }
    }
}

static void parse_bool_flag(const char* name, bool& flag, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            flag = true;
        }
    }
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

void launch_kernel_optimized(float *input, float *output, int M, int N) {
    constexpr int UNROLL = 4;
    constexpr int BLOCK_X = 8;
    constexpr int BLOCK_Y = 16;
    static dim3 blockDim(BLOCK_X, BLOCK_Y);
    static int tile_m = BLOCK_X;
    static int tile_n = BLOCK_Y * UNROLL;
    static dim3 gridDim((N + tile_n - 1) / tile_n, (M + tile_m - 1) / tile_m);
    transpose_kernel_optimized<UNROLL, BLOCK_X, BLOCK_Y><<<gridDim, blockDim>>>(input, output, M, N);
}

// Stream-enabled wrapper to integrate with our timed stream
void launch_kernel_optimized_stream(float *input, float *output, int M, int N, cudaStream_t stream) {
    constexpr int UNROLL = 4;
    constexpr int BLOCK_X = 8;
    constexpr int BLOCK_Y = 16;
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    const int tile_m = BLOCK_X;
    const int tile_n = BLOCK_Y * UNROLL;
    dim3 gridDim((N + tile_n - 1) / tile_n, (M + tile_m - 1) / tile_m);
    transpose_kernel_optimized<UNROLL, BLOCK_X, BLOCK_Y><<<gridDim, blockDim, 0, stream>>>(input, output, M, N);
}

static void benchmark_kernels(int height, int width, int iters, bool verify) {
    const size_t numel_in = static_cast<size_t>(height) * static_cast<size_t>(width);
    const size_t bytes_in = numel_in * sizeof(float);
    const size_t numel_out = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t bytes_out = numel_out * sizeof(float);

    std::printf("Matrix size: %d x %d (%.2f MB input, %.2f MB output)\n",
                height, width, bytes_in / (1024.0 * 1024.0), bytes_out / (1024.0 * 1024.0));

    // Host init
    std::vector<float> h_in(numel_in);
    for (size_t i = 0; i < numel_in; ++i) h_in[i] = static_cast<float>(i % 1337) * 0.5f;

    float *d_in = nullptr, *d_out_naive = nullptr, *d_out_tiled = nullptr;
    float *d_out_wc_naive = nullptr, *d_out_wc_blocked = nullptr;
    float *d_out_wc_blocked_opt = nullptr, *d_out_wc_blocked_tex = nullptr;
    float *d_out_opt = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out_naive, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_tiled, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_wc_naive, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_wc_blocked, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_wc_blocked_opt, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_wc_blocked_tex, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_out_opt, bytes_out));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes_in, cudaMemcpyHostToDevice));

    // Create a dedicated stream and try to set an L2 persisting window for the input region
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr = d_in;
        // Size to a fraction of device L2 if available, else fall back to 64MB
        cudaDeviceProp propA{};
        CHECK_CUDA(cudaGetDeviceProperties(&propA, 0));
        // Empirically 1/2 L2 works well for large streaming reads
        size_t window = propA.l2CacheSize ? (propA.l2CacheSize / 2) : ((size_t)64 * 1024 * 1024);
        if (window > bytes_in) window = bytes_in;
        attr.accessPolicyWindow.num_bytes = window;
        attr.accessPolicyWindow.hitRatio = 1.0f; // prefer to keep
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaError_t apw_err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        if (apw_err != cudaSuccess) {
            // ignore if unsupported
        }
    }

    dim3 block_naive(32, 8);
    dim3 grid_naive((width + block_naive.x - 1) / block_naive.x,
                    (height + block_naive.y - 1) / block_naive.y);

    dim3 block_tiled(TILE_DIM, BLOCK_ROWS);
    dim3 grid_tiled((width + TILE_DIM - 1) / TILE_DIM,
                    (height + TILE_DIM - 1) / TILE_DIM);

    // Write-coalesced variants operate in output space: width' = height, height' = width
    dim3 block_wc(32, 8);
    dim3 grid_wc((height + block_wc.x - 1) / block_wc.x,
                 (width + block_wc.y - 1) / block_wc.y);
    dim3 block_wc_blocked(WCTILE_DIM, WCBLOCK_ROWS);
    dim3 grid_wc_blocked((height + WCTILE_DIM - 1) / WCTILE_DIM,
                         (width + WCTILE_DIM - 1) / WCTILE_DIM);

    // Opt variant uses same shape by default; tweak here if needed
    dim3 block_wc_blocked_opt(WCTILE_DIM, WCBLOCK_ROWS);
    dim3 grid_wc_blocked_opt((height + WCTILE_DIM - 1) / WCTILE_DIM,
                             (width + WCTILE_DIM - 1) / WCTILE_DIM);

    // Create a texture object bound to a cudaArray (often better 2D locality handling)
    cudaTextureObject_t in_tex = 0;
    cudaArray_t arr = nullptr;
    {
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
        CHECK_CUDA(cudaMallocArray(&arr, &ch, width, height, cudaArrayDefault));
        CHECK_CUDA(cudaMemcpy2DToArray(arr, 0, 0, d_in, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToDevice));

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arr;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        CHECK_CUDA(cudaCreateTextureObject(&in_tex, &resDesc, &texDesc, nullptr));
    }

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        transpose_naive<<<grid_naive, block_naive, 0, stream>>>(d_out_naive, d_in, width, height);
        transpose_tiled<<<grid_tiled, block_tiled, 0, stream>>>(d_out_tiled, d_in, width, height);
        transpose_write_coalesced_naive<<<grid_wc, block_wc, 0, stream>>>(d_out_wc_naive, d_in, width, height);
        transpose_write_coalesced_blocked<<<grid_wc_blocked, block_wc_blocked, 0, stream>>>(d_out_wc_blocked, d_in, width, height);
        transpose_write_coalesced_blocked_opt<<<grid_wc_blocked_opt, block_wc_blocked_opt, 0, stream>>>(d_out_wc_blocked_opt, d_in, width, height);
        transpose_write_coalesced_blocked_tex<<<grid_wc_blocked, block_wc_blocked, 0, stream>>>(d_out_wc_blocked_tex, in_tex, width, height);
        launch_kernel_optimized_stream(d_in, d_out_opt, height, width, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Benchmark naive (read-coalesced, write-uncoalesced)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_naive<<<grid_naive, block_naive, 0, stream>>>(d_out_naive, d_in, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double naive_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark tiled
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_tiled<<<grid_tiled, block_tiled, 0, stream>>>(d_out_tiled, d_in, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double tiled_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark write-coalesced naive (read non-coalesced)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_write_coalesced_naive<<<grid_wc, block_wc, 0, stream>>>(d_out_wc_naive, d_in, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double wc_naive_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark write-coalesced blocked (read non-coalesced with blocking for L2)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_write_coalesced_blocked<<<grid_wc_blocked, block_wc_blocked, 0, stream>>>(d_out_wc_blocked, d_in, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double wc_blocked_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark write-coalesced blocked + prefetch/unroll
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_write_coalesced_blocked_opt<<<grid_wc_blocked_opt, block_wc_blocked_opt, 0, stream>>>(d_out_wc_blocked_opt, d_in, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double wc_blocked_opt_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark texture-based
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        transpose_write_coalesced_blocked_tex<<<grid_wc_blocked, block_wc_blocked, 0, stream>>>(d_out_wc_blocked_tex, in_tex, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double wc_blocked_tex_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Benchmark launch_kernel_optimized
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch_kernel_optimized_stream(d_in, d_out_opt, height, width, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    double opt_ms = elapsed_ms(start, stop) / static_cast<double>(iters);

    // Bandwidth: 2 * bytes moved (read + write)
    const double bytes_moved = static_cast<double>(bytes_in + bytes_out); // equal sizes
    const double naive_bw = (bytes_moved / naive_ms) * 1e-6; // GB/s
    const double tiled_bw = (bytes_moved / tiled_ms) * 1e-6; // GB/s
    const double wc_naive_bw = (bytes_moved / wc_naive_ms) * 1e-6; // GB/s
    const double wc_blocked_bw = (bytes_moved / wc_blocked_ms) * 1e-6; // GB/s
    const double wc_blocked_opt_bw = (bytes_moved / wc_blocked_opt_ms) * 1e-6; // GB/s
    const double wc_blocked_tex_bw = (bytes_moved / wc_blocked_tex_ms) * 1e-6; // GB/s
    const double opt_bw = (bytes_moved / opt_ms) * 1e-6; // GB/s

    std::printf("Naive R-coalesced/W-uncoalesced: %.3f ms, %.2f GB/s\n", naive_ms, naive_bw);
    std::printf("Tiled (shared mem):              %.3f ms, %.2f GB/s\n", tiled_ms, tiled_bw);
    std::printf("Write-coalesced naive:           %.3f ms, %.2f GB/s\n", wc_naive_ms, wc_naive_bw);
    std::printf("Write-coalesced blocked:         %.3f ms, %.2f GB/s\n", wc_blocked_ms, wc_blocked_bw);
    std::printf("WC-blocked + prefetch/unroll:    %.3f ms, %.2f GB/s\n", wc_blocked_opt_ms, wc_blocked_opt_bw);
    std::printf("WC-blocked + texture:            %.3f ms, %.2f GB/s\n", wc_blocked_tex_ms, wc_blocked_tex_bw);
    std::printf("launch_kernel_optimized:         %.3f ms, %.2f GB/s\n", opt_ms, opt_bw);
    std::printf("Tiled speedup vs naive:          %.2fx\n", naive_ms / tiled_ms);
    std::printf("WC-blocked vs WC-naive:          %.2fx\n", wc_naive_ms / wc_blocked_ms);
    std::printf("WC-opt vs WC-naive:              %.2fx\n", wc_naive_ms / wc_blocked_opt_ms);
    std::printf("WC-tex vs WC-naive:              %.2fx\n", wc_naive_ms / wc_blocked_tex_ms);

    if (verify) {
        std::vector<float> h_out_naive(numel_out);
        std::vector<float> h_out_tiled(numel_out);
        std::vector<float> h_out_wc_naive(numel_out);
        std::vector<float> h_out_wc_blocked(numel_out);
        std::vector<float> h_out_wc_blocked_opt(numel_out);
        std::vector<float> h_out_wc_blocked_tex(numel_out);
        CHECK_CUDA(cudaMemcpy(h_out_naive.data(), d_out_naive, bytes_out, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out_tiled.data(), d_out_tiled, bytes_out, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out_wc_naive.data(), d_out_wc_naive, bytes_out, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out_wc_blocked.data(), d_out_wc_blocked, bytes_out, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out_wc_blocked_opt.data(), d_out_wc_blocked_opt, bytes_out, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_out_wc_blocked_tex.data(), d_out_wc_blocked_tex, bytes_out, cudaMemcpyDeviceToHost));
        std::vector<float> h_out_opt(numel_out);
        CHECK_CUDA(cudaMemcpy(h_out_opt.data(), d_out_opt, bytes_out, cudaMemcpyDeviceToHost));

        size_t mismatches = 0;
        for (size_t i = 0; i < numel_out; ++i) {
            if (h_out_naive[i] != h_out_tiled[i]) { // exact compare OK (copy-only kernels)
                ++mismatches;
                if (mismatches < 10) {
                    std::printf("Mismatch at %zu: naive=%f, tiled=%f\n", i, h_out_naive[i], h_out_tiled[i]);
                }
            }
        }
        for (size_t i = 0; i < numel_out; ++i) {
            if (h_out_wc_naive[i] != h_out_tiled[i]) {
                ++mismatches;
                if (mismatches < 10) {
                    std::printf("Mismatch (wc_naive vs tiled) at %zu: wc_naive=%f, tiled=%f\n", i, h_out_wc_naive[i], h_out_tiled[i]);
                }
            }
        }
        for (size_t i = 0; i < numel_out; ++i) {
            if (h_out_wc_blocked[i] != h_out_tiled[i]) {
                ++mismatches;
                if (mismatches < 10) {
                    std::printf("Mismatch (wc_blocked vs tiled) at %zu: wc_blocked=%f, tiled=%f\n", i, h_out_wc_blocked[i], h_out_tiled[i]);
                }
            }
        }
        for (size_t i = 0; i < numel_out; ++i) {
            if (h_out_wc_blocked_opt[i] != h_out_tiled[i]) {
                ++mismatches;
                if (mismatches < 10) {
                    std::printf("Mismatch (wc_blocked_opt vs tiled) at %zu: wc_blocked_opt=%f, tiled=%f\n", i, h_out_wc_blocked_opt[i], h_out_tiled[i]);
                }
            }
        }
        for (size_t i = 0; i < numel_out; ++i) {
            if (h_out_wc_blocked_tex[i] != h_out_tiled[i]) {
                ++mismatches;
                if (mismatches < 10) {
                    std::printf("Mismatch (wc_blocked_tex vs tiled) at %zu: wc_blocked_tex=%f, tiled=%f\n", i, h_out_wc_blocked_tex[i], h_out_tiled[i]);
                }
            }
        }
        if (mismatches == 0) {
            std::printf("Verification: PASS\n");
        } else {
            std::printf("Verification: FAIL (mismatches=%zu)\n", mismatches);
        }
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    if (in_tex) cudaDestroyTextureObject(in_tex);
    if (arr) cudaFreeArray(arr);
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out_naive));
    CHECK_CUDA(cudaFree(d_out_tiled));
    CHECK_CUDA(cudaFree(d_out_wc_naive));
    CHECK_CUDA(cudaFree(d_out_wc_blocked));
    CHECK_CUDA(cudaFree(d_out_wc_blocked_opt));
    CHECK_CUDA(cudaFree(d_out_wc_blocked_tex));
    CHECK_CUDA(cudaFree(d_out_opt));
}

int main(int argc, char** argv) {
    int device = 0;
    int m = 8192;     // height (rows)
    int n = 8192;     // width (cols)
    int iters = 200;  // iterations per kernel
    bool verify = false;

    parse_int_arg("--device", device, argc, argv);
    parse_int_arg("--m", m, argc, argv);
    parse_int_arg("--n", n, argc, argv);
    parse_int_arg("--iters", iters, argc, argv);
    parse_bool_flag("--verify", verify, argc, argv);

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("Device %d: %s, SMs=%d, globalMem=%.1f GB, sharedMem/SM=%zu KB\n",
                device, prop.name, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.sharedMemPerMultiprocessor / 1024);

    // Try reserving part of L2 for persisting window (if supported)
    if (prop.l2CacheSize > 0) {
        size_t persist_bytes = (prop.l2CacheSize * 3) / 4; // 75% of L2
        cudaError_t lim_err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_bytes);
        if (lim_err != cudaSuccess) {
            // ignore if unsupported
        }
    }

    benchmark_kernels(m, n, iters, verify);

    return 0;
}


