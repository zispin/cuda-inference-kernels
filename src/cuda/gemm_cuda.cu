#include "raif/gemm.h"
#include "raif/cuda_utils.h"

#include <cuda_runtime.h>

#ifdef RAIF_WITH_CUBLAS
#include <cublas_v2.h>
#endif

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raif {
namespace {

constexpr int kGemmTile = 16;

__global__ void gemm_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    std::size_t m,
    std::size_t n,
    std::size_t k) {

    __shared__ float tile_a[kGemmTile][kGemmTile];
    __shared__ float tile_b[kGemmTile][kGemmTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y * kGemmTile + ty);
    const std::size_t col = static_cast<std::size_t>(blockIdx.x * kGemmTile + tx);

    float acc = 0.0f;

    for (std::size_t tile = 0; tile < k; tile += kGemmTile) {
        const std::size_t a_col = tile + static_cast<std::size_t>(tx);
        const std::size_t b_row = tile + static_cast<std::size_t>(ty);

        tile_a[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < kGemmTile; ++inner) {
            acc += tile_a[ty][inner] * tile_b[inner][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

struct DeviceFloatBuffer {
    float* ptr = nullptr;

    explicit DeviceFloatBuffer(std::size_t count) {
        if (count > 0) {
            RAIF_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
        }
    }

    DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;

    ~DeviceFloatBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

void validate_gemm_args(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const char* function_name) {

    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(std::string(function_name) + " received a null matrix pointer");
    }
    if (m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        k > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(function_name) + " currently supports at most INT_MAX rows/cols/reduction dimension");
    }
}

void fill_gemm_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int centered = static_cast<int>(i % 257) - 128;
        a[i] = static_cast<float>(centered) * 0.0025f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        const int centered = static_cast<int>((i * 17) % 251) - 125;
        b[i] = static_cast<float>(centered) * 0.0020f;
    }
}

#ifdef RAIF_WITH_CUBLAS
void check_cublas(cublasStatus_t status, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error while running `") + expr + "`, status=" + std::to_string(static_cast<int>(status)));
    }
}

#define RAIF_CUBLAS_CHECK(expr) check_cublas((expr), #expr)

struct CublasHandle {
    cublasHandle_t handle = nullptr;

    CublasHandle() {
        RAIF_CUBLAS_CHECK(cublasCreate(&handle));
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};
#endif

} // namespace

void gemm_cuda(
    const float* device_a,
    const float* device_b,
    float* device_c,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    void* stream_void) {

    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    validate_gemm_args(device_a, device_b, device_c, m, n, k, "gemm_cuda");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    const dim3 block(kGemmTile, kGemmTile);
    const dim3 grid(
        static_cast<unsigned int>((n + kGemmTile - 1) / kGemmTile),
        static_cast<unsigned int>((m + kGemmTile - 1) / kGemmTile));

    gemm_tiled_kernel<<<grid, block, 0, stream>>>(device_a, device_b, device_c, m, n, k);
    RAIF_CUDA_CHECK(cudaGetLastError());
}

void gemm_cuda_host(
    const float* host_a,
    const float* host_b,
    float* host_c,
    std::size_t m,
    std::size_t n,
    std::size_t k) {

    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    validate_gemm_args(host_a, host_b, host_c, m, n, k, "gemm_cuda_host");

    DeviceFloatBuffer device_a(m * k);
    DeviceFloatBuffer device_b(k * n);
    DeviceFloatBuffer device_c(m * n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_a.ptr, host_a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    RAIF_CUDA_CHECK(cudaMemcpy(device_b.ptr, host_b, k * n * sizeof(float), cudaMemcpyHostToDevice));
    gemm_cuda(device_a.ptr, device_b.ptr, device_c.ptr, m, n, k, nullptr);
    RAIF_CUDA_CHECK(cudaMemcpy(host_c, device_c.ptr, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());
}

double benchmark_gemm_cuda(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup) {
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (m == 0 || n == 0 || k == 0) {
        return 0.0;
    }

    std::vector<float> host_a(m * k);
    std::vector<float> host_b(k * n);
    fill_gemm_inputs(host_a, host_b);

    DeviceFloatBuffer device_a(m * k);
    DeviceFloatBuffer device_b(k * n);
    DeviceFloatBuffer device_c(m * n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_a.ptr, host_a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    RAIF_CUDA_CHECK(cudaMemcpy(device_b.ptr, host_b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        gemm_cuda(device_a.ptr, device_b.ptr, device_c.ptr, m, n, k, nullptr);
    }
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    RAIF_CUDA_CHECK(cudaEventCreate(&start));
    RAIF_CUDA_CHECK(cudaEventCreate(&stop));

    RAIF_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_cuda(device_a.ptr, device_b.ptr, device_c.ptr, m, n, k, nullptr);
    }
    RAIF_CUDA_CHECK(cudaEventRecord(stop));
    RAIF_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    RAIF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    RAIF_CUDA_CHECK(cudaEventDestroy(start));
    RAIF_CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
}

bool cublas_available() {
#ifdef RAIF_WITH_CUBLAS
    return true;
#else
    return false;
#endif
}

double benchmark_gemm_cublas_cuda(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup) {
#ifndef RAIF_WITH_CUBLAS
    (void)m;
    (void)n;
    (void)k;
    (void)iterations;
    (void)warmup;
    throw std::runtime_error("RAIF was built without the optional cuBLAS baseline");
#else
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (m == 0 || n == 0 || k == 0) {
        return 0.0;
    }

    std::vector<float> host_a(m * k);
    std::vector<float> host_b(k * n);
    fill_gemm_inputs(host_a, host_b);

    DeviceFloatBuffer device_a(m * k);
    DeviceFloatBuffer device_b(k * n);
    DeviceFloatBuffer device_c(m * n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_a.ptr, host_a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    RAIF_CUDA_CHECK(cudaMemcpy(device_b.ptr, host_b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    CublasHandle cublas;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. This computes row-major C[M,N] = A[M,K] @ B[K,N]
    // by treating the same memory as C_col[N,M] = B_col[N,K] @ A_col[K,M].
    for (int i = 0; i < warmup; ++i) {
        RAIF_CUBLAS_CHECK(cublasSgemm(
            cublas.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            device_b.ptr,
            static_cast<int>(n),
            device_a.ptr,
            static_cast<int>(k),
            &beta,
            device_c.ptr,
            static_cast<int>(n)));
    }
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    RAIF_CUDA_CHECK(cudaEventCreate(&start));
    RAIF_CUDA_CHECK(cudaEventCreate(&stop));

    RAIF_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        RAIF_CUBLAS_CHECK(cublasSgemm(
            cublas.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            device_b.ptr,
            static_cast<int>(n),
            device_a.ptr,
            static_cast<int>(k),
            &beta,
            device_c.ptr,
            static_cast<int>(n)));
    }
    RAIF_CUDA_CHECK(cudaEventRecord(stop));
    RAIF_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    RAIF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    RAIF_CUDA_CHECK(cudaEventDestroy(start));
    RAIF_CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
#endif
}

} // namespace raif
