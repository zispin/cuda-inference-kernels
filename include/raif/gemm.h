#pragma once

#include <cstddef>

namespace raif {

// Row-major FP32 matrix multiplication:
//   C[M, N] = A[M, K] @ B[K, N]
// All matrices are contiguous and stored in row-major order.
void gemm_cpu(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t n,
    std::size_t k);

double benchmark_gemm_cpu(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup);

#ifdef RAIF_WITH_CUDA
void gemm_cuda(
    const float* device_a,
    const float* device_b,
    float* device_c,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    void* stream = nullptr);

void gemm_cuda_host(
    const float* host_a,
    const float* host_b,
    float* host_c,
    std::size_t m,
    std::size_t n,
    std::size_t k);

double benchmark_gemm_cuda(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup);

// Optional vendor baseline. This is for benchmarking only; the custom GEMM
// kernel above remains implemented from scratch.
bool cublas_available();
double benchmark_gemm_cublas_cuda(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup);
#endif

} // namespace raif
