#pragma once

#include <cstddef>

namespace raif {

// Row-wise FP32 LayerNorm for a contiguous 2D tensor with shape [rows, cols].
// For each row r and column c:
//   mean = average(input[r, :])
//   var  = average((input[r, :] - mean)^2)
//   y    = (input[r, c] - mean) / sqrt(var + eps)
//   out  = gamma[c] * y + beta[c]
// gamma and beta may be nullptr, in which case gamma=1 and beta=0 are used.
void layernorm_cpu(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t cols,
    float eps);

double benchmark_layernorm_cpu(std::size_t rows, std::size_t cols, int iterations, int warmup);

#ifdef RAIF_WITH_CUDA
void layernorm_cuda(
    const float* device_input,
    const float* device_gamma,
    const float* device_beta,
    float* device_output,
    std::size_t rows,
    std::size_t cols,
    float eps,
    void* stream = nullptr);

void layernorm_cuda_host(
    const float* host_input,
    const float* host_gamma,
    const float* host_beta,
    float* host_output,
    std::size_t rows,
    std::size_t cols,
    float eps);

double benchmark_layernorm_cuda(std::size_t rows, std::size_t cols, int iterations, int warmup);
#endif

} // namespace raif
