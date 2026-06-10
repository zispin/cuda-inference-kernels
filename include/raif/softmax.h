#pragma once

#include <cstddef>

namespace raif {

// Row-wise numerically stable FP32 Softmax for a contiguous 2D tensor with
// shape [rows, cols]. For each row r:
//   m      = max(input[r, :])
//   e[c]   = exp(input[r, c] - m)
//   sum    = sum(e[:])
//   out[c] = e[c] / sum
void softmax_cpu(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols);

double benchmark_softmax_cpu(std::size_t rows, std::size_t cols, int iterations, int warmup);

#ifdef RAIF_WITH_CUDA
void softmax_cuda(
    const float* device_input,
    float* device_output,
    std::size_t rows,
    std::size_t cols,
    void* stream = nullptr);

void softmax_cuda_host(
    const float* host_input,
    float* host_output,
    std::size_t rows,
    std::size_t cols);

double benchmark_softmax_cuda(std::size_t rows, std::size_t cols, int iterations, int warmup);
#endif

} // namespace raif
