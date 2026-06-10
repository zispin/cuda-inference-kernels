#pragma once

#include <cstddef>
#include <string>

namespace raif {

// Exact GELU formulation used in many transformer implementations:
// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
void gelu_cpu(const float* input, float* output, std::size_t n);

double benchmark_gelu_cpu(std::size_t n, int iterations, int warmup);

#ifdef RAIF_WITH_CUDA
void gelu_cuda(const float* device_input, float* device_output, std::size_t n, void* stream = nullptr);
void gelu_cuda_host(const float* host_input, float* host_output, std::size_t n);
double benchmark_gelu_cuda(std::size_t n, int iterations, int warmup);
bool cuda_available();
std::string cuda_device_summary();
#endif

} // namespace raif
