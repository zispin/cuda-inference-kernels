#include "raif/layernorm.h"
#include "raif/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raif {
namespace {

constexpr int kLayerNormBlockSize = 256;

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t cols,
    float eps) {

    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    __shared__ float scratch[kLayerNormBlockSize];
    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    const int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float local_sum = 0.0f;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        local_sum += row_in[c];
    }

    scratch[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_shared = scratch[0] / static_cast<float>(cols);
    }
    __syncthreads();

    const float mean = mean_shared;
    float local_var_sum = 0.0f;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        const float centered = row_in[c] - mean;
        local_var_sum += centered * centered;
    }

    scratch[tid] = local_var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float variance = scratch[0] / static_cast<float>(cols);
        inv_std_shared = rsqrtf(variance + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        const float normalized = (row_in[c] - mean) * inv_std;
        const float scale = gamma == nullptr ? 1.0f : gamma[c];
        const float shift = beta == nullptr ? 0.0f : beta[c];
        row_out[c] = normalized * scale + shift;
    }
}

struct DeviceFloatBuffer {
    float* ptr = nullptr;

    explicit DeviceFloatBuffer(std::size_t n) {
        if (n > 0) {
            RAIF_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
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

void validate_layernorm_args(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    float eps,
    const char* function_name) {

    if (rows == 0 || cols == 0) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument(std::string(function_name) + " received a null input/output pointer");
    }
    if (!(eps > 0.0f)) {
        throw std::invalid_argument(std::string(function_name) + " requires eps > 0");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(function_name) + " currently supports at most INT_MAX rows");
    }
}

} // namespace

void layernorm_cuda(
    const float* device_input,
    const float* device_gamma,
    const float* device_beta,
    float* device_output,
    std::size_t rows,
    std::size_t cols,
    float eps,
    void* stream_void) {

    if (rows == 0 || cols == 0) {
        return;
    }
    validate_layernorm_args(device_input, device_output, rows, cols, eps, "layernorm_cuda");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(kLayerNormBlockSize);

    layernorm_kernel<<<grid, block, 0, stream>>>(
        device_input, device_gamma, device_beta, device_output, rows, cols, eps);
    RAIF_CUDA_CHECK(cudaGetLastError());
}

void layernorm_cuda_host(
    const float* host_input,
    const float* host_gamma,
    const float* host_beta,
    float* host_output,
    std::size_t rows,
    std::size_t cols,
    float eps) {

    if (rows == 0 || cols == 0) {
        return;
    }
    validate_layernorm_args(host_input, host_output, rows, cols, eps, "layernorm_cuda_host");

    const std::size_t n = rows * cols;
    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_output(n);
    DeviceFloatBuffer device_gamma(host_gamma == nullptr ? 0 : cols);
    DeviceFloatBuffer device_beta(host_beta == nullptr ? 0 : cols);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input, n * sizeof(float), cudaMemcpyHostToDevice));
    if (host_gamma != nullptr) {
        RAIF_CUDA_CHECK(cudaMemcpy(device_gamma.ptr, host_gamma, cols * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (host_beta != nullptr) {
        RAIF_CUDA_CHECK(cudaMemcpy(device_beta.ptr, host_beta, cols * sizeof(float), cudaMemcpyHostToDevice));
    }

    layernorm_cuda(
        device_input.ptr,
        device_gamma.ptr,
        device_beta.ptr,
        device_output.ptr,
        rows,
        cols,
        eps,
        nullptr);

    RAIF_CUDA_CHECK(cudaMemcpy(host_output, device_output.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());
}

double benchmark_layernorm_cuda(std::size_t rows, std::size_t cols, int iterations, int warmup) {
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (rows == 0 || cols == 0) {
        return 0.0;
    }

    const std::size_t n = rows * cols;
    std::vector<float> host_input(n);
    std::vector<float> host_gamma(cols);
    std::vector<float> host_beta(cols);

    for (std::size_t i = 0; i < n; ++i) {
        host_input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.003f);
    }
    for (std::size_t c = 0; c < cols; ++c) {
        host_gamma[c] = 0.75f + static_cast<float>(c % 17) * 0.01f;
        host_beta[c] = static_cast<float>(static_cast<int>(c % 23) - 11) * 0.001f;
    }

    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_gamma(cols);
    DeviceFloatBuffer device_beta(cols);
    DeviceFloatBuffer device_output(n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    RAIF_CUDA_CHECK(cudaMemcpy(device_gamma.ptr, host_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    RAIF_CUDA_CHECK(cudaMemcpy(device_beta.ptr, host_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    constexpr float eps = 1.0e-5f;
    for (int i = 0; i < warmup; ++i) {
        layernorm_cuda(device_input.ptr, device_gamma.ptr, device_beta.ptr, device_output.ptr, rows, cols, eps, nullptr);
    }
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    RAIF_CUDA_CHECK(cudaEventCreate(&start));
    RAIF_CUDA_CHECK(cudaEventCreate(&stop));

    RAIF_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        layernorm_cuda(device_input.ptr, device_gamma.ptr, device_beta.ptr, device_output.ptr, rows, cols, eps, nullptr);
    }
    RAIF_CUDA_CHECK(cudaEventRecord(stop));
    RAIF_CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    RAIF_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    RAIF_CUDA_CHECK(cudaEventDestroy(start));
    RAIF_CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
}

} // namespace raif
