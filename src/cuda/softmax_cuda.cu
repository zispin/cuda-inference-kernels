#include "raif/softmax.h"
#include "raif/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raif {
namespace {

constexpr int kSoftmaxBlockSize = 256;
constexpr float kNegativeInfinity = -3.4028234663852886e38f;

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    std::size_t rows,
    std::size_t cols) {

    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    __shared__ float scratch[kSoftmaxBlockSize];
    __shared__ float row_max_shared;
    __shared__ float inv_sum_shared;

    const int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float local_max = kNegativeInfinity;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, row_in[c]);
    }

    scratch[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max_shared = scratch[0];
    }
    __syncthreads();

    const float row_max = row_max_shared;
    float local_sum = 0.0f;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        const float e = expf(row_in[c] - row_max);
        row_out[c] = e;
        local_sum += e;
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
        inv_sum_shared = 1.0f / scratch[0];
    }
    __syncthreads();

    const float inv_sum = inv_sum_shared;
    for (std::size_t c = static_cast<std::size_t>(tid); c < cols; c += blockDim.x) {
        row_out[c] *= inv_sum;
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

void validate_softmax_args(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols,
    const char* function_name) {

    if (rows == 0 || cols == 0) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument(std::string(function_name) + " received a null input/output pointer");
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(function_name) + " currently supports at most INT_MAX rows");
    }
}

} // namespace

void softmax_cuda(
    const float* device_input,
    float* device_output,
    std::size_t rows,
    std::size_t cols,
    void* stream_void) {

    if (rows == 0 || cols == 0) {
        return;
    }
    validate_softmax_args(device_input, device_output, rows, cols, "softmax_cuda");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    const dim3 grid(static_cast<unsigned int>(rows));
    const dim3 block(kSoftmaxBlockSize);

    softmax_kernel<<<grid, block, 0, stream>>>(device_input, device_output, rows, cols);
    RAIF_CUDA_CHECK(cudaGetLastError());
}

void softmax_cuda_host(
    const float* host_input,
    float* host_output,
    std::size_t rows,
    std::size_t cols) {

    if (rows == 0 || cols == 0) {
        return;
    }
    validate_softmax_args(host_input, host_output, rows, cols, "softmax_cuda_host");

    const std::size_t n = rows * cols;
    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_output(n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input, n * sizeof(float), cudaMemcpyHostToDevice));
    softmax_cuda(device_input.ptr, device_output.ptr, rows, cols, nullptr);
    RAIF_CUDA_CHECK(cudaMemcpy(host_output, device_output.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());
}

double benchmark_softmax_cuda(std::size_t rows, std::size_t cols, int iterations, int warmup) {
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
    for (std::size_t i = 0; i < n; ++i) {
        host_input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.004f);
    }

    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_output(n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        softmax_cuda(device_input.ptr, device_output.ptr, rows, cols, nullptr);
    }
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    RAIF_CUDA_CHECK(cudaEventCreate(&start));
    RAIF_CUDA_CHECK(cudaEventCreate(&stop));

    RAIF_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_cuda(device_input.ptr, device_output.ptr, rows, cols, nullptr);
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
