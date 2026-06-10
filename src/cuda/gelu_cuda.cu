#include "raif/gelu.h"
#include "raif/cuda_utils.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace raif {
namespace {

__device__ __forceinline__ float gelu_erf_device(float x) {
    constexpr float inv_sqrt_2 = 0.70710678118654752440f;
    const float z = x * inv_sqrt_2;

    // Stable left-tail formulation. For x < 0, 1 + erf(z) can round to zero
    // in FP32 even when the true GELU output is a small negative value.
    if (x < 0.0f) {
        return 0.5f * x * erfcf(-z);
    }
    return 0.5f * x * (1.0f + erff(z));
}

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu_erf_device(input[idx]);
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

} // namespace

void gelu_cuda(const float* device_input, float* device_output, std::size_t n, void* stream_void) {
    if (n == 0) {
        return;
    }
    if (device_input == nullptr || device_output == nullptr) {
        throw std::invalid_argument("gelu_cuda received a null device pointer");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_void);
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(device_input, device_output, n);
    RAIF_CUDA_CHECK(cudaGetLastError());
}

void gelu_cuda_host(const float* host_input, float* host_output, std::size_t n) {
    if (n == 0) {
        return;
    }
    if (host_input == nullptr || host_output == nullptr) {
        throw std::invalid_argument("gelu_cuda_host received a null host pointer");
    }

    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_output(n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input, n * sizeof(float), cudaMemcpyHostToDevice));
    gelu_cuda(device_input.ptr, device_output.ptr, n, nullptr);
    RAIF_CUDA_CHECK(cudaMemcpy(host_output, device_output.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());
}

bool cuda_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess && count > 0) {
        return true;
    }
    cudaGetLastError();
    return false;
}

std::string cuda_device_summary() {
    int count = 0;
    RAIF_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        return "No CUDA devices found";
    }

    int device = 0;
    RAIF_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    RAIF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::ostringstream oss;
    oss << prop.name
        << " | compute capability " << prop.major << "." << prop.minor
        << " | SMs " << prop.multiProcessorCount
        << " | global memory " << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return oss.str();
}

double benchmark_gelu_cuda(std::size_t n, int iterations, int warmup) {
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (n == 0) {
        return 0.0;
    }

    std::vector<float> host_input(n);
    for (std::size_t i = 0; i < n; ++i) {
        host_input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.01f);
    }

    DeviceFloatBuffer device_input(n);
    DeviceFloatBuffer device_output(n);

    RAIF_CUDA_CHECK(cudaMemcpy(device_input.ptr, host_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        gelu_cuda(device_input.ptr, device_output.ptr, n, nullptr);
    }
    RAIF_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    RAIF_CUDA_CHECK(cudaEventCreate(&start));
    RAIF_CUDA_CHECK(cudaEventCreate(&stop));

    RAIF_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gelu_cuda(device_input.ptr, device_output.ptr, n, nullptr);
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
