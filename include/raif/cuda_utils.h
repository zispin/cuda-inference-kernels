#pragma once

#ifdef RAIF_WITH_CUDA

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace raif {
namespace cuda_detail {

inline void check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " while running `" << expr << "`: "
            << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

} // namespace cuda_detail
} // namespace raif

#define RAIF_CUDA_CHECK(expr) ::raif::cuda_detail::check((expr), #expr, __FILE__, __LINE__)

#endif // RAIF_WITH_CUDA
