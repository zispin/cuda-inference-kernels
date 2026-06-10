#include "raif/gelu.h"

#include <cmath>
#include <cstddef>

namespace raif {
namespace {

inline float gelu_erf_scalar(float x) {
    constexpr float inv_sqrt_2 = 0.70710678118654752440f;
    const float z = x * inv_sqrt_2;

    // For negative inputs, 1 + erf(z) suffers catastrophic cancellation as z
    // moves into the left tail. erfc(-z) is mathematically equivalent and keeps
    // the tiny negative GELU tail accurate instead of rounding it to -0.0.
    if (x < 0.0f) {
        return 0.5f * x * std::erfc(-z);
    }
    return 0.5f * x * (1.0f + std::erf(z));
}

} // namespace

void gelu_cpu(const float* input, float* output, std::size_t n) {
    if (n == 0) {
        return;
    }

    #pragma omp parallel for schedule(static) if(n > 4096)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        output[i] = gelu_erf_scalar(input[i]);
    }
}

} // namespace raif
