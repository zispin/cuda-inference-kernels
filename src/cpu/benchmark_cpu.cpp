#include "raif/gelu.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace raif {

double benchmark_gelu_cpu(std::size_t n, int iterations, int warmup) {
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }

    std::vector<float> input(n);
    std::vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.01f);
    }

    for (int i = 0; i < warmup; ++i) {
        gelu_cpu(input.data(), output.data(), n);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        gelu_cpu(input.data(), output.data(), n);
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    volatile float guard = n == 0 ? 0.0f : output[n / 2];
    (void)guard;

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count() / static_cast<double>(iterations);
}

} // namespace raif
