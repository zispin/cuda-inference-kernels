#include "raif/layernorm.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace raif {

void layernorm_cpu(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    std::size_t rows,
    std::size_t cols,
    float eps) {

    if (rows == 0 || cols == 0) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("layernorm_cpu received a null input/output pointer");
    }
    if (!(eps > 0.0f)) {
        throw std::invalid_argument("layernorm_cpu requires eps > 0");
    }

    const long long row_count = static_cast<long long>(rows);

    #pragma omp parallel for schedule(static) if(rows * cols > 4096)
    for (long long r = 0; r < row_count; ++r) {
        const std::size_t row = static_cast<std::size_t>(r);
        const float* row_in = input + row * cols;
        float* row_out = output + row * cols;

        double sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            sum += static_cast<double>(row_in[c]);
        }
        const double mean = sum / static_cast<double>(cols);

        double variance_sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double centered = static_cast<double>(row_in[c]) - mean;
            variance_sum += centered * centered;
        }
        const double variance = variance_sum / static_cast<double>(cols);
        const double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(eps));

        for (std::size_t c = 0; c < cols; ++c) {
            const double normalized = (static_cast<double>(row_in[c]) - mean) * inv_std;
            const double scale = gamma == nullptr ? 1.0 : static_cast<double>(gamma[c]);
            const double shift = beta == nullptr ? 0.0 : static_cast<double>(beta[c]);
            row_out[c] = static_cast<float>(normalized * scale + shift);
        }
    }
}

double benchmark_layernorm_cpu(std::size_t rows, std::size_t cols, int iterations, int warmup) {
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
    std::vector<float> input(n);
    std::vector<float> gamma(cols);
    std::vector<float> beta(cols);
    std::vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.003f);
    }
    for (std::size_t c = 0; c < cols; ++c) {
        gamma[c] = 0.75f + static_cast<float>(c % 17) * 0.01f;
        beta[c] = static_cast<float>(static_cast<int>(c % 23) - 11) * 0.001f;
    }

    constexpr float eps = 1.0e-5f;
    for (int i = 0; i < warmup; ++i) {
        layernorm_cpu(input.data(), gamma.data(), beta.data(), output.data(), rows, cols, eps);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        layernorm_cpu(input.data(), gamma.data(), beta.data(), output.data(), rows, cols, eps);
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count() / static_cast<double>(iterations);
}

} // namespace raif
