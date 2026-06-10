#include "raif/softmax.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace raif {

void softmax_cpu(
    const float* input,
    float* output,
    std::size_t rows,
    std::size_t cols) {

    if (rows == 0 || cols == 0) {
        return;
    }
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("softmax_cpu received a null input/output pointer");
    }

    const long long row_count = static_cast<long long>(rows);

    #pragma omp parallel for schedule(static) if(rows * cols > 4096)
    for (long long r = 0; r < row_count; ++r) {
        const std::size_t row = static_cast<std::size_t>(r);
        const float* row_in = input + row * cols;
        float* row_out = output + row * cols;

        double row_max = static_cast<double>(row_in[0]);
        for (std::size_t c = 1; c < cols; ++c) {
            row_max = std::max(row_max, static_cast<double>(row_in[c]));
        }

        double sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double e = std::exp(static_cast<double>(row_in[c]) - row_max);
            row_out[c] = static_cast<float>(e);
            sum += e;
        }

        const double inv_sum = 1.0 / sum;
        for (std::size_t c = 0; c < cols; ++c) {
            row_out[c] = static_cast<float>(static_cast<double>(row_out[c]) * inv_sum);
        }
    }
}

double benchmark_softmax_cpu(std::size_t rows, std::size_t cols, int iterations, int warmup) {
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
    std::vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        // Keep values in a range representative of attention logits while also
        // covering positive and negative values.
        input[i] = static_cast<float>((static_cast<int>(i % 2001) - 1000) * 0.004f);
    }

    for (int i = 0; i < warmup; ++i) {
        softmax_cpu(input.data(), output.data(), rows, cols);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        softmax_cpu(input.data(), output.data(), rows, cols);
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    volatile float guard = output[n / 2];
    (void)guard;

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count() / static_cast<double>(iterations);
}

} // namespace raif
