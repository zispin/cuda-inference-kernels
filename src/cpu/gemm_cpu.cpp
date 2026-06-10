#include "raif/gemm.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace raif {
namespace {

void validate_gemm_args(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const char* function_name) {

    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(std::string(function_name) + " received a null matrix pointer");
    }
}

void fill_gemm_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int centered = static_cast<int>(i % 257) - 128;
        a[i] = static_cast<float>(centered) * 0.0025f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        const int centered = static_cast<int>((i * 17) % 251) - 125;
        b[i] = static_cast<float>(centered) * 0.0020f;
    }
}

} // namespace

void gemm_cpu(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t n,
    std::size_t k) {

    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    validate_gemm_args(a, b, c, m, n, k, "gemm_cpu");

    const long long row_count = static_cast<long long>(m);

    // Row-parallel i-k-j loop order. For row-major B and C, this keeps the
    // innermost loop contiguous and gives a stronger CPU baseline than the
    // cache-unfriendly i-j-k order.
    #pragma omp parallel for schedule(static) if(m * n * k > 32768)
    for (long long i_ll = 0; i_ll < row_count; ++i_ll) {
        const std::size_t i = static_cast<std::size_t>(i_ll);
        float* c_row = c + i * n;
        std::fill(c_row, c_row + n, 0.0f);

        for (std::size_t kk = 0; kk < k; ++kk) {
            const float a_value = a[i * k + kk];
            const float* b_row = b + kk * n;
            for (std::size_t j = 0; j < n; ++j) {
                c_row[j] += a_value * b_row[j];
            }
        }
    }
}

double benchmark_gemm_cpu(std::size_t m, std::size_t n, std::size_t k, int iterations, int warmup) {
    if (iterations <= 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be non-negative");
    }
    if (m == 0 || n == 0 || k == 0) {
        return 0.0;
    }

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);
    fill_gemm_inputs(a, b);

    for (int i = 0; i < warmup; ++i) {
        gemm_cpu(a.data(), b.data(), c.data(), m, n, k);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        gemm_cpu(a.data(), b.data(), c.data(), m, n, k);
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    volatile float guard = c.empty() ? 0.0f : c[c.size() / 2];
    (void)guard;

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return elapsed.count() / static_cast<double>(iterations);
}

} // namespace raif
