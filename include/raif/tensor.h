#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace raif {

// Minimal host tensor helper for early tests and examples.
// This is intentionally tiny; the project is about kernels, not a tensor framework.
struct Shape2D {
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::size_t size() const noexcept { return rows * cols; }
};

template <typename T>
struct HostTensor2D {
    Shape2D shape;
    std::vector<T> data;

    HostTensor2D() = default;

    HostTensor2D(std::size_t rows, std::size_t cols)
        : shape{rows, cols}, data(rows * cols) {}

    T& operator()(std::size_t r, std::size_t c) {
        if (r >= shape.rows || c >= shape.cols) {
            throw std::out_of_range("HostTensor2D index out of range");
        }
        return data[r * shape.cols + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        if (r >= shape.rows || c >= shape.cols) {
            throw std::out_of_range("HostTensor2D index out of range");
        }
        return data[r * shape.cols + c];
    }
};

} // namespace raif
