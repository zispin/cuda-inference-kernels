#include "raif/gelu.h"
#include "raif/gemm.h"
#include "raif/layernorm.h"
#include "raif/softmax.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

py::array_t<float> make_output_like(const FloatArray& input) {
    py::buffer_info info = input.request();
    return py::array_t<float>(info.shape);
}

py::array_t<float> make_2d_output(std::size_t rows, std::size_t cols) {
    return py::array_t<float>({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
}

py::array_t<float> gelu_cpu_py(const FloatArray& input) {
    py::buffer_info in_info = input.request();
    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();

    const auto* in_ptr = static_cast<const float*>(in_info.ptr);
    auto* out_ptr = static_cast<float*>(out_info.ptr);
    const std::size_t n = static_cast<std::size_t>(in_info.size);

    raif::gelu_cpu(in_ptr, out_ptr, n);
    return output;
}

const float* optional_vector_ptr(
    const py::object& obj,
    std::size_t expected_size,
    const char* name,
    FloatArray& owner) {

    if (obj.is_none()) {
        return nullptr;
    }

    owner = FloatArray::ensure(obj);
    if (!owner) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a contiguous float32 array");
    }

    py::buffer_info info = owner.request();
    if (info.ndim != 1 || static_cast<std::size_t>(info.shape[0]) != expected_size) {
        throw std::invalid_argument(
            std::string(name) + " must be a 1D float32 array with length equal to input.shape[1]");
    }

    return static_cast<const float*>(info.ptr);
}

void parse_rowwise_2d_args(
    const FloatArray& input,
    const char* op_name,
    std::size_t& rows,
    std::size_t& cols,
    const float*& input_ptr) {

    py::buffer_info in_info = input.request();
    if (in_info.ndim != 2) {
        throw std::invalid_argument(std::string(op_name) + " input must be a 2D float32 array with shape [rows, cols]");
    }

    rows = static_cast<std::size_t>(in_info.shape[0]);
    cols = static_cast<std::size_t>(in_info.shape[1]);
    input_ptr = static_cast<const float*>(in_info.ptr);
}

void parse_layernorm_args(
    const FloatArray& input,
    const py::object& gamma_obj,
    const py::object& beta_obj,
    std::size_t& rows,
    std::size_t& cols,
    const float*& input_ptr,
    const float*& gamma_ptr,
    const float*& beta_ptr,
    FloatArray& gamma_owner,
    FloatArray& beta_owner) {

    parse_rowwise_2d_args(input, "LayerNorm", rows, cols, input_ptr);
    gamma_ptr = optional_vector_ptr(gamma_obj, cols, "gamma", gamma_owner);
    beta_ptr = optional_vector_ptr(beta_obj, cols, "beta", beta_owner);
}

void parse_gemm_args(
    const FloatArray& a,
    const FloatArray& b,
    std::size_t& m,
    std::size_t& n,
    std::size_t& k,
    const float*& a_ptr,
    const float*& b_ptr) {

    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.ndim != 2 || b_info.ndim != 2) {
        throw std::invalid_argument("GEMM inputs must be 2D float32 arrays with shapes [M, K] and [K, N]");
    }

    const std::size_t a_rows = static_cast<std::size_t>(a_info.shape[0]);
    const std::size_t a_cols = static_cast<std::size_t>(a_info.shape[1]);
    const std::size_t b_rows = static_cast<std::size_t>(b_info.shape[0]);
    const std::size_t b_cols = static_cast<std::size_t>(b_info.shape[1]);

    if (a_cols != b_rows) {
        throw std::invalid_argument("GEMM shape mismatch: A.shape[1] must equal B.shape[0]");
    }

    m = a_rows;
    k = a_cols;
    n = b_cols;
    a_ptr = static_cast<const float*>(a_info.ptr);
    b_ptr = static_cast<const float*>(b_info.ptr);
}

py::array_t<float> layernorm_cpu_py(
    const FloatArray& input,
    py::object gamma_obj,
    py::object beta_obj,
    float eps) {

    std::size_t rows = 0;
    std::size_t cols = 0;
    const float* input_ptr = nullptr;
    const float* gamma_ptr = nullptr;
    const float* beta_ptr = nullptr;
    FloatArray gamma_owner;
    FloatArray beta_owner;

    parse_layernorm_args(
        input, gamma_obj, beta_obj, rows, cols, input_ptr, gamma_ptr, beta_ptr, gamma_owner, beta_owner);

    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::layernorm_cpu(input_ptr, gamma_ptr, beta_ptr, out_ptr, rows, cols, eps);
    return output;
}

py::array_t<float> softmax_cpu_py(const FloatArray& input) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    const float* input_ptr = nullptr;

    parse_rowwise_2d_args(input, "Softmax", rows, cols, input_ptr);

    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::softmax_cpu(input_ptr, out_ptr, rows, cols);
    return output;
}

py::array_t<float> gemm_cpu_py(const FloatArray& a, const FloatArray& b) {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    const float* a_ptr = nullptr;
    const float* b_ptr = nullptr;

    parse_gemm_args(a, b, m, n, k, a_ptr, b_ptr);

    auto output = make_2d_output(m, n);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::gemm_cpu(a_ptr, b_ptr, out_ptr, m, n, k);
    return output;
}

#ifdef RAIF_WITH_CUDA
py::array_t<float> gelu_cuda_py(const FloatArray& input) {
    py::buffer_info in_info = input.request();
    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();

    const auto* in_ptr = static_cast<const float*>(in_info.ptr);
    auto* out_ptr = static_cast<float*>(out_info.ptr);
    const std::size_t n = static_cast<std::size_t>(in_info.size);

    raif::gelu_cuda_host(in_ptr, out_ptr, n);
    return output;
}

py::array_t<float> layernorm_cuda_py(
    const FloatArray& input,
    py::object gamma_obj,
    py::object beta_obj,
    float eps) {

    std::size_t rows = 0;
    std::size_t cols = 0;
    const float* input_ptr = nullptr;
    const float* gamma_ptr = nullptr;
    const float* beta_ptr = nullptr;
    FloatArray gamma_owner;
    FloatArray beta_owner;

    parse_layernorm_args(
        input, gamma_obj, beta_obj, rows, cols, input_ptr, gamma_ptr, beta_ptr, gamma_owner, beta_owner);

    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::layernorm_cuda_host(input_ptr, gamma_ptr, beta_ptr, out_ptr, rows, cols, eps);
    return output;
}

py::array_t<float> softmax_cuda_py(const FloatArray& input) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    const float* input_ptr = nullptr;

    parse_rowwise_2d_args(input, "Softmax", rows, cols, input_ptr);

    auto output = make_output_like(input);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::softmax_cuda_host(input_ptr, out_ptr, rows, cols);
    return output;
}

py::array_t<float> gemm_cuda_py(const FloatArray& a, const FloatArray& b) {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    const float* a_ptr = nullptr;
    const float* b_ptr = nullptr;

    parse_gemm_args(a, b, m, n, k, a_ptr, b_ptr);

    auto output = make_2d_output(m, n);
    py::buffer_info out_info = output.request();
    auto* out_ptr = static_cast<float*>(out_info.ptr);

    raif::gemm_cuda_host(a_ptr, b_ptr, out_ptr, m, n, k);
    return output;
}
#endif

} // namespace

PYBIND11_MODULE(raif, m) {
    m.doc() = "Reliable AI Inference Fabric native kernels";

    m.def("gelu_cpu", &gelu_cpu_py,
          py::arg("input"),
          "Run FP32 GELU on CPU using the exact erf/erfc formulation.");

    m.def("benchmark_gelu_cpu", &raif::benchmark_gelu_cpu,
          py::arg("n"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CPU GELU runtime in milliseconds.");

    m.def("layernorm_cpu", &layernorm_cpu_py,
          py::arg("input"), py::arg("gamma") = py::none(), py::arg("beta") = py::none(), py::arg("eps") = 1.0e-5f,
          "Run row-wise FP32 LayerNorm on CPU. Input shape is [rows, cols]; gamma and beta are optional length-cols vectors.");

    m.def("benchmark_layernorm_cpu", &raif::benchmark_layernorm_cpu,
          py::arg("rows"), py::arg("cols"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CPU LayerNorm runtime in milliseconds.");

    m.def("softmax_cpu", &softmax_cpu_py,
          py::arg("input"),
          "Run row-wise numerically stable FP32 Softmax on CPU. Input shape is [rows, cols].");

    m.def("benchmark_softmax_cpu", &raif::benchmark_softmax_cpu,
          py::arg("rows"), py::arg("cols"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CPU Softmax runtime in milliseconds.");

    m.def("gemm_cpu", &gemm_cpu_py,
          py::arg("a"), py::arg("b"),
          "Run row-major FP32 GEMM on CPU: C[M,N] = A[M,K] @ B[K,N].");

    m.def("benchmark_gemm_cpu", &raif::benchmark_gemm_cpu,
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("iterations") = 10, py::arg("warmup") = 2,
          "Return average CPU GEMM runtime in milliseconds.");

#ifdef RAIF_WITH_CUDA
    m.def("gelu_cuda", &gelu_cuda_py,
          py::arg("input"),
          "Run FP32 GELU on CUDA. This convenience wrapper includes host-device copies.");

    m.def("benchmark_gelu_cuda", &raif::benchmark_gelu_cuda,
          py::arg("n"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CUDA GELU kernel runtime in milliseconds. Host-device copies are excluded.");

    m.def("layernorm_cuda", &layernorm_cuda_py,
          py::arg("input"), py::arg("gamma") = py::none(), py::arg("beta") = py::none(), py::arg("eps") = 1.0e-5f,
          "Run row-wise FP32 LayerNorm on CUDA. This convenience wrapper includes host-device copies.");

    m.def("benchmark_layernorm_cuda", &raif::benchmark_layernorm_cuda,
          py::arg("rows"), py::arg("cols"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CUDA LayerNorm kernel runtime in milliseconds. Host-device copies are excluded.");

    m.def("softmax_cuda", &softmax_cuda_py,
          py::arg("input"),
          "Run row-wise FP32 Softmax on CUDA. This convenience wrapper includes host-device copies.");

    m.def("benchmark_softmax_cuda", &raif::benchmark_softmax_cuda,
          py::arg("rows"), py::arg("cols"), py::arg("iterations") = 100, py::arg("warmup") = 10,
          "Return average CUDA Softmax kernel runtime in milliseconds. Host-device copies are excluded.");

    m.def("gemm_cuda", &gemm_cuda_py,
          py::arg("a"), py::arg("b"),
          "Run row-major FP32 tiled CUDA GEMM. This convenience wrapper includes host-device copies.");

    m.def("benchmark_gemm_cuda", &raif::benchmark_gemm_cuda,
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("iterations") = 50, py::arg("warmup") = 10,
          "Return average custom CUDA GEMM kernel runtime in milliseconds. Host-device copies are excluded.");

    m.def("cublas_available", &raif::cublas_available,
          "Return true when the optional cuBLAS benchmark baseline was compiled in.");

    m.def("benchmark_gemm_cublas_cuda", &raif::benchmark_gemm_cublas_cuda,
          py::arg("m"), py::arg("n"), py::arg("k"), py::arg("iterations") = 50, py::arg("warmup") = 10,
          "Return average cuBLAS SGEMM runtime in milliseconds. Host-device copies are excluded.");

    m.def("cuda_available", &raif::cuda_available,
          "Return true when at least one CUDA device is visible.");

    m.def("cuda_device_summary", &raif::cuda_device_summary,
          "Return a short summary of the current CUDA device.");
#else
    m.def("gelu_cuda", [](const py::array_t<float>&) -> py::array_t<float> {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("benchmark_gelu_cuda", [](std::size_t, int, int) -> double {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("layernorm_cuda", [](const py::array_t<float>&, py::object, py::object, float) -> py::array_t<float> {
        throw std::runtime_error("RAIF was built without CUDA support");
    }, py::arg("input"), py::arg("gamma") = py::none(), py::arg("beta") = py::none(), py::arg("eps") = 1.0e-5f);

    m.def("benchmark_layernorm_cuda", [](std::size_t, std::size_t, int, int) -> double {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("softmax_cuda", [](const py::array_t<float>&) -> py::array_t<float> {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("benchmark_softmax_cuda", [](std::size_t, std::size_t, int, int) -> double {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("gemm_cuda", [](const py::array_t<float>&, const py::array_t<float>&) -> py::array_t<float> {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("benchmark_gemm_cuda", [](std::size_t, std::size_t, std::size_t, int, int) -> double {
        throw std::runtime_error("RAIF was built without CUDA support");
    });

    m.def("cublas_available", []() { return false; });

    m.def("benchmark_gemm_cublas_cuda", [](std::size_t, std::size_t, std::size_t, int, int) -> double {
        throw std::runtime_error("RAIF was built without CUDA/cuBLAS support");
    });

    m.def("cuda_available", []() { return false; });
    m.def("cuda_device_summary", []() { return std::string("RAIF was built without CUDA support"); });
#endif

#ifdef RAIF_WITH_OPENMP
    m.attr("openmp_enabled") = true;
#else
    m.attr("openmp_enabled") = false;
#endif

    m.attr("version") = RAIF_VERSION;
}
