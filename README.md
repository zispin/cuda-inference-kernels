# Reliable AI Inference Fabric

A local-first C++/CUDA inference-kernel library with CPU baselines, Python bindings, numerical validation, and GPU benchmarks.

This repository implements a compact set of FP32 operators commonly used in neural-network inference. Each operator includes a CPU implementation, a CUDA implementation, Python bindings through PyBind11, accuracy tests against high-precision references, and benchmark scripts for collecting reproducible local measurements.

## Status

| Operator | CPU backend | CUDA backend | Accuracy tests | Benchmark | Notes |
|---|---:|---:|---:|---:|---|
| GELU | yes | yes | yes | yes | stable erf/erfc formulation |
| LayerNorm | yes | yes | yes | yes | row-wise mean/variance reduction with optional affine parameters |
| Softmax | yes | yes | yes | yes | numerically stable max/sum/exp reduction |
| GEMM | yes | yes | yes | yes | row-major FP32 tiled shared-memory matrix multiply |

The GEMM benchmark optionally compares the custom CUDA kernel against cuBLAS SGEMM when the CUDA Toolkit exposes the `CUDA::cublas` CMake target.

## Reference environment

The project is designed for local development on NVIDIA CUDA-capable GPUs. The included benchmark tables were collected on the following system:

```text
GPU: NVIDIA GeForce RTX 2060, compute capability 7.5, 6 GB VRAM
CPU: Intel Core i7-10700
RAM: 16 GB
OS: Windows 11 with WSL2 Ubuntu
CUDA Toolkit: 12.8
```

Other CUDA-capable GPUs can be used by selecting the appropriate architecture with `scripts/build.py --arch <compute_capability>`.

## Quick start

Install Python build and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pybind11 pytest
```

Check the local toolchain:

```bash
python scripts/setup_check.py
```

Build for an RTX 2060 / compute capability 7.5:

```bash
python scripts/build.py --arch 75 --clean
```

Run all accuracy tests:

```bash
python -m pytest tests/accuracy -s
```

Run all benchmarks:

```bash
python benchmarks/bench_gelu.py
python benchmarks/bench_layernorm.py
python benchmarks/bench_softmax.py
python benchmarks/bench_gemm.py
```

CPU-only fallback:

```bash
python scripts/build.py --cpu-only --clean
python -m pytest tests/accuracy -s
```

Disable only the optional cuBLAS GEMM baseline while keeping custom CUDA kernels enabled:

```bash
python scripts/build.py --arch 75 --no-cublas --clean
```

## Python API

```python
import raif

# Elementwise activation
out = raif.gelu_cpu(x)
out = raif.gelu_cuda(x)

# Row-wise operators for x.shape == [rows, cols]
out = raif.layernorm_cpu(x, gamma=None, beta=None, eps=1e-5)
out = raif.layernorm_cuda(x, gamma=None, beta=None, eps=1e-5)
out = raif.softmax_cpu(x)
out = raif.softmax_cuda(x)

# Row-major matrix multiply: A[M,K] @ B[K,N] -> C[M,N]
c = raif.gemm_cpu(a, b)
c = raif.gemm_cuda(a, b)
```

Benchmark functions return average runtime in milliseconds. CUDA benchmark functions time GPU execution and exclude host-device copies.

## Project layout

```text
include/raif/               Public C++ headers
src/cpu/                    CPU kernels and CPU benchmarks
src/cuda/                   CUDA kernels and CUDA benchmarks
bindings/python/            PyBind11 module
scripts/                    Build and setup utilities
tests/accuracy/             Accuracy tests and ULP validation
benchmarks/                 Benchmark scripts
results/                    Generated local benchmark output
docs/                       Operator notes and project report
```

## Accuracy methodology

The tests compare native outputs against Python/NumPy float64 references rounded to FP32. Reports include:

```text
max absolute error
mean absolute error
p99 absolute error
max relative error
max ULP error
mean ULP error
p99 ULP error
significant-value ULP statistics
```

ULP distance is reported for every operator, but pass/fail criteria are adapted to the numerical pattern of the operator. Elementwise GELU has tighter ULP expectations. Reduction-heavy operators such as LayerNorm, Softmax, and GEMM can legitimately differ because CPU, CUDA, and float64 references accumulate in different orders.

## Operator notes

### GELU

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

The implementation uses the mathematically equivalent `erfc` formulation for negative inputs:

```text
0.5 * x * erfc(-x / sqrt(2))
```

This avoids cancellation in the left tail where `1 + erf(x / sqrt(2))` can round to zero in FP32.

### LayerNorm

For a contiguous 2D input with shape `[rows, cols]`, LayerNorm computes per-row mean and variance:

```text
mean = average(x[row, :])
var  = average((x[row, :] - mean)^2)
y    = (x[row, col] - mean) / sqrt(var + eps)
out  = gamma[col] * y + beta[col]
```

`gamma` and `beta` are optional. When omitted, `gamma=1` and `beta=0` are used.

### Softmax

For a contiguous 2D input with shape `[rows, cols]`, Softmax computes:

```text
m      = max(x[row, :])
e[c]   = exp(x[row, c] - m)
sum    = sum(e[:])
out[c] = e[c] / sum
```

Subtracting the row maximum keeps the exponentials numerically stable and mirrors the stability pattern used in transformer attention.

### GEMM

GEMM computes row-major FP32 matrix multiplication:

```text
C[M, N] = A[M, K] @ B[K, N]
```

The CUDA implementation uses a readable 16x16 shared-memory tiled kernel. The implementation is intended to demonstrate correct indexing, memory layout, cooperative tile loading, shared memory use, numerical validation, benchmarking, and comparison against a vendor baseline.

## Benchmark output

Benchmarks write Markdown and CSV outputs under `results/`:

```text
results/gelu_benchmark.csv
results/gelu_benchmark.md
results/layernorm_benchmark.csv
results/layernorm_benchmark.md
results/softmax_benchmark.csv
results/softmax_benchmark.md
results/gemm_benchmark.csv
results/gemm_benchmark.md
```

## Measured RTX 2060 results

### GELU CUDA speedup over CPU

| Elements | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 65,536 | 0.1057 | 0.0071 | 14.84x |
| 262,144 | 0.6463 | 0.0099 | 65.38x |
| 1,048,576 | 2.4692 | 0.0435 | 56.82x |
| 4,194,304 | 7.9548 | 0.1500 | 53.04x |
| 16,777,216 | 27.8484 | 0.5344 | 52.11x |

### LayerNorm CUDA speedup over CPU

| Shape | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1024 x 256 | 0.0587 | 0.0346 | 1.70x |
| 1024 x 768 | 0.1529 | 0.0494 | 3.10x |
| 512 x 1024 | 0.1476 | 0.0406 | 3.63x |
| 256 x 2048 | 0.1519 | 0.0384 | 3.96x |
| 128 x 4096 | 0.1527 | 0.0431 | 3.54x |

### Softmax CUDA speedup over CPU

| Shape | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1024 x 128 | 0.0965 | 0.0322 | 3.00x |
| 1024 x 256 | 0.2110 | 0.0338 | 6.25x |
| 1024 x 768 | 0.6384 | 0.0489 | 13.06x |
| 512 x 1024 | 0.4102 | 0.0300 | 13.69x |
| 256 x 2048 | 0.7344 | 0.0292 | 25.18x |

Run `python benchmarks/bench_gemm.py` to collect GEMM numbers and the optional cuBLAS comparison for the local system.

## Scope

This repository focuses on the local kernel-library layer: custom CPU/CUDA operator implementations, validation infrastructure, Python bindings, and benchmark tooling. Distributed serving, scheduler design, quantization, and multi-node reliability tooling are outside the current scope and are natural future extensions.

## License

MIT License. See `LICENSE`.
