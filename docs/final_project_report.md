# Project Report: Reliable AI Inference Fabric

## Scope

Reliable AI Inference Fabric is a local-first C++/CUDA inference-kernel library. The current implementation focuses on a compact set of FP32 operators with CPU baselines, CUDA kernels, Python bindings, numerical validation, and benchmark tooling:

```text
C++/CUDA kernels
CPU OpenMP baselines
PyBind11 Python bindings
ULP-based validation
RTX 2060 benchmark scripts
operator documentation
```

Distributed serving, scheduler design, quantization, and multi-node reliability tooling are outside the current scope. The repository is structured so those components can be added later without changing the kernel-validation workflow.

## Completed operators

| Operator | Why it matters |
|---|---|
| GELU | Elementwise activation, first CUDA vertical slice, stable FP32 formulation. |
| LayerNorm | Row-wise reduction, numerical stability, normalization used in transformers. |
| Softmax | Max/sum/exp reduction, core attention primitive, stable exponentials. |
| GEMM | Matrix multiplication, shared-memory tiling, arithmetic intensity, vendor baseline comparison. |

## Validation strategy

Each operator has a Python accuracy test that imports the compiled extension and compares native CPU/CUDA output to a high-precision NumPy reference. Reports include max absolute error, relative error, and ULP statistics.

ULP is not equally meaningful for every output. Near-zero values and reductions with cancellation can produce large ULP distances even when absolute error is tiny. The tests report ULP values while using operator-appropriate pass/fail thresholds.

## Measured RTX 2060 results

### GELU

| Elements | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 65,536 | 0.1057 | 0.0071 | 14.84x |
| 262,144 | 0.6463 | 0.0099 | 65.38x |
| 1,048,576 | 2.4692 | 0.0435 | 56.82x |
| 4,194,304 | 7.9548 | 0.1500 | 53.04x |
| 16,777,216 | 27.8484 | 0.5344 | 52.11x |

### LayerNorm

| Shape | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1024 x 256 | 0.0587 | 0.0346 | 1.70x |
| 1024 x 768 | 0.1529 | 0.0494 | 3.10x |
| 512 x 1024 | 0.1476 | 0.0406 | 3.63x |
| 256 x 2048 | 0.1519 | 0.0384 | 3.96x |
| 128 x 4096 | 0.1527 | 0.0431 | 3.54x |

### Softmax

| Shape | CPU ms | CUDA ms | Speedup |
|---:|---:|---:|---:|
| 1024 x 128 | 0.0965 | 0.0322 | 3.00x |
| 1024 x 256 | 0.2110 | 0.0338 | 6.25x |
| 1024 x 768 | 0.6384 | 0.0489 | 13.06x |
| 512 x 1024 | 0.4102 | 0.0300 | 13.69x |
| 256 x 2048 | 0.7344 | 0.0292 | 25.18x |

### GEMM

Run this command to collect local GEMM measurements:

```bash
python benchmarks/bench_gemm.py
```

The benchmark writes its table to:

```text
results/gemm_benchmark.md
```

## Reproducibility checklist

Before publishing benchmark numbers:

```text
1. Run: python scripts/build.py --arch 75 --clean
2. Run: python -m pytest tests/accuracy -s
3. Run all four benchmark scripts.
4. Record GPU model, CUDA Toolkit version, driver version, and CPU model.
5. Commit generated benchmark markdown files under results/ only when the numbers should be part of the public record.
6. Keep benchmark claims tied to the measured hardware and configuration.
```

## Repository description

```text
C++/CUDA FP32 inference kernel library with PyBind11 bindings, OpenMP CPU baselines, ULP-based accuracy validation, RTX 2060 benchmarks, and custom kernels for GELU, LayerNorm, Softmax, and tiled GEMM.
```

## Technical coverage

This project demonstrates:

```text
C++ library design
CUDA kernel development
OpenMP CPU baselines
PyBind11 bindings
CMake packaging
floating-point validation
ULP error analysis
shared-memory tiling
GPU benchmarking
vendor baseline comparison through cuBLAS
```
