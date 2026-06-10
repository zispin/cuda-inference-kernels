# Milestone 1: FP32 GELU Kernel Vertical Slice

This milestone proves the full project pipeline before moving to harder kernels.

## What is included

- C++ CPU implementation of exact GELU using `erf`.
- CUDA implementation using one thread per element.
- PyBind11 Python module named `raif`.
- Accuracy tests against a Python float64 reference rounded to FP32.
- ULP-distance reporting for CPU and CUDA outputs.
- Microbenchmark harness for CPU and CUDA.

## GELU formula

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

This project starts with the exact erf form instead of the tanh approximation because it is easier to validate numerically.

## Why GELU first

GELU is intentionally simple. It lets the project establish the development workflow:

```text
reference implementation -> CPU kernel -> CUDA kernel -> Python binding -> accuracy test -> benchmark
```

After this is stable, the same structure will be reused for LayerNorm, Softmax, and GEMM.

## Current limitations

- FP32 only.
- CUDA wrapper copies data between host and device for convenience.
- CUDA benchmark excludes host-device copies, but the public `gelu_cuda()` Python function includes them.
- The kernel is not yet optimized beyond a basic elementwise implementation.

## Next kernels

1. LayerNorm: block-level reductions, numerical stability, bandwidth analysis.
2. Softmax: max reduction, exponential, sum reduction, normalization.
3. GEMM: naive kernel, tiled shared-memory kernel, cuBLAS comparison.


## Accuracy note

The GELU implementation uses the mathematically equivalent `erfc` formulation for negative inputs: `0.5 * x * erfc(-x / sqrt(2))`. This avoids cancellation in the left tail where `1 + erf(x / sqrt(2))` can round to zero in FP32. The accuracy test reports all-value ULP statistics, but pass/fail ULP thresholds are applied to outputs with meaningful magnitude because ULP counts near zero can be misleading.
