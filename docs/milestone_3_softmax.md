# Milestone 3: FP32 Softmax

Softmax is the third operator in the local CUDA kernel library. It is the natural next step after LayerNorm because it uses the same row-wise reduction structure but adds the numerical-stability pattern required by attention.

## Implemented API

```python
raif.softmax_cpu(x)
raif.softmax_cuda(x)
raif.benchmark_softmax_cpu(rows, cols, iterations, warmup)
raif.benchmark_softmax_cuda(rows, cols, iterations, warmup)
```

Input tensors are contiguous `float32` arrays with shape `[rows, cols]`. Softmax is computed independently for each row.

## Mathematical definition

```text
m      = max(x[row, :])
e[c]   = exp(x[row, c] - m)
out[c] = e[c] / sum(e[:])
```

Subtracting the per-row max prevents overflow and is the same stability pattern used in transformer attention.

## CPU implementation

The CPU implementation uses OpenMP across rows and computes the reduction in double precision for the reference-quality CPU path.

## CUDA implementation

The CUDA implementation assigns one thread block per row. Each block performs:

1. parallel max reduction,
2. exponentiation and sum reduction,
3. normalization.

This is intentionally simple and readable. Later optimization opportunities include warp-level reductions, vectorized loads, and specialized kernels for common attention dimensions.

## Validation

The accuracy test compares against a NumPy float64 reference rounded to FP32. It reports:

- max absolute error,
- mean absolute error,
- p99 absolute error,
- max relative error,
- ULP statistics for all outputs,
- ULP statistics for probabilities above a significance threshold.

Softmax can generate very small probabilities, so ULP distance for near-zero values is reported but not used as the primary pass/fail criterion.

## Benchmark output

Run:

```bash
python benchmarks/bench_softmax.py
```

The benchmark writes:

```text
results/softmax_benchmark.csv
results/softmax_benchmark.md
```
