# Milestone 2: FP32 LayerNorm

This milestone adds a row-wise FP32 LayerNorm operator with optional affine scale and shift vectors.

For each row of a contiguous 2D tensor `[rows, cols]`:

```text
mean = average(x[row, :])
var  = average((x[row, :] - mean)^2)
y    = (x[row, col] - mean) / sqrt(var + eps)
out  = gamma[col] * y + beta[col]
```

`gamma` and `beta` are optional. When omitted, the operator computes normalization only.

## Why LayerNorm matters

LayerNorm is a core transformer operator and is more interesting than an elementwise kernel because it requires:

- per-row reductions,
- numerical stability choices,
- multi-pass memory access analysis,
- CPU/GPU comparison,
- shape-sensitive benchmarking.

## Implementations

- `src/cpu/layernorm_cpu.cpp`: OpenMP CPU implementation using double-precision accumulation for reference-grade CPU behavior.
- `src/cuda/layernorm_cuda.cu`: one CUDA block per row, shared-memory reductions for mean and variance, and affine output writeback.
- `bindings/python/raif_bindings.cpp`: PyBind11 wrappers exposed as `raif.layernorm_cpu` and `raif.layernorm_cuda`.

## Validation

Run:

```bash
python -m pytest tests/accuracy/test_layernorm_accuracy.py -s
```

The tests compare CPU and CUDA outputs against a Python float64 reference rounded to FP32. ULP statistics are reported, but thresholds are looser than GELU because LayerNorm reduction order differs across CPU and CUDA.

## Benchmark

Run:

```bash
python benchmarks/bench_layernorm.py
```

Generated files:

```text
results/layernorm_benchmark.csv
results/layernorm_benchmark.md
```

The benchmark reports CPU time, CUDA kernel time, approximate effective bandwidth, and speedup.

## Notes for the README/report

LayerNorm is memory-bandwidth-oriented. The CUDA kernel is intentionally simple and readable: one block per row with two shared-memory reductions. Future optimizations could include vectorized loads, warp-level reductions, specialized kernels for common hidden sizes, and fusing LayerNorm with following elementwise operations.
