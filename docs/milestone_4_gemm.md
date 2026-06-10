# Milestone 4: FP32 GEMM

GEMM is the final operator in the current local CUDA kernel library. It demonstrates matrix indexing, memory layout, shared-memory tiling, arithmetic intensity, and comparison against a vendor library.

## Implemented API

```python
raif.gemm_cpu(a, b)
raif.gemm_cuda(a, b)
raif.benchmark_gemm_cpu(m, n, k, iterations, warmup)
raif.benchmark_gemm_cuda(m, n, k, iterations, warmup)
raif.cublas_available()
raif.benchmark_gemm_cublas_cuda(m, n, k, iterations, warmup)
```

Inputs are contiguous `float32` arrays:

```text
A shape: [M, K]
B shape: [K, N]
C shape: [M, N]
```

All matrices are row-major.

## Mathematical definition

```text
C[i, j] = sum(A[i, kk] * B[kk, j] for kk in range(K))
```

## CPU implementation

The CPU implementation uses OpenMP over rows and an `i-k-j` loop order:

```text
for each row i in parallel:
    initialize C[i, :] to zero
    for kk in K:
        a_value = A[i, kk]
        for j in N:
            C[i, j] += a_value * B[kk, j]
```

The `i-k-j` order keeps the innermost access to `B[kk, :]` and `C[i, :]` contiguous in row-major memory, which is much better than a naive `i-j-k` loop.

## CUDA implementation

The CUDA implementation uses a readable 16x16 shared-memory tiled kernel:

```text
one block computes one 16x16 tile of C
threads cooperatively load a 16x16 tile of A and B into shared memory
each thread accumulates one output element
boundary checks handle non-multiple-of-16 matrix sizes
```

This implementation is intended to demonstrate the core CUDA optimization pattern: reducing redundant global-memory reads by staging tiles in shared memory.

## cuBLAS baseline

When available, the benchmark script also times cuBLAS SGEMM. The custom kernel remains independent; cuBLAS is used only as a vendor baseline.

Because cuBLAS assumes column-major matrices, the benchmark uses the standard row-major mapping:

```text
row-major C[M,N] = A[M,K] @ B[K,N]
interpreted as column-major C_col[N,M] = B_col[N,K] @ A_col[K,M]
```

## Validation

The accuracy test compares against:

```python
(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
```

The report includes absolute error, relative error, and ULP statistics. GEMM is a reduction over K, so ULP values can be large near zero or under cancellation. Pass/fail uses absolute and relative tolerances while still reporting ULP diagnostics.

## Benchmark output

Run:

```bash
python benchmarks/bench_gemm.py
```

The benchmark writes:

```text
results/gemm_benchmark.csv
results/gemm_benchmark.md
```

The table reports:

```text
CPU runtime
CPU GFLOP/s
custom CUDA runtime
custom CUDA GFLOP/s
optional cuBLAS runtime
optional cuBLAS GFLOP/s
CUDA speedup over CPU
custom CUDA as percent of cuBLAS
```
