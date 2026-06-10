import csv
import sys
from pathlib import Path


def import_raif():
    try:
        import raif  # type: ignore
        return raif
    except ImportError:
        root = Path(__file__).resolve().parents[1]
        patterns = [
            "build*/**/raif*.pyd",
            "build*/**/raif*.so",
            "build*/**/raif*.dylib",
        ]
        for pattern in patterns:
            for candidate in root.glob(pattern):
                sys.path.insert(0, str(candidate.parent))
                try:
                    import raif  # type: ignore
                    return raif
                except ImportError:
                    sys.path.pop(0)
        raise


def gemm_gflops(m: int, n: int, k: int, ms: float) -> float:
    if ms is None or ms <= 0:
        return 0.0
    return (2.0 * m * n * k) / (ms / 1000.0) / 1.0e9


def fmt_optional(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def cpu_iterations(m: int, n: int, k: int) -> int:
    ops = 2 * m * n * k
    if ops <= 2 * 256 * 256 * 256:
        return 5
    if ops <= 2 * 512 * 512 * 512:
        return 2
    return 1


def cuda_iterations(m: int, n: int, k: int) -> int:
    ops = 2 * m * n * k
    if ops <= 2 * 256 * 256 * 256:
        return 200
    if ops <= 2 * 512 * 512 * 512:
        return 100
    return 50


def safe_call(fn, *args):
    try:
        return float(fn(*args))
    except Exception as exc:
        print(f"    note: optional benchmark failed: {exc}")
        return None


def main() -> None:
    raif = import_raif()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (512, 1024, 512),
        (1024, 512, 1024),
    ]
    rows_out = []

    cuda_available = bool(raif.cuda_available())
    cublas_available = bool(getattr(raif, "cublas_available", lambda: False)()) if cuda_available else False

    print(f"RAIF version: {getattr(raif, 'version', 'unknown')}")
    print(f"OpenMP enabled: {getattr(raif, 'openmp_enabled', False)}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {raif.cuda_device_summary()}")
    print(f"cuBLAS baseline available: {cublas_available}")
    print()

    header = (
        f"{'M':>6} | {'N':>6} | {'K':>6} | "
        f"{'CPU ms':>10} | {'CPU GF/s':>10} | "
        f"{'CUDA ms':>10} | {'CUDA GF/s':>11} | "
        f"{'cuBLAS ms':>10} | {'cuBLAS GF/s':>12} | "
        f"{'CPU speedup':>11} | {'% cuBLAS':>8}"
    )
    print(header)
    print("-" * len(header))

    for m, n, k in shapes:
        cpu_iters = cpu_iterations(m, n, k)
        gpu_iters = cuda_iterations(m, n, k)

        cpu_ms = float(raif.benchmark_gemm_cpu(m, n, k, cpu_iters, 1))
        cpu_gflops = gemm_gflops(m, n, k, cpu_ms)

        cuda_ms = None
        cuda_gflops = None
        speedup = None
        if cuda_available:
            cuda_ms = float(raif.benchmark_gemm_cuda(m, n, k, gpu_iters, 10))
            cuda_gflops = gemm_gflops(m, n, k, cuda_ms)
            speedup = cpu_ms / cuda_ms if cuda_ms > 0 else None

        cublas_ms = None
        cublas_gflops = None
        percent_cublas = None
        if cublas_available:
            cublas_ms = safe_call(raif.benchmark_gemm_cublas_cuda, m, n, k, gpu_iters, 10)
            if cublas_ms is not None:
                cublas_gflops = gemm_gflops(m, n, k, cublas_ms)
                if cublas_gflops > 0 and cuda_gflops is not None:
                    percent_cublas = 100.0 * cuda_gflops / cublas_gflops

        rows_out.append({
            "m": m,
            "n": n,
            "k": k,
            "cpu_ms": cpu_ms,
            "cpu_gflops": cpu_gflops,
            "cuda_ms": cuda_ms,
            "cuda_gflops": cuda_gflops,
            "cublas_ms": cublas_ms,
            "cublas_gflops": cublas_gflops,
            "cpu_speedup": speedup,
            "percent_cublas": percent_cublas,
        })

        print(
            f"{m:6d} | {n:6d} | {k:6d} | "
            f"{cpu_ms:10.4f} | {cpu_gflops:10.2f} | "
            f"{cuda_ms if cuda_ms is not None else float('nan'):10.4f} | "
            f"{cuda_gflops if cuda_gflops is not None else float('nan'):11.2f} | "
            f"{cublas_ms if cublas_ms is not None else float('nan'):10.4f} | "
            f"{cublas_gflops if cublas_gflops is not None else float('nan'):12.2f} | "
            f"{speedup if speedup is not None else float('nan'):11.2f} | "
            f"{percent_cublas if percent_cublas is not None else float('nan'):8.2f}"
        )

    csv_path = results_dir / "gemm_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    md_path = results_dir / "gemm_benchmark.md"
    with md_path.open("w") as f:
        f.write("# GEMM Benchmark Results\n\n")
        f.write("| M | N | K | CPU ms | CPU GFLOP/s | CUDA ms | CUDA GFLOP/s | cuBLAS ms | cuBLAS GFLOP/s | CPU speedup | % cuBLAS |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows_out:
            f.write(
                f"| {row['m']} | {row['n']} | {row['k']} | "
                f"{fmt_optional(row['cpu_ms'], 4)} | {fmt_optional(row['cpu_gflops'], 2)} | "
                f"{fmt_optional(row['cuda_ms'], 4)} | {fmt_optional(row['cuda_gflops'], 2)} | "
                f"{fmt_optional(row['cublas_ms'], 4)} | {fmt_optional(row['cublas_gflops'], 2)} | "
                f"{fmt_optional(row['cpu_speedup'], 2)} | {fmt_optional(row['percent_cublas'], 2)} |\n"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
