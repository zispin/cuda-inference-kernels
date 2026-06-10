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


def effective_bandwidth_gbps(n: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    bytes_touched = 2 * n * 4  # one FP32 read and one FP32 write
    return bytes_touched / (ms / 1000.0) / 1e9


def fmt_optional(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main() -> None:
    raif = import_raif()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    sizes = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24]
    rows = []

    print(f"RAIF version: {getattr(raif, 'version', 'unknown')}")
    print(f"OpenMP enabled: {getattr(raif, 'openmp_enabled', False)}")
    print(f"CUDA available: {raif.cuda_available()}")
    if raif.cuda_available():
        print(f"CUDA device: {raif.cuda_device_summary()}")
    print()

    header = f"{'n':>12} | {'cpu ms':>12} | {'cpu GB/s':>10} | {'cuda ms':>12} | {'cuda GB/s':>10} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        # Keep CPU benchmark cost reasonable on older machines.
        cpu_iters = 30 if n <= (1 << 20) else 10
        cuda_iters = 200 if n <= (1 << 20) else 100

        cpu_ms = float(raif.benchmark_gelu_cpu(n, cpu_iters, 5))
        cpu_gbps = effective_bandwidth_gbps(n, cpu_ms)

        cuda_ms = None
        cuda_gbps = None
        speedup = None
        if raif.cuda_available():
            cuda_ms = float(raif.benchmark_gelu_cuda(n, cuda_iters, 20))
            cuda_gbps = effective_bandwidth_gbps(n, cuda_ms)
            speedup = cpu_ms / cuda_ms if cuda_ms > 0 else None

        rows.append({
            "n": n,
            "cpu_ms": cpu_ms,
            "cpu_gbps": cpu_gbps,
            "cuda_ms": cuda_ms,
            "cuda_gbps": cuda_gbps,
            "speedup": speedup,
        })

        print(
            f"{n:12d} | "
            f"{cpu_ms:12.4f} | {cpu_gbps:10.2f} | "
            f"{cuda_ms if cuda_ms is not None else float('nan'):12.4f} | "
            f"{cuda_gbps if cuda_gbps is not None else float('nan'):10.2f} | "
            f"{speedup if speedup is not None else float('nan'):8.2f}"
        )

    csv_path = results_dir / "gelu_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = results_dir / "gelu_benchmark.md"
    with md_path.open("w") as f:
        f.write("# GELU Benchmark Results\n\n")
        f.write("| n | CPU ms | CPU GB/s | CUDA ms | CUDA GB/s | Speedup |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['n']} | "
                f"{fmt_optional(row['cpu_ms'], 4)} | {fmt_optional(row['cpu_gbps'], 2)} | "
                f"{fmt_optional(row['cuda_ms'], 4)} | {fmt_optional(row['cuda_gbps'], 2)} | "
                f"{fmt_optional(row['speedup'], 2)} |\n"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
