import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path) -> None:
    print("+ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure and build RAIF locally with CMake.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory")
    parser.add_argument("--config", default="Release", help="Build config for multi-config generators")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA and build CPU-only")
    parser.add_argument("--arch", default="75", help="CUDA architecture. RTX 2060 uses 75.")
    parser.add_argument("--no-cublas", action="store_true", help="Disable the optional cuBLAS GEMM benchmark baseline")
    parser.add_argument("--clean", action="store_true", help="Delete the build directory before building")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = root / args.build_dir

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    cmake_configure = [
        "cmake",
        "-S", str(root),
        "-B", str(build_dir),
        f"-DRAIF_ENABLE_CUDA={'OFF' if args.cpu_only else 'ON'}",
        f"-DRAIF_ENABLE_CUBLAS={'OFF' if args.no_cublas else 'ON'}",
        f"-DRAIF_CUDA_ARCHITECTURES={args.arch}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    run(cmake_configure, root)

    cmake_build = [
        "cmake",
        "--build", str(build_dir),
        "--config", args.config,
        "--parallel",
    ]
    run(cmake_build, root)

    module_dirs = sorted({p.parent for p in build_dir.rglob("raif*.pyd")})
    module_dirs += sorted({p.parent for p in build_dir.rglob("raif*.so")})
    module_dirs += sorted({p.parent for p in build_dir.rglob("raif*.dylib")})

    print("\nBuild complete.")
    if module_dirs:
        print("Python module directory:")
        print(f"  {module_dirs[0]}")
        print("\nRun tests with:")
        print("  python -m pytest tests/accuracy -s")
        print("\nRun benchmark with:")
        print("  python benchmarks/bench_gelu.py")
        print("  python benchmarks/bench_layernorm.py")
        print("  python benchmarks/bench_softmax.py")
        print("  python benchmarks/bench_gemm.py")
    else:
        print("Could not find the Python extension module under the build directory.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
