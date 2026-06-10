import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_version(cmd):
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        output = (completed.stdout or completed.stderr).strip().splitlines()
        return output[0] if output else "found"
    except Exception as exc:
        return f"found, but version check failed: {exc}"


def python_package_status(name: str) -> str:
    return "ok" if importlib.util.find_spec(name) is not None else "missing"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Reliable AI Inference Fabric setup check")
    print(f"Project root: {root}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    print()

    tools = {
        "cmake": ["cmake", "--version"],
        "nvcc": ["nvcc", "--version"],
        "nvidia-smi": ["nvidia-smi"],
        "git": ["git", "--version"],
    }

    for name, version_cmd in tools.items():
        if command_exists(name):
            print(f"[ok] {name}: {run_version(version_cmd)}")
        else:
            print(f"[missing] {name}")

    print()
    for package in ["numpy", "pybind11", "pytest"]:
        status = python_package_status(package)
        prefix = "[ok]" if status == "ok" else "[missing]"
        print(f"{prefix} Python package {package}: {status}")

    print("\nRecommended first commands:")
    print("  python -m pip install --upgrade pip")
    print("  python -m pip install numpy pybind11 pytest")
    print("  python scripts/build.py --arch 75")
    print("  python -m pytest tests/accuracy -s")
    print("  python benchmarks/bench_gelu.py")
    print("  python benchmarks/bench_layernorm.py")
    print("  python benchmarks/bench_softmax.py")
    print("  python benchmarks/bench_gemm.py")

    print("\nNotes:")
    print("  - RTX 2060 uses CUDA architecture 75, so the default --arch 75 is correct.")
    print("  - If CUDA is not installed yet, run a CPU-only sanity build with: python scripts/build.py --cpu-only")
    print("  - If cuBLAS linking causes issues, rebuild with: python scripts/build.py --arch 75 --no-cublas --clean")


if __name__ == "__main__":
    main()
