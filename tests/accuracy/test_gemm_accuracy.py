import sys
from pathlib import Path

import numpy as np
import pytest


# GEMM is a reduction over K. CPU and CUDA accumulate in FP32, while the
# reference accumulates in float64 before rounding to FP32. ULP values are
# reported for diagnostics, but pass/fail uses absolute/relative tolerances.
SIGNIFICANT_ULP_ABS_THRESHOLD = np.float32(1e-4)


def import_raif():
    try:
        import raif  # type: ignore
        return raif
    except ImportError:
        root = Path(__file__).resolve().parents[2]
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


def float32_to_ordered_int(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    sign_mask = np.uint32(0x80000000)
    negative = (bits & sign_mask) != 0
    ordered = np.where(negative, (~bits) & np.uint32(0xFFFFFFFF), bits | sign_mask)
    return ordered.astype(np.int64)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ai = float32_to_ordered_int(a)
    bi = float32_to_ordered_int(b)
    return np.abs(ai - bi)


def summarize(actual: np.ndarray, expected: np.ndarray) -> dict:
    abs_err = np.abs(actual - expected)
    rel_err = abs_err / np.maximum(np.abs(expected), np.float32(1e-12))
    ulp = ulp_distance(actual, expected)
    significant = np.abs(expected) >= SIGNIFICANT_ULP_ABS_THRESHOLD
    ulp_sig = ulp[significant]

    report = {
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "p99_abs_error": float(np.percentile(abs_err, 99.0)),
        "max_rel_error": float(np.max(rel_err)),
        "max_ulp_all": int(np.max(ulp)),
        "mean_ulp_all": float(np.mean(ulp)),
        "p99_ulp_all": float(np.percentile(ulp, 99.0)),
        "significant_ulp_threshold": float(SIGNIFICANT_ULP_ABS_THRESHOLD),
        "significant_count": int(np.count_nonzero(significant)),
    }
    if ulp_sig.size > 0:
        report.update({
            "max_ulp_significant": int(np.max(ulp_sig)),
            "mean_ulp_significant": float(np.mean(ulp_sig)),
            "p99_ulp_significant": float(np.percentile(ulp_sig, 99.0)),
        })
    else:
        report.update({
            "max_ulp_significant": 0,
            "mean_ulp_significant": 0.0,
            "p99_ulp_significant": 0.0,
        })
    return report


def make_gemm_case(m: int, n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.25, size=(m, k)).astype(np.float32)
    b = rng.normal(loc=0.0, scale=0.20, size=(k, n)).astype(np.float32)

    # Deterministic structure catches row/column indexing bugs.
    if m > 0 and k > 0:
        a[0, :] = np.linspace(-0.5, 0.5, num=k, dtype=np.float32)
    if k > 0 and n > 0:
        b[:, 0] = np.linspace(0.25, -0.25, num=k, dtype=np.float32)
    if m > 1 and k > 1:
        a[1, :] = np.float32(0.125)
    if k > 1 and n > 1:
        b[:, 1] = np.float32(-0.0625)
    return a, b


def gemm_reference_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float64, copy=False) @ b.astype(np.float64, copy=False)).astype(np.float32)


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (3, 5, 7), (17, 19, 23), (32, 64, 48)])
def test_cpu_gemm_matches_float64_reference(m: int, n: int, k: int):
    raif = import_raif()
    a, b = make_gemm_case(m, n, k, seed=20260401 + m + n + k)

    expected = gemm_reference_fp32(a, b)
    actual = raif.gemm_cpu(a, b)

    assert actual.dtype == np.float32
    assert actual.shape == (m, n)
    report = summarize(actual, expected)
    print(f"\nCPU GEMM accuracy report ({m}x{k} @ {k}x{n}): {report}")
    assert np.allclose(actual, expected, rtol=3.0e-5, atol=2.0e-5)
    assert report["p99_abs_error"] <= 2.0e-5


def test_cuda_gemm_matches_float64_reference_if_available():
    raif = import_raif()
    if not raif.cuda_available():
        pytest.skip("CUDA device is not available to this Python process")

    m, n, k = 35, 41, 79
    a, b = make_gemm_case(m, n, k, seed=20260402)

    expected = gemm_reference_fp32(a, b)
    actual = raif.gemm_cuda(a, b)

    assert actual.dtype == np.float32
    assert actual.shape == (m, n)
    report = summarize(actual, expected)
    print(f"\nCUDA GEMM accuracy report ({m}x{k} @ {k}x{n}): {report}")
    assert np.allclose(actual, expected, rtol=3.0e-4, atol=2.0e-4)
    assert report["p99_abs_error"] <= 2.0e-4


def test_gemm_rejects_shape_mismatch():
    raif = import_raif()
    a = np.zeros((4, 5), dtype=np.float32)
    b = np.zeros((6, 7), dtype=np.float32)
    with pytest.raises(Exception):
        raif.gemm_cpu(a, b)
