import math
import sys
from pathlib import Path

import numpy as np
import pytest


# ULP counts are useful, but they become misleading for values extremely close
# to zero. The absolute error can be tiny while the ULP distance is huge because
# float32 spacing near zero is subnormal. We still report all-value ULP metrics,
# but the pass/fail ULP threshold is applied to values with meaningful magnitude.
SIGNIFICANT_ULP_ABS_THRESHOLD = np.float32(1e-8)


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


def gelu_reference_fp32(x: np.ndarray) -> np.ndarray:
    flat = x.astype(np.float64, copy=False).ravel()
    out = np.empty_like(flat, dtype=np.float64)
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    for i, value in enumerate(flat):
        z = value * inv_sqrt_2
        if value < 0.0:
            # Stable equivalent of 1 + erf(z). This avoids cancellation in the
            # reference itself for large negative values.
            out[i] = 0.5 * value * math.erfc(-z)
        else:
            out[i] = 0.5 * value * (1.0 + math.erf(z))
    return out.reshape(x.shape).astype(np.float32)


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


def make_test_values() -> np.ndarray:
    rng = np.random.default_rng(20260101)
    uniform = rng.uniform(-10.0, 10.0, size=8192).astype(np.float32)
    grid = np.linspace(-10.0, 10.0, num=4096, dtype=np.float32)
    tiny = np.array([
        0.0,
        -0.0,
        np.nextafter(np.float32(0.0), np.float32(1.0)),
        np.nextafter(np.float32(0.0), np.float32(-1.0)),
        1e-7,
        -1e-7,
        1e-4,
        -1e-4,
        1.0,
        -1.0,
        3.0,
        -3.0,
    ], dtype=np.float32)
    return np.concatenate([uniform, grid, tiny]).astype(np.float32)


def summarize(actual: np.ndarray, expected: np.ndarray) -> dict:
    abs_err = np.abs(actual - expected)
    rel_err = abs_err / np.maximum(np.abs(expected), np.float32(1e-12))
    ulp = ulp_distance(actual, expected)

    significant = np.abs(expected) >= SIGNIFICANT_ULP_ABS_THRESHOLD
    ulp_sig = ulp[significant]

    report = {
        "max_abs_error": float(np.max(abs_err)),
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


def assert_gelu_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    report = summarize(actual, expected)
    print(f"\n{label} GELU accuracy report: {report}")
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert report["p99_ulp_significant"] <= 64.0


def test_cpu_gelu_matches_float64_reference():
    raif = import_raif()
    x = make_test_values()
    expected = gelu_reference_fp32(x)
    actual = raif.gelu_cpu(x)
    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    assert_gelu_close(actual, expected, "CPU")


def test_cuda_gelu_matches_float64_reference_if_available():
    raif = import_raif()
    if not raif.cuda_available():
        pytest.skip("CUDA device is not available to this Python process")

    x = make_test_values()
    expected = gelu_reference_fp32(x)
    actual = raif.gelu_cuda(x)
    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    assert_gelu_close(actual, expected, "CUDA")
