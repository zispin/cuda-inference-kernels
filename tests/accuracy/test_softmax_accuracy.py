import sys
from pathlib import Path

import numpy as np
import pytest


# Softmax can produce very small probabilities. ULP values are useful for
# diagnostics, but pass/fail uses allclose plus ULP stats only for probabilities
# large enough to matter numerically.
SIGNIFICANT_ULP_ABS_THRESHOLD = np.float32(1e-6)


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
    significant = expected >= SIGNIFICANT_ULP_ABS_THRESHOLD
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


def make_softmax_case(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=2.0, size=(rows, cols)).astype(np.float32)

    # Deterministic rows cover uniform logits, shifted logits, low dynamic range,
    # and a numerically extreme but stable row.
    if rows >= 1:
        x[0, :] = np.linspace(-8.0, 8.0, num=cols, dtype=np.float32)
    if rows >= 2:
        x[1, :] = np.float32(7.0)
    if rows >= 3:
        x[2, :] = np.float32(-100.0)
        x[2, cols // 2] = np.float32(100.0)
    if rows >= 4:
        x[3, :] = rng.normal(loc=20.0, scale=0.05, size=(cols,)).astype(np.float32)
    return x


def softmax_reference_fp32(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    shifted = x64 - np.max(x64, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return out.astype(np.float32)


@pytest.mark.parametrize("rows, cols", [(1, 8), (7, 128), (33, 768)])
def test_cpu_softmax_matches_float64_reference(rows: int, cols: int):
    raif = import_raif()
    x = make_softmax_case(rows, cols, seed=20260301 + rows + cols)

    expected = softmax_reference_fp32(x)
    actual = raif.softmax_cpu(x)

    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    report = summarize(actual, expected)
    print(f"\nCPU Softmax accuracy report ({rows}x{cols}): {report}")
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)
    assert report["p99_ulp_significant"] <= 64.0

    row_sums = np.sum(actual.astype(np.float64), axis=1)
    assert float(np.max(np.abs(row_sums - 1.0))) < 2.0e-6


def test_cuda_softmax_matches_float64_reference_if_available():
    raif = import_raif()
    if not raif.cuda_available():
        pytest.skip("CUDA device is not available to this Python process")

    rows, cols = 33, 768
    x = make_softmax_case(rows, cols, seed=20260302)

    expected = softmax_reference_fp32(x)
    actual = raif.softmax_cuda(x)

    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    report = summarize(actual, expected)
    print(f"\nCUDA Softmax accuracy report ({rows}x{cols}): {report}")
    assert np.allclose(actual, expected, rtol=1.0e-4, atol=2.0e-6)
    assert report["p99_ulp_significant"] <= 4096.0

    row_sums = np.sum(actual.astype(np.float64), axis=1)
    assert float(np.max(np.abs(row_sums - 1.0))) < 2.0e-5
