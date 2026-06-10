import sys
from pathlib import Path

import numpy as np
import pytest


# LayerNorm uses reductions, so ULP distances are expected to be larger than
# elementwise GELU. We report ULP statistics for visibility, but rely primarily
# on absolute/relative tolerances against a float64 reference rounded to FP32.
SIGNIFICANT_ULP_ABS_THRESHOLD = np.float32(1e-3)


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


def make_layernorm_case(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(rows, cols)).astype(np.float32)

    # Include a few deterministic patterns so the test covers non-random rows,
    # shifted rows, and low-variance rows.
    if rows >= 1:
        x[0, :] = np.linspace(-2.0, 2.0, num=cols, dtype=np.float32)
    if rows >= 2:
        x[1, :] += np.float32(10.0)
    if rows >= 3:
        x[2, :] = np.float32(0.25)
        x[2, ::2] += np.float32(1.0e-3)

    gamma = rng.uniform(0.5, 1.5, size=(cols,)).astype(np.float32)
    beta = rng.uniform(-0.1, 0.1, size=(cols,)).astype(np.float32)
    return x, gamma, beta


def layernorm_reference_fp32(
    x: np.ndarray,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1.0e-5,
) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    mean = np.mean(x64, axis=1, keepdims=True)
    centered = x64 - mean
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    out = centered / np.sqrt(variance + float(eps))
    if gamma is not None:
        out = out * gamma.astype(np.float64, copy=False)[None, :]
    if beta is not None:
        out = out + beta.astype(np.float64, copy=False)[None, :]
    return out.astype(np.float32)


@pytest.mark.parametrize("rows, cols", [(1, 16), (7, 128), (33, 768)])
def test_cpu_layernorm_affine_matches_float64_reference(rows: int, cols: int):
    raif = import_raif()
    x, gamma, beta = make_layernorm_case(rows, cols, seed=20260201 + rows + cols)
    eps = 1.0e-5

    expected = layernorm_reference_fp32(x, gamma, beta, eps)
    actual = raif.layernorm_cpu(x, gamma, beta, eps)

    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    report = summarize(actual, expected)
    print(f"\nCPU LayerNorm affine accuracy report ({rows}x{cols}): {report}")
    assert np.allclose(actual, expected, rtol=5.0e-6, atol=5.0e-6)
    assert report["p99_ulp_significant"] <= 512.0


def test_cpu_layernorm_without_affine_has_zero_mean_unit_variance():
    raif = import_raif()
    x, _, _ = make_layernorm_case(17, 256, seed=20260202)
    eps = 1.0e-5

    expected = layernorm_reference_fp32(x, None, None, eps)
    actual = raif.layernorm_cpu(x, eps=eps)

    report = summarize(actual, expected)
    print(f"\nCPU LayerNorm no-affine accuracy report: {report}")
    assert np.allclose(actual, expected, rtol=5.0e-6, atol=5.0e-6)

    # Non-constant rows should be normalized close to mean 0 and variance 1.
    non_constant = np.var(x.astype(np.float64), axis=1) > 1.0e-3
    row_means = np.mean(actual[non_constant].astype(np.float64), axis=1)
    row_vars = np.var(actual[non_constant].astype(np.float64), axis=1)
    assert float(np.max(np.abs(row_means))) < 2.0e-6
    assert float(np.max(np.abs(row_vars - 1.0))) < 2.0e-4


def test_cuda_layernorm_affine_matches_float64_reference_if_available():
    raif = import_raif()
    if not raif.cuda_available():
        pytest.skip("CUDA device is not available to this Python process")

    rows, cols = 33, 768
    x, gamma, beta = make_layernorm_case(rows, cols, seed=20260203)
    eps = 1.0e-5

    expected = layernorm_reference_fp32(x, gamma, beta, eps)
    actual = raif.layernorm_cuda(x, gamma, beta, eps)

    assert actual.dtype == np.float32
    assert actual.shape == x.shape
    report = summarize(actual, expected)
    print(f"\nCUDA LayerNorm affine accuracy report ({rows}x{cols}): {report}")
    assert np.allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    assert report["p99_ulp_significant"] <= 16384.0
