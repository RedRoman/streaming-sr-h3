"""Independent NumPy-only array checks for the fixed small joint RGB probe."""
import hashlib
import math
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verified_array(path, expected_file_sha, expected_array_sha, shape, dtype):
    path = Path(path)
    require(path.suffix == ".npy" and path.stat().st_size <= 64 << 20, "unexpected array file type/size")
    require(file_sha(path) == expected_file_sha, "array file SHA mismatch")
    value = np.load(path, allow_pickle=False)
    require(str(value.dtype) == dtype and list(value.shape) == list(shape), "array shape/dtype mismatch")
    require(array_sha(value) == expected_array_sha, "array content SHA mismatch")
    return value


def compare_outputs(candidate, reference, *, exact, atol=0.0002, rtol=0.0001):
    require(candidate.dtype == reference.dtype == np.float32 and candidate.shape == reference.shape,
            "output shape/dtype mismatch")
    require(np.isfinite(candidate).all() and np.isfinite(reference).all(), "nonfinite output")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    equal = bool(np.array_equal(candidate, reference))
    # Torch's allclose performs float32 subtraction/threshold arithmetic for
    # these saved float32 outputs. Keep the declared candidate/reference order.
    within = bool(np.all(np.abs(candidate - reference) <= np.float32(atol) + np.float32(rtol) * np.abs(reference)))
    require(equal if exact else within, "strict output parity failed" if exact else "declared RGB allclose failed")
    denom = float(np.linalg.norm(reference.astype(np.float64)))
    return {"exact_equal": equal, "byte_equal": array_sha(candidate) == array_sha(reference),
            "allclose": within, "max_abs_float64": float(np.max(np.abs(delta))),
            "relative_l2_float64": float(np.linalg.norm(delta) / denom) if denom else None,
            "reference_zero_norm": denom == 0}


def verify_decoder_cast(video, cast):
    require(video.dtype == np.float32 and cast.dtype == np.float16 and video.shape == cast.shape,
            "decoder cast shape/dtype mismatch")
    require(np.isfinite(video).all() and np.isfinite(cast).all(), "nonfinite decoder input")
    require(array_sha(video.astype(np.float16)) == array_sha(cast), "saved decoder input is not the actual float16 cast")


def recompute_rgb_mse(rgb, target, expected_target_sha):
    require(rgb.dtype == target.dtype == np.float32 and np.isfinite(rgb).all() and np.isfinite(target).all(),
            "invalid RGB/target dtype or finite state")
    require(array_sha(target) == expected_target_sha, "fixed synthetic target SHA mismatch")
    require(target.shape == (1, 1, 128, 1) and rgb.shape == (5, 128, 128, 3), "fixed RGB/target geometry differs")
    difference = rgb.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(difference * difference, dtype=np.float64))


def verify_unscale(raw, unscaled, *, nonzero=False):
    require(raw.dtype == unscaled.dtype == np.float32 and raw.shape == unscaled.shape,
            "gradient pair shape/dtype mismatch")
    require(np.isfinite(raw).all() and np.isfinite(unscaled).all(), "nonfinite gradient")
    require(array_sha(raw / np.float32(1024.)) == array_sha(unscaled), "FP32 raw/1024 gradient bytes mismatch")
    count = int(np.count_nonzero(unscaled))
    require(not nonzero or count > 0, "required nonzero gradient is zero")
    return {"elements": int(raw.size), "raw_nonzero": int(np.count_nonzero(raw)),
            "unscaled_nonzero": count, "raw_l2_float64": float(np.linalg.norm(raw.astype(np.float64))),
            "unscaled_l2_float64": float(np.linalg.norm(unscaled.astype(np.float64)))}


def verify_reduction(actual, expected, label):
    """Only a saved descriptive reduction; never array/parity acceptance."""
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), "descriptive reduction differs: " + label)
