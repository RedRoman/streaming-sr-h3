"""NumPy-only byte verification for one train-role VAE cache; no model replay."""
import hashlib
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
    """Hash C-order pixel/tensor bytes without a second full-size byte string."""
    array = np.ascontiguousarray(value)
    data = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(data), 8 << 20):
        digest.update(data[offset:offset + (8 << 20)])
    return digest.hexdigest()


def array_facts(value):
    require(value.dtype == np.float32, "expected FP32 cache array")
    flat = value.reshape(-1)
    require(flat.size > 0, "empty cache array")
    lower, upper = float("inf"), float("-inf")
    for start in range(0, flat.size, 1 << 20):
        chunk = flat[start:start + (1 << 20)]
        require(np.isfinite(chunk).all(), "nonfinite cache array")
        lower, upper = min(lower, float(chunk.min())), max(upper, float(chunk.max()))
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "array_sha256": array_sha(value), "elements": int(value.size),
            "min": lower, "max": upper, "finite": True}


def verified_npy(path, record, shape, *, max_bytes=128 << 20):
    path = Path(path)
    require(path.suffix == ".npy" and path.is_file(), "expected existing NPY cache")
    require(path.stat().st_size <= max_bytes, "array exceeds explicit read budget")
    before = path.stat()
    require(file_sha(path) == record["file_sha256"], "NPY file SHA mismatch")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    require(list(value.shape) == list(shape) == record["shape"], "NPY shape mismatch")
    require(str(value.dtype) == record["dtype"] == "float32", "NPY dtype mismatch")
    facts = array_facts(value)
    require(facts["array_sha256"] == record["array_sha256"], "NPY content SHA mismatch")
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "NPY changed during CPU audit")
    if "bytes" in record:
        require(record["bytes"] == before.st_size, "NPY byte count mismatch")
    return value, {**facts, "file_sha256": record["file_sha256"], "bytes": before.st_size}


def verify_clipped_rphi(raw, clipped):
    require(raw.shape == clipped.shape and raw.dtype == clipped.dtype == np.float32,
            "Rphi clip shape/dtype mismatch")
    for index in range(len(raw)):
        require(np.isfinite(raw[index]).all() and np.isfinite(clipped[index]).all(),
                "nonfinite Rphi guide")
        require(np.array_equal(np.clip(raw[index], 0, 1), clipped[index]),
                "Rphi cache is not exact clipped prediction")


def pil_same_geometry_hashes(value):
    """Official fixed-size PIL uint8 conversion, with no resize or new model call.

    This helper applies only when output H/W equal input H/W. Pillow's same-size
    Lanczos resize is an identity on uint8 RGB pixels. The producer must attest
    actual helper identity, crop mode, geometry and observed post-helper hashes.
    """
    require(value.dtype == np.float32 and value.ndim == 4 and value.shape[-1] == 3,
            "expected FP32 THWC RGB before PIL conversion")
    u8_hash, normalized_hash = hashlib.sha256(), hashlib.sha256()
    for frame in value:
        require(np.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 1,
                "pre-PIL input is not finite clipped RGB")
        u8 = (frame * np.float32(255.0)).clip(0, 255).astype(np.uint8)
        normalized = u8.astype(np.float32) / np.float32(255.0)
        u8_hash.update(memoryview(np.ascontiguousarray(u8)).cast("B"))
        normalized_hash.update(memoryview(np.ascontiguousarray(normalized)).cast("B"))
    return {"before_float32_sha256": array_sha(value),
            "pil_uint8_sha256": u8_hash.hexdigest(),
            "after_float32_sha256": normalized_hash.hexdigest(),
            "shape": list(value.shape), "conversion": "clip(x*255,0,255).astype(uint8); uint8/255 float32",
            "same_geometry_only": True}
