"""CPU-only exact VAE input-gradient repeatability comparison/admission.

Re-read every saved array and receipt before comparing. Descriptive numerical
errors never relax the exact gate. No Torch, Comfy, NPU or model imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = "openh3.vae_gradient_repeatability_comparison.v1"
EXPECTED_INPUT_SHA = "1b53b9ebdc3358b5a5c89a639e9aac04e0b769cde9bccff71d57d4cb6e1c3ed4"
REQUIRED_GATES = {"observation_completed", "forward_parity", "finite_nonzero_scaled_gradient",
                  "frozen_hash_unchanged", "frozen_versions_unchanged", "runtime_unchanged",
                  "checkpoint_restored", "checkpoint_execution", "bounded_calls"}


def checkpoint_matches(audit, mode):
    """The fixed seven-latent-frame decode invokes three chunks of 36 blocks."""
    backwards = 1 if mode == "single" else 2
    return (audit.get("enabled") is True and audit.get("active") is False
            and audit.get("restored") is True and audit.get("block_count") == 36
            and audit.get("wrapped") == 36 and audit.get("use_reentrant") is False
            and audit.get("preserve_rng_state") is True
            and audit.get("checkpoint_forward_calls") == 108
            and audit.get("no_grad_forward_calls") == 0
            and audit.get("recompute_calls") == 108 * backwards
            and audit.get("output_gradient_guards") == 108
            and audit.get("backward_guard_calls") == 108 * backwards
            and audit.get("recomputed_block_indices") == list(range(36)))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


def save_array(root, name, array):
    root = Path(root)
    array = np.ascontiguousarray(array)
    if array.dtype != np.float32:
        raise TypeError("probe arrays must be float32")
    path = root / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temp, path)
    return {"path": name, "file_sha256": sha_file(path), "array_sha256": array_sha(array),
            "shape": list(array.shape), "dtype": str(array.dtype), "bytes": path.stat().st_size}


def read_array(root, record):
    root = Path(root).resolve()
    path = (root / record["path"]).resolve()
    if path.parent != root:
        raise ValueError("array must be directly inside its immutable run directory")
    if path.stat().st_size != record["bytes"] or sha_file(path) != record["file_sha256"]:
        raise ValueError("array file hash/size mismatch: " + str(path))
    array = np.load(path, allow_pickle=False)
    if (array.dtype != np.float32 or str(array.dtype) != record["dtype"]
            or list(array.shape) != record["shape"] or array_sha(array) != record["array_sha256"]):
        raise ValueError("array content/dtype/shape mismatch: " + str(path))
    return array


def array_statistics(array):
    finite = np.isfinite(array)
    valid = array[finite].astype(np.float64)
    magnitude = np.abs(valid)
    nonzero = magnitude[magnitude != 0]
    return {"shape": list(array.shape), "dtype": str(array.dtype), "elements": int(array.size),
            "finite_count": int(finite.sum()), "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()), "nonzero_count": int(np.count_nonzero(valid)),
            "zero_count": int(np.count_nonzero(array == 0)),
            "l2_float64": float(np.linalg.norm(valid)),
            "max_abs": float(magnitude.max()) if valid.size else None,
            "min_nonzero_abs": float(nonzero.min()) if nonzero.size else None,
            "array_sha256": array_sha(array)}


def compare_arrays(left, right):
    if left.dtype != np.float32 or right.dtype != np.float32 or left.shape != right.shape:
        raise ValueError("gradient comparison requires same-shape float32 arrays")
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    difference = b - a
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    left_nonzero, right_nonzero = left != 0, right != 0
    both = left_nonzero & right_nonzero
    return {
        "finite": finite,
        "exact_equal": bool(np.array_equal(left, right)) and array_sha(left) == array_sha(right),
        "equal_coordinates": int(np.count_nonzero(left == right)),
        "different_coordinates": int(np.count_nonzero(left != right)),
        "max_abs_elementwise_difference": float(np.max(np.abs(difference))) if finite else None,
        "l2_difference_float64": float(np.linalg.norm(difference)) if finite else None,
        "relative_l2_to_left": float(np.linalg.norm(difference) / na) if finite and na else None,
        "relative_l2_to_left_zero_norm": na == 0,
        "cosine_similarity": float(np.dot(a, b) / (na * nb)) if finite and na and nb else None,
        "cosine_zero_norm": na == 0 or nb == 0,
        "support": {"left_nonzero": int(left_nonzero.sum()), "right_nonzero": int(right_nonzero.sum()),
                    "intersection": int(both.sum()), "union": int((left_nonzero | right_nonzero).sum()),
                    "left_only": int((left_nonzero & ~right_nonzero).sum()),
                    "right_only": int((right_nonzero & ~left_nonzero).sum())},
        "opposite_sign_on_common_nonzero": int(np.count_nonzero(np.signbit(left[both]) != np.signbit(right[both]))),
        "signed_zero_difference": int(np.count_nonzero((left == 0) & (right == 0) & (np.signbit(left) != np.signbit(right)))),
    }


def load_probe(path):
    root = Path(path).resolve()
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "openh3.vae_gradient_repeatability_probe.v1"
            or report.get("complete") is not True or report.get("observations_complete") is not True):
        raise ValueError("incomplete or unsupported probe: " + str(root))
    mode = report["mode"]
    if mode not in ("single", "paired_history"):
        raise ValueError("unknown probe mode")
    if report["actual_completed_forwards"] != 2 or report["actual_completed_backwards"] != (1 if mode == "single" else 2):
        raise ValueError("unexpected probe execution count")
    if set(report["gates"]) != REQUIRED_GATES or any(v is not True for v in report["gates"].values()):
        raise ValueError("producer gate failed")
    if not checkpoint_matches(report["checkpoint"], mode):
        raise ValueError("checkpoint execution/restoration mismatch")
    frozen = report["frozen"]
    if frozen["before"]["sha256"] != frozen["after"]["sha256"] or not frozen["versions_unchanged"]:
        raise ValueError("frozen model mismatch")
    if report["runtime_before"] != report["runtime_after"]:
        raise ValueError("runtime changed within probe")
    arrays = {name: read_array(root, rec) for name, rec in report["arrays"].items()}
    expected_names = {"input", "scaled_raw", "scaled_unscaled"}
    if mode == "paired_history": expected_names.add("prior_unscaled")
    if set(arrays) != expected_names:
        raise ValueError("missing or unexpected gradient artifacts")
    if any(list(v.shape) != [1, 24, 7, 30, 52] for v in arrays.values()):
        raise ValueError("unexpected full diagnostic geometry")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite probe array")
    scale = report["common_contract"]["loss_scale"]
    if scale != 1024.0 or not np.array_equal(arrays["scaled_raw"] / np.float32(scale), arrays["scaled_unscaled"]):
        raise ValueError("raw-to-unscaled FP32 relation failed")
    if not np.count_nonzero(arrays["scaled_raw"]):
        raise ValueError("zero scaled gradient")
    if array_sha(arrays["input"]) != report["common_contract"]["input_array_sha256"]:
        raise ValueError("input differs from contract")
    if (report["common_contract"].get("expected_seed101_input_sha256") != EXPECTED_INPUT_SHA
            or array_sha(arrays["input"]) != EXPECTED_INPUT_SHA):
        raise ValueError("predeclared input hash contract missing")
    if (report["forward"]["allclose"] is not True or report["forward"].get("finite") is not True
            or report["forward"].get("requires_grad") is not True or report["forward"].get("shape_ok") is not True):
        raise ValueError("forward parity gate failed")
    return {"root": str(root), "report_sha256": sha_file(report_path), "report": report, "arrays": arrays}


def compare_runs(a1_dir, a2_dir, b_dir=None):
    a1, a2 = load_probe(a1_dir), load_probe(a2_dir)
    if a1["root"] == a2["root"] or a1["report"]["execution_id"] == a2["report"]["execution_id"]:
        raise ValueError("A1/A2 must be distinct fresh-process executions")
    if a1["report"]["mode"] != "single" or a2["report"]["mode"] != "single":
        raise ValueError("A1/A2 must execute identical single mode")
    if a1["report"]["common_contract"] != a2["report"]["common_contract"]:
        raise ValueError("A1/A2 execution contracts differ")
    if a1["report"]["frozen"]["before"]["sha256"] != a2["report"]["frozen"]["before"]["sha256"]:
        raise ValueError("loaded frozen weights differ")
    if a1["report"]["runtime_before"] != a2["report"]["runtime_before"]:
        raise ValueError("actual runtime/precision/determinism differs")
    hashes = ("reference_sha256", "prediction_sha256", "dtype", "shape", "target_sha256")
    forward_equal = all(a1["report"]["forward"][k] == a2["report"]["forward"][k] for k in hashes)
    comparisons = {k: compare_arrays(a1["arrays"][k], a2["arrays"][k]) for k in ("input", "scaled_raw", "scaled_unscaled")}
    exact = forward_equal and all(c["exact_equal"] for c in comparisons.values())
    result = {"schema": SCHEMA, "observations_complete": True, "a1_a2_exact": exact,
              "allow_paired_history": exact, "complete": exact,
              "a1": {k: a1[k] for k in ("root", "report_sha256")},
              "a2": {k: a2[k] for k in ("root", "report_sha256")},
              "common_contract": a1["report"]["common_contract"],
              "runtime_before": a1["report"]["runtime_before"],
              "frozen_before": a1["report"]["frozen"]["before"],
              "cross_process_forward_hashes_equal": forward_equal, "a1_a2": comparisons,
              "criterion": "Exact bytes/hashes; descriptive errors never create a tolerance", "b": None}
    if b_dir is not None:
        if not exact:
            raise ValueError("paired-history comparison is forbidden until A1/A2 exact")
        b = load_probe(b_dir)
        if b["root"] in (a1["root"], a2["root"]) or b["report"]["execution_id"] in (a1["report"]["execution_id"], a2["report"]["execution_id"]):
            raise ValueError("paired history must be a third fresh process")
        admission = b["report"].get("admission")
        if not admission or not admission.get("sha256") or not admission.get("a1") or not admission.get("a2"):
            raise ValueError("paired history lacks its exact A1/A2 admission receipt")
        if admission["a1"] != result["a1"] or admission["a2"] != result["a2"]:
            raise ValueError("paired history admission binds different A1/A2 evidence")
        if b["report"]["mode"] != "paired_history" or b["report"]["common_contract"] != result["common_contract"]:
            raise ValueError("paired history contract mismatch")
        if b["report"]["runtime_before"] != a1["report"]["runtime_before"] or b["report"]["frozen"]["before"]["sha256"] != a1["report"]["frozen"]["before"]["sha256"]:
            raise ValueError("paired runtime or weights differ")
        b_forward = all(a1["report"]["forward"][k] == b["report"]["forward"][k] for k in hashes)
        b_compare = {k: compare_arrays(a1["arrays"][k], b["arrays"][k]) for k in ("input", "scaled_raw", "scaled_unscaled")}
        b_exact = b_forward and all(c["exact_equal"] for c in b_compare.values())
        result["b"] = {"root": b["root"], "report_sha256": b["report_sha256"], "exact_to_a1": b_exact,
                       "forward_hashes_equal": b_forward, "comparisons": b_compare,
                       "prior_unscaled": array_statistics(b["arrays"]["prior_unscaled"])}
        result["complete"] = b_exact
    return result


def validate_admission(path):
    """Recompute from immutable arrays; a hand-edited allow=true cannot admit B."""
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    if saved.get("schema") != SCHEMA or saved.get("allow_paired_history") is not True or saved.get("b") is not None:
        raise ValueError("missing exact A1/A2 admission")
    current = compare_runs(saved["a1"]["root"], saved["a2"]["root"])
    if current != saved or not current["a1_a2_exact"]:
        raise ValueError("admission does not reproduce from original evidence")
    return {"path": str(path), "sha256": sha_file(path), "comparison": current}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = compare_runs(args.a1, args.a2, args.b)
    except BaseException as error:
        atomic_json(args.out, {"schema": SCHEMA, "complete": False, "allow_paired_history": False,
                              "observations_complete": False, "error": repr(error)})
        raise
    atomic_json(args.out, result)
    print(json.dumps({"complete": result["complete"], "allow_paired_history": result["allow_paired_history"], "out": str(args.out)}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
