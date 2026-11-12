"""RGB supervision for the audited, materialized four-task R_phi pilot.

The target is the SAME original-recording RGB reference for every task. It is
not the legacy range target D(x), and the recording is a reference proxy, not a
claim of pristine capture. ``load`` preserves native uint8 geometry. Training
uses an HR bicubic *guide* for SR; that interpolation is not the physical
forward degradation or an exact data-consistency operator.

Only public task/scale enter the condition. Oracle severity and degradation
parameters in NPZ metadata never enter it. Identity/near-clean controls retain
the nominal task/scale condition, to regularize every task toward preserving
already-clean input; their control label is returned for logging/losses only.

Current NPZ receipt/file/pixel hashes are verified by default. Original-source
provenance is explicitly a materialization attestation, not live source-video
verification; opt into verify_original_sources=True to rehash original videos.
Call preflight() before training. Stat-keyed caches avoid repeated hashing and
a bounded LRU retains uint8 assets with shared targets deduplicated by path.
No floating-point clip cache or accelerator is used here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
import re

import numpy as np


TASKS = ("sr", "deblur", "denoise", "compression")
SCHEMA = "openh3.materialized_omni_pilot.v1"
ORIGINAL_ROOT = Path("/cache/zihaohe/omni_openh3_20261111/datasets/omni_pilot_v1")
NEAR_CLEAN_SIGMA_8BIT = 1.0
GUIDE_RECIPE = {
    "implementation": "torch.nn.functional.interpolate",
    "mode": "bicubic", "align_corners": False, "antialias": True,
    "range": "float32 [0,1], clamped after resize",
    "role": "HR input guide; not the physical forward operator",
}


def _torch():
    # Loading/validating RGB assets does not require torch or its NPU extension.
    import torch
    return torch


def _pixels(value, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 ndarray")
    if value.ndim != 4 or value.shape[-1] != 3 or min(value.shape[:3]) < 1:
        raise ValueError(f"{name} must have nonempty THWC RGB geometry")
    return value


def build_condition(task, known=True):
    """Return CPU float32 [onehot(4), log2(nominal scale)/3, known_mask].

    Unknown mode removes task AND scale, rather than leaking the true task
    behind a zero known mask. The pilot's ordinary sampling uses known=True.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    values = [0.0] * 6
    if known:
        values[TASKS.index(task)] = 1.0
        values[4] = math.log2(4 if task == "sr" else 1) / 3
        values[5] = 1.0
    return _torch().tensor(values, dtype=_torch().float32, device="cpu")


def prepare_observed(observed, target_hw):
    """Native uint8 THWC -> CPU float32 TCHW HR guide in [0,1].

    Native tasks are normalized without resampling. The caller must already
    have checked geometry; only exact x4 upsampling is supported for this pilot.
    """
    observed = _pixels(observed, "observed")
    target_hw = tuple(map(int, target_hw))
    if len(target_hw) != 2 or min(target_hw) < 1:
        raise ValueError("target_hw must be positive (height, width)")
    source_hw = observed.shape[1:3]
    if source_hw != target_hw and tuple(v * 4 for v in source_hw) != target_hw:
        raise ValueError("guide input must be native target geometry or exactly x4 SR")
    torch = _torch()
    value = torch.from_numpy(np.array(observed.transpose(0, 3, 1, 2), order="C", copy=True)).float().div_(255)
    if source_hw != target_hw:
        value = torch.nn.functional.interpolate(value, size=target_hw, mode="bicubic",
                                                align_corners=False, antialias=True).clamp_(0, 1)
    return value


def make_near_clean(clean, seed, sigma_8bit=NEAR_CLEAN_SIGMA_8BIT):
    """Deterministic PCG64 Gaussian noise, then rint ties-even/clip/uint8.

    Pass each unique causal frame once, then repeat its noisy result for clip
    boundary padding. Repeated copies must not receive independent noise.
    """
    clean = _pixels(clean, "clean")
    if not np.isfinite(sigma_8bit) or sigma_8bit < 0:
        raise ValueError("sigma_8bit must be finite and nonnegative")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    noise = rng.standard_normal(clean.shape, dtype=np.float32) * np.float32(sigma_8bit)
    return np.clip(np.rint(clean.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino


class RphiDataset:
    """Strict manifest-backed RGB dataset with bounded, per-load uint8 memory."""

    def __init__(self, manifest, root_override=None, verify_hashes=True,
                 cache_bytes=6 * 1024**3, verify_original_sources=False):
        self.manifest_path = Path(manifest).expanduser().resolve()
        payload = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        self.root_override = Path(root_override).expanduser().resolve() if root_override is not None else None
        self.verify_hashes = bool(verify_hashes)
        self.verify_original_sources = bool(verify_original_sources)
        if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.cache_bytes = cache_bytes
        self._cache_used_bytes = 0
        self._asset_cache, self._receipt_cache = OrderedDict(), {}
        self.receipt_hashes = {}
        self.provenance = {
            "schema": SCHEMA, "manifest_sha256": self.manifest_sha256,
            "original_source_verification": "live_source_sha256" if self.verify_original_sources else "materialization_attestation_only",
            "current_npz_verification": "receipt_file_and_pixel_sha256" if self.verify_hashes else "disabled_explicitly",
            "target_semantics": "shared clean original-recording RGB reference proxy",
            "task_order": list(TASKS), "condition": "public onehot4 + log2(nominal scale)/3 + known_mask",
            "control_condition": "retain nominal task/scale; control label is not a model condition",
            "near_clean_sigma_8bit": NEAR_CLEAN_SIGMA_8BIT, "sr_input_preparation": dict(GUIDE_RECIPE),
        }
        self.records = []
        self._pixel_hash_cache, self._source_hash_cache = {}, {}
        seen, targets = set(), {}
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._check_record(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{self.manifest_path}:{line_number}: {exc}") from exc
            key = row["target_key"], row["task"]
            if key in seen:
                raise ValueError(f"duplicate target/task sample: {key}")
            seen.add(key)
            binding = tuple(row[name] for name in ("source_id", "source_sha256", "target_npz", "target_pixel_sha256", "split"))
            if row["target_key"] in targets and targets[row["target_key"]] != binding:
                raise ValueError("tasks do not share the same source/clean target/split")
            targets[row["target_key"]] = binding
            self.records.append(row)
        if not self.records:
            raise ValueError("manifest has no samples")
        self.by_task = {task: [i for i, row in enumerate(self.records) if row["task"] == task]
                        for task in TASKS}
        self.tasks = tuple(task for task in TASKS if self.by_task[task])

    @staticmethod
    def _check_record(row):
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise ValueError("unsupported materialized manifest schema")
        if row.get("status") != "materialized_not_vae_encoded":
            raise ValueError("expected a completed RGB materialization manifest")
        if row.get("task") not in TASKS or row.get("split") not in ("train", "dev", "test"):
            raise ValueError("unsupported task or split")
        for key in ("source_id", "target_key", "target_npz", "observation_npz"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing nonempty {key}")
        for key in ("source_sha256", "target_pixel_sha256", "observation_pixel_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError(f"invalid {key}")

    def __len__(self):
        return len(self.records)

    def resolve_path(self, path):
        """Resolve relative manifest paths and the one explicitly supported root remap."""
        path = Path(path).expanduser()
        if self.root_override is not None:
            try:
                path = self.root_override / path.relative_to(ORIGINAL_ROOT)
            except ValueError:
                pass
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        return path.resolve()

    def _receipt(self, row):
        path = self.resolve_path(row["target_npz"]).parent / "complete.json"
        before = _signature(path)
        cached = self._receipt_cache.get(str(path))
        if cached is not None and cached[0] == before:
            receipt = cached[1]
        else:
            raw = path.read_bytes()
            receipt = json.loads(raw)
            if before != _signature(path):
                raise ValueError(f"receipt changed while reading: {path}")
            self._receipt_cache[str(path)] = before, receipt
            self.receipt_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        if receipt.get("status") != "materialized" or receipt.get("materializer_version") != "omni-pilot-materializer-v1":
            raise ValueError("missing/incompatible completed materialization receipt")
        for key in ("source_id", "source_sha256", "target_key", "split"):
            if receipt.get(key) != row[key]:
                raise ValueError(f"receipt/manifest {key} mismatch")
        try:
            target_asset = receipt["target"]
            observation_asset = receipt["observations"][row["task"]]
        except KeyError as exc:
            raise ValueError("receipt lacks the requested target/task") from exc
        for asset, prefix in ((target_asset, "target"), (observation_asset, "observation")):
            if self.resolve_path(asset["path"]) != self.resolve_path(row[prefix + "_npz"]):
                raise ValueError("receipt/manifest NPZ path mismatch")
            if asset.get("pixel_sha256") != row[prefix + "_pixel_sha256"]:
                raise ValueError("receipt/manifest pixel hash mismatch")
            if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("file_sha256", ""))):
                raise ValueError("receipt lacks valid NPZ file SHA256")
        return target_asset, observation_asset

    def _read_asset(self, asset):
        path, expected_sha, file_sha = asset["path"], asset["pixel_sha256"], asset["file_sha256"]
        path = self.resolve_path(path)
        before = _signature(path)
        cache_key = str(path), before, expected_sha, file_sha
        cached = self._asset_cache.get(str(path))
        if cached is not None and cached[0] == cache_key:
            if list(cached[1].shape) != asset.get("shape") or str(cached[1].dtype) != asset.get("dtype") or before[0] != asset.get("bytes_on_disk"):
                raise ValueError("cached NPZ shape/dtype/size differs from receipt")
            self._asset_cache.move_to_end(str(path))
            return cached[1], cached[2]
        if cached is not None:
            self._cache_used_bytes -= cached[1].nbytes
            del self._asset_cache[str(path)]
        if self.verify_hashes and cache_key not in self._pixel_hash_cache:
            if _sha_file(path) != file_sha:
                raise ValueError(f"NPZ file SHA256 mismatch against receipt: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"frames", "metadata"}:
                raise ValueError(f"unsupported NPZ fields: {path}")
            frames = _pixels(archive["frames"], str(path))
            metadata_value = archive["metadata"]
            if metadata_value.ndim != 0:
                raise ValueError(f"metadata must be scalar JSON: {path}")
            metadata = json.loads(str(metadata_value))
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a JSON object: {path}")
        if list(frames.shape) != asset.get("shape") or str(frames.dtype) != asset.get("dtype") or before[0] != asset.get("bytes_on_disk"):
            raise ValueError("NPZ shape/dtype/size differs from materialization receipt")
        if self.verify_hashes:
            if cache_key not in self._pixel_hash_cache:
                actual = hashlib.sha256(frames.tobytes(order="C")).hexdigest()
                if actual != expected_sha:
                    raise ValueError(f"pixel SHA256 mismatch: {path}")
                self._pixel_hash_cache[cache_key] = True
        if before != _signature(path):
            raise ValueError(f"asset changed during load: {path}")
        frames.setflags(write=False)
        if frames.nbytes <= self.cache_bytes:
            while self._asset_cache and self._cache_used_bytes + frames.nbytes > self.cache_bytes:
                _, evicted = self._asset_cache.popitem(last=False)
                self._cache_used_bytes -= evicted[1].nbytes
            self._asset_cache[str(path)] = cache_key, frames, metadata
            self._cache_used_bytes += frames.nbytes
        return frames, metadata

    def clear_cache(self):
        """Release cached uint8 arrays; unchanged-file hash attestations remain cached."""
        self._asset_cache.clear()
        self._cache_used_bytes = 0

    def load(self, index):
        """Load one pair, returning original native uint8 arrays and manifest row."""
        row = self.records[index]
        target_asset, observation_asset = self._receipt(row)
        target, tm = self._read_asset(target_asset)
        observed, om = self._read_asset(observation_asset)
        for meta, role in ((tm, "clean_target"), (om, "observation")):
            if meta.get("role") != role:
                raise ValueError(f"expected {role} role")
            for key in ("source_id", "source_sha256", "target_key"):
                if meta.get(key) != row[key]:
                    raise ValueError(f"NPZ/manifest {key} mismatch")
            spec = meta.get("target_specification", {})
            if spec.get("pixel_target") != "clean_original_recording":
                raise ValueError("target must be the clean RGB recording reference, not D(x)")
            if tuple(spec.get(key) for key in ("frames", "height", "width")) != target.shape[:3]:
                raise ValueError("target specification/array geometry mismatch")
        if om.get("task") != row["task"] or om.get("target_pixel_sha256") != row["target_pixel_sha256"]:
            raise ValueError("observation task/clean target binding mismatch")
        if self.resolve_path(om.get("target_npz", "")) != self.resolve_path(row["target_npz"]):
            raise ValueError("observation points at a different target NPZ")
        if observed.shape[0] != target.shape[0]:
            raise ValueError("observed/target frame counts differ")
        expected_hw = tuple(v // 4 for v in target.shape[1:3]) if row["task"] == "sr" else target.shape[1:3]
        if any(v % 4 for v in target.shape[1:3]):
            raise ValueError("HR target dimensions must be divisible by four")
        if observed.shape[1:3] != expected_hw:
            raise ValueError("native observation geometry does not match task")
        source = tm.get("source_video")
        if not isinstance(source, str) or not source or om.get("source_video") != source:
            raise ValueError("source video provenance missing or inconsistent")
        if self.verify_original_sources:
            source_path = self.resolve_path(source)
            before = _signature(source_path)  # Missing source fails; never silently skipped.
            cache_key = str(source_path), before, row["source_sha256"]
            if cache_key not in self._source_hash_cache:
                if _sha_file(source_path) != row["source_sha256"]:
                    raise ValueError(f"source SHA256 mismatch: {source_path}")
                if before != _signature(source_path):
                    raise ValueError(f"source changed during hashing: {source_path}")
                self._source_hash_cache[cache_key] = True
        return {"observed": observed, "target": target, "record": dict(row)}

    def preflight(self):
        """Validate every record once, filling only the bounded uint8 LRU cache."""
        for index in range(len(self)):
            self.load(index)
        return {"manifest_sha256": self.manifest_sha256, "samples": len(self),
                "sources": len({row["source_id"] for row in self.records}),
                "tasks": {task: len(self.by_task[task]) for task in self.tasks},
                "byte_hashes_verified": self.verify_hashes,
                "provenance": dict(self.provenance), "receipt_sha256": dict(self.receipt_hashes)}

    def sample_batch(self, rng, batch_size, crop=96, k=3,
                     identity_fraction=.125, near_clean_fraction=.125):
        """Task-balanced causal RGB crops, normalized CPU tensors plus audit metadata.

        Counts across present tasks differ by at most one in each batch. Control
        fractions are per-sample probabilities, not rounded batch quotas. Noise
        seeds are drawn from the supplied NumPy Generator and recorded explicitly.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (batch_size, crop, k)):
            raise ValueError("batch_size, crop and k must be positive integers")
        if crop % 4:
            raise ValueError("crop must be divisible by four")
        probabilities = np.array([identity_fraction, near_clean_fraction], dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or probabilities.sum() > 1:
            raise ValueError("control fractions must be nonnegative and sum to at most one")
        # Randomize both the extra-task allocation and the final batch order.
        task_order = rng.permutation(len(self.tasks))
        task_indices = np.resize(task_order, batch_size)
        rng.shuffle(task_indices)
        batch = {key: [] for key in ("observed", "target", "condition", "task", "control")}
        details = []
        for task_index in task_indices:
            task = self.tasks[int(task_index)]
            index = int(rng.choice(self.by_task[task]))
            loaded = self.load(index)
            observed, target = loaded["observed"], loaded["target"]
            if min(target.shape[1:3]) < crop:
                raise ValueError("crop exceeds target dimensions; resizing/padding is not a fallback")
            frame = int(rng.integers(target.shape[0]))
            history = np.maximum(np.arange(frame - k + 1, frame + 1), 0)
            top = int(rng.integers(target.shape[1] - crop + 1))
            left = int(rng.integers(target.shape[2] - crop + 1))
            draw = rng.random()
            control = 1 if draw < identity_fraction else (2 if draw < probabilities.sum() else 0)
            recipe = None
            if control == 1:
                native = target[history]
            elif control == 2:
                seed = int(rng.integers(0, 2**63))
                unique, inverse = np.unique(history, return_inverse=True)
                native = make_near_clean(target[unique], seed)[inverse]
                recipe = {"operator": "additive_gaussian_noise", "sigma_8bit": NEAR_CLEAN_SIGMA_8BIT,
                          "rng": "NumPy Generator PCG64", "seed": seed, "noise_dtype": "float32",
                          "quantization": "rint ties-even, clip [0,255], uint8",
                          "unique_frame_indices": unique.tolist()}
            else:
                native = observed[history]
            guide = prepare_observed(native, target.shape[1:3])
            clean = prepare_observed(target[frame:frame + 1], target.shape[1:3])[0]
            batch["observed"].append(guide[:, :, top:top + crop, left:left + crop])
            batch["target"].append(clean[:, top:top + crop, left:left + crop])
            batch["condition"].append(build_condition(task))
            batch["task"].append(TASKS.index(task))
            batch["control"].append(control)
            details.append({"record_index": index, "source_id": loaded["record"]["source_id"],
                            "target_key": loaded["record"]["target_key"], "frame_index": frame,
                            "history_indices": history.tolist(), "crop_tlhw": [top, left, crop, crop],
                            "control_recipe": recipe,
                            "input_preparation": GUIDE_RECIPE if task == "sr" and control == 0 else "native RGB /255"})
        torch = _torch()
        result = {key: torch.stack(batch[key]) for key in ("observed", "target", "condition")}
        result.update({key: torch.tensor(batch[key], dtype=torch.long, device="cpu") for key in ("task", "control")})
        result["sample_metadata"] = details
        return result

    def full_clip(self, index, k=3):
        """Yield one causal full-resolution frame batch at a time, without float caching.

        Each item has observed [1,K,3,H,W], target [1,3,H,W], condition [1,6],
        task/control [1], frame_index and record. Only this clip's uint8 pair and
        the current K-frame float guide are held; callers should consume lazily.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        loaded = self.load(index)
        observed, target, row = loaded["observed"], loaded["target"], loaded["record"]
        torch = _torch()
        for frame in range(len(target)):
            history = np.maximum(np.arange(frame - k + 1, frame + 1), 0)
            yield {"observed": prepare_observed(observed[history], target.shape[1:3])[None],
                   "target": prepare_observed(target[frame:frame + 1], target.shape[1:3]),
                   "condition": build_condition(row["task"])[None],
                   "task": torch.tensor([TASKS.index(row["task"])], dtype=torch.long),
                   "control": torch.zeros(1, dtype=torch.long), "frame_index": frame,
                   "history_indices": history.tolist(), "record": dict(row)}


def assert_disjoint(train, dev):
    """Fail on shared source IDs OR byte-identical source videos across splits."""
    ids = {row["source_id"] for row in train.records} & {row["source_id"] for row in dev.records}
    hashes = {row["source_sha256"] for row in train.records} & {row["source_sha256"] for row in dev.records}
    if ids or hashes:
        raise ValueError(f"source split overlap: ids={sorted(ids)}, sha256={sorted(hashes)}")
    return {"source_id_overlap": 0, "source_sha256_overlap": 0,
            "train_manifest_sha256": train.manifest_sha256, "dev_manifest_sha256": dev.manifest_sha256}
