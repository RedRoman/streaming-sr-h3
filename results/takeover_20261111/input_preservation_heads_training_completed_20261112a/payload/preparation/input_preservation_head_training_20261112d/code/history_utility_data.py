"""P1 observation-only 96-frame data and immutable schedules (NumPy CPU only).

This is a new loader, not a modification of the four-task Rphi runtime. Receipt
entries are explicit {path, sha256, artifact_root?} bindings. A relocated copy
may supply artifact_root, but its NPZ bytes/metadata must still match the receipt.
Engineering subsets are never promoted to a 32/8-source research dataset.

``batch()['model_inputs']`` contains only observed and availability. Targets,
control IDs, source identities, profiles, severity and recipes are separate.
Teacher weights are separate explicit masks; no targets enter their construction.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROFILES = ("native_static_blur_noise_h264", "native_smooth_blur_noise_h264",
            "native_abrupt_blur_noise_h264")
SHAPE = (96, 480, 832, 3)
CROP = 96
MATERIALIZER_SCHEMA = "openh3.mixed_temporal_materializer.v1"
COMPLETE_STATUS = "engineering_smoke_complete_not_training_or_scientific_acceptance"
SEEDS = {"teacher": 20261110, "student": 20261111}
CONTROL_IDS = {"original": 0, "identity": 1, "near_clean": 2}
NEAR_RECIPE = "p1-near-clean-v1|PCG64-float32|sigma8bit=1|rint-even-clip-uint8|full-frame-before-crop"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_source_registry(plan, train_manifest, dev_manifest):
    """Read the complete role manifests even when only one source is available."""
    result, used = {}, {"source_id": {}, "source_sha256": {}, "source_group_id": {}}
    for split, path, count in (("train", train_manifest, 32), ("dev", dev_manifest, 8)):
        declared = plan["source_roles"]["existing_" + split]
        require(declared["source_count"] == count and len(set(declared["source_ids"])) == count,
                "P1 requires the existing 32/8 source registry")
        require(file_sha(path) == declared["manifest_sha256"], "role manifest SHA changed")
        actual = set()
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["source_id"]
            binding = {"source_id": sid, "source_sha256": _digest(row["source_sha256"]),
                       "source_group_id": row.get("source_group_id", "audioset-youtube:" + sid), "split": split}
            require(row["split"] == split and bool(binding["source_group_id"]), "source role mismatch")
            for key, mapping in used.items():
                value = binding[key]
                require(value not in mapping or mapping[value] == split, "source ID/hash/group crosses splits")
                mapping[value] = split
            require(sid not in result or result[sid] == binding, "conflicting source identity")
            result[sid] = binding
            actual.add(sid)
        require(actual == set(declared["source_ids"]), "registry source membership differs")
    return result


def history_indices(frame):
    require(type(frame) is int and 0 <= frame < 96, "frame must be an absolute window index 0..95")
    return [max(frame - 2, 0), max(frame - 1, 0), frame]


def true_availability(frame):
    history_indices(frame)
    return [int(frame >= 2), int(frame >= 1)]


def near_clean_seed(source_sha, window_key, frame):
    _digest(source_sha); _digest(window_key); history_indices(frame)
    text = f"{NEAR_RECIPE}|{source_sha}|{window_key}|{frame:06d}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def near_clean_frame(clean, source_sha, window_key, frame):
    require(clean.dtype == np.uint8 and clean.ndim == 3 and clean.shape[-1] == 3, "expected RGB uint8 frame")
    rng = np.random.Generator(np.random.PCG64(near_clean_seed(source_sha, window_key, frame)))
    noise = rng.standard_normal(clean.shape, dtype=np.float32)
    return np.clip(np.rint(clean.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def _bilinear_16(coarse):
    """Fixed x16, align_corners=False, float32 separable bilinear interpolation."""
    _, h, w = coarse.shape
    def coordinates(n):
        x = (np.arange(n * 16, dtype=np.float32) + np.float32(.5)) / np.float32(16) - np.float32(.5)
        lo = np.floor(x).astype(np.int64)
        low, high = np.clip(lo, 0, n - 1), np.clip(lo + 1, 0, n - 1)
        weight = np.where(low == high, np.float32(0), x - lo.astype(np.float32))
        return low, high, weight
    yl, yh, yw = coordinates(h); xl, xh, xw = coordinates(w)
    tmp = coarse[:, yl, :] * (1 - yw)[None, :, None] + coarse[:, yh, :] * yw[None, :, None]
    return tmp[:, :, xl] * (1 - xw)[None, None, :] + tmp[:, :, xh] * xw[None, None, :]


def teacher_weights(spec, availability, crop=CROP):
    require(crop == CROP, "P1 teacher crop is fixed at 96")
    a = np.asarray(availability, dtype=np.float32)
    require(a.shape == (2,) and np.isin(a, (0, 1)).all(), "invalid predecessor availability")
    require(set(spec) == {"kind", "seed"} and type(spec["seed"]) is int and spec["seed"] >= 0, "invalid mask recipe")
    fixed = {"all_available": [1, 1], "oldest_missing": [0, 1],
             "recent_missing": [1, 0], "both_missing": [0, 0]}
    kind = spec["kind"]
    if kind in fixed:
        weights = np.broadcast_to(np.asarray(fixed[kind], np.float32)[:, None, None], (2, crop, crop)).copy()
    else:
        rng = np.random.Generator(np.random.PCG64(spec["seed"]))
        if kind == "local_binary":
            coarse = (rng.random((2, 6, 6)) < .5).astype(np.float32)
            weights = np.repeat(np.repeat(coarse, 16, axis=1), 16, axis=2)
        elif kind == "local_soft":
            coarse = rng.uniform(.1, .9, (2, 6, 6)).astype(np.float32)
            weights = _bilinear_16(coarse)
        else:
            raise ValueError("unknown teacher mask")
    return np.ascontiguousarray(weights * a[:, None, None], dtype=np.float32)


def severity_phase(profile, frame):
    history_indices(frame)
    if profile in ("identity", "near_clean"):
        return profile
    require(profile in PROFILES, "undeclared P1 profile")
    if profile == PROFILES[0]: return "static"
    if profile == PROFILES[2]: return "abrupt-pre" if frame < 48 else "abrupt-post"
    return "low" if frame < 24 else "ramp-up" if frame < 48 else "high" if frame < 72 else "ramp-down"


def _validate_receipt(receipt, registry, profiles, operator_sha):
    require(receipt["schema"] == MATERIALIZER_SCHEMA and receipt["status"] == COMPLETE_STATUS,
            "only complete versioned materializer receipts are accepted")
    binding = receipt["source"]
    require(registry.get(binding["source_id"]) == binding, "receipt source differs from full role registry")
    require(receipt["source_unchanged"] is True and receipt["new_split_assignment"] is False,
            "materializer source stability/role attestation missing")
    require(value_sha(receipt["registry"]) == receipt["registry_sha256"], "receipt registry hash mismatch")
    require({r["source_id"]: r for r in receipt["registry"]["sources"]} == registry, "receipt registry differs")
    require(value_sha(receipt["software"]) == receipt["software_sha256"], "software receipt changed")
    _digest(receipt["materializer_sha256"]); _digest(receipt["plan_file_sha256"])
    spec = receipt["target_specification"]
    require(value_sha(spec) == receipt["window_key"] == receipt["target_specification_sha256"], "window hash differs")
    require(all(spec[k] == v for k, v in binding.items()), "target/source identity mismatch")
    require((spec["frames"], spec["height"], spec["width"], spec["channels"], spec["fps"], spec["dtype"], spec["color"])
            == (96, 480, 832, 3, 24, "uint8", "RGB24"), "only complete native 96-frame RGB geometry accepted")
    require(receipt["execution_contract"]["operator_contract_sha256"] == operator_sha, "operator recipe changed")
    require(value_sha(receipt["execution_contract"]) == spec["execution_contract_sha256"], "execution/target binding differs")
    require(receipt["paired_noise_equal"] is True, "paired noise replay not verified")
    require(receipt["decode"]["decoded_target_count"] == 96 and receipt["decode"]["decoded_source_count"] >= 96,
            "target/source decode count is incomplete")
    target = receipt["target"]
    require(target["shape"] == list(SHAPE) and target["dtype"] == "uint8", "target shape/dtype differs")
    for pid, profile in profiles.items():
        obs = receipt["observations"][pid]
        require(obs["task"] == "mixed_native" and obs["scale"] == 1 and obs["artifact"]["shape"] == list(SHAPE)
                and obs["artifact"]["dtype"] == "uint8", "native profile shape/task/scale differs")
        require(obs["parameter_schedule_sha256"] == value_sha(profile["frame_parameters"]), "parameter schedule changed")
        require(obs["replay_count"] == 2 and obs["pixel_replay_equal"] is True and len(obs["replays"]) == 2,
                "two actual replay receipts required")
        for replay in obs["replays"]:
            require(replay["profile_id"] == pid and replay["observation_pixel_sha256"] == obs["artifact"]["pixel_sha256"],
                    "replay/profile pixels differ")
            require(value_sha(replay["noise_frames"]) == replay["noise_schedule_sha256"], "noise receipt differs")
            noise = replay["noise_frames"]
            require(len(noise) == 96 and [r["frame"] for r in noise] == list(range(96)), "noise frame history incomplete")
            for i, row in enumerate(noise):
                text = f"omni-mixed-v1|{binding['source_sha256']}|{receipt['window_key']}|{profile['noise_stream']}|{i:06d}"
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
                require(row["seed_uint64"] == seed and row["unit_noise_shape"] == [480, 832, 3], "noise source/window/seed differs")
                _digest(row["unit_noise_pixel_sha256"])
            require(replay["keyframe_indices"] == [0], "unapproved codec reset in primary profile")
            pts = replay["decoded_pts"]
            require(len(pts) == 96 and all(r["index"] == i and r["pts"] * Fraction(r["time_base"]) == Fraction(i, 24)
                    for i, r in enumerate(pts)), "profile frame count/order/timestamps differ")
        require(obs["replays"][0]["noise_schedule_sha256"] == obs["replays"][1]["noise_schedule_sha256"], "noise replay differs")


class HistoryUtilityDataset:
    def __init__(self, receipts, *, plan, train_manifest, dev_manifest, split,
                 scope="engineering", verify_pixels=True, cache_bytes=512 << 20):
        require(split in ("train", "dev") and scope in ("engineering", "research"), "invalid split/scope")
        self.plan_path = Path(plan); self.plan = _json(plan)
        self.registry = load_source_registry(self.plan, train_manifest, dev_manifest)
        self.split, self.scope, self.verify_pixels = split, scope, verify_pixels
        self.cache_bytes = int(cache_bytes); require(self.cache_bytes >= 0, "negative cache limit")
        self._cache, self._cache_size, self._control_sha = OrderedDict(), 0, {}
        self.profiles = {p["profile_id"]: p for p in self.plan["profiles"] if p["profile_id"] in PROFILES}
        require(set(self.profiles) == set(PROFILES), "missing first-screen profiles")
        self.sources, self.receipt_bindings = {}, []
        for entry in receipts:
            require(set(entry) <= {"path", "sha256", "artifact_root"} and {"path", "sha256"} <= set(entry), "explicit receipt SHA required")
            path = Path(entry["path"])
            require(file_sha(path) == _digest(entry["sha256"]), "receipt file SHA changed")
            receipt = _json(path)
            _validate_receipt(receipt, self.registry, self.profiles, value_sha(self.plan["operator_contract"]))
            sid = receipt["source"]["source_id"]
            require(receipt["source"]["split"] == split and sid not in self.sources, "duplicate/wrong-role source receipt")
            artifacts = {"target": receipt["target"], **{pid: receipt["observations"][pid]["artifact"] for pid in PROFILES}}
            root = Path(entry["artifact_root"]).resolve() if "artifact_root" in entry else None
            for name, artifact in artifacts.items():
                require(Path(artifact["path"]).name == name + ".npz", "unexpected artifact name")
                _digest(artifact["file_sha256"]); _digest(artifact["pixel_sha256"])
            self.sources[sid] = {"receipt": receipt, "root": root, "artifacts": artifacts}
            self.receipt_bindings.append({"source_id": sid, "sha256": entry["sha256"]})
        require(bool(self.sources), "no materialized sources")
        self.source_ids = sorted(self.sources)
        expected = set(self.plan["source_roles"]["existing_" + split]["source_ids"])
        complete_sources = set(self.source_ids) == expected
        continuity = all(s["receipt"].get("scientific_continuity_approved") is True
                         and s["receipt"].get("source_natural_cut_status") == "reviewed_cut_free" for s in self.sources.values())
        self.research_ready = complete_sources and continuity and verify_pixels
        if scope == "research":
            require(self.research_ready, "research needs complete source split, approved cut-free windows and pixel verification")
        self.binding = {"schema": "openh3.history_utility_dataset.v1", "split": split, "scope": scope,
                        "registry_sha256": value_sha(self.registry), "plan_sha256": file_sha(plan),
                        "source_ids": self.source_ids, "receipts": sorted(self.receipt_bindings, key=lambda x: x["source_id"]),
                        "profiles": list(PROFILES), "shape": list(SHAPE), "research_ready": self.research_ready,
                        "research_ready_scope": "this split only; train and dev must separately pass before training",
                        "missing_materialized_source_ids": sorted(expected - set(self.source_ids)),
                        "continuity_approved": continuity,
                        "pixels_verified": verify_pixels, "near_clean_recipe": NEAR_RECIPE,
                        "numpy_version": np.__version__, "model_calls": 0, "training_steps": 0}
        self.dataset_sha256 = value_sha(self.binding)
        if verify_pixels:
            for sid in self.source_ids:
                for name in ("target", *PROFILES): self._pixels(sid, name)

    def _pixels(self, sid, name):
        key = (sid, name)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        source = self.sources[sid]; receipt = source["receipt"]; row = source["artifacts"][name]
        path = source["root"] / (name + ".npz") if source["root"] else Path(row["path"])
        require(path.stat().st_size == row["bytes_on_disk"] and file_sha(path) == row["file_sha256"], "NPZ bytes/SHA changed")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"frames", "metadata"}, "NPZ keys differ")
            pixels = z["frames"]
            metadata = json.loads(str(z["metadata"].item()))
        require(pixels.dtype == np.uint8 and pixels.shape == SHAPE and array_sha(pixels) == row["pixel_sha256"], "pixel hash/shape/dtype differs")
        common = {"schema": MATERIALIZER_SCHEMA, **receipt["source"], "window_key": receipt["window_key"],
                  "target_pixel_sha256": receipt["target"]["pixel_sha256"],
                  **{k: receipt[k] for k in ("plan_file_sha256", "registry_sha256", "materializer_sha256", "software_sha256")}}
        require(all(metadata.get(k) == v for k, v in common.items()), "NPZ source/target/provenance binding differs")
        require(metadata["role"] == ("target" if name == "target" else "observation"), "NPZ role differs")
        if name == "target":
            require(metadata["target_specification"] == receipt["target_specification"], "target specification differs")
        else:
            require(metadata["profile"] == self.profiles[name] and metadata["parameter_schedule_sha256"] == value_sha(self.profiles[name]["frame_parameters"])
                    and metadata["noise_stream"] == self.profiles[name]["noise_stream"], "NPZ profile/parameter recipe differs")
            require(metadata["blind_model_condition"] == "not_supplied_new_loader_required", "unexpected condition payload")
        pixels.setflags(write=False)
        while self._cache and self._cache_size + pixels.nbytes > self.cache_bytes:
            _, old = self._cache.popitem(last=False); self._cache_size -= old.nbytes
        if pixels.nbytes <= self.cache_bytes:
            self._cache[key] = pixels; self._cache_size += pixels.nbytes
        return pixels

    def _control_hash(self, sid, profile):
        receipt = self.sources[sid]["receipt"]
        if profile == "identity": return receipt["target"]["pixel_sha256"]
        if sid not in self._control_sha:
            clean = self._pixels(sid, "target"); h = hashlib.sha256()
            for i, frame in enumerate(clean):
                h.update(near_clean_frame(frame, receipt["source"]["source_sha256"], receipt["window_key"], i).tobytes())
            self._control_sha[sid] = h.hexdigest()
        return self._control_sha[sid]

    def _row(self, sid, profile, frame, crop, external, mask_spec, index):
        require(sid in self.sources and profile in (*PROFILES, "identity", "near_clean"), "unavailable source/profile")
        require(len(crop) == 4 and all(type(v) is int for v in crop) and crop[2:] == [96, 96]
                and crop[0] % 4 == crop[1] % 4 == 0 and 0 <= crop[0] <= 384 and 0 <= crop[1] <= 736, "invalid shared crop")
        require(len(external) == 2 and all(type(v) is int and v in (0, 1) for v in external), "invalid external mask")
        receipt = self.sources[sid]["receipt"]
        control = profile if profile in ("identity", "near_clean") else "original"
        observation_sha = receipt["observations"][profile]["artifact"]["pixel_sha256"] if control == "original" else self._control_hash(sid, profile)
        row = {**receipt["source"], "dataset_sha256": self.dataset_sha256, "window_key": receipt["window_key"],
               "profile_id": profile, "frame_index": frame, "history_indices": history_indices(frame),
               "true_availability": true_availability(frame), "crop_tlhw": crop, "external_mask": external,
               "teacher_mask": mask_spec, "control": control, "severity_phase": severity_phase(profile, frame),
               "target_pixel_sha256": receipt["target"]["pixel_sha256"], "observation_pixel_sha256": observation_sha,
               "control_recipe_and_seed": {"recipe": NEAR_RECIPE,
                   "frame_seeds": [near_clean_seed(receipt["source"]["source_sha256"], receipt["window_key"], i) for i in history_indices(frame)]}
                   if control == "near_clean" else {"recipe": control, "frame_seeds": []}, "row_index": index}
        row["row_sha256"] = value_sha(row)
        return row

    def schedule(self, role="student", steps=1000):
        require(self.verify_pixels, "metadata-only inspection cannot prepare usable schedules")
        require(self.split == "train" and role in SEEDS, "training schedule requires train split and teacher/student role")
        require(type(steps) is int and 1 <= steps <= 1000, "bounded step count required")
        require(self.scope == "engineering" or steps == 1000, "research schedule must use 1000 steps")
        rng = np.random.Generator(np.random.PCG64(SEEDS[role])); streams = {}
        families = [pid for pid in PROFILES for _ in range(4)] + ["identity"] * 2 + ["near_clean"] * 2
        rows = []
        for _ in range(steps):
            for profile in np.asarray(families)[rng.permutation(16)].tolist():
                if not streams.get(profile): streams[profile] = list(np.asarray(self.source_ids)[rng.permutation(len(self.source_ids))])
                sid = str(streams[profile].pop())
                frame = int(rng.integers(96)); crop = [int(rng.integers(97)) * 4, int(rng.integers(185)) * 4, 96, 96]
                seed = int(rng.integers(0, 2**63)); u = float(rng.random())
                if role == "student":
                    external = [1, 1] if u < .7 else [0, 1] if u < .8 else [1, 0] if u < .9 else [0, 0]
                    mask = None
                else:
                    external = [1, 1]
                    kind = "all_available" if u < .5 else "oldest_missing" if u < .6 else "recent_missing" if u < .7 else "both_missing" if u < .8 else "local_binary" if u < .9 else "local_soft"
                    mask = {"kind": kind, "seed": seed}
                rows.append(self._row(sid, profile, frame, crop, external, mask, len(rows)))
        result = {"schema": "openh3.history_utility_schedule.v1", "dataset_sha256": self.dataset_sha256,
                  "role": role, "seed": SEEDS[role], "steps": steps, "batch_size": 16, "scope": self.scope,
                  "research_ready": self.research_ready, "rows": rows}
        result["schedule_sha256"] = value_sha(result)
        return result

    def validate_schedule(self, schedule):
        require(schedule == self.schedule(schedule["role"], schedule["steps"]), "schedule differs from deterministic declared recipe")

    def evaluation_rows(self):
        require(self.split == "dev" and self.verify_pixels, "dev evaluation requires verified dev data")
        return [self._row(sid, pid, t, [192, 368, 96, 96], [1, 1], None, i)
                for i, (sid, pid, t) in enumerate((s, p, t) for s in self.source_ids
                    for p in (*PROFILES, "identity", "near_clean") for t in range(96))]

    def sample(self, row):
        require(self.verify_pixels, "metadata-only inspection cannot yield model inputs")
        expected = self._row(row["source_id"], row["profile_id"], row["frame_index"], row["crop_tlhw"],
                             row["external_mask"], row["teacher_mask"], row["row_index"])
        require(row == expected, "row provenance/crop/history changed")
        sid, profile = row["source_id"], row["profile_id"]
        clean = self._pixels(sid, "target"); top, left, h, w = row["crop_tlhw"]
        if row["control"] == "original":
            observed = self._pixels(sid, profile)[row["history_indices"], top:top+h, left:left+w]
        elif row["control"] == "identity":
            observed = clean[row["history_indices"], top:top+h, left:left+w]
        else:
            frames = {i: near_clean_frame(clean[i], row["source_sha256"], row["window_key"], i)
                      for i in set(row["history_indices"])}
            observed = np.stack([frames[i][top:top+h, left:left+w] for i in row["history_indices"]])
        target = clean[row["frame_index"], top:top+h, left:left+w]
        x = np.ascontiguousarray(observed.transpose(0, 3, 1, 2), dtype=np.float32) / np.float32(255)
        y = np.ascontiguousarray(target.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)
        a = np.asarray(row["true_availability"], np.float32) * np.asarray(row["external_mask"], np.float32)
        result = {"model_inputs": {"observed": x, "availability": a}, "target": y,
                  "control": CONTROL_IDS[row["control"]], "metadata": {**row,
                      "observed_crop_sha256_float32": array_sha(x), "target_crop_sha256_float32": array_sha(y)}}
        if row["teacher_mask"] is not None:
            result["teacher_weights"] = teacher_weights(row["teacher_mask"], a)
        return result

    def batch(self, rows):
        require(bool(rows), "empty batch")
        samples = [self.sample(row) for row in rows]
        require(len({"teacher_weights" in s for s in samples}) == 1, "teacher/student rows mixed")
        result = {"model_inputs": {k: np.stack([s["model_inputs"][k] for s in samples]) for k in ("observed", "availability")},
                  "target": np.stack([s["target"] for s in samples]),
                  "control": np.asarray([s["control"] for s in samples], np.int64), "metadata": [s["metadata"] for s in samples]}
        if "teacher_weights" in samples[0]: result["teacher_weights"] = np.stack([s["teacher_weights"] for s in samples])
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("receipts-index", "plan", "train-manifest", "dev-manifest", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev"), required=True)
    p.add_argument("--scope", choices=("engineering", "research"), default="engineering")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--schedule-role", choices=("teacher", "student"))
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError("output exists; no overwrite")
    index = _json(args.receipts_index)
    require(index["schema"] == "openh3.history_utility_receipt_index.v1", "receipt index schema differs")
    data = HistoryUtilityDataset(index["receipts"], plan=args.plan, train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest, split=args.split, scope=args.scope, verify_pixels=not args.metadata_only)
    result = {"complete": True, "binding": data.binding, "dataset_sha256": data.dataset_sha256,
              "source_sha256": file_sha(__file__), "status": "inspected_no_training"}
    if args.schedule_role: result["schedule"] = data.schedule(args.schedule_role, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f: f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
