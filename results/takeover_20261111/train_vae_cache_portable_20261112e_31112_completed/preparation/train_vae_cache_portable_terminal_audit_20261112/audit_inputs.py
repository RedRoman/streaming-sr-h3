"""Read-only binding to the already accepted ordinary TRAIN RGB guide export."""
import json
from pathlib import Path

import numpy as np

from audit_arrays import array_sha, file_sha, pil_same_geometry_hashes, require, verify_clipped_rphi


SOURCE = "2BdYaGuVAKo"
SOURCE_SHA = "ab1d437859fa6de7a25d2235bd4c144e319749bee0249f9556f1c708b09f9394"
CHECKPOINT_SHA = "61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617"
GUIDE_MANIFEST_SHA = "1c73a26f1cb81157c2445fac44d33acc925256ec77343f50aa7d54f7ab9f0008"
RGB_SHAPE = [22, 480, 832, 3]
LATENT_SHAPE = [1, 24, 7, 30, 52]
GUIDE_KEYS = {"raw_draft", "rphi_raw_rgb", "rphi_rgb"}


def selected_row(manifest):
    require(manifest.get("schema") == "rphi.train_rgb_guide_export.v1"
            and manifest.get("complete") is True and manifest.get("rgb_export_complete") is True,
            "not a completed TRAIN guide export")
    require(manifest["role"] == manifest["split"] == "train"
            and manifest["source_ids"] == [SOURCE], "wrong source or data role")
    require(manifest["controls_included"] == [] and manifest["samples"] == 4,
            "expected ordinary four-task parent cache")
    require(manifest["checkpoint_sha256"] == CHECKPOINT_SHA and manifest["checkpoint_step"] == 3000,
            "wrong frozen legacy restorer")
    require(manifest["condition_mode"] == "observations_only"
            and manifest["model_config"] == {"variant": "blind", "k": 3, "ch": 64,
                "blocks": 8, "condition_dim": 6, "observation_encoder": "legacy"},
            "wrong restorer configuration or condition")
    require(manifest["frozen_state_sha256_before"] == manifest["frozen_state_sha256_after"]
            and manifest["all_parameters_frozen"] is True, "parent restorer freeze failed")
    isolation = manifest["source_isolation"]
    require(isolation["source_id_overlap"] == isolation["source_sha256_overlap"] == 0
            and SOURCE in isolation["train_source_ids"] and SOURCE not in isolation["dev_source_ids"]
            and SOURCE_SHA in isolation["train_source_sha256"] and SOURCE_SHA not in isolation["dev_source_sha256"],
            "source is not isolated TRAIN evidence")
    require(not (set(isolation["train_source_ids"]) & set(isolation["dev_source_ids"]))
            and not (set(isolation["train_source_sha256"]) & set(isolation["dev_source_sha256"])),
            "parent train/dev source overlap")
    rows = manifest["per_sample"]
    require(len(rows) == 4 and {r["task"] for r in rows} == {"sr", "deblur", "denoise", "compression"},
            "parent task set differs")
    row = next(r for r in rows if r["task"] == "compression")
    require(row["source_id"] == SOURCE and row["source_sha256"] == SOURCE_SHA
            and row["split"] == row["role"] == "train" and row["complete"] is True
            and row.get("control") is None,
            "selected row source/role differs")
    require(row["checkpoint_sha256"] == CHECKPOINT_SHA
            and row["target_role"] == "supervision_reference_only_never_model_input",
            "selected row restorer/target role differs")
    require(row["frame_indices"] == list(range(22)) and set(row["arrays"]) == GUIDE_KEYS,
            "unexpected frame selection or guide/target keys")
    for role in GUIDE_KEYS:
        require(row["arrays"][role]["shape"] == RGB_SHAPE
                and row["arrays"][role]["dtype"] == "float32", "guide geometry/dtype differs")
    return row


def audit_guide_inputs(root, manifest_path):
    """Re-read actual observed pixels and guides; target file bytes are only hashed.

    Uses the prior accepted full manifest SHA, not self-selected export metadata.
    Original source-video SHA is a materialization attestation, not a live video
    rehash on this host. No target pixel array or model is loaded.
    """
    root, path = Path(root).resolve(), Path(manifest_path).resolve()
    require(path.is_relative_to(root) and file_sha(path) == GUIDE_MANIFEST_SHA,
            "accepted TRAIN guide manifest SHA/path differs")
    manifest = json.loads(path.read_text()); row = selected_row(manifest)
    guide = (path.parent / row["guide_npz"]).resolve()
    require(guide.is_relative_to(path.parent) and guide.suffix == ".npz"
            and guide.stat().st_size == row["guide_file_bytes"]
            and file_sha(guide) == row["guide_file_sha256"], "guide NPZ path/bytes/SHA mismatch")
    def pinned_input(key, expected):
        actual = Path(row[key]).resolve()
        require(actual.is_relative_to(root) and file_sha(actual) == row[expected], "parent input file SHA differs")
        return actual
    observation_path = pinned_input("observation_npz", "observation_file_sha256")
    target_path = pinned_input("target_npz", "target_file_sha256")
    with np.load(observation_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"frames", "metadata"}, "unexpected observation NPZ fields")
        observation = archive["frames"]
        require(observation.dtype == np.uint8 and list(observation.shape) == RGB_SHAPE
                and array_sha(observation) == row["observation_pixel_sha256"], "observation pixels differ")
    hashes = {}
    with np.load(guide, allow_pickle=False) as archive:
        require(set(archive.files) == GUIDE_KEYS, "guide NPZ contains unknown/target arrays")
        raw = archive["raw_draft"]
        require(raw.dtype == np.float32 and list(raw.shape) == RGB_SHAPE
                and array_sha(raw) == row["arrays"]["raw_draft"]["array_sha256"], "raw guide differs")
        for frame, pixels in zip(raw, observation):
            require(np.array_equal(frame, pixels.astype(np.float32) / np.float32(255)),
                    "raw draft differs from the actual ordinary observation")
        hashes["raw"] = pil_same_geometry_hashes(raw)
        del raw, observation
        rphi = archive["rphi_rgb"]; rphi_raw = archive["rphi_raw_rgb"]
        for role, value in (("rphi_rgb", rphi), ("rphi_raw_rgb", rphi_raw)):
            require(value.dtype == np.float32 and list(value.shape) == RGB_SHAPE
                    and array_sha(value) == row["arrays"][role]["array_sha256"], "Rphi guide differs")
        verify_clipped_rphi(rphi_raw, rphi)
        hashes["rphi"] = pil_same_geometry_hashes(rphi)
    return {"row": row, "guide_manifest": str(path), "guide_manifest_sha256": GUIDE_MANIFEST_SHA,
            "guide_npz": str(guide), "guide_file_sha256": row["guide_file_sha256"],
            "observation_file_sha256": row["observation_file_sha256"],
            "target_file_sha256": row["target_file_sha256"], "target_path": str(target_path),
            "target_pixel_array_loaded": False, "preprocessing": hashes,
            "source_video_verification": "prior materialization attestation, no live source-video rehash",
            "restorer_replayed": False}
