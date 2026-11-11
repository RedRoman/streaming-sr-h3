"""Synthetic fixtures for RGB target semantics, provenance, and causal sampling."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_data import (ORIGINAL_ROOT, RphiDataset, TASKS, assert_disjoint,
                       build_condition, make_near_clean, prepare_observed)

try:
    import torch
except ImportError:
    torch = None


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture(root, name="sample", split="train", remap=False, source_bytes=None):
    root.mkdir(parents=True, exist_ok=True)
    directory = root / split / name
    directory.mkdir(parents=True)
    source = root / (name + ".source")
    source.write_bytes(source_bytes or ("source:" + name).encode())
    source_sha = sha(source.read_bytes())
    target = (np.arange(4 * 16 * 20 * 3).reshape(4, 16, 20, 3) % 221 + 10).astype(np.uint8)
    target_sha = sha(target.tobytes())

    def serialized_path(path):
        return str(ORIGINAL_ROOT / path.relative_to(root)) if remap else str(path)

    spec = {"pixel_target": "clean_original_recording", "frames": 4, "height": 16, "width": 20,
            "clean_reference_caveat": "original_recording_proxy_not_pristine_truth"}
    common = {"source_id": name, "source_sha256": source_sha, "target_key": name,
              "source_video": serialized_path(source), "target_specification": spec}

    def write(path, frames, metadata):
        np.savez_compressed(path, frames=frames, metadata=json.dumps(metadata, sort_keys=True))
        return {"path": serialized_path(path), "file_sha256": sha(path.read_bytes()),
                "pixel_sha256": sha(frames.tobytes()), "shape": list(frames.shape),
                "dtype": str(frames.dtype), "bytes_on_disk": path.stat().st_size}

    target_asset = write(directory / "target.npz", target, {**common, "role": "clean_target"})
    observations, rows = {}, []
    for task in TASKS:
        if task == "sr":
            observed = target[:, ::4, ::4].copy()
        elif task == "deblur":
            observed = (target // 2).copy()
        else:
            observed = (target + 3).copy()
        meta = {**common, "role": "observation", "task": task, "target_pixel_sha256": target_sha,
                "target_npz": target_asset["path"], "actual_recipe": {"hidden_sigma": 99, "oracle_severity": 0.98}}
        observations[task] = write(directory / (task + ".npz"), observed, meta)
        rows.append({"schema": "openh3.materialized_omni_pilot.v1", "status": "materialized_not_vae_encoded",
                     "source_id": name, "source_sha256": source_sha, "target_key": name,
                     "split": split, "task": task, "target_npz": target_asset["path"],
                     "observation_npz": observations[task]["path"], "target_pixel_sha256": target_sha,
                     "observation_pixel_sha256": observations[task]["pixel_sha256"], "latent_path": None})
    receipt = {"status": "materialized", "materializer_version": "omni-pilot-materializer-v1",
               "source_id": name, "source_sha256": source_sha, "target_key": name, "split": split,
               "target": target_asset, "observations": observations}
    (directory / "complete.json").write_text(json.dumps(receipt))
    manifest = root / (name + ".jsonl")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, target, rows, directory, source


def edit_asset(path, transform):
    with np.load(path, allow_pickle=False) as archive:
        pixels, meta = archive["frames"], json.loads(str(archive["metadata"]))
    pixels, meta = transform(pixels, meta)
    np.savez_compressed(path, frames=pixels, metadata=json.dumps(meta))
    receipt_path = path.parent / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    asset = receipt["target"] if path.name == "target.npz" else receipt["observations"][path.stem]
    asset.update(file_sha256=sha(path.read_bytes()), bytes_on_disk=path.stat().st_size,
                 shape=list(pixels.shape), dtype=str(pixels.dtype))
    receipt_path.write_text(json.dumps(receipt))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.target, self.rows, self.directory, self.source = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_native_geometry_shared_rgb_target_and_attestation(self):
        dataset = RphiDataset(self.manifest)
        self.source.unlink()  # Snapshot loading explicitly does not claim live-source verification.
        summary = dataset.preflight()
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["provenance"]["original_source_verification"], "materialization_attestation_only")
        for index, task in enumerate(TASKS):
            item = dataset.load(index)
            self.assertEqual(item["observed"].shape, (4, 4, 5, 3) if task == "sr" else self.target.shape)
            np.testing.assert_array_equal(item["target"], self.target)
            self.assertEqual(item["target"].dtype, np.uint8)
            self.assertFalse(item["target"].flags.writeable)
        self.assertFalse(np.array_equal(dataset.load(1)["observed"], dataset.load(1)["target"]))

    def test_root_override_and_live_source_hash(self):
        manifest, _, _, _, source = fixture(self.root / "relocated", name="copy", remap=True)
        dataset = RphiDataset(manifest, root_override=manifest.parent, verify_original_sources=True)
        dataset.preflight()
        source.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
            dataset.load(0)

    def test_source_missing_fails_only_when_explicit_live_verification(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            RphiDataset(self.manifest, verify_original_sources=True).load(0)

    def test_receipt_file_hash_tamper_fails_even_when_pixels_unchanged(self):
        dataset = RphiDataset(self.manifest)
        dataset.load(1)  # Populate caches before tampering.
        path = self.directory / "deblur.npz"
        with np.load(path, allow_pickle=False) as archive:
            frames, meta = archive["frames"], json.loads(str(archive["metadata"]))
        meta["unexpected"] = "changed metadata"
        np.savez_compressed(path, frames=frames, metadata=json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
            dataset.load(1)

    def test_pixel_hash_tamper_fails_even_with_updated_file_receipt(self):
        path = self.directory / "deblur.npz"
        edit_asset(path, lambda frames, meta: (frames + 1, meta))
        with self.assertRaisesRegex(ValueError, "pixel SHA256 mismatch"):
            RphiDataset(self.manifest).load(1)

    def test_semantic_target_and_source_binding_fail_closed(self):
        path = self.directory / "target.npz"
        def corrupt(frames, meta):
            meta["target_specification"]["pixel_target"] = "blurred_D_x"
            return frames, meta
        edit_asset(path, corrupt)
        with self.assertRaisesRegex(ValueError, "clean RGB recording reference"):
            RphiDataset(self.manifest).load(0)

    def test_native_task_cannot_receive_lr_geometry(self):
        path = self.directory / "deblur.npz"
        edit_asset(path, lambda frames, meta: (frames[:, ::4, ::4], meta))
        with self.assertRaisesRegex(ValueError, "native observation geometry"):
            RphiDataset(self.manifest, verify_hashes=False).load(1)

    def test_uint8_lru_shared_target_cache_and_disable(self):
        dataset = RphiDataset(self.manifest)
        first = dataset.load(0)
        with mock.patch("rphi_data.np.load", wraps=np.load) as reader:
            dataset.load(0)
            second = dataset.load(1)
            self.assertEqual(reader.call_count, 1)  # New observation only; shared target stays cached.
        self.assertIs(first["target"], second["target"])
        self.assertTrue(all(value[1].dtype == np.uint8 for value in dataset._asset_cache.values()))
        small = RphiDataset(self.manifest, cache_bytes=self.target.nbytes)
        small.preflight()
        self.assertLessEqual(small._cache_used_bytes, small.cache_bytes)
        none = RphiDataset(self.manifest, cache_bytes=0)
        none.load(0)
        self.assertEqual(none._cache_used_bytes, 0)

    def test_disjoint_checks_source_content_even_under_new_id(self):
        train = RphiDataset(self.manifest)
        manifest, _, _, _, _ = fixture(self.root / "dev", name="another_id", split="dev",
                                       source_bytes=self.source.read_bytes())
        with self.assertRaisesRegex(ValueError, "source split overlap"):
            assert_disjoint(train, RphiDataset(manifest))
        clean_manifest, _, _, _, _ = fixture(self.root / "dev2", name="different", split="dev")
        self.assertEqual(assert_disjoint(train, RphiDataset(clean_manifest))["source_sha256_overlap"], 0)

    def test_manifest_rejects_cross_task_target_rebinding(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["target_pixel_sha256"] = "0" * 64
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "same source/clean target"):
            RphiDataset(self.manifest)


@unittest.skipIf(torch is None, "CPU torch is required for tensor contract tests")
class TensorTests(unittest.TestCase):
    setUp = DatasetTests.setUp
    tearDown = DatasetTests.tearDown

    def test_task_balance_no_oracle_and_aligned_crops(self):
        dataset = RphiDataset(self.manifest)
        result = dataset.sample_batch(np.random.default_rng(4), 13, crop=8, k=3,
                                      identity_fraction=0, near_clean_fraction=0)
        counts = np.bincount(result["task"].numpy(), minlength=4)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(tuple(result["observed"].shape), (13, 3, 3, 8, 8))
        self.assertEqual(tuple(result["target"].shape), (13, 3, 8, 8))
        self.assertEqual(tuple(result["condition"].shape), (13, 6))
        self.assertEqual(result["task"].dtype, torch.long)
        self.assertEqual(result["observed"].device.type, "cpu")
        for i, detail in enumerate(result["sample_metadata"]):
            item = dataset.load(detail["record_index"])
            top, left, height, width = detail["crop_tlhw"]
            expected = prepare_observed(item["observed"][detail["history_indices"]], self.target.shape[1:3])
            expected = expected[:, :, top:top + height, left:left + width]
            torch.testing.assert_close(result["observed"][i], expected, rtol=0, atol=0)
            clean = torch.from_numpy(self.target[detail["frame_index"], top:top + height, left:left + width].copy()).permute(2, 0, 1).float() / 255
            torch.testing.assert_close(result["target"][i], clean, rtol=0, atol=0)
            torch.testing.assert_close(result["condition"][i], build_condition(item["record"]["task"]), rtol=0, atol=0)
        self.assertEqual(build_condition("sr").tolist(), [1.0, 0.0, 0.0, 0.0, np.float32(2 / 3), 1.0])
        self.assertEqual(build_condition("deblur").tolist(), [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(build_condition("sr", known=False).tolist(), [0.0] * 6)

    def test_identity_is_exact_and_retains_nominal_task_condition(self):
        dataset = RphiDataset(self.manifest)
        batch = dataset.sample_batch(np.random.default_rng(10), 8, crop=8, k=3,
                                     identity_fraction=1, near_clean_fraction=0)
        torch.testing.assert_close(batch["observed"][:, -1], batch["target"], rtol=0, atol=0)
        self.assertTrue((batch["control"] == 1).all())
        for i in range(8):
            torch.testing.assert_close(batch["condition"][i], build_condition(TASKS[batch["task"][i].item()]))

    def test_near_clean_reproducible_recipe_and_repeated_history(self):
        dataset = RphiDataset(self.manifest)
        a = dataset.sample_batch(np.random.default_rng(22), 8, crop=8, k=7,
                                 identity_fraction=0, near_clean_fraction=1)
        b = dataset.sample_batch(np.random.default_rng(22), 8, crop=8, k=7,
                                 identity_fraction=0, near_clean_fraction=1)
        torch.testing.assert_close(a["observed"], b["observed"], rtol=0, atol=0)
        self.assertEqual(a["sample_metadata"], b["sample_metadata"])
        self.assertTrue((a["control"] == 2).all())
        for i, detail in enumerate(a["sample_metadata"]):
            recipe = detail["control_recipe"]
            noisy = make_near_clean(self.target[recipe["unique_frame_indices"]], recipe["seed"])
            positions = {frame: j for j, frame in enumerate(recipe["unique_frame_indices"])}
            repeated = noisy[[positions[j] for j in detail["history_indices"]]]
            top, left, height, width = detail["crop_tlhw"]
            expected = prepare_observed(repeated, self.target.shape[1:3])[:, :, top:top + height, left:left + width]
            torch.testing.assert_close(a["observed"][i], expected, rtol=0, atol=0)
            torch.testing.assert_close(a["observed"][i, 0], a["observed"][i, 1], rtol=0, atol=0)

    def test_full_clip_is_causal_lazy_and_native_targets_preserved(self):
        dataset = RphiDataset(self.manifest)
        frames = dataset.full_clip(0, k=3)
        self.assertIs(iter(frames), frames)
        first = next(frames)
        self.assertEqual(first["history_indices"], [0, 0, 0])
        torch.testing.assert_close(first["observed"][:, 0], first["observed"][:, 2], rtol=0, atol=0)
        second = next(frames)
        self.assertEqual(second["history_indices"], [0, 0, 1])
        self.assertEqual(tuple(second["target"].shape), (1, 3, 16, 20))
        self.assertEqual(len(list(frames)), 2)

    def test_crop_and_fraction_fail_without_silent_resize(self):
        dataset = RphiDataset(self.manifest)
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            dataset.sample_batch(np.random.default_rng(1), 1, crop=7)
        with self.assertRaisesRegex(ValueError, "crop exceeds"):
            dataset.sample_batch(np.random.default_rng(1), 1, crop=32)
        with self.assertRaisesRegex(ValueError, "sum to at most one"):
            dataset.sample_batch(np.random.default_rng(1), 1, crop=8, identity_fraction=.9, near_clean_fraction=.9)


if __name__ == "__main__":
    unittest.main()
