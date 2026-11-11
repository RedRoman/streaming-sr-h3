"""Synthetic CPU-only checks; no real checkpoint, source video, H3 or NPU."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_rphi_train_controls as exporter
from eval_rphi import TASKS, canonical_sha, causal_indices, sha_file
from rphi_data import RphiDataset, make_near_clean
from test_rphi_train_export import metadata, records, role_fixture


class TrainControlContractTests(unittest.TestCase):
    def test_fixed_source_sha_role_complete_window(self):
        good = records("train", exporter.SOURCE_ID, exporter.SOURCE_SHA256)
        self.assertEqual(len(exporter.selected_records(good)), 4)
        cases = [good[:-1], good + [good[0]], records("dev", exporter.SOURCE_ID, exporter.SOURCE_SHA256),
                 records("train", exporter.SOURCE_ID, "e" * 64), records("train", "some-other-source")]
        mismatch = copy.deepcopy(good)
        mismatch[0]["target_pixel_sha256"] = "e" * 64
        cases.append(mismatch)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                exporter.selected_records(case)

    def test_reused_full_train_dev_contract_rejects_leak_and_incomplete_sets(self):
        train, dev = records("train"), records("dev")
        payload = metadata(train, dev)
        validate = exporter.ordinary.validate_train_contract
        self.assertEqual(validate(payload, "1" * 64, train, "2" * 64, dev)["source_id_overlap"], 0)
        for bad_train, bad_dev in [(records("dev"), dev), (train[:-1], dev),
                (train, records("dev", "train-source")), (train, records("dev", source_hash="a" * 64))]:
            with self.subTest(), self.assertRaises(ValueError):
                validate(metadata(bad_train, bad_dev), "1" * 64, bad_train, "2" * 64, bad_dev)
        payload["training_contract"]["train_source_ids"].append("omitted-history-source")
        with self.assertRaisesRegex(ValueError, "complete source set"):
            validate(payload, "1" * 64, train, "2" * 64, dev)

    def cache_fixture(self, root):
        clean = np.full((22, 8, 12, 3), 128, np.uint8)
        record = records("train", exporter.SOURCE_ID, exporter.SOURCE_SHA256)[0]
        record["target_pixel_sha256"] = exporter.ordinary.array_sha(clean)
        control = "near_clean"
        pixels, recipe = exporter.controls.control_input(clean, record, control)
        draft = pixels.astype(np.float32) / np.float32(255)
        raw = draft + np.float32(.7)
        arrays = {"raw_draft": draft, "rphi_raw_rgb": raw, "rphi_rgb": np.clip(raw, 0, 1)}
        path = root / "cache.npz"
        np.savez_compressed(path, **arrays)
        row = {k: record[k] for k in ("source_id", "source_sha256", "target_key", "task", "split",
                                     "target_pixel_sha256", "observation_pixel_sha256")}
        row.update(schema=exporter.SCHEMA, complete=True, role="train", control=control,
            record_sha256=canonical_sha(record), parent_sample_id=exporter.sample_id(record),
            sample_id=exporter.controls.control_id(record, control), reference_derived_control=True,
            ordinary_observation_used=False, condition_mode="observations_only",
            control_recipe=recipe, control_recipe_sha256=canonical_sha(recipe),
            control_input_pixel_sha256=exporter.ordinary.array_sha(pixels), frame_indices=list(range(22)),
            history_indices=causal_indices(range(22), 3, 22).tolist(),
            guide_file_sha256=sha_file(path), guide_file_bytes=path.stat().st_size,
            arrays={k: {"shape": list(v.shape), "dtype": "float32", "array_sha256": exporter.ordinary.array_sha(v)}
                    for k, v in arrays.items()})
        return path, row, record, clean

    def test_control_recipe_seed_identity_and_native_geometry_match_dev_controls(self):
        clean = np.full((22, 8, 12, 3), 128, np.uint8)
        record = records("train", exporter.SOURCE_ID, exporter.SOURCE_SHA256)[0]
        identity, identity_recipe = exporter.controls.control_input(clean, record, "identity")
        near, recipe = exporter.controls.control_input(clean, record, "near_clean")
        np.testing.assert_array_equal(identity, clean)
        self.assertIsNot(identity, clean)
        self.assertEqual(identity_recipe["sigma_8bit"], 0)
        self.assertIsNone(identity_recipe["seed"])
        np.testing.assert_array_equal(near, make_near_clean(clean, recipe["seed"], sigma_8bit=1))
        self.assertEqual(near.shape, clean.shape)  # nominal SR is HR, no downsample
        self.assertEqual(recipe["sigma_8bit"], 1)
        self.assertFalse(recipe["ordinary_observation_used"])
        other, _ = exporter.controls.control_input(clean, {**record, "task": "deblur"}, "near_clean")
        self.assertFalse(np.array_equal(other, near))
        for control in ("task", "clean", "nearclean"):
            with self.assertRaises(ValueError):
                exporter.controls.control_input(clean, record, control)

    def test_cache_binds_recipe_record_history_pixels_and_no_target_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row, record, clean = self.cache_fixture(Path(directory))
            self.assertTrue(exporter.verify_control_cache(path, row, record, clean))
            variants = []
            for key, value in (("split", "dev"), ("reference_derived_control", False),
                               ("control", "identity"), ("record_sha256", "f" * 64),
                               ("control_recipe_sha256", "f" * 64), ("control_input_pixel_sha256", "f" * 64)):
                altered = copy.deepcopy(row)
                altered[key] = value
                variants.append(altered)
            altered = copy.deepcopy(row)
            altered["control_recipe"]["seed"] += 1
            altered["control_recipe_sha256"] = canonical_sha(altered["control_recipe"])
            variants.append(altered)
            altered = copy.deepcopy(row)
            altered["history_indices"][0][0] = 1
            variants.append(altered)
            altered = copy.deepcopy(row)
            altered["arrays"]["target"] = copy.deepcopy(altered["arrays"]["raw_draft"])
            variants.append(altered)
            for altered in variants:
                with self.subTest(), self.assertRaises(ValueError):
                    exporter.verify_control_cache(path, altered, record, clean)
            with self.assertRaisesRegex(ValueError, "reference pixel"):
                exporter.verify_control_cache(path, row, record, clean + 1)
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "file SHA256"):
                exporter.verify_control_cache(path, row, record, clean)

    def test_existing_output_invalid_checkpoint_and_bad_arguments_fail_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                exporter.export_train_controls("missing", "missing", "missing", root)
            checkpoint = root / "wrong.pt"
            checkpoint.write_bytes(b"not-the-pinned-checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                exporter.inspect_inputs(checkpoint, "missing", "missing")
            for kwargs in ({"frame_batch": True}, {"device": "cuda:0"}, {"frame_batch": 0}):
                with self.subTest(), self.assertRaises(ValueError):
                    exporter.export_train_controls("missing", "missing", "missing", root / "new", **kwargs)
            self.assertFalse((root / "new").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing CPU torch required")
class TinyTrainControlExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, name, future_from=None, *, inspect_only=False, fail=False, mutate_state=False):
        import torch
        from rphi_model import RphiModel
        train_manifest, _ = role_fixture(self.root / (name + "-train"), "train", future_from=future_from)
        dev_manifest, _ = role_fixture(self.root / (name + "-dev"), "dev")
        train, dev = RphiDataset(train_manifest), RphiDataset(dev_manifest)
        torch.manual_seed(719)
        model = RphiModel(variant="blind", ch=8, blocks=1)
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.005)
        payload = metadata(train.records, dev.records, train.manifest_sha256, dev.manifest_sha256)
        payload.update(model_config=model.model_config, state_dict=model.state_dict(),
                       implementation_sha256=exporter.ordinary._implementation())
        checkpoint = self.root / (name + ".pt")
        torch.save(payload, checkpoint)
        out = self.root / (name + "-out")
        # Only synthetic tests replace production source/config/checkpoint pins.
        with ExitStack() as stack:
            stack.enter_context(patch.object(exporter.ordinary, "CHECKPOINT_SHA256", sha_file(checkpoint)))
            stack.enter_context(patch.object(exporter.ordinary, "MODEL_CONFIG", model.model_config))
            stack.enter_context(patch.object(exporter, "SOURCE_ID", "train-source"))
            stack.enter_context(patch.object(exporter, "SOURCE_SHA256", "a" * 64))
            if inspect_only:
                stack.enter_context(patch("rphi_model.RphiModel", side_effect=AssertionError("no model in inspection")))
                return exporter.inspect_inputs(checkpoint, train_manifest, dev_manifest)[0], out
            if fail:
                stack.enter_context(patch.object(exporter.ordinary, "predict_arrays", side_effect=RuntimeError("injected failure")))
            if mutate_state:
                original = exporter.ordinary.predict_arrays
                def changing(model, *args, **kwargs):
                    result = original(model, *args, **kwargs)
                    with torch.no_grad():
                        next(model.parameters()).add_(.1)
                    return result
                stack.enter_context(patch.object(exporter.ordinary, "predict_arrays", side_effect=changing))
            if fail or mutate_state:
                with self.assertRaisesRegex((RuntimeError, ValueError), "injected failure|frozen Rphi state"):
                    exporter.export_train_controls(checkpoint, train_manifest, dev_manifest, out, frame_batch=3)
                return None, out
            result = exporter.export_train_controls(checkpoint, train_manifest, dev_manifest, out, frame_batch=3)
            for row in result["per_sample"]:
                record = next(r for r in train.records if exporter.sample_id(r) == row["parent_sample_id"])
                clean = train.load(train.records.index(record))["target"]
                self.assertTrue(exporter.verify_control_cache(out / row["guide_npz"], row, record, clean))
        return result, out

    def test_inspection_full_sets_without_forward_or_output(self):
        proof, out = self.make("inspection", inspect_only=True)
        self.assertFalse(out.exists())
        self.assertEqual(proof["model_forward_calls"], 0)
        self.assertFalse(proof["rgb_export_complete"])
        self.assertEqual(proof["planned_control_samples"], 8)
        self.assertEqual(proof["source_isolation"]["train_samples"], 4)
        self.assertEqual(proof["source_isolation"]["dev_samples"], 4)

    def test_real_tiny_export_all_controls_frozen_and_ordinary_observations_not_used(self):
        first, path = self.make("first")
        second, path2 = self.make("different-ordinary-observations", future_from=11)
        self.assertTrue(first["complete"])
        self.assertEqual(first["samples"], 8)
        self.assertEqual(first["model_forward_calls"], 64)
        self.assertEqual(first["frozen_state_sha256_before"], first["frozen_state_sha256_after"])
        self.assertEqual(first["role"], "train")
        self.assertFalse(first["ordinary_tasks_included"])
        self.assertFalse(first["vae_encoded"])
        self.assertEqual(json.loads((path / "manifest.json").read_text()), first)
        identities = []
        for a, b in zip(first["per_sample"], second["per_sample"]):
            self.assertEqual(a["history_indices"], causal_indices(range(22), 3, 22).tolist())
            self.assertNotEqual(a["observation_pixel_sha256"], b["observation_pixel_sha256"])
            self.assertEqual(a["target_pixel_sha256"], b["target_pixel_sha256"])
            self.assertEqual(sha_file(path / a["provenance_json"]), a["provenance_file_sha256"])
            with np.load(path / a["guide_npz"]) as x, np.load(path2 / b["guide_npz"]) as y:
                self.assertEqual(set(x.files), exporter.ordinary.ARRAY_KEYS)
                for key in x.files:
                    np.testing.assert_array_equal(x[key], y[key])
                if a["control"] == "identity":
                    identities.append(x["raw_draft"].copy())
        for value in identities:
            np.testing.assert_array_equal(value, identities[0])

    def test_control_predictions_causal_and_chunk_equivalent_no_condition(self):
        import torch
        from rphi_model import RphiModel
        torch.manual_seed(19)
        model = RphiModel(variant="blind", ch=8, blocks=1).eval()
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.02)
        record = records("train", exporter.SOURCE_ID, exporter.SOURCE_SHA256)[0]
        clean = np.full((22, 8, 12, 3), 128, np.uint8)
        altered = clean.copy()
        altered[11:] = 240
        for control in exporter.controls.CONTROLS:
            pixels, _ = exporter.controls.control_input(clean, record, control)
            future, _ = exporter.controls.control_input(altered, record, control)
            condition_values = []
            original = model.forward
            def capture(observed, condition=None):
                condition_values.append(condition)
                return original(observed, condition)
            with patch.object(model, "forward", side_effect=capture):
                a = exporter.ordinary.predict_arrays(model, pixels, clean.shape, frame_batch=1)
                b = exporter.ordinary.predict_arrays(model, future, clean.shape, frame_batch=3)
                c = exporter.ordinary.predict_arrays(model, pixels, clean.shape, frame_batch=3)
            self.assertTrue(all(v is None for v in condition_values))
            for key in a:
                np.testing.assert_allclose(a[key][:11], b[key][:11], atol=1e-6, rtol=1e-5)
                np.testing.assert_allclose(a[key], c[key], atol=1e-6, rtol=1e-5)
            self.assertGreater(float(np.abs(a["raw_draft"][11:] - b["raw_draft"][11:]).max()), .1)

    def test_failed_prediction_or_frozen_state_mutation_never_publishes_complete(self):
        for name, kwargs in (("failure", {"fail": True}), ("state", {"mutate_state": True})):
            _, out = self.make(name, **kwargs)
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse(json.loads((out / "failure.json").read_text())["complete"])
            with self.assertRaises(FileExistsError):
                exporter.export_train_controls("missing", "missing", "missing", out)


if __name__ == "__main__":
    unittest.main()
