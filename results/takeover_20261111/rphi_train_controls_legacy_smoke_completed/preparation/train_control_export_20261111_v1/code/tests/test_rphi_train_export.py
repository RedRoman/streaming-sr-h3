"""Isolated train-role contracts and tiny CPU export tests; no pretrained model."""
import copy
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
import export_rphi_train_guides as exporter
from eval_rphi import TASKS, causal_indices, sha_file
from rphi_data import RphiDataset
from test_rphi_export import fixture


def records(role, source=None, source_hash=None):
    source = source or role + "-source"
    return [{"schema": "openh3.materialized_omni_pilot.v1", "status": "materialized_not_vae_encoded",
             "source_id": source, "source_sha256": source_hash or ("a" if role == "train" else "b") * 64,
             "target_key": role + "-target", "split": role, "task": task,
             "target_npz": role + "/target.npz", "observation_npz": role + "/" + task + ".npz",
             "target_pixel_sha256": "c" * 64, "observation_pixel_sha256": "d" * 64,
             "latent_path": None} for task in TASKS]


def metadata(train=None, dev=None, train_sha="1" * 64, dev_sha="2" * 64):
    train, dev = train or records("train"), dev or records("dev")
    return {"schema": "rphi.checkpoint.v1", "model_config": dict(exporter.MODEL_CONFIG), "step": 3000,
            "training_contract": {"target": "clean_original_recording", "task_order": list(TASKS),
                "steps": 3000, "k": 3, "observation_encoder": "legacy",
                "train_manifest_sha256": train_sha, "dev_manifest_sha256": dev_sha,
                "train_source_ids": sorted({r["source_id"] for r in train}),
                "train_source_sha256": sorted({r["source_sha256"] for r in train}),
                "dev_source_ids": sorted({r["source_id"] for r in dev})}}


def role_fixture(root, role, **kwargs):
    """Reuse only synthetic pixels, rewriting actual receipt/metadata role bindings."""
    manifest, inputs = fixture(root, **kwargs)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    receipt_path = root / "complete.json"
    receipt = json.loads(receipt_path.read_text())
    source_hash = ("a" if role == "train" else "b") * 64
    for row in rows:
        row.update(split=role, source_id=role + "-source", source_sha256=source_hash, target_key=role + "-window")
    receipt.update(split=role, source_id=role + "-source", source_sha256=source_hash, target_key=role + "-window")
    for asset in [receipt["target"], *receipt["observations"].values()]:
        path = Path(asset["path"])
        with np.load(path, allow_pickle=False) as archive:
            pixels = archive["frames"].copy()
            meta = json.loads(str(archive["metadata"]))
        meta.update(source_id=role + "-source", source_sha256=source_hash, target_key=role + "-window")
        np.savez_compressed(path, frames=pixels, metadata=json.dumps(meta))
        asset.update(file_sha256=sha_file(path), bytes_on_disk=path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    result = root / (role + "-materialized.jsonl")
    result.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return result, inputs


class TrainContractTests(unittest.TestCase):
    def validate(self, payload=None, train=None, dev=None):
        return exporter.validate_train_contract(payload or metadata(), "1" * 64,
                                               train or records("train"), "2" * 64, dev or records("dev"))

    def test_full_train_dev_role_and_source_isolation(self):
        proof = self.validate()
        self.assertEqual(proof["train_samples"], 4)
        self.assertEqual(proof["source_sha256_overlap"], 0)
        cases = []
        wrong_role = records("train")
        wrong_role[0]["split"] = "dev"
        cases.append((wrong_role, records("dev"), "explicit train"))
        cases.append((records("train"), records("dev", source="train-source"), "source ID overlap"))
        cases.append((records("train"), records("dev", source_hash="a" * 64), "source SHA256 overlap"))
        cases.append((records("train")[:-1], records("dev"), "all four tasks"))
        for train, dev, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.validate(metadata(train, dev), train, dev)

    def test_checkpoint_final_configuration_hash_and_complete_set_contracts(self):
        modifications = [(("step",), 2999), (("training_contract", "steps"), 1000),
                         (("model_config", "observation_encoder"), "native"),
                         (("training_contract", "train_manifest_sha256"), "e" * 64),
                         (("training_contract", "dev_manifest_sha256"), "f" * 64),
                         (("training_contract", "train_source_ids"), ["not-the-training-source"]),
                         (("training_contract", "train_source_sha256"), ["e" * 64]),
                         (("training_contract", "dev_source_ids"), ["not-the-dev-source"])]
        for keys, value in modifications:
            payload = metadata()
            parent = payload
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                self.validate(payload)

    def test_output_overwrite_and_wrong_checkpoint_refused_before_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "user-data"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                exporter.export_train_guides("missing.pt", "missing-train", "missing-dev", root)
            self.assertEqual(marker.read_text(), "preserve")
            checkpoint = root / "wrong.pt"
            checkpoint.write_bytes(b"not the authorized checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                exporter.inspect_inputs(checkpoint, "missing-train", "missing-dev")

    def test_cache_float32_bytes_array_integrity_and_no_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            raw = np.linspace(-.1, 1.1, 22 * 8 * 12 * 3, dtype=np.float32).reshape(22, 8, 12, 3)
            arrays = {"raw_draft": np.full_like(raw, .5), "rphi_raw_rgb": raw, "rphi_rgb": np.clip(raw, 0, 1)}
            np.savez_compressed(path, **arrays)
            row = {"guide_file_sha256": sha_file(path), "guide_file_bytes": path.stat().st_size,
                   "arrays": {k: {"shape": list(v.shape), "dtype": "float32", "array_sha256": exporter.array_sha(v)} for k, v in arrays.items()}}
            self.assertTrue(exporter.verify_cache(path, row))
            corrupt = copy.deepcopy(row)
            corrupt["arrays"]["raw_draft"]["array_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "array SHA256"):
                exporter.verify_cache(path, corrupt)
            corrupt = copy.deepcopy(row)
            corrupt["arrays"]["target"] = corrupt["arrays"]["raw_draft"]
            with self.assertRaisesRegex(ValueError, "target/latent"):
                exporter.verify_cache(path, corrupt)
            with path.open("ab") as stream:
                stream.write(b"altered")
            with self.assertRaisesRegex(ValueError, "file SHA256"):
                exporter.verify_cache(path, row)

    def test_materialized_numpy_preflight_is_real_and_roles_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest, _ = role_fixture(root / "train", "train")
            dev_manifest, _ = role_fixture(root / "dev", "dev")
            train, dev = RphiDataset(train_manifest, cache_bytes=0), RphiDataset(dev_manifest, cache_bytes=0)
            proof = exporter.validate_train_contract(metadata(train.records, dev.records, train.manifest_sha256, dev.manifest_sha256),
                        train.manifest_sha256, train.records, dev.manifest_sha256, dev.records)
            self.assertEqual(proof["source_id_overlap"], 0)
            for dataset in (train, dev):
                preflight = dataset.preflight()
                self.assertTrue(preflight["byte_hashes_verified"])
                self.assertEqual(preflight["samples"], 4)
                self.assertEqual(preflight["provenance"]["original_source_verification"], "materialization_attestation_only")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires an existing CPU torch environment")
class TinyTrainExportTests(unittest.TestCase):
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

    def make(self, name, target_value=128, future_from=None, frame_batch=3, inspect_only=False, fail=False):
        import torch
        from rphi_model import RphiModel
        train_manifest, inputs = role_fixture(self.root / (name + "-train"), "train", target_value=target_value, future_from=future_from)
        dev_manifest, _ = role_fixture(self.root / (name + "-dev"), "dev")
        train, dev = RphiDataset(train_manifest), RphiDataset(dev_manifest)
        torch.manual_seed(52)
        model = RphiModel(variant="blind", ch=8, blocks=1)
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.005)
            model.backbone.tail.bias.fill_(.1)
        payload = metadata(train.records, dev.records, train.manifest_sha256, dev.manifest_sha256)
        payload.update(model_config=model.model_config, state_dict=model.state_dict(), implementation_sha256=exporter._implementation())
        checkpoint = self.root / (name + ".pt")
        torch.save(payload, checkpoint)
        out = self.root / (name + "-out")
        # Test-local constants authorize only the synthetic fixture. Production
        # CLI has no override for the fixed checkpoint digest/configuration.
        with patch.object(exporter, "CHECKPOINT_SHA256", sha_file(checkpoint)), patch.object(exporter, "MODEL_CONFIG", model.model_config):
            if inspect_only:
                with patch("rphi_model.RphiModel", side_effect=AssertionError("inspect must not construct a model")):
                    return exporter.inspect_inputs(checkpoint, train_manifest, dev_manifest)[0], out, inputs
            if fail:
                with patch.object(exporter, "predict_arrays", side_effect=RuntimeError("injected prediction failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected prediction failure"):
                        exporter.export_train_guides(checkpoint, train_manifest, dev_manifest, out)
                return None, out, inputs
            result = exporter.export_train_guides(checkpoint, train_manifest, dev_manifest, out, frame_batch=frame_batch)
        return result, out, inputs

    def test_inspect_does_not_construct_model_or_create_output(self):
        proof, out, _ = self.make("inspect", inspect_only=True)
        self.assertFalse(out.exists())
        self.assertTrue(proof["inspection_complete"])
        self.assertFalse(proof["rgb_export_complete"])
        self.assertFalse(proof["model_constructed"])
        self.assertEqual(proof["model_forward_calls"], 0)

    def test_full22_raw_input_frozen_target_independent_and_causal(self):
        from rphi_data import prepare_observed
        first, first_out, inputs = self.make("first", target_value=32, frame_batch=1)
        target_changed, target_out, _ = self.make("target-changed", target_value=224, frame_batch=1)
        future_changed, future_out, _ = self.make("future-changed", target_value=32, future_from=11, frame_batch=3)
        self.assertEqual(first["status"], "rgb_guides_complete_not_vae_encoded")
        self.assertEqual(first["role"], "train")
        self.assertFalse(first["vae_encoded"])
        self.assertIsNone(first["latent_path"])
        self.assertEqual(first["samples"], 4)
        self.assertEqual(first["frozen_state_sha256_before"], first["frozen_state_sha256_after"])
        self.assertTrue(first["all_parameters_frozen"])
        self.assertEqual(json.loads((first_out / "manifest.json").read_text()), first)
        for a, b, c in zip(first["per_sample"], target_changed["per_sample"], future_changed["per_sample"]):
            self.assertEqual(a["history_indices"], causal_indices(range(22), 3, 22).tolist())
            self.assertEqual(a["geometry"]["frames"], 22)
            self.assertEqual(a["split"], "train")
            self.assertNotEqual(a["target_pixel_sha256"], b["target_pixel_sha256"])
            self.assertEqual(a["observation_pixel_sha256"], b["observation_pixel_sha256"])
            self.assertTrue(exporter.verify_cache(first_out / a["guide_npz"], a))
            self.assertEqual(sha_file(first_out / a["provenance_json"]), a["provenance_file_sha256"])
            with np.load(first_out / a["guide_npz"]) as x, np.load(target_out / b["guide_npz"]) as y, np.load(future_out / c["guide_npz"]) as z:
                self.assertEqual(set(x.files), exporter.ARRAY_KEYS)
                expected_raw = prepare_observed(inputs[a["task"]], (8, 12)).permute(0, 2, 3, 1).numpy()
                np.testing.assert_array_equal(x["raw_draft"], expected_raw)
                for key in x.files:
                    np.testing.assert_array_equal(x[key], y[key])
                    np.testing.assert_allclose(x[key][:11], z[key][:11], rtol=1e-5, atol=1e-6)
                self.assertGreater(float(np.abs(x["raw_draft"][11:] - z["raw_draft"][11:]).max()), .01)

    def test_failure_never_publishes_complete_manifest(self):
        _, out, _ = self.make("failure", fail=True)
        self.assertFalse((out / "manifest.json").exists())
        self.assertFalse(json.loads((out / "failure.json").read_text())["complete"])
        with self.assertRaises(FileExistsError):
            exporter.export_train_guides("missing", "missing", "missing", out)


if __name__ == "__main__":
    unittest.main()
