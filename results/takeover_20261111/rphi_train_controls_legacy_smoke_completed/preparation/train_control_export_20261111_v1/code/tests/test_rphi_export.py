"""Tiny CPU guide-export tests; no real checkpoints, H3, or NPU execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_rphi_guides import export_guides, verify_export_file
from eval_rphi import TASKS, causal_indices, sha_file


def fixture(root, target_value=128, future_from=None):
    """Independent clean references and observations, with real dataset receipts."""
    root.mkdir(parents=True)
    source_sha = hashlib.sha256(b"synthetic-development-source").hexdigest()
    target = np.full((22, 8, 12, 3), target_value, dtype=np.uint8)
    observed = (np.arange(target.size).reshape(target.shape) % 210 + 20).astype(np.uint8)
    if future_from is not None:
        observed[future_from:] = 255 - observed[future_from:]
    target_sha = hashlib.sha256(target.tobytes()).hexdigest()
    common = {"source_id": "dev-source", "source_sha256": source_sha, "target_key": "dev-window",
              "source_video": str(root / "attested-source.mp4"),
              "target_specification": {"pixel_target": "clean_original_recording", "frames": 22,
                                       "height": 8, "width": 12}}

    def write(path, pixels, metadata):
        np.savez_compressed(path, frames=pixels, metadata=json.dumps(metadata))
        return {"path": str(path), "file_sha256": sha_file(path),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "shape": list(pixels.shape), "dtype": "uint8", "bytes_on_disk": path.stat().st_size}

    target_asset = write(root / "target.npz", target, {**common, "role": "clean_target"})
    rows, assets, inputs = [], {}, {}
    for task in TASKS:
        inputs[task] = observed[:, ::4, ::4].copy() if task == "sr" else observed.copy()
        assets[task] = write(root / (task + ".npz"), inputs[task], {
            **common, "role": "observation", "task": task, "target_pixel_sha256": target_sha,
            "target_npz": target_asset["path"], "actual_recipe": {"hidden_severity": 999}})
        rows.append({"schema": "openh3.materialized_omni_pilot.v1", "status": "materialized_not_vae_encoded",
                     "source_id": common["source_id"], "source_sha256": source_sha,
                     "target_key": common["target_key"], "split": "dev", "task": task,
                     "target_npz": target_asset["path"], "observation_npz": assets[task]["path"],
                     "target_pixel_sha256": target_sha, "observation_pixel_sha256": assets[task]["pixel_sha256"],
                     "latent_path": None})
    receipt = {"status": "materialized", "materializer_version": "omni-pilot-materializer-v1",
               "source_id": common["source_id"], "source_sha256": source_sha,
               "target_key": common["target_key"], "split": "dev", "target": target_asset,
               "observations": assets}
    (root / "complete.json").write_text(json.dumps(receipt))
    manifest = root / "dev.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, inputs


def checkpoint_for(path, manifest, variant="blind_confidence"):
    import torch
    from rphi_model import RphiModel
    torch.manual_seed(77)
    model = RphiModel(variant=variant, ch=8, blocks=1)
    with torch.no_grad():
        model.backbone.tail.weight.normal_(std=.005)
        model.backbone.tail.bias.fill_(.3)
    torch.save({"schema": "rphi.checkpoint.v1", "model_config": model.model_config,
                "state_dict": model.state_dict(), "step": 1000, "training_contract": {
                    "target": "clean_original_recording", "task_order": list(TASKS),
                    "dev_manifest_sha256": sha_file(manifest), "train_source_ids": ["different-train-source"],
                    "train_source_sha256": ["e" * 64], "steps": 1000}}, path)
    return path


class ExportPathTests(unittest.TestCase):
    def test_existing_output_refused_before_checkpoint_or_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "user-file"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                export_guides("missing.pt", "missing.jsonl", directory)
            self.assertEqual(marker.read_text(), "keep")
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["user-file"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing CPU torch environment")
class ExportModelTests(unittest.TestCase):
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

    def create_export(self, name, target_value=128, future_from=None, variant="blind_confidence", **kwargs):
        manifest, inputs = fixture(self.root / name, target_value=target_value, future_from=future_from)
        checkpoint = checkpoint_for(self.root / (name + ".pt"), manifest, variant)
        out = self.root / (name + "-export")
        result = export_guides(checkpoint, manifest, out, **kwargs)
        return result, out, inputs

    def test_all_22_frames_causal_histories_float_arrays_and_provenance(self):
        import torch
        from rphi_model import RphiModel
        from rphi_data import prepare_observed
        calls = []
        original = RphiModel.forward

        def observe_call(model, observed, condition=None):
            calls.append(observed.detach().cpu().clone())
            return original(model, observed, condition)

        with patch.object(RphiModel, "forward", new=observe_call):
            report, out, inputs = self.create_export("complete", frame_batch=3)
        self.assertTrue(report["complete"])
        self.assertEqual(report["scope"], "full_development_set")
        self.assertEqual(report["condition_mode"], "observations_only")
        self.assertEqual(report["samples"], 4)
        self.assertEqual(json.loads((out / "manifest.json").read_text()), report)
        expected_indices = causal_indices(list(range(22)), 3, 22)
        captured = torch.cat(calls)
        for index, row in enumerate(report["per_sample"]):
            self.assertEqual(row["split"], "dev")
            self.assertEqual(row["frame_indices"], list(range(22)))
            self.assertEqual(row["history_indices"], expected_indices.tolist())
            self.assertEqual(row["geometry"]["frames"], 22)
            guide = prepare_observed(inputs[row["task"]], (8, 12))
            torch.testing.assert_close(captured[index * 22:(index + 1) * 22],
                                       guide[torch.from_numpy(expected_indices)], rtol=0, atol=0)
            self.assertTrue(verify_export_file(out / row["guide_npz"], row))
            self.assertEqual(sha_file(row["observation_npz"]), row["observation_file_sha256"])
            self.assertEqual(sha_file(row["checkpoint"]), row["checkpoint_sha256"])
            self.assertEqual(sha_file(out / row["provenance_json"]), row["provenance_file_sha256"])
            with np.load(out / row["guide_npz"], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"rgb", "raw_rgb", "log_variance"})
                self.assertEqual(archive["rgb"].shape, (22, 8, 12, 3))
                self.assertEqual(archive["log_variance"].shape, (22, 8, 12, 1))
                self.assertEqual(archive["rgb"].dtype, np.float32)
                np.testing.assert_array_equal(archive["rgb"], np.clip(archive["raw_rgb"], 0, 1))
                # Float prediction survives export without an RGB8 round trip.
                quantization_error = np.abs(archive["rgb"] * 255 - np.rint(archive["rgb"] * 255))
                self.assertGreater(float(quantization_error.max()), .001)

    def test_target_pixel_change_cannot_change_any_guide_array(self):
        first, first_out, _ = self.create_export("target-one", target_value=40, variant="known")
        second, second_out, _ = self.create_export("target-two", target_value=210, variant="known")
        self.assertEqual(first["condition_mode"], "known")
        for a, b in zip(first["per_sample"], second["per_sample"]):
            self.assertNotEqual(a["target_pixel_sha256"], b["target_pixel_sha256"])
            self.assertEqual(a["observation_pixel_sha256"], b["observation_pixel_sha256"])
            with np.load(first_out / a["guide_npz"]) as x, np.load(second_out / b["guide_npz"]) as y:
                self.assertEqual(set(x.files), {"rgb", "raw_rgb"})
                for key in x.files:
                    np.testing.assert_array_equal(x[key], y[key])

    def test_future_observations_do_not_change_earlier_exports(self):
        first, first_out, _ = self.create_export("future-one", frame_batch=1)
        second, second_out, _ = self.create_export("future-two", future_from=11, frame_batch=3)
        for a, b in zip(first["per_sample"], second["per_sample"]):
            with np.load(first_out / a["guide_npz"]) as x, np.load(second_out / b["guide_npz"]) as y:
                for key in x.files:
                    np.testing.assert_allclose(x[key][:11], y[key][:11], rtol=1e-5, atol=1e-6)
                self.assertGreater(float(np.max(np.abs(x["raw_rgb"][11:] - y["raw_rgb"][11:]))), .01)

    def test_output_tamper_and_incomplete_run_are_not_accepted(self):
        report, out, _ = self.create_export("integrity")
        row = report["per_sample"][0]
        path = out / row["guide_npz"]
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
            verify_export_file(path, row)
        with self.assertRaises(FileExistsError):
            export_guides(row["checkpoint"], report["dev_manifest"], out)
        failed_out = self.root / "failed-export"
        with patch("export_rphi_guides.infer_selected", side_effect=RuntimeError("injected export failure")):
            with self.assertRaisesRegex(RuntimeError, "injected export failure"):
                export_guides(row["checkpoint"], report["dev_manifest"], failed_out)
        self.assertFalse((failed_out / "manifest.json").exists())
        failure = json.loads((failed_out / "failure.json").read_text())
        self.assertFalse(failure["complete"])
        self.assertEqual(failure["completed_samples"], 0)


if __name__ == "__main__":
    unittest.main()
