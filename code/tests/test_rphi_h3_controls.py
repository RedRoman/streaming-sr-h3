"""CPU-only control provenance and fake-generator checks; no H3/NPU imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval_rphi import TASKS, control_seed, sha_file
from export_rphi_guides import array_sha
from rphi_data import make_near_clean
from rphi_h3_controls import (SCHEMA, control_id, control_input, export_controls,
                              inspect_controls, score_control_pair, verify_control_file,
                              write_partial_atomic)


def fixture(root):
    root.mkdir()
    clean = (np.arange(22 * 32 * 32 * 3).reshape(22, 32, 32, 3) % 220 + 16).astype(np.uint8)
    source_sha, target_sha = "a" * 64, array_sha(clean)
    common = {"source_id": "control-dev-source", "source_sha256": source_sha,
              "target_key": "window0", "source_video": str(root / "attested.mp4"),
              "target_specification": {"pixel_target": "clean_original_recording",
                                       "frames": 22, "height": 32, "width": 32}}
    def save(name, pixels, metadata):
        path = root / (name + ".npz")
        np.savez_compressed(path, frames=pixels, metadata=json.dumps(metadata))
        return {"path": str(path), "file_sha256": sha_file(path), "pixel_sha256": array_sha(pixels),
                "shape": list(pixels.shape), "dtype": "uint8", "bytes_on_disk": path.stat().st_size}
    target = save("target", clean, {**common, "role": "clean_target"})
    observations, rows = {}, []
    for task in TASKS:
        degraded = np.zeros((22, 8, 8, 3) if task == "sr" else clean.shape, np.uint8)
        observations[task] = save(task, degraded, {**common, "role": "observation", "task": task,
            "target_pixel_sha256": target_sha, "target_npz": target["path"], "actual_recipe": {}})
        rows.append({"schema": "openh3.materialized_omni_pilot.v1", "status": "materialized_not_vae_encoded",
            "source_id": common["source_id"], "source_sha256": source_sha, "target_key": "window0",
            "split": "dev", "task": task, "target_npz": target["path"],
            "observation_npz": observations[task]["path"], "target_pixel_sha256": target_sha,
            "observation_pixel_sha256": observations[task]["pixel_sha256"], "latent_path": None})
    (root / "complete.json").write_text(json.dumps({"status": "materialized",
        "materializer_version": "omni-pilot-materializer-v1", "source_id": common["source_id"],
        "source_sha256": source_sha, "target_key": "window0", "split": "dev",
        "target": target, "observations": observations}))
    manifest = root / "dev.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, rows, clean


class ControlRecipeTests(unittest.TestCase):
    def test_partial_metrics_atomic_replacement_and_failed_write_preserves_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "partial_metrics.json"
            result = {"complete": False, "per_sample": [{"sample_id": "first", "mae": .1}]}
            write_partial_atomic(path, result)
            self.assertEqual(json.loads(path.read_text()), result)
            result["per_sample"].append({"sample_id": "second", "mae": .2})
            write_partial_atomic(path, result)
            saved = path.read_bytes()
            self.assertEqual(len(json.loads(saved)["per_sample"]), 2)
            with self.assertRaises(ValueError):
                write_partial_atomic(path, {"complete": False, "invalid": float("nan")})
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["partial_metrics.json"])

    def test_exact_eval_recipe_and_task_keyed_draws(self):
        clean = np.full((22, 32, 32, 3), 127, np.uint8)
        record = {"source_id": "a", "target_key": "b", "task": "sr"}
        identity, recipe = control_input(clean, record, "identity")
        self.assertTrue(np.array_equal(identity, clean))
        self.assertIsNot(identity, clean)
        self.assertFalse(recipe["ordinary_observation_used"])
        near, recipe = control_input(clean, record, "near_clean")
        self.assertEqual(recipe["seed"], control_seed(record))
        self.assertEqual(recipe["sigma_8bit"], 1.0)
        self.assertTrue(np.array_equal(near, make_near_clean(clean, seed=control_seed(record), sigma_8bit=1.0)))
        other, _ = control_input(clean, {**record, "task": "deblur"}, "near_clean")
        self.assertFalse(np.array_equal(near, other))
        with self.assertRaisesRegex(ValueError, "explicit"):
            control_id(record, "task")

    def test_existing_output_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                export_controls("absent", "absent", temp)

    def test_input_reconstruction_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            pixels = np.full((22, 32, 32, 3), 100, np.uint8)
            rgb = pixels.astype(np.float32) / np.float32(255)
            arrays = {"input_rgb": rgb, "rgb": rgb, "raw_rgb": rgb}
            path = Path(temp) / "guide.npz"
            np.savez_compressed(path, **arrays)
            row = {"guide_file_sha256": sha_file(path), "arrays": {
                k: {"shape": list(v.shape), "dtype": "float32", "array_sha256": array_sha(v)} for k, v in arrays.items()}}
            self.assertTrue(verify_control_file(path, row, pixels))
            with self.assertRaisesRegex(ValueError, "deterministic"):
                verify_control_file(path, row, pixels + 1)
            path.write_bytes(path.read_bytes() + b"unexpected")
            with self.assertRaisesRegex(ValueError, "file SHA"):
                verify_control_file(path, row, pixels)


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing CPU torch required")
class ControlExportTests(unittest.TestCase):
    def checkpoint(self, path, manifest):
        import torch
        from rphi_model import RphiModel
        torch.manual_seed(8101)
        model = RphiModel(variant="blind", ch=8, blocks=1)
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.001)
        code = Path(__file__).resolve().parents[1]
        torch.save({"schema": "rphi.checkpoint.v1", "step": 3, "model_config": model.model_config,
            "state_dict": model.state_dict(), "training_contract": {"steps": 3,
                "dev_manifest_sha256": sha_file(manifest), "target": "clean_original_recording",
                "task_order": list(TASKS), "train_source_ids": ["train"], "train_source_sha256": ["b" * 64]},
            "implementation_sha256": {"code/" + n: sha_file(code / n) for n in ("rphi_model.py", "rphi_data.py")}}, path)
        return model

    def test_full_export_inspection_and_recipe_tampering(self):
        import torch
        from eval_rphi import infer_selected
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, records, clean = fixture(root / "data")
            checkpoint = root / "model.pt"
            model = self.checkpoint(checkpoint, manifest).eval()
            out = root / "export"
            exported = export_controls(checkpoint, manifest, out, frame_batch=2)
            self.assertEqual(exported["schema"], SCHEMA)
            self.assertEqual(exported["samples"], 8)
            self.assertFalse(exported["ordinary_tasks_included"])
            _, _, samples = inspect_controls(out / "manifest.json", manifest)
            self.assertEqual(len(samples), 8)
            for sample in samples:
                row, record = sample["export"], sample["record"]
                pixels, _ = control_input(clean, record, row["control"])
                with np.load(sample["guide_file"], allow_pickle=False) as data:
                    self.assertEqual(data["rgb"].shape, (22, 32, 32, 3))
                    self.assertTrue(np.array_equal(data["input_rgb"], pixels.astype(np.float32) / np.float32(255)))
                    direct = infer_selected(model, torch.from_numpy(data["input_rgb"].transpose(0, 3, 1, 2).copy()),
                                            None, list(range(22)), "cpu", batch_size=1)
                    np.testing.assert_allclose(data["raw_rgb"], direct["rgb"], atol=1e-6, rtol=1e-6)
            row = exported["per_sample"][1]
            row["control_recipe"]["seed"] += 1
            sidecar = out / row["provenance_json"]
            sidecar.write_text(json.dumps({k: v for k, v in row.items() if k not in ("provenance_json", "provenance_file_sha256")}))
            row["provenance_file_sha256"] = sha_file(sidecar)
            (out / "manifest.json").write_text(json.dumps(exported))
            with self.assertRaisesRegex(ValueError, "recipe/input/history"):
                inspect_controls(out / "manifest.json", manifest)

    def test_fake_h3_fixed_draft_and_no_metric_target_feedback(self):
        import torch
        class FakeH3:
            def __init__(self):
                self.calls = []
            def encode_guide_pixels(self, rgb, hw):
                return torch.full((1, 24, 7, 2, 2), float(rgb.mean()))
            def sample(self, guide, mode, **kw):
                self.calls.append((guide.clone(), kw["warm_start"].clone(), kw["seed"], mode))
                return {"video": guide}, {"steps": 3, "sigmas": [0.909375, 6/7, 2/3, 0], "mode": mode}
            def decode(self, latent):
                return torch.full((22, 32, 32, 3), float(latent.mean()))
        raw = np.full((22, 32, 32, 3), .2, np.float32)
        restored = np.full(raw.shape, .7, np.float32)
        outputs = []
        for reference_value in (0, 255):
            fake = FakeH3()
            scored, decoded = score_control_pair(fake, raw, restored, np.full(raw.shape, reference_value, np.uint8),
                                                 "denoise", require_flow=False)
            self.assertEqual(len(fake.calls), 2)
            self.assertTrue(torch.equal(fake.calls[0][1], fake.calls[1][1]))
            self.assertEqual(fake.calls[0][2:], fake.calls[1][2:])
            self.assertEqual(scored["sampling"]["denoiser_evaluations"], 6)
            self.assertEqual(len(scored["arms"]), 6)
            self.assertEqual(scored["frame_indices"], list(range(22)))
            outputs.append((scored, decoded))
        for key in ("raw", "rphi"):
            self.assertTrue(torch.equal(outputs[0][1][key], outputs[1][1][key]))
        self.assertNotEqual(outputs[0][0]["arms"]["h3_raw"]["quality"]["mae"],
                            outputs[1][0]["arms"]["h3_raw"]["quality"]["mae"])


if __name__ == "__main__":
    unittest.main()
