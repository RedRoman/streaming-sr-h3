"""CPU regressions for RGB scoring, controls, calibration, and causal grouping."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_rphi import (TASKS, calibration_statistics, canonical_sha, causal_indices,
                       control_seed, error_attribution, rgb_metrics, sample_id, source_summary,
                       summarize_calibration, temporal_metrics, validate_contract)


class RphiMetricTests(unittest.TestCase):
    def test_known_rgb_error_and_identity_control(self):
        target = np.full((2, 8, 8, 3), .5)
        identity = rgb_metrics(target, target)
        changed = rgb_metrics(target + .1, target)
        self.assertEqual(identity["mse"], 0)
        self.assertEqual(identity["perfect_frames"], 2)
        self.assertAlmostEqual(changed["psnr"], 20)
        self.assertAlmostEqual(changed["mae"], .1)
        self.assertIsNone(changed["lpips"])
        harm = error_attribution(target + .1, target, target)
        self.assertAlmostEqual(harm["mse_change"], .01)
        self.assertEqual(harm["cross_term"], 0)

    def test_corrective_and_harmful_residuals_have_exact_attribution(self):
        gt = np.zeros((1, 4, 4, 3))
        base = gt + .2
        improved = error_attribution(gt + .1, base, gt)
        worsened = error_attribution(gt + .3, base, gt)
        self.assertAlmostEqual(improved["mse_change"], -.03)
        self.assertAlmostEqual(improved["correction_energy"], .01)
        self.assertAlmostEqual(improved["cross_term"], -.04)
        self.assertAlmostEqual(worsened["mse_change"], .05)
        self.assertLess(improved["identity_abs_error"], 1e-12)
        self.assertLess(worsened["identity_abs_error"], 1e-12)

    def test_source_weighting_does_not_reward_many_windows(self):
        def row(source, psnr, mse):
            return {"source_id": source, "model": {"psnr": psnr, "mse": mse, "mae": .1},
                    "baseline": {"psnr": 20, "mse": .01, "mae": .2},
                    "attribution": {"mse_change": mse - .01}}
        rows = [row("a", 30, .001)] * 4 + [row("b", 10, .1)]
        result = source_summary(rows)
        self.assertEqual(result["model"]["psnr"], 20)
        self.assertEqual(result["model_minus_baseline"]["psnr"]["model_better_sources"], 1)
        self.assertEqual(result["damage"]["damaged_samples"], 1)
        self.assertIsNone(result["model"]["lpips"])

    def test_temporal_scores_compare_rgb_error_not_raw_motion(self):
        gt = np.stack([np.zeros((4, 4, 3)), np.ones((4, 4, 3)) * .5])
        stable_bias = gt + .1
        flickering_bias = gt.copy(); flickering_bias[1] += .2
        self.assertAlmostEqual(temporal_metrics(stable_bias, gt, None)["unwarped_rgb_error_change_mae"], 0)
        self.assertAlmostEqual(temporal_metrics(flickering_bias, gt, None)["unwarped_rgb_error_change_mae"], .2)

    def test_nan_and_shape_mismatch_are_rejected(self):
        gt = np.zeros((1, 4, 4, 3))
        with self.assertRaises(ValueError):
            rgb_metrics(gt, gt[0])
        with self.assertRaises(ValueError):
            rgb_metrics(gt + np.nan, gt)


class CalibrationTests(unittest.TestCase):
    def test_known_variance_coverage_and_pooling(self):
        # Within one fixed bin: +/-sigma gives exact MSE=variance.
        gt = np.zeros((1, 4, 4, 3))
        pred = np.full_like(gt, .1)
        lv = np.full((1, 4, 4, 1), np.log(.01))
        stats = calibration_statistics(pred, gt, lv)
        report = summarize_calibration([stats, stats])
        self.assertEqual(report["pixels"], 32)
        self.assertLess(report["variance_calibration_mae"], 1e-15)
        self.assertEqual(report["coverage_1_96sigma"], 1)
        nonempty = [b for b in report["bins"] if b["pixels"]]
        self.assertEqual(len(nonempty), 1)
        self.assertAlmostEqual(nonempty[0]["empirical_mse"], .01)

    def test_overconfidence_is_not_hidden_by_confidence_weighting(self):
        gt = np.zeros((1, 4, 4, 3))
        pred = gt + .1
        small = summarize_calibration([calibration_statistics(pred, gt, np.full((1, 4, 4, 1), np.log(.0001)))])
        matched = summarize_calibration([calibration_statistics(pred, gt, np.full((1, 4, 4, 1), np.log(.01)))])
        self.assertEqual(small["coverage_1_96sigma"], 0)
        self.assertGreater(small["variance_calibration_mae"], matched["variance_calibration_mae"])
        self.assertGreater(small["gaussian_nll_without_constant"], matched["gaussian_nll_without_constant"])

    def test_calibration_uses_raw_error_outside_rgb_bounds(self):
        gt = np.ones((1, 4, 4, 3))
        raw = gt + .2
        report = summarize_calibration([calibration_statistics(raw, gt, np.full((1, 4, 4, 1), np.log(.04)))])
        self.assertAlmostEqual(next(b for b in report["bins"] if b["pixels"])["empirical_mse"], .04)
        self.assertEqual(rgb_metrics(np.clip(raw, 0, 1), gt)["mse"], 0)

    def test_invalid_variance_is_not_silently_clipped(self):
        gt = np.zeros((1, 4, 4, 3))
        with self.assertRaises(ValueError):
            calibration_statistics(gt, gt, gt)
        with self.assertRaises(ValueError):
            calibration_statistics(gt, gt, np.full((1, 4, 4, 1), np.nan))
        self.assertFalse(summarize_calibration([])["available"])


class ContractTests(unittest.TestCase):
    def make_contract(self):
        record = {"sample_id": "dev-sr", "source_id": "dev", "source_sha256": "a" * 64,
                  "target_key": "target1", "target_pixel_sha256": "b" * 64,
                  "observation_pixel_sha256": "c" * 64, "task": "sr", "split": "dev"}
        checkpoint = {"schema": "rphi.checkpoint.v1", "training_contract": {
            "dev_manifest_sha256": "d" * 64, "target": "clean_original_recording", "task_order": list(TASKS),
            "train_source_ids": ["train"], "train_source_sha256": {"train": "e" * 64}}}
        return checkpoint, record

    def test_hash_target_and_source_contracts(self):
        ckpt, record = self.make_contract()
        validate_contract(ckpt, "d" * 64, [record])
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            validate_contract(ckpt, "f" * 64, [record])
        with self.assertRaisesRegex(ValueError, "content overlap"):
            validate_contract(ckpt, "d" * 64, [{**record, "source_sha256": "e" * 64}])
        with self.assertRaisesRegex(ValueError, "identical RGB target"):
            validate_contract(ckpt, "d" * 64, [record, {**record, "sample_id": "dev-denoise", "task": "denoise", "target_pixel_sha256": "f" * 64}])

    def test_control_seed_ignores_hidden_parameters(self):
        _, record = self.make_contract()
        self.assertEqual(control_seed(record), control_seed({**record, "degradation": {"sigma": 999}}))
        self.assertNotEqual(control_seed(record), control_seed({**record, "task": "denoise"}))
        self.assertEqual(canonical_sha({"a": 1, "b": 2}), canonical_sha({"b": 2, "a": 1}))
        without_optional_id = {k: v for k, v in record.items() if k != "sample_id"}
        self.assertEqual(sample_id(record), sample_id(without_optional_id))
        ckpt, _ = self.make_contract()
        validate_contract(ckpt, "d" * 64, [without_optional_id])

    def test_causal_context_is_consecutive_at_stride_four(self):
        np.testing.assert_array_equal(causal_indices([0, 4, 8], 3, 9), [[0, 0, 0], [2, 3, 4], [6, 7, 8]])
        with self.assertRaises(ValueError):
            causal_indices([9], 3, 9)

    def test_existing_output_is_refused_before_any_model_load(self):
        from eval_rphi import main
        with tempfile.TemporaryDirectory() as directory:
            args = ["eval_rphi.py", "--checkpoint", "missing.pt", "--dev-manifest", "missing.jsonl", "--out", directory]
            with patch.object(sys, "argv", args), self.assertRaises(FileExistsError):
                main()


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed; run in existing remote CPU environment")
class TorchInferenceTests(unittest.TestCase):
    def test_actual_model_chunk_equivalence_and_missing_condition(self):
        import torch
        from rphi_model import RphiModel
        from eval_rphi import infer_selected, verify_chunk_equivalence
        torch.manual_seed(11)
        model = RphiModel(variant="known", k=3, ch=8, blocks=1).eval()
        torch.nn.init.normal_(model.backbone.tail.weight, std=.02)
        guide = torch.rand(10, 3, 8, 12)
        condition = torch.tensor([1., 0, 0, 0, 2 / 3, 1])
        result = verify_chunk_equivalence(model, guide, condition, [0, 4, 8], "cpu")
        self.assertTrue(result["passed"])
        output = infer_selected(model, guide, torch.zeros(6), [0, 4, 8], "cpu")
        self.assertEqual(output["rgb"].shape, (3, 8, 12, 3))

    def test_lpips_missing_cache_never_constructs_a_network(self):
        from tempfile import TemporaryDirectory
        from eval_rphi import load_local_lpips
        with TemporaryDirectory() as directory:
            metric, info = load_local_lpips(Path(directory) / "absent.pth")
        self.assertIsNone(metric)
        self.assertFalse(info["available"])

    def test_cli_four_task_control_and_calibration_results(self):
        """Exercise orchestration on tiny RGB tensors with the real model API."""
        import contextlib
        import io
        import torch
        from eval_rphi import main, sha_file
        from rphi_model import RphiModel
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = directory / "dev.jsonl"
            manifest.write_text("fixture; dataset validation is covered separately\n")
            records = [{"source_id": "dev", "source_sha256": "a" * 64, "target_key": "window0",
                        "target_pixel_sha256": "b" * 64, "observation_pixel_sha256": "c" * 64,
                        "task": task, "split": "dev"} for task in TASKS]
            clean = np.full((3, 8, 12, 3), 128, dtype=np.uint8)
            class Fixture:
                provenance = {"original_source_verification": "fixture_only"}
                receipt_hashes = {"fixture": "f" * 64}
                def __init__(self):
                    self.records = records
                def load(self, i):
                    obs = clean[:, ::4, ::4] if records[i]["task"] == "sr" else clean
                    return {"observed": obs, "target": clean, "record": dict(records[i])}
            model = RphiModel(variant="blind_confidence", ch=8, blocks=1)
            checkpoint = directory / "model.pt"
            torch.save({"schema": "rphi.checkpoint.v1", "model_config": model.model_config,
                        "state_dict": model.state_dict(), "step": 0, "training_contract": {
                            "target": "clean_original_recording", "task_order": list(TASKS),
                            "dev_manifest_sha256": sha_file(manifest), "train_source_ids": ["train"],
                            "train_source_sha256": ["e" * 64]}}, checkpoint)
            out = directory / "eval"
            args = ["eval_rphi.py", "--checkpoint", str(checkpoint), "--dev-manifest", str(manifest),
                    "--out", str(out), "--frame-stride", "1", "--skip-flow", "--device", "cpu"]
            with patch.object(sys, "argv", args), patch("rphi_data.RphiDataset", return_value=Fixture()), \
                    patch("eval_rphi.load_local_lpips", return_value=(None, {"available": False, "reason": "test"})), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((out / "metrics.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(set(result["per_task"]), set(TASKS))
            self.assertEqual(len(result["per_sample"]), 4)
            self.assertTrue(result["calibration_per_task"]["sr"]["available"])
            self.assertEqual(result["controls"]["identity"]["sr"]["damage"]["damaged_samples"], 0)
            self.assertTrue(all(r["chunk_equivalence"]["passed"] for r in result["per_sample"]))
            self.assertIsNone(result["per_task"]["sr"]["model"]["lpips"])
            self.assertEqual(len(list((out / "examples").rglob("*.png"))), 12)


if __name__ == "__main__":
    unittest.main()
