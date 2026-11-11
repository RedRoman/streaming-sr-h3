"""CPU-only integrity, exact-admission and actual tiny autograd tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_vae_gradient_repeatability as comparison
from probe_vae_gradient_repeatability import run_backwards


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (1, 24, 7, 30, 52)
        self.input = np.zeros(self.shape, dtype=np.float32)
        # Fixtures exercise integrity without pretending to be actual seed101 data.
        self.hash_patch = patch.object(comparison, "EXPECTED_INPUT_SHA", comparison.array_sha(self.input))
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def fixture(self, name, paired=False, admission=None):
        root = self.root / name
        root.mkdir()
        raw = np.full(self.shape, .125, dtype=np.float32)
        arrays = {"input": self.input, "scaled_raw": raw, "scaled_unscaled": raw / np.float32(1024)}
        if paired:
            arrays["prior_unscaled"] = np.zeros(self.shape, dtype=np.float32)
        mode = "paired_history" if paired else "single"
        backwards = 2 if paired else 1
        frozen = {"sha256": "verified-weight-hash", "parameter_count": 559}
        report = {
            "schema": "openh3.vae_gradient_repeatability_probe.v1", "complete": True,
            "observations_complete": True, "execution_id": name, "mode": mode,
            "actual_completed_forwards": 2, "actual_completed_backwards": backwards,
            "gates": {key: True for key in comparison.REQUIRED_GATES},
            "checkpoint": {"enabled": True, "active": False, "restored": True,
                           "block_count": 36, "wrapped": 36, "use_reentrant": False,
                           "preserve_rng_state": True, "checkpoint_forward_calls": 108,
                           "no_grad_forward_calls": 0, "recompute_calls": 108 * backwards,
                           "output_gradient_guards": 108, "backward_guard_calls": 108 * backwards,
                           "recomputed_block_indices": list(range(36))},
            "frozen": {"before": frozen, "after": copy.deepcopy(frozen), "versions_unchanged": True},
            "runtime_before": {"precision": "fp16", "deterministic": False},
            "runtime_after": {"precision": "fp16", "deterministic": False},
            "common_contract": {"loss_scale": 1024.0, "seed": 101,
                                "input_array_sha256": comparison.array_sha(self.input),
                                "expected_seed101_input_sha256": comparison.EXPECTED_INPUT_SHA},
            "forward": {"allclose": True, "finite": True, "requires_grad": True, "shape_ok": True,
                        "reference_sha256": "reference", "prediction_sha256": "prediction",
                        "target_sha256": "target", "dtype": "torch.float32", "shape": [22, 480, 832, 3]},
            "arrays": {key: comparison.save_array(root, key + ".npy", value) for key, value in arrays.items()},
            "admission": admission,
        }
        comparison.atomic_json(root / "report.json", report)
        return root, report

    def save_report(self, root, report):
        comparison.atomic_json(root / "report.json", report)

    def test_actual_elementwise_error_support_sign_and_signed_zero(self):
        left = np.array([2, -1, 0, 0], np.float32)
        right = np.array([-2, 0, 1, -0.0], np.float32)
        result = comparison.compare_arrays(left, right)
        self.assertEqual(np.abs(left).max(), np.abs(right).max())
        self.assertEqual(result["max_abs_elementwise_difference"], 4)
        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["support"]["left_only"], 1)
        self.assertEqual(result["support"]["right_only"], 1)
        self.assertEqual(result["opposite_sign_on_common_nonzero"], 1)
        self.assertEqual(result["signed_zero_difference"], 1)
        self.assertAlmostEqual(result["relative_l2_to_left"], np.sqrt(18 / 5))

    def test_exact_a1_a2_admit_and_paired_is_bound_to_them(self):
        a1, _ = self.fixture("a1")
        a2, _ = self.fixture("a2")
        result = comparison.compare_runs(a1, a2)
        path = self.root / "admission.json"
        comparison.atomic_json(path, result)
        receipt = comparison.validate_admission(path)
        self.assertTrue(receipt["comparison"]["allow_paired_history"])
        b, report = self.fixture("b", paired=True, admission={"sha256": receipt["sha256"],
                               "a1": result["a1"], "a2": result["a2"]})
        self.assertTrue(comparison.compare_runs(a1, a2, b)["complete"])
        report["admission"]["a1"]["report_sha256"] = "other"
        self.save_report(b, report)
        with self.assertRaisesRegex(ValueError, "different A1/A2"):
            comparison.compare_runs(a1, a2, b)

    def test_saved_array_corruption_and_overwrite_are_rejected(self):
        root, report = self.fixture("a1")
        path = root / report["arrays"]["scaled_raw"]["path"]
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "file hash"):
            comparison.load_probe(root)
        with self.assertRaises(FileExistsError):
            comparison.save_array(root, "input.npy", self.input)

    def test_nonexact_gradient_cannot_admit_even_if_flag_is_forged(self):
        a1, _ = self.fixture("a1")
        a2, report = self.fixture("a2")
        for name, value in (("scaled_raw", .25), ("scaled_unscaled", .25 / 1024)):
            report["arrays"][name] = comparison.save_array(a2, name + "_changed.npy", np.full(self.shape, value, np.float32))
        self.save_report(a2, report)
        result = comparison.compare_runs(a1, a2)
        self.assertFalse(result["allow_paired_history"])
        self.assertGreater(result["a1_a2"]["scaled_raw"]["max_abs_elementwise_difference"], 0)
        result["allow_paired_history"] = True
        path = self.root / "forged.json"
        comparison.atomic_json(path, result)
        with self.assertRaises(ValueError):
            comparison.validate_admission(path)

    def test_contract_runtime_weight_and_forward_mismatches_fail(self):
        a1, _ = self.fixture("a1")
        a2, original = self.fixture("a2")
        for key in ("contract", "runtime", "weights"):
            report = copy.deepcopy(original)
            if key == "contract": report["common_contract"]["seed"] = 102
            if key == "runtime": report["runtime_before"]["precision"] = report["runtime_after"]["precision"] = "bf16"
            if key == "weights": report["frozen"]["before"]["sha256"] = report["frozen"]["after"]["sha256"] = "other"
            self.save_report(a2, report)
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparison.compare_runs(a1, a2)
        report = copy.deepcopy(original)
        report["forward"]["prediction_sha256"] = "different-output"
        self.save_report(a2, report)
        self.assertFalse(comparison.compare_runs(a1, a2)["allow_paired_history"])

    def test_incomplete_checkpoint_bypass_and_empty_gates_rejected(self):
        root, original = self.fixture("a1")
        for field, value in (("recompute_calls", 0), ("restored", False), ("no_grad_forward_calls", 1), ("backward_guard_calls", 0)):
            report = copy.deepcopy(original)
            report["checkpoint"][field] = value
            self.save_report(root, report)
            with self.subTest(field=field), self.assertRaises(ValueError): comparison.load_probe(root)
        original["gates"] = {}
        self.save_report(root, original)
        with self.assertRaisesRegex(ValueError, "producer gate"):
            comparison.load_probe(root)

    def test_unscale_relation_and_distinct_process_receipts_required(self):
        a1, _ = self.fixture("a1")
        a2, report = self.fixture("a2")
        report["arrays"]["scaled_unscaled"] = comparison.save_array(a2, "bad-unscale.npy", np.ones(self.shape, np.float32))
        self.save_report(a2, report)
        with self.assertRaisesRegex(ValueError, "unscaled"):
            comparison.load_probe(a2)
        with self.assertRaisesRegex(ValueError, "distinct"):
            comparison.compare_runs(a1, a1)


class TinyAutogradTests(unittest.TestCase):
    def test_actual_backward_single_and_paired_clear_leaf_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch not installed locally; run this test in existing remote CPU environment")
        values = torch.tensor([.125, -.25, .5], dtype=torch.float32)
        observations = []
        for paired in (False, True):
            latent = values.clone().requires_grad_(True)
            loss = (latent.square() + latent * .25).sum()
            saved = {}
            run_backwards(loss, latent, torch, paired, lambda key, grad: saved.__setitem__(key, grad.clone()))
            self.assertEqual(set(saved), {"prior_unscaled", "scaled_raw"} if paired else {"scaled_raw"})
            self.assertTrue(torch.equal(saved["scaled_raw"], (2 * values + .25) * 1024))
            if paired:
                self.assertTrue(torch.equal(saved["prior_unscaled"], 2 * values + .25))
            observations.append(saved["scaled_raw"])
        self.assertTrue(torch.equal(*observations))


if __name__ == "__main__":
    unittest.main()
