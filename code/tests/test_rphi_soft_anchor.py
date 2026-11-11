"""CPU contracts for the isolated post-decode soft-anchor experiment."""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import rphi_soft_anchor as anchors
    from rphi_model import canonical_q4


@unittest.skipIf(torch is None, "CPU torch is required")
class SoftAnchorTests(unittest.TestCase):
    def pair(self, height=16, width=20, dtype=None):
        generator = torch.Generator().manual_seed(73)
        dtype = dtype or torch.float64
        a = .35 + .3 * torch.rand((2, 3, height, width), generator=generator, dtype=dtype)
        b = .35 + .3 * torch.rand((2, 3, height, width), generator=generator, dtype=dtype)
        return a, b

    def test_alpha_zero_is_original_object_without_clamp_or_operator(self):
        a, b = self.pair(dtype=torch.float32)
        a[0, 0, 0, 0], a[0, 0, 0, 1] = -.5, 1.5
        original = a.clone()
        for task in ("sr", "deblur", "denoise", "compression"):
            with mock.patch.object(anchors, "canonical_q4", side_effect=AssertionError("must not run")):
                output, report = anchors.apply_soft_anchor(a, b, task, 0)
            self.assertIs(output, a)
            self.assertTrue(torch.equal(output, original))
            self.assertTrue(report["identity_passthrough"])
            self.assertFalse(report["clip_applied"])
            self.assertIsNone(report["residuals"])

    def test_native_alpha_one_is_full_rgb_anchor_not_low_frequency(self):
        a, b = self.pair(height=7, width=9, dtype=torch.float32)
        for task in ("deblur", "denoise", "compression"):
            with mock.patch.object(anchors, "canonical_q4", side_effect=AssertionError("native cannot use Q4")):
                output, report = anchors.apply_soft_anchor(a, b, task, 1)
            torch.testing.assert_close(output, b, rtol=0, atol=0)
            self.assertEqual(report["constraint_definition"], "full-resolution R_phi RGB fidelity anchor")
            self.assertEqual(report["residuals"]["output"]["rmse"], 0)
            self.assertFalse(report["physical_observation_consistency"])

    def test_native_fractional_blend(self):
        a, b = self.pair()
        output, report = anchors.apply_soft_anchor(a, b, "denoise", .25)
        torch.testing.assert_close(output, a + .25 * (b - a), rtol=0, atol=0)
        self.assertAlmostEqual(report["residuals"]["output"]["rmse"] /
                               report["residuals"]["before"]["rmse"], .75, places=12)

    def test_sr_residual_decreases_with_documented_canonical_operator(self):
        a, b = self.pair()
        output, report = anchors.apply_soft_anchor(a, b, "sr", .4, clip=False)
        residual = report["residuals"]
        self.assertAlmostEqual(residual["after_soft_update_preclip"]["rmse"] /
                               residual["before"]["rmse"], .6, places=10)
        self.assertLess(residual["output"]["rmse"], residual["before"]["rmse"])
        expected = (canonical_q4(output) - canonical_q4(b)).square().mean().sqrt().item()
        self.assertAlmostEqual(residual["output"]["rmse"], expected, places=14)
        self.assertLessEqual(report["operator_check"]["probe_max_abs_error"], 1e-12)
        self.assertFalse(report["strict_constraint_claimed"])
        json.dumps(report)  # The report is suitable for an immutable experiment artifact.

    def test_sr_alpha_one_does_not_replace_all_rgb(self):
        a, b = self.pair()
        output, report = anchors.apply_soft_anchor(a, b, "sr", 1, clip=False)
        torch.testing.assert_close(canonical_q4(output), canonical_q4(b), rtol=0, atol=1e-12)
        self.assertGreater(float((output - b).abs().max()), .01)
        self.assertLess(report["residuals"]["output"]["max_abs"], 1e-12)

    def test_clipping_breaks_attained_sr_anchor_and_is_reported(self):
        y = torch.arange(16)[:, None]
        x = torch.arange(20)[None, :]
        checker = ((x + y) % 2).double()[None, None].repeat(1, 3, 1, 1)
        target = torch.zeros_like(checker)
        output, report = anchors.apply_soft_anchor(checker, target, "sr", 1)
        self.assertGreater(report["clipped_component_fraction"], 0)
        self.assertLess(report["residuals"]["after_soft_update_preclip"]["max_abs"], 1e-12)
        self.assertGreater(report["residuals"]["output"]["rmse"], .05)
        self.assertTrue(report["clipping_increased_residual"])
        self.assertTrue(report["clipping_broke_attained_anchor"])
        self.assertFalse(report["output_within_anchor_tolerance"])
        self.assertGreaterEqual(output.min().item(), 0)
        self.assertLessEqual(output.max().item(), 1)

    def test_dtype_rounding_is_measured_separately(self):
        a, b = self.pair(dtype=torch.float32)
        output, report = anchors.apply_soft_anchor(a, b, "sr", 1, clip=False)
        self.assertEqual(output.dtype, a.dtype)
        self.assertLess(report["residuals"]["after_soft_update_preclip"]["max_abs"], 1e-12)
        expected = (canonical_q4(output.double()) - canonical_q4(b.double())).square().mean().sqrt().item()
        self.assertAlmostEqual(report["residuals"]["output"]["rmse"], expected, places=14)

    def test_operator_mismatch_fails_instead_of_silent_substitution(self):
        a, b = self.pair()
        original = anchors.BicubicConsistency.down
        with mock.patch.object(anchors.BicubicConsistency, "down", lambda self, value: original(self, value) + .01):
            with self.assertRaisesRegex(ValueError, "canonical Q4"):
                anchors.apply_soft_anchor(a, b, "sr", .5)

    def test_invalid_contracts_fail(self):
        a, b = self.pair()
        for alpha in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                anchors.apply_soft_anchor(a, b, "sr", alpha)
        with self.assertRaises(TypeError):
            anchors.apply_soft_anchor(a, b, "sr", True)
        with self.assertRaises(ValueError):
            anchors.apply_soft_anchor(a, b[..., :16], "sr", .5)
        with self.assertRaises(ValueError):
            anchors.apply_soft_anchor(a, b, "not_a_task", .5)
        with self.assertRaises(ValueError):
            anchors.apply_soft_anchor(a[..., :15], b[..., :15], "sr", .5)
        invalid = a.clone()
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            anchors.apply_soft_anchor(invalid, b, "denoise", .5)


if __name__ == "__main__":
    unittest.main()
