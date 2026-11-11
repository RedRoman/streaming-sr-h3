"""CPU tests of an isolated, untrained conditioning candidate.

Run: python -m unittest discover -s code/tests -p 'test_rphi_condition_v2.py' -v
No integration into the running RphiModel or NPU task is performed.
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_condition_v2 import CANDIDATE_STATUS, NativeObservationEncoder
from rphi_model import _ObservationEncoder


class NativeObservationEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_v1_parameter_layout_initialization_and_strict_loading(self):
        torch.manual_seed(41)
        previous = _ObservationEncoder(k=3)
        torch.manual_seed(41)
        candidate = NativeObservationEncoder()
        old, new = previous.state_dict(), candidate.state_dict()
        self.assertEqual(old.keys(), new.keys())
        for key in old:
            self.assertTrue(torch.equal(old[key], new[key]), key)
        candidate.load_state_dict(old, strict=True)
        # Below the old input-pooling threshold, this change is exactly neutral.
        observed = torch.rand(2, 3, 3, 24, 40)
        torch.testing.assert_close(candidate(observed), previous(observed), rtol=0, atol=0)
        broken = dict(old)
        broken.pop("net.0.weight")
        with self.assertRaises(RuntimeError):
            candidate.load_state_dict(broken, strict=True)
        self.assertEqual(CANDIDATE_STATUS, "integrated_opt_in_quality_unverified")

    def test_native_resolution_output_is_bounded_and_sample_independent(self):
        model = NativeObservationEncoder().eval()
        observed = torch.rand(3, 3, 3, 96, 132)
        with torch.no_grad():
            together = model(observed)
            pieces = torch.cat([model(x[None]) for x in observed])
            changed = observed.clone()
            changed[1] = 10 * torch.randn_like(changed[1])
            changed_output = model(changed)
        self.assertEqual(together.shape, (3, 16))
        self.assertTrue(torch.isfinite(together).all())
        self.assertLessEqual(float(together.abs().max()), 1.)
        torch.testing.assert_close(together, pieces, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(together[[0, 2]], changed_output[[0, 2]], rtol=0, atol=0)
        self.assertGreater(float((together[1] - changed_output[1]).abs().max()), 0.)
        self.assertEqual(model.dense_features(observed).shape, (3, 16, 24, 33))

    def test_fixed_stride_local_features_survive_larger_canvas(self):
        model = NativeObservationEncoder().eval()
        # Both canvases exceed 64, so reintroducing observation pre-pooling
        # would break both the dense geometry and this local-feature check.
        patch = torch.rand(1, 3, 3, 80, 96)
        canvas = torch.rand(1, 3, 3, 128, 176)
        top, left = 16, 28  # Preserve the stride-four convolution phase.
        canvas[..., top:top + 80, left:left + 96] = patch
        with torch.no_grad():
            local = model.dense_features(patch)
            expanded = model.dense_features(canvas)
        self.assertEqual(local.shape[-2:], (20, 24))
        self.assertEqual(expanded.shape[-2:], (32, 44))
        # Exclude the 7x7 receptive field's boundary margin. This assertion
        # concerns local dense features, never the final content-dependent code.
        inner = local[..., 2:-2, 2:-2]
        shifted = expanded[..., top // 4 + 2:top // 4 + 18,
                           left // 4 + 2:left // 4 + 22]
        torch.testing.assert_close(inner, shifted, rtol=1e-5, atol=1e-6)

    def test_observation_only_k3_contract_and_finite_gradients(self):
        model = NativeObservationEncoder()
        observed = torch.rand(2, 3, 3, 36, 44, requires_grad=True)
        result = model(observed)
        result.square().mean().backward()
        self.assertIsNotNone(observed.grad)
        self.assertTrue(torch.isfinite(observed.grad).all())
        self.assertGreater(float(observed.grad.abs().sum()), 0.)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for malformed in (torch.rand(1, 4, 3, 8, 8), torch.rand(1, 2, 3, 8, 8),
                          torch.rand(1, 3, 1, 8, 8), torch.rand(0, 3, 3, 8, 8),
                          torch.zeros(1, 3, 3, 8, 8, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                model(malformed)
        with self.assertRaises(TypeError):
            model(observed, target=observed[:, -1])
        with self.assertRaises(TypeError):
            model(observed, torch.zeros(2, 6))
        with self.assertRaises(TypeError):
            model({"observed": observed, "task": "denoise"})
        for k in (1, 4, True):
            with self.assertRaises(ValueError):
                NativeObservationEncoder(k=k)


if __name__ == "__main__":
    unittest.main()
