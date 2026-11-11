"""CPU-only model-contract tests; no H3, assets, or accelerator imports.

Run: python -m unittest discover -s code/tests -p 'test_rphi_model.py' -v
"""
import io
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_model import RphiModel, canonical_q4, uncertainty_nll


class RphiModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(91)
        self.observed = torch.rand(2, 3, 3, 8, 12)
        self.condition = torch.tensor([[1., 0., 0., 0., 2. / 3., 1.],
                                       [0., 1., 0., 0., 0., 1.]])

    def model(self, variant):
        torch.manual_seed(123)
        return RphiModel(variant=variant, ch=8, blocks=1)

    def test_zero_initialization_identity_shapes_and_canonical_anchor(self):
        for variant in ("known", "blind", "blind_confidence"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                result = model(self.observed, self.condition)
                self.assertTrue(torch.equal(result["rgb"], self.observed[:, -1]))
                self.assertEqual(result["rgb"].shape, (2, 3, 8, 12))
                self.assertEqual(result["anchor"].shape, (2, 3, 2, 3))
                self.assertEqual(result["condition_code"].shape, (2, 16))
                expected = F.interpolate(result["rgb"], (2, 3), mode="bicubic",
                                         align_corners=False, antialias=True)
                self.assertTrue(torch.equal(result["anchor"], expected))
                if variant == "blind_confidence":
                    self.assertEqual(result["log_variance"].shape, (2, 1, 8, 12))
                    self.assertTrue(torch.isfinite(result["log_variance"]).all())
                else:
                    self.assertIsNone(result["log_variance"])

    def test_backbone_initialization_is_matched_across_variants(self):
        models = [self.model(v) for v in ("known", "blind", "blind_confidence")]
        reference = models[0].backbone.state_dict()
        for model in models[1:]:
            self.assertEqual(reference.keys(), model.backbone.state_dict().keys())
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, model.backbone.state_dict()[key]), key)
        for key, value in models[1].condition_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, models[2].condition_encoder.state_dict()[key]), key)

    def test_blind_predictions_and_codes_ignore_all_external_condition(self):
        for variant in ("blind", "blind_confidence"):
            model = self.model(variant)
            with torch.no_grad():
                model.backbone.tail.weight.normal_(std=.01)
            expected = model(self.observed)
            for metadata in (self.condition, torch.full((2, 6), float("nan")), object()):
                actual = model(self.observed, metadata)
                for key in ("rgb", "anchor", "condition_code"):
                    self.assertTrue(torch.equal(actual[key], expected[key]), (variant, key))
            changed = model(torch.zeros_like(self.observed))
            self.assertFalse(torch.equal(changed["condition_code"], expected["condition_code"]))

    def test_known_condition_is_required_and_changes_code(self):
        model = self.model("known")
        with self.assertRaises(ValueError):
            model(self.observed)
        with self.assertRaises(ValueError):
            model(self.observed, torch.zeros(2, 7))
        first = model(self.observed, self.condition)["condition_code"]
        second = model(self.observed, self.condition.flip(0))["condition_code"]
        self.assertFalse(torch.equal(first, second))

    def test_fidelity_prediction_is_unclamped_and_gradient_is_finite(self):
        for variant in ("known", "blind", "blind_confidence"):
            model = self.model(variant)
            with torch.no_grad():
                model.backbone.tail.bias.fill_(2.)
            result = model(self.observed, self.condition)
            self.assertGreater(float(result["rgb"].detach().min()), 1.)
            loss = (result["rgb"] - .5).square().mean() + result["anchor"].square().mean()
            loss.backward()
            grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(float(model.backbone.tail.weight.grad.abs().sum()), 0.)
            if model.confidence_head is not None:
                self.assertTrue(all(p.grad is None for p in model.confidence_head.parameters()))

    def test_condition_encoder_can_learn_after_identity_layers_open(self):
        # At exact identity the zero tail intentionally blocks gradients into
        # earlier layers. Once the tail and FiLM learn, the code path must have
        # a real fidelity gradient, including the blind observation encoder.
        for variant in ("known", "blind", "blind_confidence"):
            model = self.model(variant)
            with torch.no_grad():
                model.backbone.tail.weight.normal_(std=.01)
                model.backbone.blocks[0].film.weight.normal_(std=.01)
            prediction = model(self.observed, self.condition)["rgb"]
            prediction.square().mean().backward()
            grads = [p.grad for p in model.condition_encoder.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_uncertainty_gradient_is_isolated_to_confidence_head(self):
        model = self.model("blind_confidence")
        observed = self.observed.clone().requires_grad_()
        target = torch.zeros_like(observed[:, -1], requires_grad=True)
        result = model(observed)
        loss = uncertainty_nll(result, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(observed.grad)
        self.assertIsNone(target.grad)
        for name, parameter in model.named_parameters():
            if not name.startswith("confidence_head."):
                self.assertIsNone(parameter.grad, name)
        grads = [p.grad for p in model.confidence_head.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_gaussian_nll_prefers_variance_matching_actual_error(self):
        # Same image error and difficulty for all candidates: the proper score
        # must prefer its measured MSE to both a smaller and a larger variance.
        target = torch.zeros(1, 3, 4, 4, requires_grad=True)
        rgb = torch.full_like(target, .1, requires_grad=True)
        losses = []
        for variance in (.001, .01, .1):
            log_variance = torch.full((1, 1, 4, 4), float(torch.log(torch.tensor(variance))),
                                      requires_grad=True)
            loss = uncertainty_nll({"rgb": rgb, "log_variance": log_variance}, target)
            losses.append(float(loss.detach()))
            if variance == .01:
                loss.backward()
                self.assertLess(float(log_variance.grad.abs().max()), 1e-6)
                self.assertIsNone(rgb.grad)
                self.assertIsNone(target.grad)
        self.assertLess(losses[1], losses[0])
        self.assertLess(losses[1], losses[2])

    def test_variance_bounds_and_missing_head_contract(self):
        model = self.model("blind_confidence")
        for bias, expected in ((-20., -12.), (20., 0.)):
            with torch.no_grad():
                model.confidence_head[-1].bias.fill_(bias)
            actual = model(self.observed)["log_variance"]
            self.assertTrue(torch.equal(actual, torch.full_like(actual, expected)))
        with self.assertRaises(ValueError):
            uncertainty_nll(self.model("blind")(self.observed), self.observed[:, -1])

    def test_strict_save_reload_and_batch_partition_parity(self):
        for variant in ("known", "blind", "blind_confidence"):
            model = self.model(variant).eval()
            with torch.no_grad():
                model.backbone.tail.weight.normal_(std=.01)
            expected = model(self.observed, self.condition)
            buffer = io.BytesIO()
            torch.save({"model_config": model.model_config, "state": model.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True, map_location="cpu")
            clone = RphiModel(**checkpoint["model_config"]).eval()
            clone.load_state_dict(checkpoint["state"], strict=True)
            actual = clone(self.observed, self.condition)
            for key in expected:
                if expected[key] is not None:
                    self.assertTrue(torch.equal(expected[key], actual[key]), (variant, key))
                    pieces = [clone(self.observed[i:i+1], self.condition[i:i+1])[key] for i in range(2)]
                    torch.testing.assert_close(torch.cat(pieces), actual[key], rtol=1e-5, atol=1e-6)
            broken = dict(checkpoint["state"])
            broken.pop("backbone.tail.bias")
            with self.assertRaises(RuntimeError):
                clone.load_state_dict(broken, strict=True)

    def test_geometry_and_temporal_width_fail_loudly(self):
        with self.assertRaises(ValueError):
            RphiModel(k=1)
        model = self.model("blind")
        for x in (torch.zeros(1, 2, 3, 8, 12), torch.zeros(1, 3, 3, 9, 12),
                  torch.zeros(1, 3, 3, 4, 2)):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(ValueError):
            canonical_q4(torch.zeros(1, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
