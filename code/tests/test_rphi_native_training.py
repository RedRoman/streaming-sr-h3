"""CPU regressions for optional native conditioning and recorded training setup."""
import copy
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_condition_v2 import NativeObservationEncoder
from rphi_model import RphiModel, _ObservationEncoder, uncertainty_nll
from train_rphi import (ADAMW_OPTIONS, build_contract, configure_determinism,
                        determinism_snapshot, learning_rate)


class NativeModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self, variant='blind', encoder='legacy'):
        torch.manual_seed(34)
        return RphiModel(variant=variant, ch=8, blocks=1, observation_encoder=encoder)

    def test_old_checkpoint_config_defaults_to_exact_legacy_path(self):
        reference = self.model()
        old_config = {k: v for k, v in reference.model_config.items() if k != 'observation_encoder'}
        restored = RphiModel(**old_config)
        restored.load_state_dict(reference.state_dict(), strict=True)
        self.assertIsInstance(restored.condition_encoder, _ObservationEncoder)
        self.assertEqual(restored.model_config['observation_encoder'], 'legacy')
        observed = torch.rand(2, 3, 3, 80, 96)
        for key in ('rgb', 'anchor', 'condition_code'):
            torch.testing.assert_close(reference(observed)[key], restored(observed)[key], rtol=0, atol=0)

    def test_shared_initialization_matches_both_encoders_and_all_variants(self):
        models = [self.model(variant, encoder) for encoder in ('legacy', 'native')
                  for variant in ('known', 'blind', 'blind_confidence')]
        reference = models[0].backbone.state_dict()
        for model in models[1:]:
            for name, value in reference.items():
                self.assertTrue(torch.equal(value, model.backbone.state_dict()[name]), name)
        blind = [model for model in models if model.variant != 'known']
        reference = blind[0].condition_encoder.state_dict()
        for model in blind[1:]:
            self.assertEqual(reference.keys(), model.condition_encoder.state_dict().keys())
            for name, value in reference.items():
                self.assertTrue(torch.equal(value, model.condition_encoder.state_dict()[name]), name)

    def test_known_encoder_selection_does_not_change_weights_or_outputs(self):
        legacy, native = self.model('known'), self.model('known', 'native')
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, native.state_dict()[name]), name)
        observed, condition = torch.rand(2, 3, 3, 12, 16), torch.rand(2, 6)
        for key in ('rgb', 'anchor', 'condition_code'):
            torch.testing.assert_close(legacy(observed, condition)[key], native(observed, condition)[key],
                                       rtol=0, atol=0)

    def test_native_geometry_identity_metadata_isolation_and_strict_reload(self):
        model = self.model('blind_confidence', 'native')
        self.assertIsInstance(model.condition_encoder, NativeObservationEncoder)
        observed = torch.rand(2, 3, 3, 80, 96)
        expected = model(observed)
        self.assertTrue(torch.equal(expected['rgb'], observed[:, -1]))
        self.assertEqual(expected['anchor'].shape, (2, 3, 20, 24))
        self.assertEqual(model.condition_encoder.dense_features(observed).shape, (2, 16, 20, 24))
        ignored = model(observed, object())
        for key, value in expected.items():
            torch.testing.assert_close(value, ignored[key], rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save({'model_config': model.model_config, 'state': model.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        restored = RphiModel(**saved['model_config'])
        restored.load_state_dict(saved['state'], strict=True)
        self.assertIsInstance(restored.condition_encoder, NativeObservationEncoder)
        actual = restored(observed)
        for key, value in expected.items():
            torch.testing.assert_close(value, actual[key], rtol=0, atol=0)
        for bad in (torch.rand(1, 4, 3, 8, 8), torch.rand(1, 3, 3, 9, 12)):
            with self.assertRaises(ValueError):
                model(bad)
        with self.assertRaises(ValueError):
            RphiModel(observation_encoder='unknown')

    def test_native_fidelity_gradient_and_confidence_isolation(self):
        model = self.model('blind_confidence', 'native')
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.01)
            model.backbone.blocks[0].film.weight.normal_(std=.01)
        observed = torch.rand(2, 3, 3, 12, 16, requires_grad=True)
        target = torch.zeros_like(observed[:, -1], requires_grad=True)
        outputs = model(observed)
        outputs['rgb'].square().mean().backward()
        grads = [p.grad for p in model.condition_encoder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        model.zero_grad(set_to_none=True)
        observed.grad = None
        outputs = model(observed)
        uncertainty_nll(outputs, target).backward()
        self.assertIsNone(observed.grad)
        self.assertIsNone(target.grad)
        for name, parameter in model.named_parameters():
            if not name.startswith('confidence_head.'):
                self.assertIsNone(parameter.grad, name)


class TrainingContractTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(k=3, crop=96, seed=20261111, steps=3000, batch=16,
                               observation_encoder='native', determinism='inherit', lr=1e-4, warmup=50,
                               identity_fraction=.125, near_clean_fraction=.125, edge_weight=.1,
                               guard_weight=.5)

    def contract(self, args):
        train = SimpleNamespace(manifest_sha256='a' * 64,
                                records=[{'source_id': 'train', 'source_sha256': 'b' * 64}])
        dev = SimpleNamespace(manifest_sha256='c' * 64, records=[{'source_id': 'dev'}])
        return build_contract(args, train, dev)

    def test_contract_records_optimizer_schedule_guard_and_encoder(self):
        args = self.args()
        contract = self.contract(args)
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=args.lr, **ADAMW_OPTIONS)
        for key, value in ADAMW_OPTIONS.items():
            self.assertEqual(optimizer.param_groups[0][key], value)
            expected = list(value) if isinstance(value, tuple) else value
            self.assertEqual(contract['optimizer'][key], expected)
        self.assertEqual(contract['optimizer']['base_lr'], 1e-4)
        self.assertEqual(contract['lr_schedule']['warmup_steps'], 50)
        self.assertEqual(contract['observation_encoder'], 'native')
        self.assertEqual(contract['determinism']['requested'], 'inherit')
        self.assertIsNone(contract['determinism']['runtime'])
        for field, value in (('lr', 2e-4), ('warmup', 25), ('guard_weight', 2.0),
                             ('observation_encoder', 'legacy'), ('determinism', 'warn')):
            changed = copy.copy(args)
            setattr(changed, field, value)
            self.assertNotEqual(contract, self.contract(changed), field)

    def test_schedule_matches_original_arithmetic_including_tiny_smoke(self):
        for steps, warmup in ((3000, 50), (2, 50), (3000, 0)):
            for step in (1, min(steps, max(1, warmup)), steps):
                warm = min(1., step / max(1, warmup))
                decay = .2 + .8 * .5 * (1 + math.cos(math.pi * max(0, step-warmup) / max(1, steps-warmup)))
                self.assertEqual(learning_rate(step, steps, 1e-4, warmup), 1e-4 * warm * decay)
        self.assertAlmostEqual(learning_rate(3000, 3000, 1e-4, 50), 2e-5)

    def test_determinism_default_is_nonmutating_and_warn_is_not_strict(self):
        with patch('train_rphi.torch.use_deterministic_algorithms') as setting:
            configure_determinism('inherit')
            setting.assert_not_called()
            configure_determinism('warn')
            setting.assert_called_once_with(True, warn_only=True)
        with self.assertRaises(ValueError):
            configure_determinism('strict')
        before = (torch.are_deterministic_algorithms_enabled(),
                  torch.is_deterministic_algorithms_warn_only_enabled())
        actual = determinism_snapshot('cpu', 'inherit')
        self.assertEqual(actual['torch_deterministic_algorithms'], before[0])
        self.assertEqual(actual['torch_deterministic_warn_only'], before[1])
        self.assertEqual(actual['device'], 'cpu')
        self.assertIsNone(actual['npu'])
        self.assertFalse(actual['bitwise_reproducibility_claimed'])
        self.assertEqual(before, (torch.are_deterministic_algorithms_enabled(),
                                  torch.is_deterministic_algorithms_warn_only_enabled()))


if __name__ == '__main__':
    unittest.main()
