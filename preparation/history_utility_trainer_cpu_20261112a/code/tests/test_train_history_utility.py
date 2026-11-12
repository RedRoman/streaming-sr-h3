"""Small actual Torch CPU fixtures; no real checkpoints or data files."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_history_utility as T


class ToyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 3, 1)
        self.model_config = {'fixture': '11channel_stochastic_RGB'}

    def forward(self, observed, weights):
        # Deliberately consume torch RNG so resume tests exercise its restore.
        x = torch.cat((observed.flatten(1, 2), weights), 1)
        return {'rgb': self.conv(x) + torch.rand_like(observed[:, -1])*.001}


def batch():
    # Exercise Python and NumPy streams as well as Torch during resume.
    x = np.random.default_rng(int(random.random()*1e6)).random((2, 3, 3, 16, 16), dtype=np.float32)
    return {'observed': torch.from_numpy(x), 'teacher_weights': torch.from_numpy(np.random.random((2, 2, 16, 16)).astype(np.float32)),
            'target': torch.rand(2, 3, 16, 16)}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def initialize(self):
        random.seed(37); np.random.seed(38); torch.manual_seed(39)
        model = ToyTeacher()
        return model, T.make_optimizer(model)

    def test_exact_resume_model_optimizer_all_rng_and_schedule_offset(self):
        model, opt = self.initialize()
        for step in range(1, 7): T.train_step(model, opt, batch(), step)
        expected_model = T.cpu_tree(model.state_dict()); expected_opt = T.cpu_tree(opt.state_dict())
        expected_rng = T.capture_rng(torch.device('cpu'))
        model, opt = self.initialize()
        for step in (1, 2): T.train_step(model, opt, batch(), step)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'step2.pt'; binding = {'schedule_sha256': 'fixed', 'initial_sha': 'fixed'}; runtime = {'device': 'cpu'}
            T.save_checkpoint(p, model, opt, 2, binding, runtime, torch.device('cpu'))
            loaded = torch.load(p, weights_only=True)
            self.assertEqual(loaded['next_row_index'], 32)
            other = ToyTeacher(); other_opt = T.make_optimizer(other)
            step = T.restore_checkpoint(p, other, other_opt, binding, runtime, torch.device('cpu'))
            self.assertEqual(step, 2)
            for current in range(step+1, 7): T.train_step(other, other_opt, batch(), current)
            self.assertTrue(T.tree_equal(expected_model, T.cpu_tree(other.state_dict())))
            self.assertTrue(T.tree_equal(expected_opt, T.cpu_tree(other_opt.state_dict())))
            self.assertTrue(T.tree_equal(expected_rng, T.capture_rng(torch.device('cpu'))))

    def test_fixed_step_boundary_and_actual_first_lr(self):
        model, opt = self.initialize(); before = T.state_sha(model.state_dict())
        for step in (0, 1001, True):
            with self.assertRaisesRegex(ValueError, 'budget'): T.train_step(model, opt, batch(), step)
        self.assertEqual(before, T.state_sha(model.state_dict()))
        result = T.train_step(model, opt, batch(), 1)
        self.assertEqual(result['lr'], 1e-4*(1/50)); self.assertEqual(result['row_end_exclusive'], 16)
        self.assertGreater(result['gradient_norm_before_clip'], 0)
        self.assertNotEqual(before, T.state_sha(model.state_dict()))
        with self.assertRaisesRegex(ValueError, 'LR|step'): T.train_step(model, opt, batch(), 1)

    def test_nonfinite_input_or_gradient_cannot_update_model(self):
        model, opt = self.initialize(); before = T.state_sha(model.state_dict()); b = batch()
        b['target'][0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite input'): T.train_step(model, opt, b, 1)
        self.assertEqual(before, T.state_sha(model.state_dict())); self.assertEqual(len(opt.state), 0)
        # Failed calls are not retried by the trainer. Fresh fixture here tests
        # failure after backward but before the optimizer mutation.
        model, opt = self.initialize(); before = T.state_sha(model.state_dict())
        handle = model.conv.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
        try:
            with self.assertRaisesRegex(ValueError, 'nonfinite gradient'): T.train_step(model, opt, batch(), 1)
        finally: handle.remove()
        self.assertEqual(before, T.state_sha(model.state_dict())); self.assertEqual(len(opt.state), 0)

    def test_strict_checkpoint_identity_dtype_lr_and_no_overwrite(self):
        model, opt = self.initialize(); T.train_step(model, opt, batch(), 1)
        binding, runtime = {'schedule': 'one'}, {'device': 'cpu'}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'step.pt'; T.save_checkpoint(p, model, opt, 1, binding, runtime, torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'exists'): T.save_checkpoint(p, model, opt, 1, binding, runtime, torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'identity'): T.restore_checkpoint(p, model, opt, {'schedule': 'two'}, runtime, torch.device('cpu'))
            payload = torch.load(p, weights_only=True); payload['model_state']['conv.weight'] = payload['model_state']['conv.weight'].half()
            tampered = Path(td)/'wrong.pt'; torch.save(payload, tampered)
            with self.assertRaisesRegex(ValueError, 'dtype'): T.restore_checkpoint(tampered, model, opt, binding, runtime, torch.device('cpu'))
            payload = torch.load(p, weights_only=True); payload['optimizer_state']['param_groups'][0]['lr'] = .3
            torch.save(payload, tampered)
            with self.assertRaisesRegex(ValueError, 'LR'): T.restore_checkpoint(tampered, model, opt, binding, runtime, torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'only step1000'): T.save_checkpoint(Path(td)/'fakefinal.pt', model, opt, 1, binding, runtime, torch.device('cpu'), final=True)

    def test_resume_binding_excludes_only_new_admission_and_checkpoint(self):
        c = {'schema': T.SCHEMA, 'optimizer': T.OPTIMIZER, 'runtime': {'device': 'cpu'}, 'files': {'teacher': 'fixed'},
             'admission': {'path': 'old'}, 'resume': None}
        other = copy.deepcopy(c); other['admission'] = {'path': 'new'}; other['resume'] = {'path': 'step2'}
        self.assertEqual(T.training_identity(c), T.training_identity(other))
        other['files']['teacher'] = 'changed'
        self.assertNotEqual(T.training_identity(c), T.training_identity(other))

    def test_batch_rejects_oracle_or_dev_and_unavailable_weights(self):
        from history_utility_data import PROFILES
        families = [p for p in PROFILES for _ in range(4)] + ['identity']*2 + ['near_clean']*2
        rows = [{'profile_id': p, 'split': 'train', 'row_index': i} for i, p in enumerate(families)]
        b = {'model_inputs': {'observed': np.zeros((16, 3, 3, 96, 96), np.float32), 'availability': np.ones((16, 2), np.float32)},
             'teacher_weights': np.ones((16, 2, 96, 96), np.float32), 'target': np.zeros((16, 3, 96, 96), np.float32),
             'metadata': copy.deepcopy(rows)}
        T.validate_batch(b, rows)
        b['model_inputs']['target'] = b['target']
        with self.assertRaisesRegex(ValueError, 'oracle'): T.validate_batch(b, rows)
        del b['model_inputs']['target']; b['model_inputs']['availability'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'revives'): T.validate_batch(b, rows)
        b['teacher_weights'][0, 0] = 0; rows[0]['split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'non-train'): T.validate_batch(b, rows)


if __name__ == '__main__': unittest.main()
