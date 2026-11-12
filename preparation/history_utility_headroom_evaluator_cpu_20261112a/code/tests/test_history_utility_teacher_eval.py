"""Tiny real Torch CPU tests; no model assets, actual dev data or NPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_history_utility_teacher as E
import train_history_utility as T


class Toy(torch.nn.Module):
    def __init__(self, fail_at=None, mutate=False):
        super().__init__(); self.gain = torch.nn.Parameter(torch.tensor(.2))
        self.calls = []; self.fail_at = fail_at; self.mutate = mutate
        self.eval().requires_grad_(False)

    def forward(self, observed, weights):
        self.calls.append((observed.clone(), weights.clone()))
        if len(self.calls) == self.fail_at: raise RuntimeError('fixture forward failure')
        if self.mutate: self.gain.add_(.001)
        delta = ((observed[:, :2]-observed[:, -1:])*weights[:, :, None]).sum(1)
        return {'rgb': observed[:, -1]+self.gain*delta}


def row(t, index=None):
    r = {'source_id': 'one', 'profile_id': 'identity', 'frame_index': t,
         'true_availability': [int(t >= 2), int(t >= 1)], 'severity_phase': 'identity',
         'row_index': t if index is None else index, 'crop_tlhw': [192, 368, 96, 96]}
    r['row_sha256'] = T.value_sha(r); return r


class Data:
    def batch(self, rows):
        x = np.zeros((len(rows), 3, 3, 16, 16), np.float32)
        x[:, 0] = .1; x[:, 1] = .2; x[:, 2] = .4
        return {'model_inputs': {'observed': x, 'availability': np.asarray([r['true_availability'] for r in rows], np.float32)},
                'target': np.full((len(rows), 3, 16, 16), .3, np.float32), 'metadata': copy.deepcopy(rows)}


def counts():
    return {'crop_forward_attempts': 0, 'crop_forwards_completed': 0,
            'model_invocation_attempts': 0, 'model_invocations_completed': 0, 'unavailable_LOO_crops_skipped': 0}


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_clipped_exact_psnr_null_does_not_hide_raw_harm(self):
        y = np.zeros((3, 16, 16), np.float32); pred = y-.1
        m = E.metrics(pred, y, y)
        self.assertIsNone(m['clipped_psnr']); self.assertTrue(m['clipped_exact_match'])
        self.assertGreater(m['unclamped_mae'], .09); self.assertEqual(m['clipped_mae'], 0)
        self.assertEqual(m['clip_fraction'], 1.); self.assertEqual(m['changed_pixel_fraction'], 1.)
        json.dumps(m, allow_nan=False)

    def test_boundary_skip_and_fixed_other_history_weights(self):
        model = Toy(); c = counts(); records = E.evaluate_batch(model, Data().batch([row(t) for t in range(3)]), torch.device('cpu'), c)
        self.assertEqual(c['crop_forwards_completed'], 9); self.assertEqual(c['unavailable_LOO_crops_skipped'], 3)
        self.assertEqual(c['model_invocations_completed'], 4)
        self.assertEqual(model.calls[2][1].shape[0], 1)
        self.assertTrue(torch.all(model.calls[2][1][:, 0] == 0))
        self.assertTrue(torch.all(model.calls[2][1][:, 1] == 1))
        self.assertTrue(torch.all(model.calls[3][1][:, 1] == 0))
        self.assertEqual(model.calls[3][1][:, 0, 0, 0].tolist(), [0., 1.])
        self.assertEqual(records[0]['raw_utility'], [[[0.]], [[0.]]])
        self.assertEqual(records[0]['metrics']['all_history']['prediction_float32_sha256'],
                         records[0]['metrics']['remove_0']['prediction_float32_sha256'])

    def test_target_change_never_enters_predictions(self):
        b = Data().batch([row(2)]); changed = copy.deepcopy(b); changed['target'][:] = .9
        a = E.evaluate_batch(Toy(), b, torch.device('cpu'), counts())[0]
        d = E.evaluate_batch(Toy(), changed, torch.device('cpu'), counts())[0]
        for mode in a['metrics']:
            self.assertEqual(a['metrics'][mode]['prediction_float32_sha256'], d['metrics'][mode]['prediction_float32_sha256'])
        self.assertNotEqual(a['all_history_mae'], d['all_history_mae'])
        self.assertNotEqual(a['raw_utility'], d['raw_utility'])

    def test_actual_freeze_records_and_partial_survive_failure(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); report = {}
            with self.assertRaisesRegex(RuntimeError, 'fixture forward'):
                E.run_evaluation(Toy(fail_at=4), Data(), [row(t) for t in range(3)], torch.device('cpu'), out, report, batch_size=1)
            saved = json.loads((out/'partial.json').read_text())
            self.assertEqual(saved['completed_rows'], 1)
            self.assertEqual(len((out/'records.jsonl').read_text().splitlines()), 1)
            self.assertEqual(saved['counts']['crop_forward_attempts'], 4)
            self.assertEqual(saved['counts']['crop_forwards_completed'], 3)
            self.assertTrue(saved['state_sha_preserved'])

    def test_mutation_is_reported_and_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            report = {}
            with self.assertRaisesRegex(ValueError, 'mutated'):
                E.run_evaluation(Toy(mutate=True), Data(), [row(2)], torch.device('cpu'), Path(td), report)
            self.assertFalse(report['state_sha_preserved']); self.assertFalse(report['parameter_versions_preserved'])

    def test_no_training_and_control_harm_summary(self):
        with tempfile.TemporaryDirectory() as td:
            model = Toy(); before = T.state_sha(model.state_dict()); report = {}
            records = E.run_evaluation(model, Data(), [row(t) for t in range(3)], torch.device('cpu'), Path(td), report)
            self.assertEqual(before, T.state_sha(model.state_dict())); self.assertTrue(report['frozen_flags_preserved'])
            total = next(x for x in report['descriptive_by_source_profile_phase'] if x['severity_phase'] == 'all_frames')
            self.assertEqual(total['frames'], 3)
            self.assertAlmostEqual(total['all_history_mae_harm_vs_input'], np.mean([r['all_history_mae_harm_vs_input'] for r in records]))
            self.assertNotIn('assessment', report)  # A tiny fixture cannot accept the real headroom gate.

    def test_full_coverage_count_and_future_row_rejected(self):
        from history_utility_data import PROFILES, history_indices, true_availability
        p = {'profile_ids': [*PROFILES, 'identity', 'near_clean'], 'dev_source_ids': list('abcdef')}
        rows = []
        for s in p['dev_source_ids']:
            for profile in p['profile_ids']:
                for t in range(96):
                    r = {'source_id': s, 'profile_id': profile, 'frame_index': t, 'split': 'dev', 'row_index': len(rows),
                         'crop_tlhw': [192, 368, 96, 96], 'history_indices': history_indices(t),
                         'true_availability': true_availability(t), 'external_mask': [1, 1], 'teacher_mask': None}
                    r['row_sha256'] = T.value_sha(r); rows.append(r)
        E.validate_rows(rows, p)
        self.assertEqual(sum(2+sum(r['true_availability']) for r in rows), 11430)
        rows[1]['history_indices'][-1] = 2
        rows[1]['row_sha256'] = T.value_sha({k: v for k, v in rows[1].items() if k != 'row_sha256'})
        with self.assertRaisesRegex(ValueError, 'chronology'): E.validate_rows(rows, p)
        with self.assertRaisesRegex(ValueError, 'coverage'): E.validate_rows(rows[:-1], p)

    def test_strict_final_checkpoint_no_optimizer_or_model_selection(self):
        def constructor(_):
            model = Toy(); model.model_config = {'fixture': 'frozen_teacher'}; return model
        model = constructor(None); model_sha = T.state_sha(model.state_dict())
        terminal = {'binding': {'proof': 'actual_final'}, 'runtime': {'train': 'fixture'}, 'final_model_sha256': model_sha}
        payload = {'schema': T.CHECKPOINT_SCHEMA, 'phase': 'teacher', 'step': 1000, 'final': True,
                   'next_row_index': 16000, 'binding': terminal['binding'], 'runtime': terminal['runtime'],
                   'optimizer_specification': T.OPTIMIZER, 'headroom_accepted': False, 'student_training_admitted': False,
                   'model_config': model.model_config, 'parameter_names': ['gain'], 'model_state': model.state_dict(),
                   'model_sha256': model_sha}
        runtime = {'cpu_threads': 1, 'device': 'cpu', 'determinism': 'inherit',
                   'expected_versions': {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': str(np.__version__),
                                         'torch': str(torch.__version__), 'torch_npu': None}}
        with tempfile.TemporaryDirectory() as td, patch('rphi_history_utility.MaskAwareRestorer', constructor):
            p = Path(td)/'final.pt'; torch.save(payload, p)
            actual, device, _ = E.load_teacher(p, terminal, runtime)
            self.assertEqual(T.state_sha(actual.state_dict()), model_sha)
            self.assertFalse(actual.training); self.assertFalse(actual.gain.requires_grad)
            payload['step'] = 999; torch.save(payload, p)
            with self.assertRaisesRegex(ValueError, 'final1000'): E.load_teacher(p, terminal, runtime)


if __name__ == '__main__': unittest.main()
