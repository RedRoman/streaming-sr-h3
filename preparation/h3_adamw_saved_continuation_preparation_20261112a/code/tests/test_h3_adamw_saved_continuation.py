"""CPU-only continuation boundaries; no pretrained asset/model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_h3_adamw_saved_continuation as M


class MemoryEvidence:
    def __init__(self):
        self.data = {'phase': 'fixture', 'actual': {'merge_calls': 0},
            'block_forward_calls': {'DiT': {}, 'VAE': {}}}
        self.saves = 0
    def save(self): self.saves += 1
    def progress(self, *args): self.save()
    def phase(self, label, *args): self.data['phase'] = label; self.save()


def acceptance():
    contract = {'failed_run': {'partial': {'sha256': M.PARTIAL_SHA}, 'queue': {'sha256': M.QUEUE_SHA},
        'root_review': {'sha256': 'review'}}}
    prior = {'contract': {'execution_contract_sha256': 'execution', 'accepted_full22': {'report': 'bound'}},
        'checkpoint': {'bytes': 20}}
    audit = {'schema': 'openh3.saved_adamw_partial_audit.v1', 'audit_complete': True, 'complete': True,
        'partial_artifact_acceptance': 'verified_saved_one_update_and_zero_updated_outputs_only',
        'saved_subset_ready_for_continuation': True,
        'original_run_accepted': False, 'strict_reload_accepted': False, 'merge_accepted': False,
        'functional_deployment_ready': False,
        'components': {k: {'status': 'verified', 'facts': {'passed': True, 'actual_arrays_verified': True}}
            for k in ('saved_update', 'outputs')},
        'binding': {'partial_sha256': M.PARTIAL_SHA, 'queue_sha256': M.QUEUE_SHA,
            'root_review_sha256': 'review', 'original_execution_contract_sha256': 'execution',
            'producer_sha256': M.PRODUCER_SHA, 'protocol_sha256': M.A.PROTOCOL_SHA,
            'checkpoint': {'path': 'adapter', 'sha256': M.ADAPTER_SHA, 'bytes': 20},
            'optimizer_checkpoint': {'path': 'optimizer', 'sha256': M.OPTIMIZER_SHA, 'bytes': 40},
            'accepted_full22': {'report': 'bound'}}}
    return audit, contract, prior


def complete_counts():
    d = {'actual': {k: v for k, v in M.PLANNED.items() if k not in ('NFE', 'VAE_decodes', 'stage_decodes')},
        'block_forward_calls': {'DiT': {m: {str(i): 3 for i in range(50)} for m in M.MODES},
            'VAE': {m + '_decode': {str(i): 3 for i in range(36)} for m in M.MODES}},
        'decoder_context_restorations': [{'phase': m + '_decode', 'restored': True} for m in M.MODES],
        'noise_observations': [{'phase': m} for m in M.MODES], 'math_observations': [{}] * 516,
        'native_tile_rows': [{'phase': m + '_decode', 'row_index': i, 'pixel_calls': 1,
            'yielded_tiles': 4, 'groups': [{'shape': [4, 24, 7, 16, 16]}]} for m in M.MODES for i in range(3)]}
    d['actual'].update(NFE={m: 3 for m in M.MODES}, VAE_decode_calls={m: 1 for m in M.MODES},
        stage_decode_calls={m + '_decode': 1 for m in M.MODES}, forbidden_calls=[])
    return d


class PureBoundaries(unittest.TestCase):
    def test_accept_only_actual_saved_subset_without_relabel(self):
        a, c, p = acceptance(); M.validate_subset_audit(a, c, p)
        for key in ('complete', 'audit_complete'):
            bad = copy.deepcopy(a); bad[key] = False
            with self.assertRaises(ValueError): M.validate_subset_audit(bad, c, p)
        bad = copy.deepcopy(a); bad['original_run_accepted'] = True
        with self.assertRaises(ValueError): M.validate_subset_audit(bad, c, p)
        bad = copy.deepcopy(a); bad['components']['outputs']['facts']['actual_arrays_verified'] = False
        with self.assertRaises(ValueError): M.validate_subset_audit(bad, c, p)
        bad = copy.deepcopy(a); bad['binding']['checkpoint']['sha256'] = 'other'
        with self.assertRaises(ValueError): M.validate_subset_audit(bad, c, p)

    def test_strict_bytes_include_signed_zero_and_dtype(self):
        a = np.array([0., 1.], dtype=np.float32)
        self.assertTrue(M.byte_parity(a, a.copy())['byte_exact'])
        b = a.copy(); b[0] = np.float32(-0.)
        result = M.byte_parity(a, b)
        self.assertTrue(result['array_equal']); self.assertFalse(result['byte_exact'])
        self.assertFalse(M.byte_parity(a, a.astype(np.float64))['byte_exact'])
        b[1] = np.nan; self.assertFalse(M.byte_parity(a, b)['finite'])

    def test_budget_no_backward_update_extra_noise_or_tile(self):
        d = complete_counts(); self.assertTrue(M.budget_passed(d))
        for key in ('backward_attempts', 'optimizer_steps', 'VAE_encode_calls', 'Rphi_calls', 'checkpoint_saves'):
            b = copy.deepcopy(d); b['actual'][key] += 1
            self.assertFalse(M.budget_passed(b))
        b = copy.deepcopy(d); b['noise_observations'].append({'phase': 'reloaded'})
        self.assertFalse(M.budget_passed(b))
        b = copy.deepcopy(d); b['native_tile_rows'][0]['groups'][0]['shape'][0] = 2
        self.assertFalse(M.budget_passed(b))
        b = copy.deepcopy(d); b['block_forward_calls']['DiT']['backward'] = {'0': 3}
        self.assertFalse(M.budget_passed(b))

    def test_success_no_flush_failure_durable_and_guard_restoration(self):
        e = MemoryEvidence(); M.require(True, 'ok', e); self.assertEqual(e.saves, 0)
        class Block:
            def forward(self, x): return x + 1
        b = Block(); before = b.forward.__func__; e.data['phase'] = 'reloaded'
        with self.assertRaises(RuntimeError):
            with M.bounded_blocks([b], 'DiT', e, lambda *args: None):
                for _ in range(4): b.forward(0)
        self.assertIs(b.forward.__func__, before); self.assertNotIn('forward', b.__dict__)
        self.assertEqual(e.saves, 1); self.assertEqual(len(e.data['gate_failures']), 1)

    def test_functional_failure_not_hidden_by_reload(self):
        x = {'reference__zero': {'rgb': {'byte_exact': True}},
            'reference__merged': {'rgb': {'changed_coordinates': 0}}}
        self.assertFalse(M.functional_result(x)['readiness_gate_passed'])
        x['reference__merged']['rgb']['changed_coordinates'] = 1
        self.assertTrue(M.functional_result(x)['readiness_gate_passed'])
        x['reference__zero']['rgb']['byte_exact'] = False
        self.assertEqual(M.functional_result(x)['attribution'], 'numerical_mode_confounded')


TORCH = importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH, 'Torch unavailable locally; run exact CPU fixture snapshot in existing remote environment')
class TorchRestoration(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.params = {'p.lora_a': torch.nn.Parameter(torch.tensor([[.1, .2]], dtype=torch.float32)),
            'p.lora_b': torch.nn.Parameter(torch.tensor([[.3]], dtype=torch.float32))}
        self.expected = {n: tuple(p.shape) for n, p in self.params.items()}
        self.saved, self.inventory = {}, {}
        for n, p in self.params.items():
            self.inventory[n] = {}
            for k in ('step', 'exp_avg', 'exp_avg_sq'):
                value = torch.tensor(1.) if k == 'step' else torch.full_like(p, .01)
                self.saved[n + '::' + k] = value
                self.inventory[n][k] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'device': 'cpu', 'sha256': M.tensor_sha(value, torch)}

    def restore(self):
        return M.restore_optimizer(self.params, self.expected, self.saved, self.inventory, list(self.params), self.t)

    def test_exact_fresh_all_fields_load_without_update(self):
        before = {n: p.detach().clone() for n, p in self.params.items()}
        saved_rows, loaded_events = [], []
        opt = M.restore_optimizer(self.params, self.expected, self.saved, self.inventory, list(self.params), self.t,
            on_restored=lambda n,k,v,cpu: saved_rows.append((n,k,str(cpu.device),M.tensor_sha(cpu,self.t))),
            on_loaded=lambda: loaded_events.append('returned'))
        self.assertEqual(len(opt.state), 2); self.assertEqual(loaded_events, ['returned'])
        self.assertEqual(len(saved_rows), 6)
        self.assertTrue(all(device == 'cpu' and sha == self.inventory[n][k]['sha256'] for n,k,device,sha in saved_rows))
        for n, p in self.params.items():
            self.assertTrue(self.t.equal(p, before[n])); self.assertIsNone(p.grad)
            for k, value in opt.state[p].items(): self.assertTrue(self.t.equal(value, self.saved[n + '::' + k]))

    def test_wrong_step_shape_order_and_nonfinite_rejected(self):
        key = 'p.lora_a::step'; original = self.saved[key]
        self.saved[key] = self.t.tensor(2.)
        with self.assertRaises(ValueError): self.restore()
        self.saved[key] = original
        with self.assertRaises(ValueError): M.restore_optimizer(self.params, self.expected, self.saved,
            self.inventory, list(reversed(self.params)), self.t)
        self.saved['p.lora_a::exp_avg'] = self.t.full((1, 2), float('nan'))
        with self.assertRaises(ValueError): self.restore()

    def test_real_200_matrix_BF16_merge_preserves_all_other_state(self):
        t = self.t
        import lora as L
        class Tiny(t.nn.Module):
            def __init__(self):
                super().__init__(); self.blocks = t.nn.ModuleList()
                for _ in range(50):
                    b = t.nn.Module(); b.attn = t.nn.Module(); b.mlp = t.nn.Module()
                    for target in ('qkv_proj', 'out_proj'): setattr(b.attn, target, t.nn.Linear(2, 2, dtype=t.bfloat16))
                    for target in ('fc1', 'fc2'): setattr(b.mlp, target, t.nn.Linear(2, 2, dtype=t.bfloat16))
                    self.blocks.append(b)
                self.register_buffer('untouched', t.ones(2)); self._openh3_pretrained_record = {'fixture': True}
        model = Tiny().eval().requires_grad_(False)
        snap = M.J.freeze_snapshot(model, t); names = L.inject(model, 8)
        with t.no_grad():
            for name, value in L.lora_parameters(model):
                if name.endswith('lora_b'): value.fill_(.125)
        e = MemoryEvidence(); expected = M.merge_once(model, names, snap, e, t)
        self.assertEqual(e.data['actual']['merge_calls'], 1)
        self.assertEqual(e.data['merge']['runtime_receipt']['merged_modules'], 200)
        self.assertEqual(len(L.lora_parameters(model)), 0)
        self.assertTrue(M.J.check_snapshot(model, expected, t, names)['passed'])
        self.assertTrue(all(row['exact_declared_write'] for row in e.data['merge']['layers'].values()))
        with self.assertRaises(RuntimeError): M.merge_once(model, names, expected, e, t)

    def test_actual_save_no_overwrite_and_canonical_snapshot_identity(self):
        t = self.t
        with tempfile.TemporaryDirectory() as d:
            e = M.Evidence(Path(d) / 'fresh', {'fixture': True})
            value = t.tensor([1., 2.]); row, array = M.save_actual(e, 'a', value, t)
            self.assertTrue(np.array_equal(M.N.load_array(e.path, row), array))
            with self.assertRaises(RuntimeError): M.save_actual(e, 'a', value, t)
        model = t.nn.Linear(2, 2).requires_grad_(False)
        inventory = M.J.frozen_inventory(model, t)
        snap = M.snapshot_from_inventory(model, inventory, t)
        self.assertTrue(M.J.check_snapshot(model, snap, t)['passed'])
        with t.no_grad(): model.weight.add_(1.)
        self.assertFalse(M.J.check_snapshot(model, snap, t)['passed'])


if __name__ == '__main__': unittest.main(verbosity=2)
