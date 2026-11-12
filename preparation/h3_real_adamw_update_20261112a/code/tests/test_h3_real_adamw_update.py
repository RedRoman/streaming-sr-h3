"""CPU AdamW/serialization and actual sampler-AST toy; no real H3 or NPU."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try: import torch
except ImportError: torch = None
import check_h3_real_adamw_update as P
import check_h3_joint_rgb_grad as J
if torch is not None:
    import lora as L
    from test_h3_nonzero_adapter import tiny_model


@unittest.skipUnless(torch is not None, 'requires Torch CPU')
class AdamWTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.e = P.Evidence(Path(self.tmp.name) / 'run', {'fixture': True})
        self.model, self.names = tiny_model(); self.expected = P.N.expected_adapters(self.model, torch, blocks=1)
        self.pretrained = self.model._openh3_pretrained_record; self.diagnostic = {'fixture': 'one_AdamW', 'lr': P.LR}
    def tearDown(self): self.tmp.cleanup()

    def gradients(self):
        for n, p in L.lora_parameters(self.model): p.grad = torch.full_like(p, 0. if n.endswith('.lora_a') else 1024.)
        video = torch.ones(1, requires_grad=True) * 2; video.retain_grad(); video.sum().backward()
        J.save_gradients(self.model, video, self.e, torch, expected=len(self.expected))

    def test_first_actual_warmup_LR_and_closed_first_AdamW(self):
        self.gradients(); before = {n: p.clone() for n, p in L.lora_parameters(self.model)}
        opt = P.one_adamw_update(self.model, self.expected, self.e, torch)
        self.assertEqual(P.LR, 1e-4 * min(1., 1 / 100))
        self.assertEqual(len(opt.state), len(self.expected))
        for n, p in L.lora_parameters(self.model):
            state = opt.state[p]
            self.assertEqual(set(state), {'step', 'exp_avg', 'exp_avg_sq'}); self.assertEqual(float(state['step']), 1.)
            self.assertEqual(state['step'].device.type, 'cpu')
            if n.endswith('.lora_a'):
                self.assertTrue(torch.equal(p, before[n])); self.assertEqual(int(torch.count_nonzero(state['exp_avg'])), 0)
                self.assertEqual(int(torch.count_nonzero(state['exp_avg_sq'])), 0)
            else:
                expected = before[n] - P.LR * p.grad / (p.grad.abs() + 1e-8)
                self.assertTrue(torch.allclose(p, expected, atol=2e-13, rtol=0))
        with self.assertRaisesRegex(RuntimeError, 'second update'): P.one_adamw_update(self.model, self.expected, self.e, torch)

    def test_None_or_NaN_rejected_before_unscale_or_step(self):
        for invalid in ('none', 'nan'):
            self.gradients(); p = L.lora_parameters(self.model)[0][1]
            p.grad = None if invalid == 'none' else torch.full_like(p, float('nan'))
            with self.assertRaisesRegex(RuntimeError, 'missing/nonfinite'): P.one_adamw_update(self.model, self.expected, self.e, torch)
            self.assertEqual(self.e.data['actual']['optimizer_steps'], 0); self.assertEqual(self.e.data['actual']['unscale_calls'], 0)

    def test_actual_global_clip_then_one_step(self):
        self.gradients(); opt = P.one_adamw_update(self.model, self.expected, self.e, torch)
        norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for _, p in L.lora_parameters(self.model)]))
        self.assertLessEqual(float(norm), 1.000001)
        self.assertGreater(self.e.data['gradient_clip']['pre_clip_total_norm'], 1.)
        self.assertEqual(self.e.data['actual']['clip_calls'], 1)
        self.assertTrue(all(torch.isfinite(v).all() for st in opt.state.values() for v in st.values()))

    def test_adapter_and_optimizer_save_clear_reload_exact(self):
        self.gradients(); opt = P.one_adamw_update(self.model, self.expected, self.e, torch)
        files = P.save_checkpoint(self.model, opt, self.expected, self.pretrained, self.diagnostic, self.e, torch)
        P.clear_reload(self.model, opt, self.expected, files, self.pretrained, self.diagnostic, self.e, torch)
        self.assertTrue(all(self.e.data['reload']['cleared'].values()))
        self.assertTrue(all(self.e.data['reload']['exact_adapter'].values()))
        self.assertTrue(all(self.e.data['reload']['exact_optimizer'].values()))
        self.assertFalse(self.e.data['reload']['resume_next_step_trajectory_equivalence_claim'])
        with self.assertRaisesRegex(RuntimeError, 'second clear'): P.clear_reload(self.model, opt, self.expected, files, self.pretrained, self.diagnostic, self.e, torch)

    def test_optimizer_extra_key_dtype_and_byte_tampering_rejected(self):
        self.gradients(); opt = P.one_adamw_update(self.model, self.expected, self.e, torch)
        params = dict(L.lora_parameters(self.model)); first = next(iter(params.values()))
        opt.state[first]['extra'] = torch.tensor(0.)
        with self.assertRaisesRegex(RuntimeError, 'unexpected AdamW state'): P.optimizer_inventory(opt, params, self.expected, self.e, torch, 'bad')
        del opt.state[first]['extra']; original = opt.state[first]['exp_avg']; opt.state[first]['exp_avg'] = original.half()
        with self.assertRaisesRegex(RuntimeError, 'invalid FP32'): P.optimizer_inventory(opt, params, self.expected, self.e, torch, 'bad2')
        opt.state[first]['exp_avg'] = original
        files = P.save_checkpoint(self.model, opt, self.expected, self.pretrained, self.diagnostic, self.e, torch)
        with files[1].open('ab') as f: f.write(b'tamper')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'): P.clear_reload(self.model, opt, self.expected, files, self.pretrained, self.diagnostic, self.e, torch)

    def test_strict_reload_includes_FP16_actual_decoder_cast(self):
        for mode in ('updated', 'reloaded'):
            for stream in ('video', 'audio', 'rgb', 'decoder_input'):
                value = torch.zeros(1, dtype=torch.float16 if stream == 'decoder_input' else torch.float32)
                if mode == 'reloaded' and stream == 'decoder_input': value += .001
                J.save_array(self.e, mode + '_' + stream, value, torch)
        with self.assertRaisesRegex(RuntimeError, 'strict reload output'): P.compare_modes('updated', 'reloaded', self.e, torch, strict=True)

    def test_unchanged_RGB_does_not_become_functional_readiness(self):
        row = P.functional_result({'zero__updated': {'rgb': {'changed_coordinates': 2}},
                                   'zero__merged': {'rgb': {'changed_coordinates': 0}},
                                   'reference__zero': {'rgb': {'changed_coordinates': 0,
                                       'reference_sha256_float32':'a','candidate_sha256_float32':'a'}},
                                   'reference__merged': {'rgb': {'changed_coordinates': 0}}})
        self.assertTrue(row['resident_unmerged_RGB_changed']); self.assertFalse(row['deployed_merged_RGB_changed'])
        self.assertTrue(row['no_LR_rank_step_adjustment']); self.assertFalse(row['quality_or_generalization_claim'])
        self.assertFalse(row['readiness_gate_passed'])

    def test_numerical_mode_confounding_blocks_functional_readiness(self):
        comparisons={k:{'rgb':{'changed_coordinates':1}} for k in
            ('zero__updated','zero__merged','reference__zero','reference__merged')}
        comparisons['reference__zero']['rgb'].update(reference_sha256_float32='a',candidate_sha256_float32='b')
        row=P.functional_result(comparisons)
        self.assertEqual(row['attribution'],'numerical_mode_confounded');self.assertFalse(row['readiness_gate_passed'])
        comparisons['reference__zero']['rgb']['changed_coordinates']=0
        comparisons['reference__zero']['rgb']['candidate_sha256_float32']='a'
        row=P.functional_result(comparisons)
        self.assertEqual(row['attribution'],'observed_deployed_RGB_change');self.assertTrue(row['readiness_gate_passed'])
        self.assertFalse(row['repeatability_or_statistical_quality_claim'])

    def test_signed_zero_RGB_is_not_byte_exact_control(self):
        positive=torch.tensor([0.],dtype=torch.float32);negative=torch.tensor([-0.],dtype=torch.float32)
        self.assertTrue(torch.equal(positive,negative))
        for mode,value in (('reference',positive),('zero',negative)):
            for stream in ('video','audio','decoder_input','rgb'):J.save_array(self.e,mode+'_'+stream,value,torch)
        P.compare_modes('reference','zero',self.e,torch)
        comparisons=self.e.data['comparisons']
        for key in ('zero__updated','zero__merged','reference__merged'):
            comparisons[key]={'rgb':{'changed_coordinates':1}}
        row=P.functional_result(comparisons)
        self.assertFalse(row['zero_vs_accepted_reference_RGB_byte_exact'])
        self.assertFalse(row['readiness_gate_passed']);self.assertEqual(row['attribution'],'numerical_mode_confounded')

    def test_real_target_parity_gate_forbids_backward(self):
        video = torch.ones(1, requires_grad=True) * 2; video.retain_grad()
        rgb = video.reshape(1, 1, 1, 1).expand(2, 4, 4, 3); target = torch.zeros_like(rgb).detach()
        digest = hashlib.sha256(target.numpy().tobytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, 'RGB geometry/parity'):
            P.F.full_rgb_backward(torch.zeros_like(rgb), rgb, video, target, self.e, torch,
                                  expected_shape=tuple(target.shape), target_sha256=digest)
        self.assertEqual(self.e.data['actual']['backward_attempts'], 0)


@unittest.skipUnless(torch is not None and os.environ.get('OPENH3_COMFY_REFERENCE_ROOT'), 'requires pinned sampler AST CPU fixture')
class CompleteTinyPath(unittest.TestCase):
    def test_400_states_four_modes_real_sampler_AST_one_AdamW(self):
        from test_openh3_train_trajectory import fixture
        from openh3_train_trajectory import ResidentH3Trajectory
        with tempfile.TemporaryDirectory() as tmp:
            e = P.Evidence(Path(tmp) / 'run', {'CPU_fixture_only': True})
            # Persist every actual array/checkpoint. Batch only toy metadata I/O;
            # small tests above retain production atomic metadata/error writes.
            flush = e.save; e.save = lambda: None
            h3, backend, state = fixture(); model, names = tiny_model(50)
            h3.patcher.model.diffusion_model = model; expected = P.N.expected_adapters(model, torch)
            self.assertEqual(len(expected), 400)
            math_state = {'kind': 'dit', 'entries': 0, 'exits': 0}
            class Block(torch.nn.Module):
                def forward(self, x):
                    if math_state['kind'] != 'vae': raise RuntimeError('decoder math missing')
                    return x + x.sin() * .001
            class Stage(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.frozen = torch.nn.Parameter(torch.tensor(.1), requires_grad=False)
                    self.decoder = torch.nn.Module(); self.decoder.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(36)])
                def decode(self, x):
                    for b in self.decoder.transformer_blocks: x = b(x)
                    return x.float().mean().sigmoid().reshape(1, 1, 1, 1).expand(2, 4, 4, 3)
            stage = Stage().eval()
            h3.vae = SimpleNamespace(first_stage_model=stage, device=torch.device('cpu'), offload_device=torch.device('cpu'),
                                     disable_offload=True, vae_dtype=torch.float16, decode=stage.decode)
            @contextmanager
            def decoder_context():
                previous = math_state['kind']; math_state['kind'] = 'vae'; math_state['entries'] += 1
                try: yield
                finally: math_state['kind'] = previous; math_state['exits'] += 1
            def observe(group, phase, index):
                self.assertTrue(state['math']); self.assertEqual(math_state['kind'], 'vae' if group == 'VAE' else 'dit')
            generator = torch.Generator().manual_seed(101)
            shape = (1, 24, 2, 8, 8); guide = torch.randn(shape, generator=generator); raw = torch.randn(shape, generator=generator)
            target = torch.arange(96, dtype=torch.uint8).reshape(2, 4, 4, 3).float() / 255.
            target_sha = hashlib.sha256(target.numpy().tobytes()).hexdigest()
            snapshot = J.freeze_snapshot(model, torch, names); diagnostic = {'fixture': 'AdamW_engineering', 'lr': P.LR}
            actual = {'NFE': 0, 'decode': 0}; optimizer = None
            wrappers = J.wrapper_snapshot([(b, 'forward') for b in model.blocks] + [(b, 'forward') for b in stage.decoder.transformer_blocks])
            with J.counted_blocks(model.blocks, 'DiT', e, observe), J.counted_blocks(stage.decoder.transformer_blocks, 'VAE', e, observe):
                for mode in P.MODES:
                    if mode == 'updated':
                        optimizer = P.one_adamw_update(model, expected, e, torch)
                        self.assertEqual(len(optimizer.state), 400)
                        self.assertEqual(e.data['optimizer_after_update']['state_tensors'], 1200)
                        paths = P.save_checkpoint(model, optimizer, expected, model._openh3_pretrained_record, diagnostic, e, torch)
                    elif mode == 'reloaded':
                        P.clear_reload(model, optimizer, expected, paths, model._openh3_pretrained_record, diagnostic, e, torch)
                    elif mode == 'merged':
                        optimizer.state.clear(); optimizer = None
                        snapshot = P.N.merge_once(model, names, snapshot, e, torch)
                    session = ResidentH3Trajectory(h3, _backend=backend, checkpoint_blocks=mode == 'zero',
                        checkpoint_decoder=mode == 'zero', decoder_context=decoder_context)
                    with (nullcontext() if mode == 'zero' else torch.no_grad()), session:
                        e.phase(mode); result = session.sample(guide, raw, seed=101); actual['NFE'] += result['contract']['actual_forward_NFE']
                        if mode == 'zero': result['video'].retain_grad()
                        e.phase(mode + '_decode'); rgb = session.decode(result['video']); actual['decode'] += 1
                        for stream in ('video', 'audio'): J.save_array(e, mode + '_' + stream, result[stream], torch)
                        J.save_array(e, mode + '_decoder_input', result['video'].to(torch.float16), torch)
                        J.save_array(e, mode + '_rgb', rgb, torch)
                        if mode == 'zero':
                            for stream in ('video','audio','decoder_input','rgb'):
                                J.save_array(e,'reference_'+stream,torch.from_numpy(P.N.load_array(e.path,e.data['arrays']['zero_'+stream])),torch)
                            P.F.full_rgb_backward(rgb.detach().clone(), rgb, result['video'], target, e, torch,
                                expected_shape=tuple(target.shape), target_sha256=target_sha)
                            J.save_gradients(model, result['video'], e, torch)
                            self.assertTrue(e.data['adapter_gradients']['passed'])
                    self.assertFalse(session.active); self.assertEqual(math_state['kind'], 'dit')
                    self.assertTrue(J.check_snapshot(model, snapshot, torch, names)['passed'])
                    if mode == 'reloaded': P.compare_modes('updated', 'reloaded', e, torch, strict=True)
                    if mode == 'zero': P.compare_modes('reference', 'zero', e, torch)
                    if mode != 'zero': P.compare_modes('zero', mode, e, torch)
                    if mode == 'merged': P.compare_modes('reference', 'merged', e, torch)
                    del result, rgb
            self.assertEqual(actual, {'NFE': 12, 'decode': 4})
            self.assertEqual(e.data['actual']['optimizer_steps'], 1); self.assertEqual(e.data['actual']['completed_backward_calls'], 1)
            self.assertEqual(e.data['actual']['optimizer_state_clear_reload'], 1)
            self.assertTrue(J.wrappers_restored(wrappers)); self.assertEqual(math_state['entries'], math_state['exits'])
            self.assertFalse(state['math']); self.assertIsNone(h3.patcher.model.current_patcher)
            self.assertTrue(all(e.data['reload']['exact_optimizer'].values()))
            e.data['functional_effect'] = P.functional_result(e.data['comparisons']); flush()
            print(json.dumps({'CPU_toy_actual': actual, 'AdamW_steps': 1, 'backwards': 1, 'adapter_tensors': 400,
                'optimizer_named_entries': 400, 'optimizer_state_tensors': 1200, 'strict_reload': True,
                'native_full22_tile_geometry_tested_here': False, 'functional': e.data['functional_effect']}), flush=True)


if __name__ == '__main__': unittest.main(verbosity=2)
