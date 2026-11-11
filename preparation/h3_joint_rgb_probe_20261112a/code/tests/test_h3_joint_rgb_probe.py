"""Real Torch CPU joint-graph tests; no model weights, Comfy import or NPU.

The H3 fixture executes pinned official sampler function bodies through AST.
The tiny DiT and 36 tiny VAE blocks exercise the actual retained/checkpoint API.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import torch
except ImportError:
    torch = None
import check_h3_joint_rgb_grad as probe

if torch is not None:
    import openh3_train_trajectory as trajectory
    from test_openh3_train_trajectory import fixture


@unittest.skipUnless(torch is not None, 'requires Torch CPU')
class ProbeHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e = probe.Evidence(Path(self.tmp.name) / 'probe', {'test': True})
    def tearDown(self): self.tmp.cleanup()

    def test_actual_fixed_target_and_roundtrip_records(self):
        target = probe.fixed_target(torch)
        self.assertEqual(tuple(target.shape), (1, 1, 128, 1))
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(bool(torch.all(target[..., 1:, :] > target[..., :-1, :])))
        row = probe.save_array(self.e, 'target', target, torch)
        import numpy as np
        arr = np.load(self.e.path / row['path'], allow_pickle=False)
        self.assertEqual(hashlib.sha256(arr.tobytes()).hexdigest(), row['array_sha256'])
        self.assertEqual(probe.sha_file(self.e.path / row['path']), row['file_sha256'])
        print(json.dumps({'actual_torch_target_sha256': row['array_sha256'], 'shape': row['shape'],
                          'torch': torch.__version__, 'numpy': np.__version__}), flush=True)

    def test_fail_fast_video_or_audio_parity_before_any_backward(self):
        for stream in ('video', 'audio'):
            reference = {'video': torch.zeros(2), 'audio': torch.zeros(2)}
            candidate = {s: v.clone().requires_grad_() for s,v in reference.items()}
            candidate[stream] = candidate[stream] + .001
            with self.assertRaisesRegex(RuntimeError, 'strict video/audio'):
                probe.save_latent_parity(reference, candidate, self.e, torch)
            self.assertEqual(self.e.data['actual']['backward_attempts'], 0)
            self.assertEqual(self.e.data['actual']['RGB_loss_expressions'], 0)

    def test_rgb_failure_saves_pixels_and_blocks_backward(self):
        parameter = torch.nn.Parameter(torch.tensor(.2))
        video = parameter * torch.ones(1, 1, 128, 3)
        video.retain_grad()
        rgb = video.sin()
        with self.assertRaisesRegex(RuntimeError, 'RGB geometry/parity'):
            probe.rgb_backward(torch.zeros_like(rgb), rgb, video, self.e, torch,
                               expected_shape=(1,1,128,3))
        self.assertIn('candidate_rgb', self.e.data['arrays'])
        self.assertEqual(self.e.data['actual']['backward_attempts'], 0)
        self.assertIsNone(parameter.grad)

    def test_candidate_reference_tolerance_direction_is_asymmetric(self):
        # Exact representable pair straddles the directional rtol boundary.
        reference = torch.full((1,1,128,3), .01)
        candidate = torch.full_like(reference, .01020101, requires_grad=True) * 1
        self.assertTrue(torch.allclose(reference, candidate, atol=probe.ATOL, rtol=probe.RTOL))
        self.assertFalse(torch.allclose(candidate, reference, atol=probe.ATOL, rtol=probe.RTOL))
        candidate.retain_grad()
        with self.assertRaisesRegex(RuntimeError, 'RGB geometry/parity'):
            probe.rgb_backward(reference, candidate, candidate, self.e, torch, expected_shape=(1,1,128,3))
        self.assertEqual(self.e.data['actual']['backward_attempts'], 0)

    def test_session_entry_failure_and_bad_restoration_not_hidden(self):
        original = object()
        model = SimpleNamespace(current_patcher=original)
        failed_session = SimpleNamespace(active=False)  # _validate_resident failed before saved attrs
        row = probe.restore_patcher_state(model, failed_session, original, (False, None))
        self.assertFalse(row['session_state_initialized'])
        self.assertIsNone(row['session_shapes_restored'])
        self.assertTrue(row['whole_run_shapes_restored'])
        entry = object()
        session = SimpleNamespace(saved_patcher=entry, had_shapes=True, saved_shapes=[(1,2)])
        model.current_patcher = object()
        model.latent_shapes = [(3,4)]
        row = probe.restore_patcher_state(model, session, original, (False, None))
        self.assertFalse(row['session_current_patcher_restored'])
        self.assertFalse(row['session_shapes_restored'])
        self.assertTrue(row['whole_run_current_patcher_restored'])
        self.assertTrue(row['whole_run_shapes_restored'])

    def test_parameter_and_nonpersistent_buffer_identity_version_bytes(self):
        model = torch.nn.Linear(2, 2).requires_grad_(False)
        model.register_buffer('temporary_buffer', torch.ones(2), persistent=False)
        snap = probe.freeze_snapshot(model, torch)
        inv = probe.frozen_inventory(model, torch)
        self.assertEqual(len(inv['registered_buffers']), 1)
        self.assertTrue(probe.check_snapshot(model, snap, torch, device='cpu')['passed'])
        model.temporary_buffer.add_(1)
        self.assertFalse(probe.check_snapshot(model, snap, torch)['passed'])
        self.assertNotEqual(inv['registered_buffers_sha256'], probe.frozen_inventory(model, torch)['registered_buffers_sha256'])

    def test_missing_gradient_is_failure_zero_A_allowed(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_a = torch.nn.Parameter(torch.ones(2))
                self.lora_b = torch.nn.Parameter(torch.zeros(2))
        model = torch.nn.Module()
        model.block = Block()
        model.block.lora_b.grad = torch.ones(2)
        video = (model.block.lora_b * 2).sum().reshape(1)
        video.retain_grad()
        video.backward()
        probe.save_gradients(model, video, self.e, torch, expected=2)
        self.assertFalse(self.e.data['adapter_gradients']['passed'])
        model.block.lora_a.grad = torch.zeros(2)
        probe.save_gradients(model, video, self.e, torch, expected=2)
        self.assertTrue(self.e.data['adapter_gradients']['passed'])
        self.assertEqual(self.e.data['adapter_gradients']['B_nonzero_tensors'], 1)

    def test_guard_preserves_native_tiling_and_restores_on_error(self):
        class Stage:
            def tiled_decode(self, x): return x + 1
            def decode(self, x): return self.tiled_decode(x)
        class Vae:
            def __init__(self): self.first_stage_model = Stage()
            def decode(self, x): return self.first_stage_model.decode(x)
            def encode(self, x): return x
            def decode_tiled_3d(self, x): return x
            def _decode_tiled_owned(self, x): return x
        h3 = SimpleNamespace(vae=Vae())
        methods = [(h3.vae, 'decode'), (h3.vae, 'encode'), (h3.vae, 'decode_tiled_3d'),
                   (h3.vae, '_decode_tiled_owned'), (h3.vae.first_stage_model, 'decode')]
        snap = probe.wrapper_snapshot(methods)
        with self.assertRaisesRegex(RuntimeError, 'second VAE decode'):
            with probe.guarded_vae(h3, self.e, torch):
                self.e.phase('reference_decode')
                self.assertTrue(torch.equal(h3.vae.decode(torch.zeros(2)), torch.ones(2)))
                h3.vae.decode(torch.zeros(2))
        self.assertTrue(probe.wrappers_restored(snap))
        with probe.guarded_vae(h3, self.e, torch):
            with self.assertRaisesRegex(RuntimeError, 'OOM fallback'):
                h3.vae.decode_tiled_3d(torch.zeros(2))
            with self.assertRaisesRegex(RuntimeError, 'OOM fallback'):
                h3.vae._decode_tiled_owned(torch.zeros(2))
        self.assertTrue(probe.wrappers_restored(snap))


@unittest.skipUnless(torch is not None and os.environ.get('OPENH3_COMFY_REFERENCE_ROOT'),
                     'requires Torch CPU and pinned Comfy sources for AST fixture')
class JointPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e = probe.Evidence(Path(self.tmp.name) / 'joint', {'test_fixture_not_real_H3': True})
        self.h3, self.backend, self.state = fixture()
        self.dit = self.h3.patcher.model.diffusion_model
        with torch.no_grad(): self.dit.blocks[0].lora_b.zero_()
        self.math = {'kind': 'dit', 'contexts': 0, 'restores': 0}
        original = self.dit.blocks[0].forward
        def dit_forward(*a, **kw):
            if self.math['kind'] != 'dit': raise AssertionError('VAE math leaked to DiT')
            return original(*a, **kw)
        self.dit.blocks[0].forward = dit_forward
        runtime = self.math
        class VBlock(torch.nn.Module):
            def forward(self, x):
                if runtime['kind'] != 'vae': raise AssertionError('VAE checkpoint math missing')
                return x + x.sin() * .001
        class Stage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Parameter(torch.tensor(.1), requires_grad=False)
                self.register_buffer('buffer', torch.tensor(.001), persistent=False)
                self.decoder = torch.nn.Module()
                self.decoder.transformer_blocks = torch.nn.ModuleList([VBlock() for _ in range(36)])
            def decode(self, value):
                for block in self.decoder.transformer_blocks: value = block(value)
                # Native-like cast + differentiable returned-pixel operation.
                return value.float().mean().reshape(1,1,1,1).expand(1,1,128,3).sigmoid()
        self.stage = Stage().eval()
        self.h3.vae = SimpleNamespace(first_stage_model=self.stage, device='cpu', offload_device='cpu',
            disable_offload=True, vae_dtype=torch.float16, decode=self.stage.decode)
        self.guide = torch.full((1,24,2,2,2), .2)
        self.raw = torch.full_like(self.guide, .4)
    def tearDown(self): self.tmp.cleanup()

    @contextmanager
    def decoder_context(self):
        prior = self.math['kind']
        self.math['kind'] = 'vae'
        self.math['contexts'] += 1
        try: yield
        finally:
            self.math['kind'] = prior
            self.math['restores'] += 1

    def session(self, checkpoint):
        return trajectory.ResidentH3Trajectory(self.h3, _backend=self.backend,
            checkpoint_blocks=checkpoint, checkpoint_decoder=checkpoint,
            decoder_context=self.decoder_context)

    def observe(self, group, phase, index):
        self.assertTrue(self.state['math'])
        self.assertEqual(self.math['kind'], 'vae' if group == 'VAE' else 'dit')

    def test_real_torch_same_graph_two_decodes_single_RGB_backward(self):
        original = probe.wrapper_snapshot([(self.dit.blocks[0], 'forward')] +
                    [(b, 'forward') for b in self.stage.decoder.transformer_blocks])
        baseline = probe.freeze_snapshot(self.stage, torch)
        adapters = probe.adapter_snapshot(self.dit, torch)
        with probe.counted_blocks(self.dit.blocks, 'DiT', self.e, self.observe), \
             probe.counted_blocks(self.stage.decoder.transformer_blocks, 'VAE', self.e, self.observe):
            self.e.phase('reference')
            with torch.no_grad(), self.session(False) as session:
                reference = session.sample(self.guide, self.raw)
                self.e.phase('reference_decode')
                reference_rgb = session.decode(reference['video'])
            with self.session(True) as session:
                self.e.phase('candidate')
                candidate = session.sample(self.guide, self.raw)
                probe.save_latent_parity(reference, candidate, self.e, torch)
                candidate['video'].retain_grad()
                self.e.phase('candidate_decode')
                pixels = session.decode(candidate['video'])
                self.assertEqual(self.math['kind'], 'dit')
                probe.rgb_backward(reference_rgb, pixels, candidate['video'], self.e, torch,
                                   expected_shape=(1,1,128,3))
                probe.save_gradients(self.dit, candidate['video'], self.e, torch, expected=2)
                self.assertTrue(self.e.data['adapter_gradients']['passed'])
                self.assertTrue(self.e.data['retained_video_gradient']['passed'])
                self.assertEqual(self.math['kind'], 'dit')
            with self.assertRaises(RuntimeError): session._guard_backward(torch.zeros(()), session.generation)
        self.assertTrue(probe.wrappers_restored(original))
        self.assertFalse(self.state['math'])
        self.assertEqual(self.math['contexts'], self.math['restores'])
        self.assertTrue(probe.check_snapshot(self.stage, baseline, torch)['passed'])
        self.assertTrue(probe.check_adapters(self.dit, adapters, torch, 'cpu')['passed'])
        self.assertEqual(self.e.data['actual']['backward_attempts'], 1)
        self.assertEqual(self.e.data['actual']['optimizer_steps'], 0)
        self.assertEqual(self.e.data['actual']['adapter_updates'], 0)
        for phase in ('reference', 'candidate', 'backward'):
            self.assertEqual(self.e.data['block_forward_calls']['DiT'][phase], {'0':3})
        for phase in ('reference_decode', 'candidate_decode', 'backward'):
            self.assertEqual(self.e.data['block_forward_calls']['VAE'][phase], {str(i):1 for i in range(36)})
        import numpy as np
        role = self.e.data['arrays']
        raw = np.load(self.e.path / role['video_gradient_raw']['path'])
        unscaled = np.load(self.e.path / role['video_gradient_unscaled']['path'])
        np.testing.assert_array_equal(raw / 1024., unscaled)
        self.assertEqual(self.e.data['loss']['scale'], 1024.)
        self.assertFalse(self.e.data['loss']['auxiliary_latent_audio_loss'])

    def test_exception_restores_checkpoint_math_and_patcher(self):
        original = probe.wrapper_snapshot([(self.dit.blocks[0], 'forward')] +
                    [(b, 'forward') for b in self.stage.decoder.transformer_blocks])
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            with probe.counted_blocks(self.dit.blocks, 'DiT', self.e, self.observe), \
                 probe.counted_blocks(self.stage.decoder.transformer_blocks, 'VAE', self.e, self.observe):
                with self.session(True) as session:
                    self.e.phase('candidate')
                    candidate = session.sample(self.guide, self.raw)
                    self.e.phase('candidate_decode')
                    session.decode(candidate['video'])
                    raise RuntimeError('injected failure')
        self.assertTrue(probe.wrappers_restored(original))
        self.assertEqual(self.math['kind'], 'dit')
        self.assertFalse(self.state['math'])
        self.assertIsNone(self.h3.patcher.model.current_patcher)
        self.assertFalse(session.active)
        self.assertEqual(self.e.data['actual']['backward_attempts'], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
