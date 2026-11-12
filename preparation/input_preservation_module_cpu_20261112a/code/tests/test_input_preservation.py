"""Small CPU fixtures; no dataset/checkpoint assets, NPU or optimization."""
import copy
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_input_preservation import (InputPreservationHead, compose_input_preservation,
                                     input_benefit_labels, normalize_input_benefit)
from rphi_history_utility import HistoryUtilityGate, MaskAwareRestorer, UTILITY_SCALE
from rphi_model import RphiModel


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8), b.detach().cpu().contiguous().view(torch.uint8))


class InputPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = torch.Generator().manual_seed(801)
        self.x = torch.rand(2, 3, 3, 32, 48, generator=self.rng)
        self.a = torch.tensor([[1., 1.], [0., 1.]])

    def active_head(self):
        head = InputPreservationHead(19)
        with torch.no_grad():
            head.gate.net[-1].weight.copy_(torch.randn(head.gate.net[-1].weight.shape, generator=self.rng) * .1)
        return head

    def test_exact_inherited_parameters_seed_and_zero_identity(self):
        before = torch.random.get_rng_state().clone()
        head = InputPreservationHead(17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        reference = HistoryUtilityGate(17)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 1745)
        for k, v in reference.state_dict().items(): self.assertTrue(byte_equal(v, head.gate.state_dict()[k]))
        score = head(self.x, self.a)
        self.assertEqual(tuple(score['benefit_score'].shape), (2, 1, 2, 3))
        self.assertEqual(int(torch.count_nonzero(score['benefit_score'])), 0)
        self.assertFalse(bool(score['correction_mask'].any()))
        actual = compose_input_preservation(self.x[:, -1], torch.full_like(self.x[:, -1], 2.), score['benefit_score'])
        self.assertTrue(byte_equal(actual['rgb'], self.x[:, -1]))

    def test_shared_slot_mean_and_cached_features_are_exact(self):
        head = self.active_head(); features = head.features(self.x, self.a)
        old = head.gate.features(self.x, self.a)
        self.assertTrue(byte_equal(features.reshape_as(old), old))
        expected = head.gate.net(old).reshape(2, 2, 2, 3).mean(1, keepdim=True)
        self.assertTrue(byte_equal(expected, head(self.x, self.a)['benefit_score']))
        self.assertTrue(byte_equal(expected, head.forward_features(features)['benefit_score']))

    def test_unavailable_past_ignored_and_both_absent_still_decides(self):
        head = self.active_head(); a = torch.tensor([[0., 1.], [0., 0.]])
        changed = self.x.clone()
        for b in range(2):
            for s in range(2):
                if a[b, s] == 0: changed[b, s] = 1 - changed[b, s]
        self.assertTrue(byte_equal(head(self.x, a)['benefit_score'], head(changed, a)['benefit_score']))
        with torch.no_grad():
            for p in head.parameters(): p.zero_()
            head.gate.net[0].weight[0, 0, 1, 1] = 1
            head.gate.net[-1].weight[0, 0, 1, 1] = 1
        bright = torch.full_like(self.x, .75); dark = torch.full_like(self.x, .25)
        missing = torch.zeros(2, 2)
        self.assertTrue(bool(head(bright, missing)['correction_mask'].all()))
        self.assertFalse(byte_equal(head(bright, missing)['benefit_score'], head(dark, missing)['benefit_score']))

    def test_raw_only_signatures_and_strict_reload(self):
        self.assertEqual(list(inspect.signature(InputPreservationHead.forward).parameters), ['self', 'observed', 'availability'])
        head = self.active_head()
        for extra in ({'target': self.x[:, -1]}, {'metadata': {'source': 'x'}}):
            with self.assertRaises(TypeError): head(self.x, self.a, **extra)
        saved = copy.deepcopy(head.state_dict()); other = InputPreservationHead(111)
        other.load_state_dict(saved, strict=True)
        self.assertTrue(byte_equal(head(self.x, self.a)['benefit_score'], other(self.x, self.a)['benefit_score']))

    def test_signed_zero_ties_and_unclamped_candidate_bytes(self):
        current = self.x[:, -1].clone(); current[0, 0, 0, 0] = -0.
        k1 = torch.full_like(current, 1.25); k1[1, 0, 16, 0] = -.25
        score = torch.tensor([[[[0., -0., 1.], [-1., 1., 0.]]], [[[1., 0., 0.], [1., -0., 1.]]]])
        actual = compose_input_preservation(current, k1, score)
        expected = current.clone()
        for b in range(2):
            for y in range(2):
                for x in range(3):
                    if score[b, 0, y, x] > 0: expected[b, :, y*16:(y+1)*16, x*16:(x+1)*16] = k1[b, :, y*16:(y+1)*16, x*16:(x+1)*16]
        self.assertTrue(byte_equal(actual['rgb'], expected))
        self.assertTrue(byte_equal(actual['rgb_clipped'], expected.clamp(0, 1)))
        self.assertTrue(torch.signbit(actual['rgb'][0, 0, 0, 0]))
        self.assertEqual(float(actual['rgb'][0, 0, 0, 32]), 1.25)
        self.assertEqual(float(actual['rgb_clipped'][0, 0, 0, 32]), 1.)

    def test_random_masks_preserve_each_selected_FP32_candidate(self):
        current = self.x[:, -1]; k1 = torch.randn(current.shape, generator=self.rng)
        score = torch.randn(2, 1, 2, 3, generator=self.rng)
        result = compose_input_preservation(current, k1, score)
        m = result['pixel_mask'].expand_as(current)
        self.assertTrue(byte_equal(result['rgb'][~m], current[~m]))
        self.assertTrue(byte_equal(result['rgb'][m], k1[m]))

    def test_reject_nonfinite_dtype_geometry_and_availability(self):
        head = InputPreservationHead(); current = self.x[:, -1]; z = torch.zeros(2, 1, 2, 3)
        for value in (float('nan'), float('inf'), -float('inf')):
            bad = self.x.clone(); bad[0, 0, 0, 0, 0] = value
            with self.assertRaises(ValueError): head(bad, torch.zeros_like(self.a))
            k = current.clone(); k[0, 0, 0, 0] = value
            with self.assertRaises(ValueError): compose_input_preservation(current, k, z)
            s = z.clone(); s[0, 0, 0, 0] = value
            with self.assertRaises(ValueError): compose_input_preservation(current, current, s)
        for obs, availability in ((self.x.half(), self.a), (self.x, self.a * .5),
                                  (self.x[..., :31, :], self.a), (self.x, self.a.bool())):
            with self.assertRaises(ValueError): head(obs, availability)
        with self.assertRaises(ValueError): compose_input_preservation(current, current.half(), z)
        with self.assertRaises(ValueError): head.forward_features(torch.full((2, 2, 11, 2, 3), float('nan')))
        with torch.no_grad(): head.gate.net[-1].bias.fill_(float('nan'))
        with self.assertRaises(ValueError): head(self.x, self.a)

    def test_FP64_benefit_matches_independent_tile_loops_and_signs(self):
        rng = np.random.default_rng(2026)
        current = rng.random((2, 3, 32, 48), dtype=np.float32)
        target = rng.random(current.shape, dtype=np.float32)
        k1 = rng.normal(.5, .4, current.shape).astype(np.float32)
        result = input_benefit_labels(current, k1, target)
        for b in range(2):
            for y in range(2):
                for x in range(3):
                    box = (b, slice(None), slice(y*16,(y+1)*16), slice(x*16,(x+1)*16))
                    ei = np.abs(current[box].astype(np.float64)-target[box].astype(np.float64)).mean()
                    ek = np.abs(k1[box].astype(np.float64)-target[box].astype(np.float64)).mean()
                    self.assertAlmostEqual(result['raw_benefit'][b,0,y,x], ei-ek, places=14)
        self.assertEqual(result['raw_benefit'].dtype, np.float64)
        self.assertEqual(result['normalized_benefit'].dtype, np.float32)
        zero = np.zeros((1,3,16,16),np.float32); one = np.ones_like(zero)
        self.assertEqual(float(input_benefit_labels(one,zero,zero)['normalized_benefit'].item()), 4.)
        self.assertEqual(float(input_benefit_labels(zero,one,zero)['normalized_benefit'].item()), -4.)
        self.assertEqual(float(input_benefit_labels(zero,zero,zero)['raw_benefit'].item()), 0.)
        with self.assertRaises(ValueError): input_benefit_labels(zero.astype(np.float64),zero,zero)
        with self.assertRaises(ValueError): normalize_input_benefit(np.full((1,1,1,1),np.nan,np.float64))

    def test_K1_zero_weights_invariance_with_nontrivial_tiny_restorer(self):
        torch.manual_seed(7)
        old = RphiModel(variant='blind', ch=4, blocks=1, observation_encoder='legacy')
        with torch.no_grad(): old.backbone.tail.weight.normal_(std=.01)
        model = MaskAwareRestorer.from_rphi(old.model_config, old.state_dict()).eval().requires_grad_(False)
        changed = self.x.clone(); changed[:, :2] = 1-changed[:, :2]
        zero = torch.zeros(2,2,32,48)
        before = {n: p._version for n,p in model.named_parameters()}
        with torch.inference_mode():
            a = model(self.x,zero)['rgb']; b = model(changed,zero)['rgb']
        self.assertTrue(byte_equal(a,b))
        self.assertFalse(byte_equal(a,self.x[:,-1]))
        self.assertEqual(before,{n:p._version for n,p in model.named_parameters()})
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main(verbosity=2)
