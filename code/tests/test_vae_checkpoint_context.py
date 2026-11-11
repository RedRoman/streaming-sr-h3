"""Actual Torch CPU regression for 36-block scoped VAE checkpointing."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_checkpoint_context import checkpoint_decoder_blocks


class Block(torch.nn.Module):
    def __init__(self, index, dropout=False):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4, dtype=torch.float64)
        self.scale = .01 + index * .0001
        self.dropout = torch.nn.Dropout(.2) if dropout else torch.nn.Identity()
        self.fail = False

    def forward(self, x, *, gain=1.):
        if self.fail:
            raise RuntimeError("intentional block failure")
        return x + self.dropout(torch.tanh(self.layer(x))) * self.scale * gain


class Decoder(torch.nn.Module):
    def __init__(self, count=36, dropout=False):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(Block(i, dropout) for i in range(count))

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x, gain=1.25)
        return x


def stage(count=36, dropout=False):
    torch.manual_seed(17)
    decoder = Decoder(count, dropout)
    decoder.requires_grad_(False)
    return SimpleNamespace(decoder=decoder)


def attribute_state(decoder):
    return [("forward" in b.__dict__, b.__dict__.get("forward")) for b in decoder.transformer_blocks]


class CheckpointContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_36_blocks_exact_output_gradient_frozen_weights_and_attributes(self):
        s = stage()
        before = {k:v.clone() for k,v in s.decoder.state_dict().items()}
        versions = [p._version for p in s.decoder.parameters()]
        attributes = attribute_state(s.decoder)
        x = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(3,4).requires_grad_()
        y = s.decoder(x)
        y.square().sum().backward()
        expected, gradient = y.detach().clone(), x.grad.clone()
        xx = x.detach().clone().requires_grad_()
        with checkpoint_decoder_blocks(s, enabled=True) as state:
            actual = s.decoder(xx)
            self.assertTrue(torch.equal(actual, expected))
            actual.square().sum().backward()
            self.assertTrue(torch.equal(xx.grad, gradient))
            self.assertEqual(state["block_count"], 36)
            self.assertEqual(state["checkpoint_forward_calls"], 36)
            self.assertEqual(state["recompute_calls"], 36)
            self.assertEqual(state["backward_guard_calls"], 36)
            json.dumps(state)
        self.assertFalse(state["active"])
        self.assertTrue(state["restored"])
        self.assertEqual(state["recomputed_block_indices"], list(range(36)))
        self.assertEqual(attributes, attribute_state(s.decoder))
        self.assertEqual(versions, [p._version for p in s.decoder.parameters()])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in s.decoder.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in s.decoder.state_dict().items()))

    def test_rng_preserved_with_dropout(self):
        s = stage(dropout=True)
        x = torch.ones(2,4,dtype=torch.float64,requires_grad=True)
        torch.manual_seed(123)
        expected = s.decoder(x)
        expected.sum().backward()
        expected_rng = torch.get_rng_state().clone()
        xx = x.detach().clone().requires_grad_()
        torch.manual_seed(123)
        with checkpoint_decoder_blocks(s, True):
            actual = s.decoder(xx)
            actual.sum().backward()
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(xx.grad,x.grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_forward_exception_restores_every_attribute(self):
        s = stage()
        attrs = attribute_state(s.decoder)
        s.decoder.transformer_blocks[17].fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with checkpoint_decoder_blocks(s, True) as state:
                s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True))
        self.assertTrue(state["restored"])
        self.assertEqual(attrs, attribute_state(s.decoder))

    def test_recompute_exception_restores_and_scope_can_reenter(self):
        s = stage()
        attrs = attribute_state(s.decoder)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with checkpoint_decoder_blocks(s, True) as state:
                out = s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True))
                s.decoder.transformer_blocks[35].fail = True
                out.sum().backward()
        self.assertTrue(state["restored"])
        self.assertEqual(attrs, attribute_state(s.decoder))
        s.decoder.transformer_blocks[35].fail = False
        with checkpoint_decoder_blocks(s, True) as retry:
            s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True)).sum().backward()
        self.assertTrue(retry["restored"])

    def test_backward_outside_scope_rejected(self):
        s = stage()
        with checkpoint_decoder_blocks(s, True):
            y = s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "exited before backward"):
            y.sum().backward()

    def test_backward_without_recomputation_still_rejected(self):
        s = SimpleNamespace(decoder=SimpleNamespace(transformer_blocks=torch.nn.ModuleList(torch.nn.Identity() for _ in range(36))))
        x = torch.ones(2,4,requires_grad=True)
        with checkpoint_decoder_blocks(s, True):
            y = s.decoder.transformer_blocks[0](x)
        with self.assertRaisesRegex(RuntimeError, "exited before backward"):
            y.sum().backward()

    def test_reenter_cannot_reactivate_old_output(self):
        s = stage()
        with checkpoint_decoder_blocks(s, True):
            y = s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True))
        with checkpoint_decoder_blocks(s, True):
            with self.assertRaisesRegex(RuntimeError, "exited before backward"):
                y.sum().backward()

    def test_no_grad_bypasses_checkpoint(self):
        s = stage()
        x = torch.ones(2,4,dtype=torch.float64)
        with torch.no_grad():
            expected = s.decoder(x)
            with patch("torch.utils.checkpoint.checkpoint", side_effect=AssertionError("must bypass")):
                with checkpoint_decoder_blocks(s, True) as state:
                    actual = s.decoder(x)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(state["no_grad_forward_calls"], 36)
        self.assertEqual(state["checkpoint_forward_calls"], 0)
        self.assertEqual(state["output_gradient_guards"], 0)

    def test_disabled_does_not_inspect_or_patch_stage(self):
        class Poison:
            def __getattr__(self, key):
                raise AssertionError("disabled stage touched")
        with checkpoint_decoder_blocks(Poison()) as state:
            self.assertFalse(state["enabled"])
            self.assertEqual(state["wrapped"], 0)
        s = stage()
        attrs = attribute_state(s.decoder)
        with checkpoint_decoder_blocks(s, False):
            self.assertEqual(attrs, attribute_state(s.decoder))
        self.assertEqual(attrs, attribute_state(s.decoder))

    def test_nested_and_shared_scopes_rejected_without_disturbing_outer(self):
        s = stage()
        alias = SimpleNamespace(decoder=s.decoder)
        attrs = attribute_state(s.decoder)
        with checkpoint_decoder_blocks(s, True) as outer:
            wrapped = attribute_state(s.decoder)
            for target in (s, alias):
                with self.assertRaisesRegex(RuntimeError, "nested or overlapping"):
                    with checkpoint_decoder_blocks(target, True):
                        pass
            self.assertTrue(outer["active"])
            self.assertEqual(wrapped, attribute_state(s.decoder))
            s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True)).sum().backward()
        self.assertEqual(attrs, attribute_state(s.decoder))

    def test_exact_count_and_no_duplicate_blocks(self):
        for count in (0,35,37):
            with self.assertRaisesRegex(ValueError, "exactly 36"):
                with checkpoint_decoder_blocks(stage(count), True):
                    pass
        s = stage()
        s.decoder.transformer_blocks[1] = s.decoder.transformer_blocks[0]
        with self.assertRaisesRegex(ValueError, "distinct"):
            with checkpoint_decoder_blocks(s, True):
                pass

    def test_existing_instance_forward_restored_by_identity(self):
        s = stage()
        b = s.decoder.transformer_blocks[0]
        prior = b.forward
        custom = lambda *args,**kwargs: prior(*args,**kwargs)
        b.forward = custom
        with checkpoint_decoder_blocks(s, True):
            s.decoder(torch.ones(2,4,dtype=torch.float64,requires_grad=True)).sum().backward()
        self.assertIs(b.__dict__["forward"], custom)


if __name__ == "__main__":
    unittest.main()
