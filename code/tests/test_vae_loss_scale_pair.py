"""Torch CPU tests of retained-graph paired scaling; no VAE or NPU import."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vae_checkpoint_context import checkpoint_decoder_blocks


class TinyBlock(torch.nn.Module):
    def __init__(self,index):
        super().__init__()
        self.linear=torch.nn.Linear(4,4,dtype=torch.float64)
        self.scale=.01+index*.0001

    def forward(self,x):
        return x+torch.tanh(self.linear(x))*self.scale


class LossScalePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_retained_graph_36_blocks_twice_backward_exact_unscale(self):
        torch.manual_seed(101)
        decoder=torch.nn.Module()
        decoder.transformer_blocks=torch.nn.ModuleList(TinyBlock(i) for i in range(36))
        decoder.requires_grad_(False)
        stage=SimpleNamespace(decoder=decoder)
        weights={k:v.clone() for k,v in decoder.state_dict().items()}
        versions=[p._version for p in decoder.parameters()]
        x=torch.linspace(-1,1,12,dtype=torch.float64).reshape(3,4).requires_grad_()
        with checkpoint_decoder_blocks(stage,True) as state:
            y=x
            for block in decoder.transformer_blocks:y=block(y)
            loss=(y-.25).square().mean()
            loss.backward(retain_graph=True)
            first=x.grad.detach().clone()
            self.assertTrue(torch.isfinite(first).all())
            self.assertGreater(torch.count_nonzero(first),0)
            x.grad=None
            (loss*1024).backward()
            scaled=x.grad.detach().clone()
            self.assertTrue(torch.equal(scaled/1024,first))
            self.assertEqual(state['checkpoint_forward_calls'],36)
            self.assertEqual(state['recompute_calls'],72)
            self.assertEqual(state['backward_guard_calls'],72)
            self.assertEqual(state['output_gradient_guards'],36)
        self.assertTrue(state['restored'])
        self.assertEqual(state['recomputed_block_indices'],list(range(36)))
        self.assertTrue(all(torch.equal(weights[k],v) for k,v in decoder.state_dict().items()))
        self.assertEqual(versions,[p._version for p in decoder.parameters()])
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in decoder.parameters()))
        self.assertTrue(all('forward' not in b.__dict__ for b in decoder.transformer_blocks))

    def test_missing_clear_would_accumulate_is_not_the_tested_protocol(self):
        x=torch.tensor([.5],dtype=torch.float64,requires_grad=True)
        loss=x.square().mean();loss.backward(retain_graph=True)
        first=x.grad.clone();(loss*1024).backward()
        self.assertTrue(torch.equal(x.grad,first*1025))
        self.assertFalse(torch.equal(x.grad/1024,first))

    def test_half_gradient_cast_underflow_recovered_but_not_exact_real_gradient(self):
        x=torch.ones(4,dtype=torch.float32,requires_grad=True)
        half=x.to(torch.float16)
        loss=half.float().square().mean()*1e-10
        loss.backward(retain_graph=True)
        first=x.grad.clone()
        self.assertEqual(torch.count_nonzero(first),0)
        x.grad=None
        (loss*1024).backward()
        raw=x.grad.clone()
        self.assertTrue(torch.isfinite(raw).all())
        self.assertGreater(torch.count_nonzero(raw),0)
        unscaled=raw.float()/1024
        self.assertGreater(torch.count_nonzero(unscaled),0)
        exact=torch.full_like(unscaled,2/4*1e-10)
        # Recovering nonzero arithmetic is not proof of exact mathematical
        # gradients; the FP16 gradient cast still quantizes this toy example.
        self.assertFalse(torch.equal(unscaled,exact))


if __name__=='__main__':unittest.main()
