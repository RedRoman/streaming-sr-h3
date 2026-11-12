"""Meaningful CPU-only P1 contract tests. No teacher/student training or NPU."""
import copy
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rphi_model import RphiModel
from rphi_history_utility import (MaskAwareRestorer, HistoryUtilityGate, HistoryUtilityStudent,
    causal_indices, apply_external_mask, fill_history, remove_history, tile_mae,
    utility_targets, utility_loss, matched_losses, UTILITY_SCALE, GATE_PARAMETERS)
from train_rphi import fidelity_losses


class HistoryUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(410)
        self.observed=torch.rand(2,3,3,32,48)
        self.available=torch.ones(2,2)

    def inherited(self, encoder='legacy', ch=4):
        torch.manual_seed(123)
        model=RphiModel(variant='blind',ch=ch,blocks=1,observation_encoder=encoder)
        # Avoid the trivial identity-tail test; exercise all inherited layers.
        with torch.no_grad():
            model.backbone.tail.weight.normal_(std=.015)
            model.backbone.tail.bias.normal_(std=.01)
            model.backbone.blocks[0].film.weight.normal_(std=.01)
        return model

    def restorer(self, encoder='legacy', ch=4):
        old=self.inherited(encoder,ch)
        return MaskAwareRestorer.from_rphi(old.model_config,old.state_dict())

    def test_all_one_head_expansion_preserves_real_legacy_and_native_paths(self):
        # Explicit CPU backend tolerance; zero-added channels can change kernel
        # accumulation order. This is not a claim of accelerator bit-equality.
        for encoder in ('legacy','native'):
            old=self.inherited(encoder);state=copy.deepcopy(old.state_dict())
            new=MaskAwareRestorer.from_rphi(old.model_config,state)
            w=torch.ones(2,2,32,48);prior=old(self.observed);actual=new(self.observed,w)
            torch.testing.assert_close(actual['rgb'],prior['rgb'],rtol=1e-6,atol=1e-6)
            self.assertTrue(torch.equal(actual['condition_code'],prior['condition_code']))
            self.assertEqual(sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters()),2*4*3*3)
            for name,value in new.state_dict().items():
                if name=='backbone.head.weight':
                    self.assertTrue(torch.equal(value[:,:9],state[name]));self.assertEqual(int(torch.count_nonzero(value[:,9:])),0)
                else:self.assertTrue(torch.equal(value,state[name]),name)
                self.assertNotEqual(value.data_ptr(),state[name].data_ptr())
            self.assertEqual(new.backbone.head.in_channels,11)

    def test_strict_warmstart_rejects_unknown_config_keys_shapes_dtype_finiteness(self):
        old=self.inherited();state=old.state_dict()
        cases=[]
        bad=copy.deepcopy(state);bad.pop(next(iter(bad)));cases.append(bad)
        bad=copy.deepcopy(state);bad['unknown']=torch.zeros(1);cases.append(bad)
        bad=copy.deepcopy(state);bad['backbone.head.weight']=bad['backbone.head.weight'][:,:8];cases.append(bad)
        bad=copy.deepcopy(state);bad['backbone.head.weight']=bad['backbone.head.weight'].half();cases.append(bad)
        bad=copy.deepcopy(state);bad['backbone.tail.bias'][0]=float('nan');cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):MaskAwareRestorer.from_rphi(old.model_config,bad)
        for key,value in [('variant','known'),('variant','blind_confidence')]:
            with self.assertRaises(ValueError):MaskAwareRestorer.from_rphi({**old.model_config,key:value},state)
        missing=dict(old.model_config);missing.pop('observation_encoder')
        with self.assertRaises(ValueError):MaskAwareRestorer.from_rphi(missing,state)

    def test_missing_is_current_plus_explicit_zero_not_black_or_real_padding(self):
        weights=torch.zeros(2,2,32,48)
        changed=self.observed.clone();changed[:,:2]=float('nan')
        filled=fill_history(changed,weights)
        self.assertTrue(torch.equal(filled,self.observed[:,-1:].expand(-1,3,-1,-1,-1)))
        restorer=self.restorer();a=restorer(self.observed,weights);b=restorer(changed,weights)
        self.assertTrue(torch.equal(a['rgb'],b['rgb']))
        # An explicit mask remains visible even when the pixel stack is identical.
        observed=self.observed[:,-1:].expand(-1,3,-1,-1,-1).clone()
        with torch.no_grad():restorer.backbone.head.weight[:,9:].fill_(.2)
        self.assertFalse(torch.equal(restorer(observed,weights)['rgb'],restorer(observed,torch.ones_like(weights))['rgb']))

    def test_causal_boundaries_and_external_masks_never_create_history(self):
        indices,true=causal_indices([0,1,2,95])
        self.assertEqual(indices.tolist(),[[0,0,0],[0,0,1],[0,1,2],[93,94,95]])
        self.assertEqual(true.tolist(),[[0,0],[0,1],[1,1],[1,1]])
        actual=apply_external_mask(true,torch.tensor([[1.,1.],[1.,0.],[0.,1.],[1.,1.]]))
        self.assertEqual(actual.tolist(),[[0,0],[0,0],[0,1],[1,1]])
        for invalid in ([-1],[1.5],[True],[]):
            with self.assertRaises(ValueError):causal_indices(invalid)
        with self.assertRaises(ValueError):apply_external_mask(true,torch.ones_like(true)*.5)

    def test_gate_1745_shared_features_and_initial_half_weights(self):
        gate=HistoryUtilityGate(42)
        self.assertEqual(sum(p.numel() for p in gate.parameters()),GATE_PARAMETERS)
        availability=torch.tensor([[0.,1.],[1.,1.]])
        features=gate.features(self.observed,availability)
        self.assertEqual(tuple(features.shape),(4,11,2,3))
        pooled=F.avg_pool2d(self.observed[0,-1:],16,16)
        self.assertTrue(torch.equal(features[0,:3],pooled[0]))
        self.assertTrue(torch.equal(features[0,3:6],pooled[0]))
        self.assertEqual(float(features[0,6:9].abs().sum()),0)
        self.assertTrue(torch.equal(features[:,9,0,0],torch.tensor([1.,.5,1.,.5])))
        self.assertTrue(torch.equal(features[:,10,0,0],availability.flatten()))
        actual=gate(self.observed,availability)
        self.assertTrue(torch.equal(actual['g'],torch.full_like(actual['g'],.5)))
        self.assertTrue(torch.equal(actual['weights'],availability[:,:,None,None].expand(2,2,32,48)*.5))

    def test_gate_is_slot_separable_and_ignores_removed_pixels(self):
        gate=HistoryUtilityGate(10)
        with torch.no_grad():gate.net[-1].weight.normal_(std=.1)
        original=gate(self.observed,self.available)
        mask=self.available.clone();mask[:,0]=0
        changed=self.observed.clone();changed[:,0]=float('nan')
        removed=gate(changed,mask)
        self.assertTrue(torch.equal(original['g'][:,1],removed['g'][:,1]))
        self.assertTrue(torch.equal(original['z'][:,1],removed['z'][:,1]))
        self.assertEqual(float(removed['weights'][:,0].abs().sum()),0)
        other=gate(torch.cat((torch.zeros_like(self.observed[:,:1]),self.observed[:,1:]),1),mask)
        self.assertTrue(torch.equal(removed['z'],other['z']))

    def test_gate_is_same_operator_on_crop_and_native_grid(self):
        gate=HistoryUtilityGate(3)
        for h,w,grid in ((96,96,(6,6)),(480,832,(30,52))):
            result=gate(torch.rand(1,3,3,h,w),torch.ones(1,2))
            self.assertEqual(tuple(result['z'].shape),(1,2,*grid));self.assertEqual(tuple(result['weights'].shape),(1,2,h,w))

    def test_three_student_parameter_initializations_match_without_aliases(self):
        teacher=self.restorer(ch=64).eval().requires_grad_(False)
        students=[HistoryUtilityStudent(teacher,20261111) for _ in range(3)]
        self.assertTrue(all(p.requires_grad for s in students for p in s.parameters()))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        teacher_count=sum(p.numel() for p in teacher.parameters())
        self.assertEqual(sum(p.numel() for p in students[0].parameters()),teacher_count+1745)
        for name,value in students[0].state_dict().items():
            for other in students[1:]:
                self.assertTrue(torch.equal(value,other.state_dict()[name]));self.assertNotEqual(value.data_ptr(),other.state_dict()[name].data_ptr())
        for name,value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value,students[0].restorer.state_dict()[name]));self.assertNotEqual(value.data_ptr(),students[0].restorer.state_dict()[name].data_ptr())
        half=copy.deepcopy(teacher).half()
        with self.assertRaises(ValueError):HistoryUtilityStudent(half,3)

    def test_gate_seed_is_local_and_not_a_sample_rng_reset(self):
        before=torch.random.get_rng_state().clone();a=HistoryUtilityGate(23)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b=HistoryUtilityGate(23);c=HistoryUtilityGate(24)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertFalse(torch.equal(a.net[0].weight,c.net[0].weight))

    def test_normal_call_budget_and_self_intervention_keeps_other_gate_fixed(self):
        student=HistoryUtilityStudent(self.restorer(),99);calls={'gate':0,'restorer':0}
        def count(key):
            def hook(*args):calls[key]+=1
            return hook
        hooks=[student.gate.register_forward_hook(count('gate')),student.restorer.register_forward_hook(count('restorer'))]
        result=student(self.observed,self.available)
        self.assertEqual(calls,{'gate':1,'restorer':1})
        removed=student.without_history(self.observed,result['weights'],0)
        self.assertEqual(calls,{'gate':1,'restorer':2})
        self.assertTrue(torch.equal(removed['weights'][:,1],result['weights'][:,1]))
        self.assertEqual(float(removed['weights'][:,0].abs().sum()),0)
        self.assertGreater(float(result['weights'][:,0].sum()),0)
        for h in hooks:h.remove()

    def test_soft_masks_have_finite_fidelity_gradients_and_output_is_unclamped(self):
        restorer=self.restorer();weights=torch.full((2,2,32,48),.4,requires_grad=True)
        with torch.no_grad():restorer.backbone.tail.bias.add_(2.)
        output=restorer(self.observed,weights)['rgb'];self.assertGreater(float(output.detach().max()),1.)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(weights.grad).all());self.assertGreater(float(weights.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in restorer.parameters()))

    def test_utility_sign_raw_unclamped_and_nonexistent_history_no_labels(self):
        baseline=torch.full((2,3,32,48),.001,requires_grad=True);target=torch.zeros_like(baseline,requires_grad=True)
        removed=torch.stack((torch.full_like(baseline,.003),torch.zeros_like(baseline)),1).requires_grad_()
        availability=torch.tensor([[1.,1.],[0.,0.]])
        with torch.no_grad():removed[1]=float('nan')
        labels=utility_targets(baseline,removed,target,availability)
        self.assertTrue((labels['raw_utility'][0,0]>0).all());self.assertTrue((labels['raw_utility'][0,1]<0).all())
        self.assertEqual(float(labels['raw_utility'][1].abs().sum()),0);self.assertFalse(labels['valid'][1].any())
        self.assertFalse(labels['normalized'].requires_grad)
        torch.testing.assert_close(labels['normalized'][0,0],torch.full((2,3),.002/UTILITY_SCALE))
        self.assertEqual(float(tile_mae(torch.full_like(baseline,2.),target).mean()),2.)

    def test_SmoothL1_available_tile_mean_empty_zero_and_GT_detach(self):
        z=torch.zeros(2,2,2,3,requires_grad=True);labels=torch.ones_like(z,requires_grad=True)
        valid=torch.tensor([[True,False],[False,False]])
        loss=utility_loss(z,labels,valid);self.assertEqual(float(loss),.5);loss.backward()
        self.assertIsNone(labels.grad);self.assertEqual(int(torch.count_nonzero(z.grad)),6)
        empty=torch.randn(2,2,2,3,requires_grad=True);zero=utility_loss(empty,labels,torch.zeros_like(valid))
        self.assertTrue(zero.requires_grad);self.assertEqual(float(zero),0.);zero.backward()
        self.assertTrue(torch.equal(empty.grad,torch.zeros_like(empty)))

    def test_matched_losses_reuse_fixed_fidelity_and_compute_zero_arm_aux_graph(self):
        student=HistoryUtilityStudent(self.restorer(),8);out=student(self.observed,self.available)
        target=torch.zeros_like(out['rgb']);labels=torch.ones_like(out['z']);valid=torch.ones(2,2,dtype=torch.bool)
        expected,parts=fidelity_losses(out['rgb'],target,self.observed[:,-1],.1,.5)
        generic,c=matched_losses(out,target,self.observed[:,-1],labels,valid,'generic_gate')
        utility,d=matched_losses(out,target,self.observed[:,-1],labels,valid,'utility_gate')
        shuffled,_=matched_losses(out,target,self.observed[:,-1],labels,valid,'shuffled_utility_gate')
        self.assertTrue(torch.equal(generic,expected));self.assertTrue(torch.equal(utility,expected+.01*c['utility']))
        self.assertTrue(torch.equal(utility,shuffled));self.assertTrue(c['utility'].requires_grad)
        for k in parts:self.assertTrue(torch.equal(c[k],parts[k]))
        utility.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        self.assertGreater(float(student.gate.net[-1].weight.grad.abs().sum()),0.)

    def test_reject_geometry_nonfloat_oracle_and_invalid_weights(self):
        student=HistoryUtilityStudent(self.restorer(),7)
        for observed in (self.observed.half(),self.observed[:,:,:,:,:36],self.observed[:,:2]):
            with self.assertRaises(ValueError):student(observed,self.available)
        with self.assertRaises(ValueError):student(self.observed,torch.ones(2,2)*.5)
        with self.assertRaises(TypeError):student(self.observed,self.available,target=self.observed[:,-1])
        with self.assertRaises(TypeError):student(self.observed,self.available,condition=torch.zeros(2,6))
        self.assertEqual(list(inspect.signature(student.forward).parameters),['observed','availability'])
        with self.assertRaises(ValueError):student.restorer(self.observed,torch.full((2,2,32,48),1.1))
        with self.assertRaises(ValueError):remove_history(self.available,0.0)


if __name__=='__main__':unittest.main(verbosity=2)
