import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import adamw_equations as E
from audit_arrays import array_sha,compare_outputs
from audit_checkpoint import SafeAdapter
import audit_terminal as A
import finalize_once as F

def equation_fixture(lr=E.LR,beta1=.9,beta2=.999,eps=1e-8,scale=1.):
    before=np.zeros(6,np.float32);g=np.array([0.,1e-8,-1e-7,.002,-.3,1.],np.float32)
    m=np.asarray((1-beta1)*(g*scale),np.float32);v=np.asarray((1-beta2)*(g*scale)**2,np.float32)
    after=np.asarray(-lr*(m.astype(float)/(1-beta1))/(np.sqrt(v.astype(float)/(1-beta2))+eps),np.float32)
    after[g==0]=before[g==0]
    d=after.astype(float)-before.astype(float)
    row={'before_sha256':array_sha(before),'after_sha256':array_sha(after),'all_finite':True,
         'zero_gradient_coordinates_unchanged':True,'changed_coordinates':int(np.count_nonzero(d)),
         'update_l2':float(np.linalg.norm(d)),'update_rms':float(np.sqrt(np.mean(d*d)))}
    return before,g,after,m,v,row

class Numeric(unittest.TestCase):
    def test_correct_first_step_and_error_distribution(self):
        r=E.first_step(*equation_fixture());self.assertEqual(r['update']['failed_coordinates'],0)
        self.assertLessEqual(r['update']['worst_normalized_residual'],1)
        self.assertEqual(set(r['update']['absolute_error_quantiles']),{'p0','p50','p95','p99','p100'})
    def test_wrong_LR_scale_betas_eps_rejected(self):
        for kwargs in ({'lr':1e-4},{'scale':1024.},{'beta1':.8},{'beta2':.99},{'eps':1e-6}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'equation differs'):
                E.first_step(*equation_fixture(**kwargs))
    def test_zero_gradient_coordinates_are_byte_hard(self):
        args=list(equation_fixture());args[2][0]=np.float32(-0.)
        with self.assertRaisesRegex(ValueError,'zero-gradient parameter bytes'):E.first_step(*args)
    def test_nonfinite_state_rejected(self):
        args=list(equation_fixture());args[4][1]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):E.first_step(*args)
    def test_output_parity_is_not_AdamW_equation_tolerance(self):
        a=np.ones(2,np.float32);b=a.copy();b[0]+=.0001
        with self.assertRaisesRegex(ValueError,'strict output parity'):compare_outputs(b,a,exact=True)
    def test_signed_zero_numeric_equal_does_not_claim_byte_equal(self):
        a=np.asarray([0.],np.float32);b=np.asarray([-0.],np.float32)
        row={'torch_equal':True,'same_shape':True,'finite':True,'reference_shape':[1],'candidate_shape':[1],
            'reference_sha256_float32':array_sha(a),'candidate_sha256_float32':array_sha(b),
            'max_abs':0.,'relative_l2':0.,'changed_coordinates':0}
        result=A.parity_record(row,a,b)
        self.assertTrue(result['exact_equal']);self.assertFalse(result['byte_equal'])
    def test_state_file_extra_key_tampering_rejected(self):
        import struct
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.safetensors';header={'__metadata__':{'fixture':'x'},'a':{'dtype':'F32','shape':[],'data_offsets':[0,4]}}
            raw=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(raw))+raw+np.float32(1).tobytes())
            r=SafeAdapter(p,A.file_sha(p),{'a':[]},{'fixture':'x'},provenance=False);self.assertEqual(float(r.load('a')),1)
            with self.assertRaisesRegex(ValueError,'exact adapter key'):SafeAdapter(p,A.file_sha(p),{'b':[]},{'fixture':'x'},provenance=False)
    def test_wait_covers3630_terminal_and_stops4200(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';clock=[0.]
            def sleep(t):
                clock[0]+=t
                if clock[0]>=3630:path.write_text(json.dumps({'state':'completed'}))
            r=F.wait_terminal(path,lambda *x:None,clock=lambda:clock[0],sleep=sleep)
            self.assertTrue(r['terminal']);self.assertEqual(clock[0],3630)
            path.unlink();clock[0]=0.
            r=F.wait_terminal(path,lambda *x:None,clock=lambda:clock[0],sleep=lambda t:clock.__setitem__(0,clock[0]+t))
            self.assertFalse(r['terminal']);self.assertEqual(clock[0],4200)
    def test_total900_not_reset_before_archive(self):
        self.assertEqual(F.remaining(100,clock=lambda:990),10)
        with self.assertRaisesRegex(TimeoutError,'900s exhausted'):F.remaining(100,clock=lambda:1000)
    def test_actual_role_sets_include1200states_each_and_fourcasts(self):
        self.assertEqual(len([n for n in A.EXPECTED_ARRAYS if n.startswith('optimizer_state_after_update__')]),1200)
        self.assertEqual(len([n for n in A.EXPECTED_ARRAYS if n.startswith('optimizer_state_after_reload__')]),1200)
        self.assertEqual(len([n for n in A.EXPECTED_ARRAYS if n.endswith('_stage_decoder_input')]),4)
        self.assertNotIn('zero_rgb',A.ARCHIVE_SMALL)

@unittest.skipUnless(os.environ.get('ADAMW_CPU_PRODUCER_FIXTURE'),'requires isolated Torch producer fixture path')
class ActualSavedState(unittest.TestCase):
    def test_400_actual_Torch_states_independent_numpy_audit(self):
        root=Path(os.environ['ADAMW_CPU_PRODUCER_FIXTURE']);sys.path[:0]=[str(root/'code'),str(root/'code/tests')]
        import torch
        import check_h3_real_adamw_update as P
        import check_h3_joint_rgb_grad as J
        import lora as L
        from test_h3_nonzero_adapter import tiny_model
        with tempfile.TemporaryDirectory() as tmp:
            e=P.Evidence(Path(tmp)/'run',{'execution_contract_sha256':'e'*64});flush=e.save;e.save=lambda:None
            model,names=tiny_model(50);shapes=P.N.expected_adapters(model,torch)
            e.data['adapter_initial']=J.adapter_audit(model,names,torch)
            for n,p in L.lora_parameters(model):p.grad=torch.full_like(p,0. if n.endswith('.lora_a') else 1024.)
            video=torch.ones(1,requires_grad=True)*2;video.retain_grad();video.sum().backward()
            J.save_gradients(model,video,e,torch)
            contract={'implementation_sha256':{'fixture':'x'},'manifest_sha256':'m'*64}
            e.data['pretrained']={'openh3_pretrained':model._openh3_pretrained_record}
            e.data['input_inspection']={'identity':{'fixture':True}};e.data['noise_observations']=[{'fixture':True}]
            for n in ('guide','raw_warm_start','target'):J.save_array(e,n,torch.zeros(1),torch)
            opt=P.one_adamw_update(model,shapes,e,torch)
            diagnostic={'role':'one_real_TRAIN_AdamW_engineering_step_not_quality_checkpoint',
                'execution_contract_sha256':'e'*64,'protocol_sha256':A.CONTRACT_SHA,'manifest_sha256':'m'*64,
                'implementation_sha256':contract['implementation_sha256'],'input_identity':e.data['input_inspection']['identity'],
                'input_arrays':{n:e.data['arrays'][n] for n in ('guide','raw_warm_start','target')},
                'noise':e.data['noise_observations'][0],'sigmas':[.909375,6/7,2/3,0.],
                'sigmas_float32_sha256':array_sha(np.asarray([.909375,6/7,2/3,0.],np.float32)),
                'rank':8,'alpha':8,'steps':1,'loss_scale':1024.,'optimizer':e.data['optimizer'],'objective':'original clean proxy RGB MSE only'}
            paths=P.save_checkpoint(model,opt,shapes,model._openh3_pretrained_record,diagnostic,e,torch)
            P.clear_reload(model,opt,shapes,paths,model._openh3_pretrained_record,diagnostic,e,torch)
            opt.state.clear();e.data['optimizer_released_before_merge']=True;flush()
            report=json.loads(json.dumps(e.data)) # audit the actual JSON serialization, including betas tuple->list
            reader=A.Arrays(e.path,report['arrays']);result=A.gradients_update_checkpoint(report,reader,{k:list(v) for k,v in shapes.items()},contract,expected_device='cpu',video_shape=[1])
            self.assertEqual(result['optimizer_after_update_and_reload_arrays'],2400)
            self.assertEqual(result['optimizer_named_entries'],400);self.assertTrue(result['zero_and_serialization_byte_exact'])
            # Negative receipt mismatch, no repeat model or optimizer calls.
            report['reload']['exact_optimizer'][next(iter(A.NAMES))]=False
            with self.assertRaisesRegex(ValueError,'reload coverage'):
                A.gradients_update_checkpoint(report,reader,{k:list(v) for k,v in shapes.items()},contract,expected_device='cpu',video_shape=[1])
            print(json.dumps({'actual_CPU_optimizer_states_independently_audited':1200,'actual_reload_arrays':1200,
              'actual_adapter_tensors':400,'kernel_reexecution_claim':False,'NPU_calls':0}))
if __name__=='__main__':unittest.main(verbosity=2)
