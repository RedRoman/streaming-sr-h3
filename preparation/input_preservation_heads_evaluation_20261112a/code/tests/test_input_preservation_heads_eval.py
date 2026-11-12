"""NumPy protocol fixtures only: no actual heads, LPIPS, caches or NPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import eval_input_preservation_heads as E


def fake_metric(error,mse,lpips,coverage,seam):
    d={k:0. for k in E.MEAN_KEYS}
    d.update(clipped_mae=error,unclamped_mae=error,clipped_mse=mse,unclamped_mse=mse,
             lpips=lpips,corrected_tile_fraction=coverage,residual_boundary_jump=seam)
    return d


def screen_rows():
    rows=[]
    for sid in range(6):
        for profile in ('p1','p2','p3','identity','near_clean'):
            if profile=='identity':
                arms={'raw':fake_metric(0.,0.,0.,0.,0.),'K1':fake_metric(.001,1e-6,.001,1.,.001),
                      'matched':fake_metric(.0001,1e-8,.0001,.02,.0001),'shuffled':fake_metric(.001,1e-6,.001,.5,.001)}
            elif profile=='near_clean':
                arms={'raw':fake_metric(.0005,2.5e-7,.0005,0.,.001),'K1':fake_metric(.0007,4.9e-7,.0007,1.,.002),
                      'matched':fake_metric(.0006,3.6e-7,.0006,.5,.001),'shuffled':fake_metric(.0007,4.9e-7,.0007,.5,.002)}
            else:
                arms={'raw':fake_metric(.02,.001,.03,0.,.002),'K1':fake_metric(.01,.0005,.02,1.,.0015),
                      'matched':fake_metric(.0098,.00048,.0199,.5,.001),'shuffled':fake_metric(.0102,.00052,.0201,.5,.0015)}
            for t in range(96):
                phase=profile if profile in ('identity','near_clean') else ('pre' if t<48 else 'post')
                rows.append({'source_id':str(sid),'profile_id':profile,'severity_phase':phase,'frame_index':t,'arms':copy.deepcopy(arms)})
    return rows


def checkpoints():
    result={}
    for arm in E.HEADS:
        c={'schema':E.H.SCHEMA,'steps':1000,'batch_size':16,'seed':20261111,'mode':'research','arm':arm,
           'optimizer':copy.deepcopy(E.H.OPTIMIZER),'loss':copy.deepcopy(E.H.LOSS),'model_config':copy.deepcopy(E.H.CONFIG),
           'runtime':{'device':'cpu','threads':2,'dtype':'float32','autocast':False,'grad_scaler':False,'expected_versions':{'torch_npu':None}},
           'implementation_sha256':{'/fixed/train_input_preservation_heads.py':E.TRAINER_SHA,'/fixed/rphi_input_preservation.py':E.A.INPUT_MODULE_SHA},
           'files':{},'backend_smokes':{},'admission':{'arm':arm}}
        result[arm]={'schema':'rphi.input_preservation_head_checkpoint.v1','artifact_role':'research_final_head','arm':arm,
                     'completed_step':1000,'row_cursor':16000,'training_contract':c,'execution_payload_sha256':E.T.execution_sha(c),
                     'binding':E.H.binding(c),'optimizer_specification':copy.deepcopy(E.H.OPTIMIZER),'initial_state_sha256':'a'*64}
    return result


class EvaluationTests(unittest.TestCase):
    def test_row_metrics_actual_clipping_null_psnr_and_bit_changes(self):
        raw=np.zeros((3,96,96),np.float32);pred=np.full_like(raw,-.5);mask=np.ones((1,6,6),bool)
        m=E.row_metrics(pred,raw,raw,mask,0.)
        self.assertEqual(m['unclamped_mae'],.5);self.assertEqual(m['clipped_mae'],0)
        self.assertIsNone(m['clipped_psnr']);self.assertTrue(m['clipped_exact_match'])
        self.assertEqual(m['clipped_component_fraction'],1);self.assertEqual(m['changed_pixel_fraction_unclamped'],1)
        signed=-raw;m=E.row_metrics(signed,raw,raw,np.zeros_like(mask),0.,signed)
        self.assertEqual(m['changed_pixel_fraction_clipped'],1)
        self.assertNotEqual(m['clipped_rgb_sha256'],E.asha(raw))
        with self.assertRaises(ValueError):E.row_metrics(pred,raw,raw,mask,float('nan'))

    def test_nearest16_selection_signedzero_and_no_target_input(self):
        raw=np.zeros((2,3,96,96),np.float32);raw[1]=-raw[1];k1=np.ones_like(raw);score=np.zeros((2,1,6,6),np.float32)
        score[1]=-score[1];out,mask=E.compose_numpy(raw,k1,score)
        self.assertEqual(E.asha(out),E.asha(raw));self.assertFalse(mask.any())
        score[0,0,2,3]=1;out,mask=E.compose_numpy(raw,k1,score)
        self.assertEqual(int(np.count_nonzero(out)),3*16*16)
        self.assertTrue((out[0,:,32:48,48:64]==1).all())
        score[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):E.compose_numpy(raw,k1,score)

    def test_actual_seam_fixture_distinguishes_residual_from_image_edge(self):
        pred=np.zeros((3,96,96),np.float32)
        for x in range(6):pred[:,:,x*16:(x+1)*16]=x%2
        m=E.row_metrics(pred,np.zeros_like(pred),np.zeros_like(pred),np.ones((1,6,6),bool),0.)
        self.assertEqual(m['residual_boundary_jump'],.5);self.assertEqual(m['residual_nonboundary_jump'],0.)
        m=E.row_metrics(pred,np.zeros_like(pred),pred.copy(),np.ones((1,6,6),bool),0.)
        self.assertEqual(m['residual_boundary_jump'],0.);self.assertEqual(m['output_boundary_jump'],.5)

    def test_equal_source_profile_and_separate_controls(self):
        rows=screen_rows();s=E.aggregate_rows(rows)
        self.assertAlmostEqual(s['native_macro']['matched']['clipped_mae'],.0098)
        self.assertAlmostEqual(s['controls']['identity']['arms']['matched']['clipped_mae'],.0001)
        self.assertIsNone(s['controls']['identity']['arms']['raw']['psnr_macro_finite'])
        self.assertEqual(s['controls']['identity']['arms']['raw']['psnr_exact_count'],6)
        self.assertAlmostEqual(s['native_macro']['matched']['psnr_macro_finite'],-10*np.log10(.00048))
        with self.assertRaises(ValueError):E.aggregate_rows(rows[:-1])

    def test_prospective_numeric_gates_and_pending_not_quality(self):
        s=E.aggregate_rows(screen_rows());a=E.screen_decision(s)
        self.assertTrue(all(a['numeric_gates'].values()));self.assertEqual(a['status'],'eligible_for_next_bounded_fullframe_validation')
        self.assertFalse(a['quality_proven']);self.assertFalse(a['automatic_training_or_fullframe_admission'])
        self.assertEqual(E.screen_decision(s,False)['status'],'pending_review')
        b=copy.deepcopy(s);b['native_macro']['matched']['residual_boundary_jump']=.1
        self.assertTrue(E.screen_decision(b)['seam_requires_review']);self.assertEqual(E.screen_decision(b)['status'],'pending_review')
        b=copy.deepcopy(s);b['native_macro']['matched']['corrected_tile_fraction']=0
        self.assertEqual(E.screen_decision(b)['status'],'failed_dev_screen')
        b=copy.deepcopy(s);b['controls']['near_clean']['arms']['matched']['clipped_mae']=1.
        self.assertFalse(E.screen_decision(b)['numeric_gates']['nearclean_mean_harm'])

    def test_no_gate_tolerance_or_oracle_can_manufacture_head_pass(self):
        s=E.aggregate_rows(screen_rows());s['native_macro']['matched']['clipped_mae']=s['native_macro']['K1']['clipped_mae']
        self.assertFalse(E.screen_decision(s)['numeric_gates']['native_gain_vs_K1'])
        s=E.aggregate_rows(screen_rows());s['native_macro']['matched']['unclamped_mae']=1.
        self.assertFalse(E.screen_decision(s)['numeric_gates']['unclamped_gain_not_reversed'])
        s=E.aggregate_rows(screen_rows());s['native_macro']['matched']['lpips']=s['native_macro']['K1']['lpips']+.001000001
        self.assertFalse(E.screen_decision(s)['numeric_gates']['native_LPIPS_noninferior'])

    def test_temporal_real_adjacent_masks_and_declared_transition(self):
        zero=np.zeros((3,96,96),np.float32);a={'source_id':'s','profile_id':'p','window_key':'w','crop_tlhw':[0,0,96,96],
            'frame_index':47,'severity_phase':'pre','arrays':{k:zero for k in E.MODES},'masks':{k:np.zeros((1,6,6),bool) for k in E.MODES},'target':zero}
        b=copy.deepcopy(a);b.update(frame_index=48,severity_phase='post');b['masks']['matched'][:]=True;b['arrays']['matched'][:]=1
        p=E.pair_metrics(a,b);self.assertTrue(p['declared_phase_transition'])
        self.assertEqual(p['arms']['matched']['mask_flip_fraction'],1.);self.assertEqual(p['arms']['matched']['unwarped_output_change_mae'],1.)
        b['frame_index']=49
        with self.assertRaisesRegex(ValueError,'nonadjacent'):E.pair_metrics(a,b)

    def test_temporal_complete95_pairs_not_sparse_frames(self):
        pairs=[]
        for sid in range(6):
            for profile in range(5):
                for t in range(1,96):pairs.append({'source_id':str(sid),'profile_id':str(profile),'from_frame':t-1,'to_frame':t,
                    'declared_phase_transition':t==48 and profile<3,'arms':{a:{k:0. for k in ('mask_flip_fraction','unwarped_output_change_mae','unwarped_error_change_mae')} for a in E.MODES}})
        result=E.aggregate_pairs(pairs);self.assertEqual(result['pairs'],2850)
        with self.assertRaises(ValueError):E.aggregate_pairs(pairs[:-1])

    def test_checkpoint_final_pair_recipe_and_binding(self):
        cp=checkpoints();E.validate_checkpoint_pair(cp)
        bad=copy.deepcopy(cp);bad['matched']['completed_step']=999
        with self.assertRaises(ValueError):E.validate_checkpoint_pair(bad)
        bad=copy.deepcopy(cp)
        for x in bad.values():
            x['training_contract']['optimizer']['lr']=.01;x['binding']=E.H.binding(x['training_contract'])
            x['execution_payload_sha256']=E.T.execution_sha(x['training_contract'])
        with self.assertRaisesRegex(ValueError,'recipe'):E.validate_checkpoint_pair(bad)
        bad=copy.deepcopy(cp);bad['shuffled']['initial_state_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'matched pair'):E.validate_checkpoint_pair(bad)
        bad=copy.deepcopy(cp);bad['matched']['artifact_role']='backend_smoke_not_research_resume'
        with self.assertRaises(ValueError):E.validate_checkpoint_pair(bad)

    def test_saved_scores_masks_LPIPS_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out';sink=E.Output(out)
            values={'matched_score':np.zeros((16,1,6,6),np.float32),'matched_mask':np.zeros((16,1,6,6),np.uint8),'lpips':np.zeros((16,4),np.float64)}
            row=[{'row_sha256':str(i)} for i in range(16)];entry=sink.arrays(0,values,row)
            with np.load(entry['path'],allow_pickle=False) as z:
                for name in values:self.assertEqual(E.asha(z[name]),entry['arrays'][name]['sha256'])
            self.assertEqual(E.T.file_sha(entry['path']),entry['sha256'])
            with self.assertRaises(ValueError):sink.arrays(0,values,row)
            with self.assertRaises(FileExistsError):E.Output(out)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'local Torch unavailable; no installation')
class TorchInterfaceTests(unittest.TestCase):
    def test_real_head_strict_load_frozen_and_raw_ties(self):
        import torch
        from rphi_input_preservation import InputPreservationHead,compose_input_preservation
        torch.set_num_threads(2);base=InputPreservationHead(seed=20261111)
        initial=E.T.state_sha(base.state_dict());cp=checkpoints();basefiles={k:{'path':'/fixture/'+k,'sha256':'f'*64}
            for k in ('proposal','screen','metric_readiness','train_cache_manifest','train_cache_audit','dev_cache_manifest','dev_cache_audit')}
        with tempfile.TemporaryDirectory() as directory:
            paths={};files=copy.deepcopy(basefiles)
            for arm,x in cp.items():
                x['training_contract']['files']=copy.deepcopy(basefiles);x['binding']=E.H.binding(x['training_contract'])
                x['execution_payload_sha256']=E.T.execution_sha(x['training_contract'])
                x.update(initial_state_sha256=initial,model_config=base.model_config,state_dict=E.T.cpu_tree(base.state_dict()),final_state_sha256=initial)
                p=Path(directory)/(arm+'.pt');torch.save(x,p);paths[arm+'_checkpoint']=p;files[arm+'_checkpoint']={'path':str(p),'sha256':E.T.file_sha(p)}
            models=E.load_heads(paths,{'files':files},{})
            raw=torch.rand((16,3,96,96));k1=torch.ones_like(raw);features=torch.zeros((16,2,11,6,6))
            for model in models.values():
                self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                with torch.inference_mode():score=model.forward_features(features)['benefit_score'];out=compose_input_preservation(raw,k1,score)
                expected,mask=E.compose_numpy(raw.numpy(),k1.numpy(),score.numpy())
                self.assertEqual(E.asha(out['rgb'].numpy()),E.asha(raw.numpy()));self.assertEqual(E.asha(expected),E.asha(out['rgb'].numpy()))
                self.assertFalse(mask.any())

    def test_actual_CPU_metric_wrapper_range_and_complete_batch(self):
        import torch
        class FakeMetric(torch.nn.Module):
            def forward(self,p,y):
                self.seen=(p.clone(),y.clone());return (p-y).square().mean((1,2,3),keepdim=True)
        metric=FakeMetric();p=np.zeros((16,3,96,96),np.float32);y=np.ones_like(p)
        d=E.lpips_batch(metric,p,y)
        self.assertEqual(d.dtype,np.float64);self.assertEqual(d.shape,(16,));self.assertTrue((d==4).all())
        self.assertTrue(bool((metric.seen[0]==-1).all()));self.assertTrue(bool((metric.seen[1]==1).all()))
        self.assertFalse(metric.seen[0].requires_grad);self.assertEqual(metric.seen[0].device.type,'cpu')
        p[0,0,0,0]=-1
        with self.assertRaises(ValueError):E.lpips_batch(metric,p,y)


if __name__=='__main__':unittest.main()
