"""Bounded CPU-only fixture tests; no real assets or research updates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import check_history_utility_one_step as probe
from rphi_model import RphiModel


class OneStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixtures(self):
        torch.manual_seed(17)
        baseline=RphiModel(variant='blind',ch=4,blocks=1).eval().requires_grad_(False)
        with torch.no_grad():baseline.backbone.tail.weight.normal_(std=.02)
        rng=np.random.default_rng(52);batches={}
        for role in ('teacher','student'):
            observed=rng.random((2,3,3,16,16),dtype=np.float32)
            batches[role]={'model_inputs':{'observed':observed,'availability':np.ones((2,2),np.float32)},
                'target':rng.random((2,3,16,16),dtype=np.float32)}
        batches['teacher']['teacher_weights']=rng.random((2,2,16,16),dtype=np.float32)
        return baseline,batches

    def test_real_torch_fixture_seven_forwards_two_backwards_one_update(self):
        baseline,batches=self.fixtures()
        with tempfile.TemporaryDirectory() as td:
            report={};ev=probe.Evidence(td,report);probe.run_core(baseline,batches,ev,torch)
            self.assertEqual(report['actual'],probe.COUNTS);self.assertEqual(report['completed_backward_calls'],2)
            self.assertTrue(all(row=={'pre':0,'post':0} for row in report['cloned_restorer_hooks'].values()))
            self.assertEqual(report['completed_restorer_calls'],7);self.assertEqual(report['optimizer']['actual_lr'],1e-4*(1/50))
            self.assertEqual(set(report['optimizer']['state_steps'].values()),{1.})
            self.assertNotEqual(report['teacher_before_sha256'],report['teacher_after_sha256'])
            self.assertEqual(report['teacher_after_sha256'],report['teacher_final_sha256'])
            self.assertEqual(report['student_initial_sha256'],report['student_final_sha256'])
            self.assertEqual(report['input_tensor_sha256_before'],report['input_tensor_sha256_after'])
            self.assertTrue(report['hooks_removed']);self.assertFalse(report['torch_npu_imported_after'])
            first=[v['nonzero'] for n,v in report['student_grad']['parameters'].items() if n.startswith('gate.net.0.')]
            self.assertEqual(sum(first),0)
            self.assertGreater(sum(v['nonzero'] for n,v in report['student_grad']['parameters'].items() if n.startswith('gate.net.2.')),0)
            for row in report['arrays'].values():self.assertEqual(probe.sha_file(Path(td)/row['path']),row['file_sha256'])
            labels=np.load(Path(td)/report['arrays']['teacher_labels/normalized']['path'],allow_pickle=False)
            self.assertEqual(labels.shape,(2,2,1,1));self.assertTrue(np.isfinite(labels).all())

    def test_failure_after_teacher_update_preserves_states_and_hook_cleanup(self):
        baseline,batches=self.fixtures()
        with tempfile.TemporaryDirectory() as td:
            report={};ev=probe.Evidence(td,report)
            import rphi_history_utility as model
            with patch.object(model,'HistoryUtilityStudent',side_effect=ValueError('injected clone failure')):
                with self.assertRaisesRegex(ValueError,'clone failure'):probe.run_core(baseline,batches,ev,torch)
            saved=json.loads((Path(td)/'partial.json').read_text())
            self.assertEqual(saved['actual']['teacher_optimizer_steps'],1)
            self.assertEqual(saved['actual']['restorer_total'],1);self.assertEqual(saved['actual']['backward'],1)
            self.assertTrue(saved['hooks_removed']);self.assertIn('teacher_after_sha256',saved)
            self.assertEqual(saved['teacher_after_sha256'],saved['teacher_final_sha256'])
            self.assertEqual(saved['student_final_sha256'],{})

    def test_missing_gradient_rejected_after_evidence_is_saved(self):
        model=torch.nn.Linear(2,1)
        model.weight.grad=torch.ones_like(model.weight)
        with tempfile.TemporaryDirectory() as td:
            report={};ev=probe.Evidence(td,report)
            with self.assertRaisesRegex(ValueError,'missing/zero'):probe.grad_inventory(model,ev,'gradient')
            self.assertEqual(report['gradient']['missing'],['bias'])
            self.assertTrue((Path(td)/'partial.json').exists())

    def valid_batch(self,teacher):
        families=['native_static_blur_noise_h264']*4+['native_smooth_blur_noise_h264']*4+['native_abrupt_blur_noise_h264']*4+['identity']*2+['near_clean']*2
        rows=[{'split':'train','profile_id':f,'source_id':('2BdYaGuVAKo','4EKWTiWjPTQ')[i%2]} for i,f in enumerate(families)]
        batch={'model_inputs':{'observed':np.zeros((16,3,3,96,96),np.float32),'availability':np.ones((16,2),np.float32)},
            'target':np.zeros((16,3,96,96),np.float32),'metadata':rows,'control':np.array([0]*12+[1]*2+[2]*2)}
        if teacher:batch['teacher_weights']=np.ones((16,2,96,96),np.float32)
        return batch

    def test_oracle_keys_dev_rows_and_revived_history_fail_before_model(self):
        valid=self.valid_batch(True);probe.validate_batch(valid,'teacher',np)
        bad=copy.deepcopy(valid);bad['model_inputs']['target']=bad['target']
        with self.assertRaisesRegex(ValueError,'oracle'):probe.validate_batch(bad,'teacher',np)
        bad=copy.deepcopy(valid);bad['metadata'][0]['split']='dev'
        with self.assertRaisesRegex(ValueError,'non-train'):probe.validate_batch(bad,'teacher',np)
        bad=copy.deepcopy(valid);bad['model_inputs']['availability'][0,0]=0
        with self.assertRaisesRegex(ValueError,'revive'):probe.validate_batch(bad,'teacher',np)
        probe.validate_batch(self.valid_batch(False),'student',np)

    def test_recovery_is_explicit_not_renamed_original_or_unknown_schema(self):
        recovery={'schema':'openh3.history_utility_materialization_recovery.v1','complete':True,'status':'completed_engineering_only',
            'model_calls':0,'NPU_calls':0,'original_data_contract_file_sha256':'13b6e03f8b5387a101997e1545214a03b43d281454e1d1848a589a57fd9c47ad',
            'reused_sources':1,'newly_materialized_sources':3,'new_target_decodes':3,'new_observation_codec_passes':18,
            'inherited_completed_target_decodes':1,'inherited_completed_codec_passes':6,'total_verified_target_NPZ':4,'total_verified_observation_NPZ':12,
            'phases_completed':[{'complete':True} for _ in range(5)],'independent_worker_PIDs':[1,2,3,4,5],
            'scientific_continuity_approved':False,'natural_cut_status':'unknown_engineering_only'}
        self.assertEqual(probe.validate_materialization(recovery),recovery['schema'])
        bad=copy.deepcopy(recovery);bad['new_target_decodes']=4
        with self.assertRaisesRegex(ValueError,'accounting'):probe.validate_materialization(bad)
        bad=copy.deepcopy(recovery);bad['schema']='unknown'
        with self.assertRaisesRegex(ValueError,'unrecognized'):probe.validate_materialization(bad)
        bad=copy.deepcopy(recovery);bad['phases_completed'][0]['complete']=False
        with self.assertRaisesRegex(ValueError,'completion'):probe.validate_materialization(bad)

    def test_admission_payload_binds_data_pins_but_avoids_admission_hash_cycle(self):
        original={'files':{'teacher_schedule':{'sha256':'a'}},'root_admission':{'sha256':'old'}}
        digest=probe.execution_payload_sha256(original)
        replacement=copy.deepcopy(original);replacement['root_admission']['sha256']='new'
        self.assertEqual(digest,probe.execution_payload_sha256(replacement))
        replacement['files']['teacher_schedule']['sha256']='changed'
        self.assertNotEqual(digest,probe.execution_payload_sha256(replacement))

    def test_default_inspection_never_calls_execute_and_existing_out_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            contract=Path(td)/'contract.json';contract.write_text('{}');out=Path(td)/'inspect'
            with patch.object(probe,'verify_contract',return_value={}),patch.object(probe,'load_inputs',return_value=({},{})),patch.object(probe,'execute') as execute:
                with patch.object(sys,'argv',['probe','--execution-contract',str(contract),'--out',str(out)]):self.assertEqual(probe.main(),0)
                execute.assert_not_called()
                self.assertEqual(json.loads((out/'report.json').read_text())['model_calls'],0)
                with patch.object(sys,'argv',['probe','--execution-contract',str(contract),'--out',str(out)]):
                    with self.assertRaises(FileExistsError):probe.main()

    def test_execute_exception_never_sets_complete(self):
        with tempfile.TemporaryDirectory() as td:
            contract=Path(td)/'contract.json';contract.write_text('{}');out=Path(td)/'failed'
            with patch.object(probe,'verify_contract',return_value={}),patch.object(probe,'execute',side_effect=ValueError('injected')):
                with patch.object(sys,'argv',['probe','--execution-contract',str(contract),'--out',str(out),'--execute']):self.assertEqual(probe.main(),2)
            report=json.loads((out/'failure.json').read_text());self.assertFalse(report['complete']);self.assertFalse((out/'report.json').exists())


if __name__=='__main__':unittest.main()
