"""CPU producer tests: strict inheritance, recorded tolerances and finite scope."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import check_history_utility_integration as probe
from rphi_model import RphiModel
from rphi_history_utility import MaskAwareRestorer


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(19)
        old=RphiModel(variant='blind',ch=4,blocks=1)
        with torch.no_grad():old.backbone.tail.weight.normal_(std=.01)
        return old

    def test_warmstart_inventory_catches_changed_inherited_or_added_weights(self):
        old=self.model();new=MaskAwareRestorer.from_rphi(old.model_config,old.state_dict())
        evidence=probe.warmstart_evidence(old,new,torch)
        self.assertTrue(evidence['added_all_zero']);self.assertEqual(evidence['added_parameters'],72)
        with torch.no_grad():new.backbone.head.weight[0,9,0,0]=.1
        with self.assertRaisesRegex(ValueError,'not zero'):probe.warmstart_evidence(old,new,torch)
        new=MaskAwareRestorer.from_rphi(old.model_config,old.state_dict())
        with torch.no_grad():new.backbone.tail.bias[0]+=.1
        with self.assertRaisesRegex(ValueError,'tensor changed'):probe.warmstart_evidence(old,new,torch)

    def test_output_tolerance_is_predeclared_directional_and_absence_strict(self):
        a=torch.tensor([1.],dtype=torch.float32);b=torch.tensor([1.+1e-5])
        close=probe.comparison(a,b,torch)
        self.assertFalse(close['torch_equal']);self.assertTrue(close['engineering_close'])
        equal=probe.comparison(a,a,torch)
        rows={k:copy.deepcopy(close if 'baseline' in k else equal) for k in
            ('baseline_all_one','repeated_baseline_all_one','missing_content_invariance','repeated_mask_identity')}
        probe.validate_comparisons(rows)
        rows['missing_content_invariance']=close
        with self.assertRaisesRegex(ValueError,'bitexact'):probe.validate_comparisons(rows)
        self.assertFalse(probe.comparison(a,a+.0001,torch)['engineering_close'])

    def test_eight_actual_cpu_forward_fixture_with_no_update_and_saved_arrays(self):
        old=self.model()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);checkpoint=root/'fixture.pt';out=root/'out';out.mkdir()
            torch.save({'schema':'rphi.checkpoint.v1','step':3000,'model_config':old.model_config,
                'state_dict':old.state_dict(),'training_contract':{'fixture':True},'implementation_sha256':{}},checkpoint)
            budget={'baseline_forwards':2,'mask_aware_forwards':6,'student_forwards':1,'gate_forwards':1,'backward':0,'optimizer_steps':0,'NPU_calls':0}
            contract={'cpu_threads':1,'checkpoint':{'path':str(checkpoint)},'input':{'kind':'synthetic','seed':22},'budget':budget}
            report={}
            with patch.object(probe,'CHECKPOINT_SHA',probe.sha_file(checkpoint)),patch.object(probe,'EXPECTED_CONFIG',old.model_config),patch.dict(os.environ,{'TORCH_DEVICE_BACKEND_AUTOLOAD':'0'}):
                probe.execute(contract,out,report)
            self.assertEqual(report['actual'],budget);self.assertTrue(report['all_frozen_unchanged'])
            self.assertFalse(report['torch_npu_imported_after']);self.assertEqual(len(report['arrays']),10)
            for row in report['arrays'].values():self.assertEqual(probe.sha_file(out/row['path']),row['file_sha256'])
            self.assertTrue((out/'partial.json').exists())

    def test_contract_rejects_changed_resource_budget_or_source_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'contract.json'
            c={'schema':'rphi.history_utility_integration_execution.v1','checkpoint':{'sha256':probe.CHECKPOINT_SHA},
                'model_config':probe.EXPECTED_CONFIG,'budget':{'baseline_forwards':2,'mask_aware_forwards':6,'student_forwards':1,'gate_forwards':1,'backward':0,'optimizer_steps':0,'NPU_calls':0},
                'cpu_threads':1,'hard_timeout_seconds':180,'rss_limit_bytes':2<<30,'shape':probe.SHAPE,'output_policy':probe.OUTPUT_POLICY,
                'input':{'kind':'synthetic'},'implementation_sha256':{str(Path(probe.__file__).resolve()):probe.sha_file(probe.__file__)}}
            path.write_text(json.dumps(c));probe.verify_contract(path)
            c['budget']['backward']=1;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'budget'):probe.verify_contract(path)
            c['budget']['backward']=0;c['implementation_sha256'][str(Path(probe.__file__).resolve())]='0'*64;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'pin mismatch'):probe.verify_contract(path)

    def test_main_refuses_existing_output_without_calling_model(self):
        with tempfile.TemporaryDirectory() as td,patch.object(probe,'verify_contract',return_value={}),patch.object(probe,'execute') as execute:
            with patch.object(sys,'argv',['probe','--execution-contract','dummy','--out',td]):
                with self.assertRaises(FileExistsError):probe.main()
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
