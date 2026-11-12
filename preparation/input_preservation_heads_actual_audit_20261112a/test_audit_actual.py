import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

S=importlib.util.spec_from_file_location('audit_actual',Path(__file__).with_name('audit_actual.py'))
A=importlib.util.module_from_spec(S);S.loader.exec_module(A)
try:import torch
except ImportError:torch=None

def metric_fixture():
    rows=[A.value_sha(i) for i in range(16000)];donors=list(range(8000,16000))+list(range(8000))
    batches=[];pair={'matched':[],'shuffled':[]}
    for i in range(1000):
        inds=list(range(16*i,16*i+16));fh=A.value_sha(['feature',i]);lh=A.value_sha(['label',i]);step=i+1
        batches.append({'batch_index':i,'row_indices':inds,'row_sha256':rows[16*i:16*i+16],
            'arrays':{'raw_features':{'sha256':fh},'normalized_benefit':{'sha256':lh}}})
        for arm in A.ARMS:
            pair[arm].append({'step':step,'row_end_exclusive':step*16,'feature_rows':inds,'feature_batch_sha256':fh,
                'label_rows':inds if arm=='matched' else donors[16*i:16*i+16],
                'label_batch_sha256':lh if arm=='matched' else A.value_sha(['shufflelabel',i]),
                'lr':A.lr(step),'lr_hex':A.lr(step).hex(),'loss':.1,'gradient_norm_before_clip':.2,
                'correction_tile_fraction':.5,'tail_gradient_norms':{'gate.net.2.weight':.3,'gate.net.2.bias':.1}})
    return pair,donors,batches,rows

class Tests(unittest.TestCase):
    def test_complete_1000_pair_and_no_quality_claim(self):
        proof=A.check_metrics(*metric_fixture());self.assertEqual(proof['paired_feature_rows'],16000)
        self.assertFalse(proof['shuffled_label_batch_hashes_independently_recomputed'])
    def test_wrong_label_indices_feature_hash_lr_and_nan_rejected(self):
        for field,value in [('label_rows',list(range(16))),('feature_batch_sha256','0'*64),('lr',.0001),('loss',float('nan'))]:
            fixture=metric_fixture();fixture[0]['shuffled'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):A.check_metrics(*fixture)
    def test_missing_step_and_wrong_cache_metadata_rejected(self):
        fixture=metric_fixture();fixture[0]['matched'].pop()
        with self.assertRaises(ValueError):A.check_metrics(*fixture)
        fixture=metric_fixture();fixture[2][2]['row_sha256'][0]='0'*64
        with self.assertRaises(ValueError):A.check_metrics(*fixture)
    def test_bound_file_hash_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{}');f=A.Files(d);b={'path':str(p),'sha256':A.sha(p)}
            f.get(b);p.write_text('[]')
            with self.assertRaises(ValueError):f.unchanged()
            with self.assertRaises(ValueError):f.get(b)
    def test_wrong_donor_semantic_hash_rejected_without_model(self):
        m={'total_rows':16000,'mapped_rows':16000,'donor_ready':True,'impossible_strata':[], 'entries':[], 'donor_map_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'semantic'):A.donor_indices(m,[{}]*16000)
    @unittest.skipIf(torch is None,'local Torch unavailable; no installation')
    def test_cpu_weights_only_state_and_optimizer_finite_step_shapes(self):
        state={k:torch.ones(shape,dtype=torch.float32) for k,shape in A.SHAPES.items()}
        opt={'state':{i:{'step':torch.tensor(1000.),'exp_avg':torch.ones(shape),'exp_avg_sq':torch.ones(shape)} for i,shape in enumerate(A.SHAPES.values())},
            'param_groups':[{'lr':A.lr(1000),'betas':(.9,.999),'eps':1e-8,'weight_decay':0.,'amsgrad':False,'foreach':False,'fused':False,
                'maximize':False,'capturable':False,'differentiable':False,'params':[0,1,2,3]}]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.pt';torch.save({'model':state,'optimizer':opt},p);loaded=torch.load(p,map_location='cpu',weights_only=True)
            self.assertEqual(A.state_check(loaded['model'],torch),A.state_sha(state))
            self.assertEqual(A.optimizer_check(loaded['optimizer'],state,torch)['tensor_states'],12)
        opt['state'][0]['step'].fill_(999.)
        with self.assertRaises(ValueError):A.optimizer_check(opt,state,torch)
        state['gate.net.2.bias'].fill_(float('nan'))
        with self.assertRaises(ValueError):A.state_check(state,torch)
    @unittest.skipIf(torch is None,'local Torch unavailable; no installation')
    def test_tensor_hash_distinguishes_signed_zero(self):
        self.assertNotEqual(A.tensor_sha(torch.tensor([0.])),A.tensor_sha(torch.tensor([-0.])))

if __name__=='__main__':unittest.main()
