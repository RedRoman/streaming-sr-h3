import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('feature_probe',Path(__file__).with_name('probe.py'))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def rows(n=20):
    out=[]
    for i in range(n):
        r={'row_index':i,'split':'train','source_id':'s'+str(i%10),'source_group_id':'g'+str((i%10)//2)}
        r['row_sha256']=P.value_sha(r);out.append(r)
    return out

class ProbeTests(unittest.TestCase):
    def test_group_source_exclusion_deterministic_independent_of_order(self):
        r=rows();s=P.split_rows(r,20,10);self.assertEqual(s,P.split_rows(r,20,10))
        self.assertEqual(len(s['folds']),5)
        for fold in s['folds']:
            self.assertFalse(set(fold['training_sources'])&set(fold['held_out_sources']))
            train_groups={s['source_to_group'][sid] for sid in fold['training_sources']}
            self.assertFalse(train_groups&set(fold['held_out_groups']))
        reverse=[]
        for i,old in enumerate(reversed(r)):
            item={k:v for k,v in old.items() if k!='row_sha256'};item['row_index']=i;item['row_sha256']=P.value_sha(item);reverse.append(item)
        self.assertEqual(s['source_to_fold'],P.split_rows(reverse,20,10)['source_to_fold'])
    def test_dev_row_and_changed_group_rejected(self):
        r=rows();r[0]['split']='dev';r[0]['row_sha256']=P.value_sha({k:v for k,v in r[0].items() if k!='row_sha256'})
        with self.assertRaises(ValueError):P.split_rows(r,20,10)
        r=rows();r[-1]['source_group_id']='other';r[-1]['row_sha256']=P.value_sha({k:v for k,v in r[-1].items() if k!='row_sha256'})
        with self.assertRaises(ValueError):P.split_rows(r,20,10)
    def test_fold_training_standardization_not_heldout_and_zero_variance(self):
        train=np.array([[1.,3.],[3.,3.]])
        stats=P.standardizer(train);a=P.transform(np.array([[100.,999.]]),stats)
        np.testing.assert_array_equal(stats[0],[2,3]);np.testing.assert_array_equal(stats[1],[1,1])
        np.testing.assert_array_equal(a,[[98,0,1]])
        P.transform(np.array([[-999.,-999.]]),stats)
        np.testing.assert_array_equal(P.standardizer(train)[0],stats[0])
    def test_constant_huber_flat_interval_unique_and_zero_tie(self):
        flat=P.constant_huber(np.array([-4.,4.]));np.testing.assert_allclose(flat['minimizer_interval'],[-3,3],atol=1e-12)
        self.assertEqual(flat['selected_nearest_zero'],0.)
        unique=P.constant_huber(np.array([.25,.25,.25]));self.assertAlmostEqual(unique['selected_nearest_zero'],.25,places=12)
        m=P.score_summary(np.array([-1.,0.,1.]),np.zeros(3));self.assertEqual(m['K1_tile_fraction'],0.)
    def test_fixed_solver_improves_linear_signal_no_hyperparameter_search(self):
        x=np.linspace(-1,1,64)[:,None];z=P.transform(x,P.standardizer(x));y=.5*x[:,0]
        w,r=P.linear_probe(z,y);self.assertEqual(r['iterations'],200);self.assertFalse(r['convergence_claimed'])
        self.assertLess(P.huber(z@w-y).mean(),P.constant_huber(y)['loss']/100)
        with self.assertRaises(ValueError):P.linear_probe(z,y,201)
    def test_no_dev_file_permission_and_file_hash_rejection(self):
        c={'schema':P.SCHEMA,'resources':P.RESOURCES,'protocol':P.PROTOCOL,'files':{'dev_cache_manifest':{}}}
        with self.assertRaises(ValueError):P.metadata(c)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metadata.json';p.write_text('{}');fs=P.Files(d)
            with self.assertRaises(ValueError):fs.bound({'path':str(p),'sha256':'0'*64})
    def test_split_persisted_before_load_and_failure_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d).resolve()/'fresh';cpath=Path(d)/'contract.json';cpath.write_text(json.dumps({'out':str(out),
                'runtime':{'python':'.'.join(map(str,P.sys.version_info[:3])),'numpy':np.__version__,'BLAS_threads':2,'Torch_imported':False}}))
            split=P.split_rows(rows(),20,10)
            def fail_after_split(fs,m,r,p,h):
                self.assertTrue(p.is_file());self.assertEqual(P.file_sha(p),h)
                self.assertEqual(json.loads(p.read_text())['feature_members_loaded_before_split'],0)
                raise ValueError('synthetic load failure')
            args=['probe.py','--execution-contract',str(cpath),'--execution-contract-sha256',P.file_sha(cpath),'--out',str(out),'--execute']
            with patch.object(P.sys,'argv',args),patch.object(P,'set_limits'),patch.object(P,'metadata',return_value=(None,None,rows(),split)),patch.object(P,'read_training_arrays',side_effect=fail_after_split):
                self.assertEqual(P.main(),1)
            report=json.loads((out/'report.json').read_text());self.assertFalse(report['complete']);self.assertEqual(report['status'],'failed_fixed_budget_no_retry')
            self.assertEqual(report['error'],{'type':'ValueError','message':'synthetic load failure'})

if __name__=='__main__':unittest.main()
