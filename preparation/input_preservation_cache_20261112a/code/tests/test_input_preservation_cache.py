"""Small NumPy fixtures; no real data/checkpoint/Torch/NPU/model export."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import export_input_preservation_cache as C


def frozen_labels():
    # Execute only the already-tested pure NumPy label functions, not Torch or
    # the model module. The production path imports these original functions.
    path = CODE/'rphi_input_preservation.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != '7a52656a3119dcaad515dee0ec591bccbb21ee7756ba150b7f59827ad3b8100d':
        raise AssertionError('frozen label source changed')
    tree = ast.parse(path.read_text())
    functions = [x for x in tree.body if isinstance(x,ast.FunctionDef) and
                 x.name in ('normalize_input_benefit','input_benefit_labels')]
    env = {'np':np, 'TILE':16, 'UTILITY_SCALE':.25/255}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),env)
    return env['input_benefit_labels']


LABELS = frozen_labels()


def train_fixture(b=2):
    rng = np.random.default_rng(103)
    x = rng.random((b,3,3,96,96),dtype=np.float32)
    target = rng.random((b,3,96,96),dtype=np.float32)
    rows, metadata = [], []
    for i in range(b):
        r={'row_index':i,'split':'train','teacher_mask':None,'true_availability':[1,1],
           'external_mask':[0,0] if i == 0 else [1,1],'source_id':str(i)}
        r['row_sha256']=C.T.value_sha(r);rows.append(r)
        metadata.append({**r,'observed_crop_sha256_float32':C.asha(x[i]),
                         'target_crop_sha256_float32':C.asha(target[i])})
    return {'model_inputs':{'observed':x,'availability':np.asarray([r['external_mask'] for r in rows],np.float32)},
            'target':target,'metadata':metadata}, rows


def feature_fixture(x,a):
    # A deterministic raw-only stand-in, not a claim of Torch feature parity.
    return np.full((len(x),2,11,6,6),float(x[:,-1].mean()),np.float32)


def group_fixture():
    rng=np.random.default_rng(104)
    current=rng.random((96,3,96,96),dtype=np.float32);records=[]
    for t in range(96):
        h=[max(t-2,0),max(t-1,0),t]
        records.append({'split':'dev','frame_index':t,'source_id':'one','source_sha256':'a'*64,
            'source_group_id':'group','window_key':'window','profile_id':'near_clean',
            'crop_tlhw':[192,368,96,96],'history_indices':h,
            'observed_crop_sha256_float32':C.asha(current[h]),
            'metrics':{'input':{'prediction_float32_sha256':C.asha(current[t])}}})
    return records,current


class CacheTests(unittest.TestCase):
    def test_workspace_root_mismatch_fails_before_evidence_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'contract.json'
            p.write_text(json.dumps({'root':'/one','evidence_root':'/two'}))
            with self.assertRaisesRegex(ValueError,'same absolute workspace'):
                C.verify_contract(p,'train')

    def test_train_target_changes_labels_never_predictor_or_features(self):
        batch,rows=train_fixture();seen=[]
        def predict(x):
            seen.append(x.copy());return x[:,-1]*np.float32(.9)
        a,r=C.make_train_batch(batch,rows,predict,feature_fixture,LABELS)
        changed=copy.deepcopy(batch);changed['target']=1-changed['target']
        for i,m in enumerate(changed['metadata']):m['target_crop_sha256_float32']=C.asha(changed['target'][i])
        b,s=C.make_train_batch(changed,rows,predict,feature_fixture,LABELS)
        self.assertEqual(len(seen),2)
        self.assertEqual(C.asha(seen[0]),C.asha(seen[1]))
        for key in ('k1_rgb','raw_features','availability'):self.assertEqual(C.asha(a[key]),C.asha(b[key]))
        self.assertNotEqual(C.asha(a['raw_benefit']),C.asha(b['raw_benefit']))
        self.assertTrue(r[0]['usable_for_training'])
        self.assertEqual(a['raw_benefit'].dtype,np.float64)
        # Both-absent rows retain complete, nonzero benefit labels.
        self.assertTrue(np.any(a['raw_benefit'][0] != 0))

    def test_wrong_role_input_hash_and_oracle_input_keys_fail_before_forward(self):
        for case in ('dev','hash','oracle'):
            b,r=train_fixture();called=[]
            if case=='dev':
                r[0]['split']='dev';r[0]['row_sha256']=C.T.value_sha({k:v for k,v in r[0].items() if k!='row_sha256'})
                b['metadata'][0].update(r[0])
            elif case=='hash':b['metadata'][0]['observed_crop_sha256_float32']='0'*64
            else:b['model_inputs']['target']=b['target']
            with self.assertRaises(ValueError):C.make_train_batch(b,r,lambda x:called.append(x),feature_fixture,LABELS)
            self.assertEqual(called,[])

    def test_nonfinite_K1_and_feature_shape_are_rejected(self):
        b,r=train_fixture()
        for predict,features in ((lambda x:np.full_like(x[:,-1],np.nan),feature_fixture),
                                  (lambda x:x[:,-1],lambda x,a:np.zeros((len(x),1,6,6),np.float32))):
            with self.assertRaises(ValueError):C.make_train_batch(b,r,predict,features,LABELS)

    def test_exact_full96_K3_reconstruction_and_causal_boundaries(self):
        r,current=group_fixture();x=C.reconstruct_group(r,current)
        self.assertEqual(x.shape,(96,3,3,96,96))
        self.assertEqual(C.asha(x[0]),C.asha(current[[0,0,0]]))
        self.assertEqual(C.asha(x[1]),C.asha(current[[0,0,1]]))
        self.assertEqual(C.asha(x[95]),C.asha(current[[93,94,95]]))

    def test_missing_neighbor_nearclean_tamper_and_mixed_source_fail_closed(self):
        r,current=group_fixture()
        with self.assertRaises(ValueError):C.reconstruct_group(r[:-1],current[:-1])
        changed=current.copy();changed[5,0,0,0]=np.float32(.99)
        with self.assertRaises(ValueError):C.reconstruct_group(r,changed)
        changed_rows=copy.deepcopy(r);changed_rows[20]['source_id']='other'
        with self.assertRaises(ValueError):C.reconstruct_group(changed_rows,current)
        changed_rows=copy.deepcopy(r);changed_rows[20]['history_indices']=[19,20,21]
        with self.assertRaises(ValueError):C.reconstruct_group(changed_rows,current)

    def test_two_candidate_oracle_ties_raw_and_clipped_are_distinct(self):
        current=np.full((1,3,96,96),.4,np.float32);target=np.zeros_like(current);k1=np.full_like(current,-.5)
        arrays,rows=C.two_candidate_diagnostic(current,k1,target,LABELS)
        self.assertTrue((arrays['oracle_unclamped_index']==0).all())
        self.assertTrue((arrays['oracle_clipped_index']==1).all())
        self.assertEqual(rows[0]['clipped_tile_oracle']['clipped_mse'],0)
        self.assertIsNone(rows[0]['clipped_tile_oracle']['clipped_psnr'])
        arrays,_=C.two_candidate_diagnostic(current,current.copy(),target,LABELS)
        self.assertTrue((arrays['oracle_unclamped_index']==0).all())
        self.assertTrue((arrays['oracle_clipped_index']==0).all())

    def test_disk_array_hashes_no_overwrite_and_actual_byte_total(self):
        batch,rows=train_fixture();arrays,records=C.make_train_batch(batch,rows,lambda x:x[:,-1],feature_fixture,LABELS)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'new';s=C.Output(out)
            entry=s.array_batch(0,arrays,rows);s.append('records.jsonl',records)
            with np.load(entry['path'],allow_pickle=False) as z:
                self.assertEqual(set(z.files),set(arrays))
                for k in arrays:self.assertEqual(C.asha(z[k]),entry['arrays'][k]['sha256'])
            self.assertEqual(C.T.file_sha(entry['path']),entry['sha256'])
            with self.assertRaises(ValueError):s.array_batch(0,arrays,rows)
            with self.assertRaises(FileExistsError):C.Output(out)
            report={'complete':True};s.finish(report)
            self.assertEqual(report['actual_output_bytes'],sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))

    def test_budget_rejects_before_write_and_failure_receipt_can_persist(self):
        with tempfile.TemporaryDirectory() as folder:
            s=C.Output(Path(folder)/'new',limit=C.RESERVE+32)
            with self.assertRaises(ValueError):s.put('too_large',b'x'*33)
            self.assertFalse((s.path/'too_large').exists())
            report={'complete':False,'failure':{'message':'budget exceeded'}}
            s.partial(report);s.finish(report)
            self.assertFalse(json.loads((s.path/'failure.json').read_text())['complete'])

    def test_payload_bytes_include_availability_and_fit_with_metadata_reserve(self):
        b=C.payload_budget();self.assertEqual(b['total_array_payload_bytes'],1836416000)
        self.assertEqual(b['arrays']['k1_rgb'],1769472000)
        self.assertLess(b['total_array_payload_bytes']+C.RESERVE,C.MAX_BYTES)
        self.assertGreater(16000*3*3*96*96*4,C.MAX_BYTES) # fullK3 cannot be silently added

    def test_old_cache_exact_six_array_hashes_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'arrays').mkdir();p=out/'arrays'/'batch_0000.npz'
            a={n:np.zeros((16,3,96,96),np.float32) for n in C.SOURCE_ARRAYS}
            np.savez(p,**a)
            e={'path':str(p),'sha256':C.T.file_sha(p),'bytes':p.stat().st_size,
               'arrays':{n:{'dtype':'float32','shape':[16,3,96,96],'sha256':C.asha(v)} for n,v in a.items()}}
            self.assertEqual(set(C.load_old_cache(e,out)),{'current','target','current_only'})
            wrong=copy.deepcopy(e);wrong['arrays']['remove_0']['sha256']='0'*64
            with self.assertRaises(ValueError):C.load_old_cache(wrong,out)
            with p.open('ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):C.load_old_cache(e,out)


if __name__ == '__main__':unittest.main()
