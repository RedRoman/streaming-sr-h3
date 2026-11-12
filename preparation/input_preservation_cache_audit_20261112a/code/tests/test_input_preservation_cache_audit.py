"""Small independent NumPy fixtures; no real data, model, Torch or NPU."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import audit_input_preservation_cache as A


def frozen_label_math():
    # Reference is only the frozen pure NumPy functions; never import the
    # exporter or the model/feature head. The auditor does not use this code.
    p=CODE/'rphi_input_preservation.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==A.INPUT_MODULE_SHA
    tree=ast.parse(p.read_text())
    methods=[n for n in tree.body if isinstance(n,ast.FunctionDef) and
             n.name in ('normalize_input_benefit','input_benefit_labels')]
    env={'np':np,'TILE':16,'UTILITY_SCALE':.25/255}
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(p),'exec'),env)
    return env['input_benefit_labels']


LABELS=frozen_label_math()


def numeric_fixture(batch=2):
    rng=np.random.default_rng(107)
    current=rng.random((batch,3,96,96),dtype=np.float32)
    target=rng.random(current.shape,dtype=np.float32)
    k1=(rng.random(current.shape,dtype=np.float32)-np.float32(.1))*np.float32(1.2)
    return current,k1,target


def batch_fixture(current,k1,target):
    return {'k1_rgb':k1,'raw_features':np.zeros((len(current),2,11,6,6),np.float32),
            'availability':np.ones((len(current),2),np.float32),**LABELS(current,k1,target)}


def npz_fixture(folder,arrays):
    (folder/'arrays').mkdir(exist_ok=True);p=folder/'arrays'/'batch_0000.npz';np.savez(p,**arrays)
    return {'path':str(p),'sha256':A.T.file_sha(p),'bytes':p.stat().st_size,
            'arrays':{k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':A.asha(v)} for k,v in arrays.items()}}


def diagnostic_fixture(current,k1,target):
    raw=LABELS(current,k1,target);clipped=LABELS(current,np.clip(k1,0,1),target)
    return {'diagnostic_input_mae':raw['input_mae'],'diagnostic_K1_mae':raw['k1_mae'],
            'diagnostic_clipped_input_mae':clipped['input_mae'],'diagnostic_clipped_K1_mae':clipped['k1_mae'],
            'oracle_unclamped_index':(raw['raw_benefit']>0).astype(np.uint8),
            'oracle_clipped_index':(clipped['raw_benefit']>0).astype(np.uint8)}


class CacheAuditTests(unittest.TestCase):
    def test_independent_FP64_tiles_match_frozen_vectorized_reduction(self):
        current,k1,target=numeric_fixture();a=batch_fixture(current,k1,target);errors={}
        A.verify_labels(a,current,target,errors)
        self.assertEqual(A.independent_tiles(current,target).shape,(2,1,6,6))
        self.assertLessEqual(max(errors.values()),1e-12)
        self.assertGreater(float(np.max(a['k1_rgb'])),1.)
        self.assertLess(float(np.min(a['k1_rgb'])),0.)
        changed=copy.deepcopy(a);changed['normalized_benefit'][0,0,0,0]+=np.float32(.01)
        with self.assertRaisesRegex(ValueError,'normalization'):A.verify_labels(changed,current,target,{})

    def test_actual_target_and_K1_tamper_fail_label_recompute(self):
        current,k1,target=numeric_fixture();a=batch_fixture(current,k1,target)
        for key in ('target','K1'):
            b=copy.deepcopy(a);y=target.copy()
            if key=='target':y[0,0,0,0]+=np.float32(.2)
            else:b['k1_rgb'][0,0,0,0]+=np.float32(.2)
            with self.assertRaisesRegex(ValueError,'reduction differs'):A.verify_labels(b,current,y,{})
        invalid=k1.copy();invalid[0,0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):A.independent_tiles(invalid,target)

    def test_two_candidate_unclipped_clipped_tie_and_null_PSNR(self):
        current=np.full((1,3,96,96),.4,np.float32);target=np.zeros_like(current);k1=np.full_like(current,-.5)
        a=diagnostic_fixture(current,k1,target);rows=A.dev_diagnostic(a,current,k1,target,{})
        self.assertFalse(a['oracle_unclamped_index'].any());self.assertTrue(a['oracle_clipped_index'].all())
        self.assertIsNone(rows[0]['clipped_tile_oracle']['clipped_psnr'])
        self.assertTrue(rows[0]['clipped_tile_oracle']['clipped_exact_match'])
        self.assertEqual(rows[0]['clipped_tile_oracle']['unclamped_mae'],.5)
        # A +0/-0 input/K1 tie must select input, not restoration.
        current=np.zeros_like(current);k1=-current
        a=diagnostic_fixture(current,k1,target);A.dev_diagnostic(a,current,k1,target,{})
        self.assertFalse(a['oracle_unclamped_index'].any())
        a['oracle_unclamped_index'][0,0,0,0]=1
        with self.assertRaisesRegex(ValueError,'threshold/tie'):A.dev_diagnostic(a,current,k1,target,{})

    def test_actual_NPZ_bytes_all_arrays_and_declared_shape_are_checked(self):
        current,k1,target=numeric_fixture();a=batch_fixture(current,k1,target)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);entry=npz_fixture(folder,a)
            loaded=A.load_npz(entry,folder,A.expected_shapes('train',2))
            self.assertEqual(set(loaded),A.TRAIN_NAMES)
            bad=copy.deepcopy(entry);bad['arrays']['raw_features']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'array SHA'):A.load_npz(bad,folder,A.expected_shapes('train',2))
            bad=copy.deepcopy(entry);bad['arrays']['extra']={}
            with self.assertRaisesRegex(ValueError,'keys differ'):A.load_npz(bad,folder,A.expected_shapes('train',2))
            with Path(entry['path']).open('ab') as f:f.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'file binding'):A.load_npz(entry,folder,A.expected_shapes('train',2))

    def test_per_row_role_teacher_and_payload_bindings_are_exact(self):
        current,k1,target=numeric_fixture();a=batch_fixture(current,k1,target);rows=[]
        e={'sha256':'c'*64,'row_indices':[0,1],'row_sha256':['a'*64,'b'*64]}
        for i in range(2):
            r={'row_index':i,'row_sha256':e['row_sha256'][i],'cache_role':'train','usable_for_training':True,
               'teacher_checkpoint_sha256':A.TEACHER_SHA,'cache_file_sha256':e['sha256'],
               'feature_sha256_float32':A.asha(a['raw_features'][i]),'availability':a['availability'][i].tolist(),
               'array_row_sha256':{k:A.asha(v[i]) for k,v in a.items()}}
            r['cache_record_sha256']=A.T.value_sha(r);rows.append(r)
        A.verify_record_records(e,rows,a,'train')
        bad=copy.deepcopy(rows);bad[0]['cache_role']='dev';bad[0]['cache_record_sha256']=A.T.value_sha({k:v for k,v in bad[0].items() if k!='cache_record_sha256'})
        with self.assertRaisesRegex(ValueError,'identity/role'):A.verify_record_records(e,bad,a,'train')
        bad=copy.deepcopy(a);bad['raw_features'][0,0,0,0,0]=1
        with self.assertRaisesRegex(ValueError,'per-row'):A.verify_record_records(e,rows,bad,'train')

    def test_native_source_profile_balancing_excludes_controls(self):
        rows=[]
        for source,values in [('a',[1.,2.,3.]),('b',[3.,4.,5.])]:
            for profile,value in list(zip(('p1','p2','p3'),values))+[('identity',100.),('near_clean',200.)]:
                for frame in range(1 if source=='a' else 2):
                    m={k:value for k in ('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse')}
                    d={k:dict(m) for k in ('input','K1','unclamped_tile_oracle','clipped_tile_oracle')}
                    d.update(raw_oracle_K1_tile_fraction=.5,clipped_oracle_K1_tile_fraction=.25)
                    rows.append({'source_id':source,'profile_id':profile,'severity_phase':'mild','two_candidate_diagnostic':d})
        s=A.summary(rows)
        self.assertEqual(s['native_equal_source_equal_profile']['K1']['unclamped_mae'],3.)
        self.assertEqual(len(s['per_source_profile_phase']),20)
        self.assertTrue(any(r['profile_id']=='identity' for r in s['per_source_profile_phase']))
        with self.assertRaisesRegex(ValueError,'three native profiles'):A.summary([r for r in rows if r['profile_id']!='p3'])

    def test_float_tolerance_does_not_relax_roles_or_boolean_gates(self):
        errors={};A.compare({'metric':1.+1e-13,'psnr':30.+1e-11,'passed':True},
                          {'metric':1.,'psnr':30.,'passed':True},errors)
        with self.assertRaisesRegex(ValueError,'exact value'):A.compare({'passed':1},{'passed':True},{})
        with self.assertRaisesRegex(ValueError,'metric differs'):A.compare(1.+2e-12,1.,{})
        with self.assertRaisesRegex(ValueError,'nonfinite'):A.compare(float('nan'),1.,{})

    def test_output_inventory_rejects_extra_or_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for n in ('manifest.json','partial.json','records.jsonl','batches.jsonl'):(p/n).write_text('{}\n')
            entry=npz_fixture(p,{'x':np.zeros(1,np.float32)})
            r={'actual_output_bytes':sum(q.stat().st_size for q in p.rglob('*') if q.is_file())}
            v=A.verify_output_inventory(r,p,[entry]);self.assertEqual(v['files'],5)
            (p/'extra').write_text('not part of producer')
            with self.assertRaisesRegex(ValueError,'membership'):A.verify_output_inventory(r,p,[entry])

    def test_small_archive_contains_no_NPZ_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'audit';out.mkdir()
            for n in ('audit.json','partial.json'):(out/n).write_text('{"complete":true}\n')
            c=root/'contract.json';c.write_text('{}\n');m=root/'producer.json';m.write_text('{}\n')
            A.pack_small(out,c,m)
            receipt=json.loads((out/'archive_receipt.json').read_text())
            self.assertEqual(receipt['sha256'],A.T.file_sha(out/'evidence.tar.gz'))
            self.assertFalse(receipt['NPZ_or_checkpoint_arrays_included'])
            with tarfile.open(out/'evidence.tar.gz') as tar:
                self.assertEqual(len(tar.getmembers()),6)
                self.assertFalse(any(n.endswith(('.npz','.pt','.safetensors')) for n in tar.getnames()))

    def test_old_teacher_data_chain_is_independent_of_producer_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=('teacher_contract','teacher_report','teacher_checkpoint','dataset_audit','headroom_contract',
                   'headroom_report','headroom_audit','schedule_manifest','student_schedule','headroom_records')
            files={n:{'path':str(root/(n+'.json')),'sha256':'a'*64} for n in names}
            files['teacher_checkpoint']['sha256']=A.TEACHER_SHA;files['headroom_report']['sha256']=A.HEADROOM_SHA
            files['headroom_audit']['sha256']=A.HEADROOM_AUDIT_SHA
            teacher={'model_config':A.T.CONFIG,'optimizer':A.T.OPTIMIZER,'loss':A.T.LOSS,'phase':'teacher','steps':1000,
                     'files':{k:files[k] for k in ('schedule_manifest','student_schedule','dataset_audit')}}
            terminal={'complete':True,'completed_step':1000,'final_frozen':True,'execution_contract':copy.deepcopy(teacher),
                      'execution_contract_file_sha256':files['teacher_contract']['sha256'],'final_model_sha256':'b'*64,
                      'data':{'pair':{'train_sources':19,'dev_sources':6}}}
            h={'complete':True,'status':'complete_headroom_fail','assessment':{'headroom_rule_passed':False},
               'execution_contract':{k:files[k] for k in ('teacher_contract','teacher_report','teacher_checkpoint','dataset_audit','headroom_contract')},
               'frozen_before_sha256':'b'*64,'frozen_after_sha256':'b'*64,'records':{**files['headroom_records'],'rows':2880},
               'actual_rgb_caches':[{}]*180}
            audit={'complete':True,'report_sha256':A.HEADROOM_SHA,'actual_cache_files':180,'actual_arrays':1080,'actual_rows':2880}
            data={'complete':True,'revised_pair_ready':True,'pair':terminal['data']['pair']}
            paths={k:Path(v['path']) for k,v in files.items()}
            for k,value in [('headroom_report',h),('headroom_audit',audit),('dataset_audit',data)]:paths[k].write_text(json.dumps(value))
            A.verify_identity_chain({'files':files},paths,teacher,terminal)
            changed=copy.deepcopy(files);changed['teacher_contract']['sha256']='c'*64
            with self.assertRaisesRegex(ValueError,'identity link'):A.verify_identity_chain({'files':changed},paths,teacher,terminal)
            changed=copy.deepcopy(terminal);changed['final_frozen']=False
            with self.assertRaisesRegex(ValueError,'final checkpoint'):A.verify_identity_chain({'files':files},paths,teacher,changed)
            changed=copy.deepcopy(terminal);changed['execution_contract']['steps']=999
            with self.assertRaisesRegex(ValueError,'final checkpoint'):A.verify_identity_chain({'files':files},paths,teacher,changed)

    def test_audit_location_cannot_mutate_or_nest_within_old_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'producer';old.mkdir()
            for out in (old,old/'child'):
                with self.assertRaisesRegex(ValueError,'inside producer'):A.fresh_audit_location(out,old)
            with self.assertRaisesRegex(ValueError,'already exists'):A.fresh_audit_location(root,old)
            self.assertEqual(A.fresh_audit_location(root/'fresh',old),(root/'fresh').resolve())


if __name__=='__main__':unittest.main()
