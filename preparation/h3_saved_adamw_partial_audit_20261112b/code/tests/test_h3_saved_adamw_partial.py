"""Independent NumPy/safetensors fixtures: no Torch, H3, dataset or remote IO."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('partial_audit',ROOT/'code/audit_h3_saved_adamw_partial.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
A=P.load_numeric_helpers(ROOT/'preparation/h3_real_adamw_terminal_audit_20261112a')
import adamw_equations as E


def stats(x):
    return {'present':True,'elements':x.size,'finite_count':x.size,'all_finite':True,'nonzero_finite_count':int(np.count_nonzero(x)),
            'nan_count':0,'inf_count':0,'finite_l2':float(np.linalg.norm(x.astype(np.float64)))}


def write_safe(path,values,metadata,provenance=False):
    meta={'openh3_provenance':json.dumps(metadata)} if provenance else metadata
    header={'__metadata__':meta}; offset=0; payload=[]
    for name,value in values.items():
        b=value.tobytes();header[name]={'dtype':'F32','shape':list(value.shape),'data_offsets':[offset,offset+len(b)]};offset+=len(b);payload.append(b)
    raw=json.dumps(header,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(payload))


def fixture(run):
    (run/'arrays').mkdir();r={'arrays':{},'adapter_initial':{'modules':200,'trainable_tensors':400,'rank':8,'tensors':{}},
        'adapter_gradients':{'tensors':{},'expected':400,'present_tensors':400,'all_finite':True,'passed':True,'B_nonzero_tensors':200},
        'parameters_before_update':{},'parameters_after_update':{},'gradient_clipped':{},'optimizer_state_after_update':{},
        'update':{'tensors':{}},'pretrained':{'openh3_pretrained':{'fingerprint':'fixture'}},'contract':{'execution_contract_sha256':'e'*64},
        'input_inspection':{'identity':{'train_source':'fixture'}},'noise_observations':[{'phase':'zero','fixture':True}]}
    def save(role,v):
        v=np.asarray(v,dtype=np.float32);p=run/'arrays'/(role.replace('::','_')+'.npy');np.save(p,v,allow_pickle=False)
        row={'path':str(p.relative_to(run)),'role':role,'shape':list(v.shape),'dtype':'float32','file_sha256':P.file_sha(p),'array_sha256':A.array_sha(v)};r['arrays'][role]=row;return row
    recipe={'class':'torch.optim.AdamW','base_lr':1e-4,'warmup_steps':100,'global_step':1,'actual_lr':E.LR,
        'lr_rule':'base_lr * min(1.0, global_step / warmup_steps)','betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
        'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False,'foreach':False,'fused':False,
        'clip_grad_norm':{'max_norm':1.,'norm_type':2.,'error_if_nonfinite':True,'foreach':False},'loss_scale':1024.}
    r['optimizer']=recipe;shapes={};values={};states={};inv={};b2=0.;changed=0;gnorm=0.
    for n in sorted(P.NAMES):
        is_a=n.endswith('.lora_a');shape=[8,2] if is_a else [2,8];shapes[n]=shape
        before=np.full(shape,.125 if is_a else 0,np.float32);g=np.zeros(shape,np.float32) if is_a else np.full(shape,.001,np.float32)
        raw=g*np.float32(1024);m=np.asarray(.1*g,np.float32);v=np.asarray(.001*g*g,np.float32)
        after=before.copy() if is_a else np.asarray(-E.LR*(m.astype(float)/.1)/(np.sqrt(v.astype(float)/.001)+1e-8),np.float32)
        d=after.astype(float)-before.astype(float);gnorm+=float(np.sum(g.astype(float)**2))
        r['adapter_initial']['tensors'][n]={'shape':shape,'dtype':'torch.float32','device':'cpu','sha256_bytes':A.array_sha(before),'all_zero':not bool(np.count_nonzero(before))}
        r['adapter_gradients']['tensors'][n]={'present':True,'raw':save('gradient_raw__'+n,raw),'unscaled':save('gradient_unscaled__'+n,g),'statistics_raw':stats(raw),'statistics_unscaled':stats(g)}
        r['parameters_before_update'][n]=save('parameters_before_update__'+n,before);r['parameters_after_update'][n]=save('parameters_after_update__'+n,after);r['gradient_clipped'][n]=save('gradient_clipped__'+n,g)
        r['update']['tensors'][n]={'before_sha256':A.array_sha(before),'after_sha256':A.array_sha(after),'all_finite':True,'zero_gradient_coordinates_unchanged':True,'changed_coordinates':int(np.count_nonzero(d)),'update_l2':float(np.linalg.norm(d)),'update_rms':float(np.sqrt(np.mean(d*d)))}
        values[n]=after;inv[n]={}
        for k,x in [('step',np.asarray(1,np.float32)),('exp_avg',m),('exp_avg_sq',v)]:
            nk=n+'::'+k;states[nk]=x;r['optimizer_state_after_update'][nk]=save('optimizer_state_after_update__'+nk,x)
            inv[n][k]={'shape':list(x.shape),'dtype':'torch.float32','device':'cpu','sha256':A.array_sha(x)}
        if not is_a:b2+=stats(raw)['finite_l2']**2;changed+=int(np.count_nonzero(d))
    r['gradient_clip']={'recipe':recipe['clip_grad_norm'],'applied_once':True,'pre_clip_total_norm':float(np.float32(math.sqrt(gnorm)))}
    r['adapter_gradients'].update(B_l2_raw=math.sqrt(b2),B_l2_unscaled=math.sqrt(b2)/1024)
    r['update']['B_changed_coordinates']=changed
    raw=np.array([1024],np.float32);g=np.array([1],np.float32)
    r['retained_video_gradient']={'raw':save('video_gradient_raw',raw),'unscaled':save('video_gradient_unscaled',g),'statistics_raw':stats(raw),'statistics_unscaled':stats(g),'passed':True}
    for n in ('guide','raw_warm_start','target'):save(n,np.zeros(1,np.float32))
    c={'manifest_sha256':'m'*64,'implementation_sha256':{'fixture.py':'s'*64}}
    training={'role':'one_real_TRAIN_AdamW_engineering_step_not_quality_checkpoint','execution_contract_sha256':'e'*64,
        'protocol_sha256':P.CONTRACT_SHA,'manifest_sha256':c['manifest_sha256'],'implementation_sha256':c['implementation_sha256'],
        'input_identity':r['input_inspection']['identity'],'input_arrays':{k:r['arrays'][k] for k in ('guide','raw_warm_start','target')},
        'optimizer':recipe,'steps':1,'loss_scale':1024,'rank':8,'alpha':8,'objective':'original clean proxy RGB MSE only',
        'noise':r['noise_observations'][0],'sigmas':[.909375,6/7,2/3,0.]}
    training['sigmas_float32_sha256']=A.array_sha(np.asarray(training['sigmas'],np.float32))
    meta={'rank':8,'pretrained_fingerprint':'fixture','training':training};p=run/'real_train_AdamW_step1.safetensors';write_safe(p,values,meta,True)
    order=sorted(P.NAMES);ometa={'schema':'openh3.named_AdamW_state.v1','adapter_sha256':P.file_sha(p),'parameter_order':json.dumps(order),'optimizer_recipe':json.dumps(recipe,sort_keys=True),'training_contract':json.dumps(training,sort_keys=True)}
    op=run/'real_train_AdamW_step1_optimizer.safetensors';write_safe(op,states,ometa)
    r['checkpoint']={'path':p.name,'sha256':P.file_sha(p),'bytes':p.stat().st_size,'expected_shapes':shapes,'exact_saved_live_tensors':dict.fromkeys(P.NAMES,True),'metadata':meta,'role':training['role'],
        'optimizer':{'path':op.name,'sha256':P.file_sha(op),'bytes':op.stat().st_size,'state_tensors':1200,'parameter_order':order,'metadata':ometa}}
    r['optimizer_after_update']={'named_entries':400,'state_tensors':1200,'inventory':inv,'parameter_order':order,'arrays':r['optimizer_state_after_update'],
        'parameter_group':dict(lr=E.LR,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,capturable=False,differentiable=False,foreach=False,fused=False)}
    r['complete']=False;r['phase']='clear_and_strict_reload';r['reload']={'cleared':{'parameters_all_zero':True,'optimizer_tensors_all_zero':True,'optimizer_mapping_empty':True}}
    return r,shapes,c


class SavedSubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.artifacts=Path(cls.tmp.name);cls.report,cls.shapes,cls.contract=fixture(cls.artifacts)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_failed_run_still_verifies400_and1200_without_reload(self):
        reader=A.Arrays(self.artifacts,self.report['arrays']);calls=[];load=reader.load
        def once(role,*a,**kw): calls.append(role);return load(role,*a,**kw)
        reader.load=once
        result=P.saved_update(self.report,reader,self.shapes,self.contract,expected_device='cpu',video_shape=[1])
        self.assertTrue(result['passed']);self.assertEqual(result['optimizer_safetensors_keys'],1200);self.assertEqual(len(result['equations']),400)
        self.assertEqual(len(reader.manifest),3202);self.assertEqual(len(calls),len(set(calls)));self.assertFalse(result['strict_reload_accepted']);self.assertFalse(result['merge_accepted'])
    def test_missing_postupdate_state_rejected(self):
        r=copy.deepcopy(self.report);r['optimizer_state_after_update'].pop(next(iter(r['optimizer_state_after_update'])))
        with self.assertRaisesRegex(ValueError,'optimizer inventory'):P.saved_update(r,A.Arrays(self.artifacts,r['arrays']),self.shapes,self.contract,expected_device='cpu',video_shape=[1])
    def test_wrong_scale_and_clip_recipe_rejected(self):
        r=copy.deepcopy(self.report);r['optimizer']['loss_scale']=1.
        with self.assertRaisesRegex(ValueError,'recipe'):P.saved_update(r,A.Arrays(self.artifacts,r['arrays']),self.shapes,self.contract,expected_device='cpu',video_shape=[1])
    def test_saved_array_actual_bytes_not_summary(self):
        role=next(iter(self.report['parameters_after_update'].values()))['role'];row=self.report['arrays'][role];p=self.artifacts/row['path'];old=p.read_bytes()
        try:
            p.write_bytes(old[:-1]+bytes([old[-1]^1]))
            with self.assertRaisesRegex(ValueError,'file SHA'):A.Arrays(self.artifacts,self.report['arrays']).load(role)
        finally:p.write_bytes(old)
    def test_safetensors_content_checked_against_array(self):
        p=self.artifacts/self.report['checkpoint']['path'];old=p.read_bytes()
        try:
            p.write_bytes(old[:-1]+bytes([old[-1]^1]))
            with self.assertRaisesRegex(ValueError,'SHA'):P.saved_update(self.report,A.Arrays(self.artifacts,self.report['arrays']),self.shapes,self.contract,expected_device='cpu',video_shape=[1])
        finally:p.write_bytes(old)
    def test_array_escape_refused(self):
        rows=copy.deepcopy(self.report['arrays']);role=next(iter(rows));rows[role]['path']='../outside.npy'
        with self.assertRaisesRegex(ValueError,'escaped'):A.Arrays(self.artifacts,rows).load(role)
    def test_exact_role_subset_excludes_reload_and_merge(self):
        self.assertEqual(len(P.EXPECTED_ARRAYS),3222)
        self.assertFalse(any('reload' in n or 'merged' in n for n in P.EXPECTED_ARRAYS))


class Equations(unittest.TestCase):
    def args(self,lr=E.LR,beta1=.9,beta2=.999,eps=1e-8,scale=1.):
        x=np.zeros(6,np.float32);g=np.array([0,1e-8,-1e-7,.002,-.3,1],np.float32)
        m=np.asarray((1-beta1)*(g*scale),np.float32);v=np.asarray((1-beta2)*(g*scale)**2,np.float32)
        y=np.asarray(-lr*(m.astype(float)/(1-beta1))/(np.sqrt(v.astype(float)/(1-beta2))+eps),np.float32);y[0]=0
        d=y.astype(float)-x.astype(float);r={'before_sha256':A.array_sha(x),'after_sha256':A.array_sha(y),'all_finite':True,'zero_gradient_coordinates_unchanged':True,'changed_coordinates':int(np.count_nonzero(d)),'update_l2':float(np.linalg.norm(d)),'update_rms':float(np.sqrt(np.mean(d*d)))}
        return x,g,y,m,v,r
    def test_correct_and_wrong_optimizer_meaningful(self):
        self.assertEqual(E.first_step(*self.args())['update']['failed_coordinates'],0)
        for kw in ({'lr':1e-4},{'scale':1024.},{'beta1':.8},{'beta2':.99},{'eps':1e-6}):
            with self.subTest(kw=kw),self.assertRaisesRegex(ValueError,'equation differs'):E.first_step(*self.args(**kw))
    def test_signed_zero_is_not_byte_exact(self):
        args=list(self.args());args[2][0]=np.float32(-0.)
        with self.assertRaisesRegex(ValueError,'bytes changed'):E.first_step(*args)
    def test_unscale_is_exact_and_once(self):
        raw=np.asarray([1,2,3],np.float32);once=raw/np.float32(1024);A.verify_unscale(raw,once)
        with self.assertRaises(ValueError):A.verify_unscale(raw,once/np.float32(1024))
    def test_independent_actual_reference_signedzero_diagnostic(self):
        x=np.asarray([0.],np.float32);y=np.asarray([-0.],np.float32)
        row={'torch_equal':True,'same_shape':True,'finite':True,'reference_shape':[1],'candidate_shape':[1],
             'reference_sha256_float32':A.array_sha(x),'candidate_sha256_float32':A.array_sha(y),'max_abs':0.,'relative_l2':0.,'changed_coordinates':0}
        fact=A.parity_record(row,x,y);self.assertTrue(fact['exact_equal']);self.assertFalse(fact['byte_equal'])

class Scope(unittest.TestCase):
    def test_actual_failed_metadata_does_not_skip_saved_scope(self):
        base=ROOT/'results/takeover_20261111/h3_real_adamw_failed_20261112a'
        r=json.loads((base/'partial.json').read_text());q=json.loads((base/'queue_status.json').read_text())
        value=P.partial_scope(r,q);self.assertTrue(value['passed']);self.assertFalse(value['strict_reload_accepted'])
        r['actual']['NFE']['reloaded']=1
        with self.assertRaisesRegex(ValueError,'budget'):P.partial_scope(r,q)
    def test_actual_partial_metadata_schema_without_arrays(self):
        base=ROOT/'results/takeover_20261111/h3_real_adamw_failed_20261112a'
        r=json.loads((base/'partial.json').read_text());facts=P.metadata_schema_check(r)
        self.assertEqual(facts['array_records'],3222);self.assertEqual(facts['actual_array_reads'],0)
        r['optimizer_after_update']['inventory'][next(iter(P.NAMES))]['step']['shape']=[1]
        with self.assertRaisesRegex(ValueError,'inventory'):P.metadata_schema_check(r)
    def test_actual_clipping_metadata_semantics_and_numeric_values(self):
        base=ROOT/'results/takeover_20261111/h3_real_adamw_failed_20261112a'
        actual=json.loads((base/'partial.json').read_text())
        for mode in ('zero','updated'):
            row=actual['mode_outputs'][mode]['clipping'];numeric={k:row[k] for k in ('at_zero','at_one','outside_unit_interval')}
            self.assertEqual(P.verify_clipping_summary(row,numeric),row)
            wrong=dict(row,at_zero=row['at_zero']+1)
            with self.assertRaisesRegex(ValueError,'summary'):P.verify_clipping_summary(wrong,numeric)
            wrong=dict(row,semantics='extra clamp allowed')
            with self.assertRaisesRegex(ValueError,'semantics'):P.verify_clipping_summary(wrong,numeric)
            wrong=dict(row);wrong.pop('semantics')
            with self.assertRaisesRegex(ValueError,'semantics'):P.verify_clipping_summary(wrong,numeric)
    def test_continuation_needs_byte_exact_each_stream_and_full_audit(self):
        pair={s:{'byte_equal':True,'exact_equal':True} for s in ('video','audio','decoder_input','rgb')}
        facts={'paired_outputs':{'reference__zero':pair}}
        self.assertTrue(P.continuation_readiness(True,facts)['ready'])
        pair['rgb']['byte_equal']=False # numeric equal signed-zero still ineligible
        self.assertFalse(P.continuation_readiness(True,facts)['ready'])
        pair['rgb']['byte_equal']=True;self.assertFalse(P.continuation_readiness(False,facts)['ready'])
        del pair['audio'];self.assertFalse(P.continuation_readiness(True,facts)['ready'])

if __name__=='__main__':unittest.main(verbosity=2)
