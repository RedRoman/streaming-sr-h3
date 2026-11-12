"""NumPy-only evidence fixtures. No Torch/model/NPU import or dispatch."""
import copy
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_terminal as A
import audit_checkpoint as C
import finalize_once as F
import small_archive as S
from audit_arrays import array_sha, file_sha, verify_unscale, compare_outputs


def jwrite(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True))
    return file_sha(path)


def save_safe(path,values,meta,mutate=None):
    head={'__metadata__':{'openh3_provenance':json.dumps(meta,sort_keys=True)}};chunks=[];offset=0
    for name,value in sorted(values.items()):
        data=value.astype('<f4').tobytes();head[name]={'dtype':'F32','shape':list(value.shape),'data_offsets':[offset,offset+len(data)]}
        chunks.append(data);offset+=len(data)
    if mutate:mutate(head)
    raw=json.dumps(head,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(chunks))
    return file_sha(path)


def update_row(before,grad,after):
    delta=(after-before).astype(np.float64)
    return {'before_sha256':array_sha(before),'gradient_unscaled_sha256':array_sha(grad),'after_sha256':array_sha(after),
        'expected_after_sha256':array_sha(after),'exact_SGD_arithmetic':True,'all_finite':True,
        'update_l2':float(np.linalg.norm(delta)),'update_rms':float(np.sqrt(np.mean(delta*delta))),
        'changed_coordinates':int(np.count_nonzero(delta)),'version':1}


def stats(value):
    return {'present':True,'elements':int(value.size),'finite_count':int(value.size),'all_finite':True,
        'nonzero_finite_count':int(np.count_nonzero(value)),'nan_count':0,'inf_count':0,
        'finite_l2':float(np.linalg.norm(value.astype(np.float64)))}


def budget_report():
    singles=('RGB_loss_expressions','backward_attempts','completed_backward_calls','optimizer_steps',
        'adapter_updates','checkpoint_saves','clear_reload_cycles','merge_calls','unscale_calls')
    return {'actual':{'NFE':{m:3 for m in A.MODES},'VAE_decode_calls':{m:1 for m in A.MODES},
        **{k:1 for k in singles},'VAE_encode_calls':0,'stage_decode_calls':{m+'_decode':1 for m in A.MODES},'forbidden_calls':[]},
        'block_forward_calls':{'DiT':{m:{str(i):3 for i in range(50)} for m in (*A.MODES,'backward')},
        'VAE':{m:{str(i):1 for i in range(36)} for m in (*(x+'_decode' for x in A.MODES),'backward')}},
        'budget_passed':True,'block_budget_passed':True}


def canonical(rows):
    data=[{k:r[k] for k in ('name','shape','dtype','sha256_bytes')} for r in rows]
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inventories():
    initial={}
    for group,count,buffers in (('DiT',530,2),('VAE',559,42)):
        names=sorted(m+'.weight' for m in A.MODULES) if group=='DiT' else []
        names += ['untargeted_'+str(i) for i in range(count-len(names))]
        def row(name):
            return {'name':name,'shape':[1,1],'dtype':'torch.bfloat16' if group=='DiT' else 'torch.float16',
                'sha256_bytes':hashlib.sha256(name.encode()).hexdigest(),'version':1,'device':'npu:0',
                'requires_grad':False,'grad_is_none':True}
        initial[group]={'parameters':[row(n) for n in names],'registered_buffers':[row('buffer_'+str(i)) for i in range(buffers)]}
        for kind in ('parameters','registered_buffers'):initial[group][kind+'_sha256']=canonical(initial[group][kind])
    final=copy.deepcopy(initial);layers={}
    for row in final['DiT']['parameters']:
        if row['name'].removesuffix('.weight') not in A.MODULES:continue
        name=row['name'].removesuffix('.weight');before=row['sha256_bytes'];row['sha256_bytes']=hashlib.sha256((name+'updated').encode()).hexdigest();row['version']=2
        layers[name]={'base_before_sha256':before,'base_before_version':1,'expected_after_sha256':row['sha256_bytes'],
            'actual_after_sha256':row['sha256_bytes'],'actual_after_version':2,'exact_declared_write':True,'all_finite':True,
            'shape':[1,1],'dtype':'torch.bfloat16','scale':1.,'ideal_delta_l2':.1,'storage_delta_l2':.05,
            'rounding_residual_l2':.05,'changed_weight_count':1,'changed_weight_fraction':1.,
            'nonzero_ideal_rounded_to_zero_count':0,'zero_denominator':False,'rounding_residual_relative_l2':.5}
    final['DiT']['parameters_sha256']=canonical(final['DiT']['parameters'])
    stages={'after_injection','after_RGB_backward','after_update_and_save','after_reload','after_declared_merge','finally'}|{m+s for m in A.MODES for s in ('_before_decode','_after_decode','_complete')}
    state={}
    for label in stages:
        merged=label in {'after_declared_merge','merged_before_decode','merged_after_decode','merged_complete','finally'}
        state[label]={}
        for group,count in [('DiT',532),('VAE',601)]+([] if merged else [('adapters',400)]):
            r={'passed':True,'failures':[],'tensors_checked':count}
            if group!='adapters':
                full=label in {'after_RGB_backward','after_update_and_save','updated_complete','reloaded_complete','merged_complete','finally'}
                if full:r['checks']=['canonical name','object','version','device','bytes','frozen','grad None']
                else:r['byte_hash']='separately checked at full stage gates, not recomputed at this boundary'
            state[label][group]=r
    report={'state_checks':state,'merge':{'layers':layers,'expected_writes':200,'remaining_adapter_tensors':0,'equivalence_claim':False,
        'runtime_receipt':{'merged_modules':200,'live_adapter_parameters':0,'storage_dtypes':['torch.bfloat16']},
        'declared_mutation_check':{'passed':True,'failures':[],'tensors_checked':532}}}
    return initial,final,report


def gradient_fixture(directory):
    directory.mkdir(parents=True);(directory/'arrays').mkdir()
    report={'arrays':{},'adapter_initial':{'modules':200,'trainable_tensors':400,'rank':8,'tensors':{}},
        'parameters_before_update':{},'parameters_after_update':{},'adapter_gradients':{'tensors':{}},
        'update':{'tensors':{},'optimizer_state_entries_after':0},'reload':{'cleared':{},'reloaded':{},'loaded_count':400},
        'pretrained':{'openh3_pretrained':{'fingerprint':'synthetic_metadata'}},'contract':{'execution_contract_sha256':'1'*64},
        'noise_observations':[{'phase':'zero','synthetic':'no runtime'}]}
    def put(role,value):
        p=directory/'arrays'/(role+'.npy');np.save(p,value,allow_pickle=False)
        row={'role':role,'path':'arrays/'+p.name,'file_sha256':file_sha(p),'array_sha256':array_sha(value),
            'shape':list(value.shape),'dtype':str(value.dtype)};report['arrays'][role]=row;return row
    for role,shape in [('guide',A.VIDEO),('raw_warm_start',A.VIDEO),('target',[1,1,128,1])]:put(role,np.ones(shape,np.float32))
    shapes={n:([8,1] if n.endswith('.lora_a') else [1,8]) for n in A.NAMES};values={};b2=0.;B=0
    for name in sorted(A.NAMES):
        b=name.endswith('.lora_b');shape=shapes[name]
        before=np.full(shape,0 if b else .25,np.float32);grad=np.full(shape,1 if b else 0,np.float32)
        raw=grad*np.float32(1024);after=before+np.float32(-1e-4)*grad;values[name]=after
        report['adapter_initial']['tensors'][name]={'shape':shape,'dtype':'torch.float32','all_zero':b,'sha256_bytes':array_sha(before)}
        report['adapter_gradients']['tensors'][name]={'present':True,'raw':put('gradient_raw__'+name,raw),
            'unscaled':put('gradient_unscaled__'+name,grad),'statistics_raw':stats(raw),'statistics_unscaled':stats(grad)}
        for prefix,value in [('parameters_before_update',before),('parameters_after_update',after)]:report[prefix][name]=put(prefix+'__'+name,value)
        report['update']['tensors'][name]=update_row(before,grad,after)
        report['reload']['cleared'][name]={'all_zero':True,'sha256':array_sha(np.zeros(shape,np.float32)),'version':2}
        report['reload']['reloaded'][name]={'equals_saved':True,'sha256':array_sha(after),'version':3}
        if b:b2+=stats(raw)['finite_l2']**2;B+=int(np.count_nonzero(after))
    report['adapter_gradients'].update(expected=400,present_tensors=400,all_finite=True,passed=True,B_nonzero_tensors=200,
        B_l2_raw=b2**.5,B_l2_unscaled=b2**.5/1024.)
    report['update']['B_nonzero_coordinates']=B
    raw=np.ones(A.VIDEO,np.float32)*1024;unscaled=raw/np.float32(1024)
    report['retained_video_gradient']={'raw':put('video_gradient_raw',raw),'unscaled':put('video_gradient_unscaled',unscaled),
        'statistics_raw':stats(raw),'statistics_unscaled':stats(unscaled),'passed':True}
    opt={'type':'torch.optim.SGD','lr':1e-4,'momentum':0.,'dampening':0.,'weight_decay':0.,'nesterov':False,
        'maximize':False,'foreach':False,'fused':False,'loss_scale':1024.,
        'expected_arithmetic':'device FP32 before.clone().add_(unscaled_grad, alpha=-1e-4)','loss_descent_required':False}
    report['optimizer']=opt;contract={'manifest_sha256':'2'*64,'implementation_sha256':{'synthetic.py':'3'*64}}
    training={'execution_contract_sha256':'1'*64,'source_contract_sha256':A.CONTRACT_SHA,**contract,'pretrained':report['pretrained'],
        'input_arrays':{r:report['arrays'][r] for r in ('guide','raw_warm_start','target')},'noise':report['noise_observations'][0],
        'rank':8,'alpha':8,'lr':1e-4,'steps':1,'loss_scale':1024,'optimizer':opt,'objective':'synthetic RGB MSE only',
        'role':'diagnostic_synthetic_plain_SGD_step1_not_a_restoration_checkpoint','sigmas':[.909375,6/7,2/3,0.]}
    training['sigmas_float32_sha256']=array_sha(np.asarray(training['sigmas'],np.float32))
    meta={'pretrained_fingerprint':'synthetic_metadata','rank':8,'training':training}
    path=directory/'diagnostic_synthetic_SGD_step1.safetensors';digest=save_safe(path,values,meta)
    report['checkpoint']={'path':path.name,'sha256':digest,'bytes':path.stat().st_size,'expected_shapes':shapes,
        'exact_saved_live_tensors':{n:True for n in A.NAMES},'metadata':meta,'role':training['role']}
    report['reload'].update(metadata=meta,file_sha_after=digest)
    return report,shapes,contract


class ArithmeticAndFiles(unittest.TestCase):
    def test_zero_coordinate_hard_bytes_and_same_device_attestation(self):
        before=np.asarray([.25,0.,0.],np.float32);grad=np.asarray([0.,1.,-2.],np.float32);after=before+grad*np.float32(-1e-4)
        result=C.update_tensor(before,grad,after,update_row(before,grad,after));self.assertEqual(result['independent_zero_gradient_coordinates'],1)
        for index in (0,1):
            corrupt=after.copy();corrupt[index]=np.nextafter(corrupt[index],np.float32(np.inf))
            with self.assertRaisesRegex(ValueError,'zero-'):C.update_tensor(before,grad,corrupt,update_row(before,grad,corrupt))
        row=update_row(before,grad,after);row['expected_after_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'same-device'):C.update_tensor(before,grad,after,row)

    def test_nontrivial_CPU_candidates_descriptive_not_union_gate(self):
        before=np.asarray([1.],np.float32);grad=np.asarray([1.],np.float32);after=np.asarray([.75],np.float32)
        result=C.update_tensor(before,grad,after,update_row(before,grad,after))
        self.assertEqual(result['descriptive_nontrivial_split_matches'],0)
        self.assertFalse(result['nontrivial_candidate_union_is_acceptance_gate'])
        # Passing here proves only the reported same-device attestation plus saved
        # array integrity. It intentionally does not prove a unique NPU FMA.

    def test_unscale_exact_and_nonfinite_rejected(self):
        raw=np.asarray([1024.,-2048.],np.float32);unscaled=raw/1024
        verify_unscale(raw,unscaled,nonzero=True)
        with self.assertRaisesRegex(ValueError,'bytes'):verify_unscale(raw,unscaled/1024)
        raw[0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):verify_unscale(raw,unscaled)

    def test_strict_safetensor_key_dtype_shape_metadata_offsets_and_finite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.safetensors';v={'x':np.arange(8,dtype=np.float32).reshape(1,8)};shapes={'x':[1,8]};meta={'role':'fixture'}
            C.SafeAdapter(p,save_safe(p,v,meta),shapes,meta).load('x')
            mutations=[lambda h:h['x'].update(dtype='F16'),lambda h:h['x'].update(shape=[8,1]),
                lambda h:h['x'].update(data_offsets=[4,36]),lambda h:h.update(extra=h['x']),lambda h:h['__metadata__'].update(extra='bad')]
            for mutate in mutations:
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):C.SafeAdapter(p,save_safe(p,v,meta,mutate),shapes,meta)
            with self.assertRaisesRegex(ValueError,'provenance'):C.SafeAdapter(p,save_safe(p,v,meta),shapes,{})
            v['x'][0,0]=np.inf
            with self.assertRaisesRegex(ValueError,'nonfinite'):C.SafeAdapter(p,save_safe(p,v,meta),shapes,meta).load('x')

    def test_complete_400_key_gradient_update_checkpoint_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'eval';report,shapes,contract=gradient_fixture(run)
            result=A.gradients_update_checkpoint(report,A.Arrays(run,report['arrays']),shapes,contract)
            self.assertEqual(result['adapter_pairs_checked'],400);self.assertEqual(result['B_nonzero_gradient_tensors'],200)
            bad=copy.deepcopy(report);name=sorted(A.NAMES)[0];bad['reload']['reloaded'][name]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'reloaded bytes'):A.gradients_update_checkpoint(bad,A.Arrays(run,bad['arrays']),shapes,contract)
            bad=copy.deepcopy(report);bad['adapter_gradients']['tensors'][name]['present']=False
            with self.assertRaisesRegex(ValueError,'presence'):A.gradients_update_checkpoint(bad,A.Arrays(run,bad['arrays']),shapes,contract)
            bad=copy.deepcopy(report);bad['optimizer']['weight_decay']=.01
            with self.assertRaisesRegex(ValueError,'optimizer contract'):A.gradients_update_checkpoint(bad,A.Arrays(run,bad['arrays']),shapes,contract)

    def test_actual_npy_tamper_and_shape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'arrays').mkdir();value=np.zeros((2,3),np.float32);np.save(p/'arrays/x.npy',value)
            row={'role':'x','path':'arrays/x.npy','dtype':'float32','shape':[2,3],'file_sha256':file_sha(p/'arrays/x.npy'),'array_sha256':array_sha(value)}
            A.Arrays(p,{'x':row}).load('x',[2,3])
            with self.assertRaisesRegex(ValueError,'metadata'):A.Arrays(p,{'x':row}).load('x',[3,2])
            value[0,0]=1;np.save(p/'arrays/x.npy',value)
            with self.assertRaisesRegex(ValueError,'file SHA'):A.Arrays(p,{'x':row}).load('x',[2,3])

    def test_all_four_RGB_arrays_MSE_decoder_cast_and_strict_reload(self):
        # Fixed actual CPU Torch-linspace bytes, also bound by TARGET_SHA.
        encoded="zcxMPkejUT7AeVY+OlBbPrQmYD4u/WQ+qNNpPiGqbj6bgHM+FVd4Po4tfT4EAoE+QW2DPn7YhT67Q4g++K6KPjQajT5xhY8+rvCRPutblD4ox5Y+ZTKZPqKdmz7fCJ4+HHSgPljfoj6VSqU+0rWnPg8hqj5MjKw+ifeuPsZisT4CzrM+Pzm2PnykuD65D7s+9nq9PjPmvz5wUcI+rbzEPuonxz4mk8k+Y/7LPqBpzj7d1NA+GkDTPler1T6UFtg+0IHaPg3t3D5KWN8+h8PhPsQu5D4BmuY+PgXpPntw6z642+0+9EbwPjGy8j5uHfU+q4j3Pujz+T4lX/w+Ysr+Ps+aAD9u0AE/DAYDP6s7BD9JcQU/6KYGP4bcBz8kEgk/w0cKP2F9Cz8Asww/nugNPz0eDz/bUxA/eokRPxi/Ej+29BM/VSoVP/NfFj+SlRc/MMsYP88AGj9tNhs/DGwcP6qhHT9I1x4/5wwgP4VCIT8keCI/wq0jP2HjJD//GCY/nU4nPzyEKD/auSk/ee8qPxclLD+2Wi0/VJAuP/LFLz+R+zA/LzEyP85mMz9snDQ/C9I1P6kHNz9IPTg/5nI5P4SoOj8j3js/wRM9P2BJPj/+fj8/nbRAPzvqQT/aH0M/eFVEPxaLRT+1wEY/U/ZHP/IrST+QYUo/L5dLP83MTD8="
        target=np.frombuffer(base64.b64decode(encoded),np.float32).reshape(1,1,128,1)
        loaded={role:np.full(shape,.4,dtype=dtype) for role,(shape,dtype) in A.SMALL.items()};loaded['target']=target
        def parity(a,b):
            delta=(a-b).astype(np.float64)
            return {'torch_equal':bool(np.array_equal(a,b)),'same_shape':True,'finite':True,
                'reference_shape':list(a.shape),'candidate_shape':list(b.shape),'reference_sha256_float32':array_sha(a),
                'candidate_sha256_float32':array_sha(b),'max_abs':float(np.max(np.abs(a-b))),
                'relative_l2':float(np.linalg.norm(delta)/max(float(np.linalg.norm(a.astype(np.float64))),1e-30))}
        report={'latent_parity':{x:parity(loaded['reference_'+x],loaded['zero_'+x]) for x in ('video','audio')},
            'rgb_parity':{**parity(loaded['reference_rgb'],loaded['zero_rgb']),'allclose':True,'atol':.0002,'rtol':.0001},
            'graph':{k:True for k in ('video_requires_grad','video_has_grad_fn','rgb_requires_grad','rgb_has_grad_fn')},'mode_outputs':{},'comparisons':{}}
        for mode in A.MODES:
            pixel=loaded[mode+'_rgb'];mse=float(np.mean((pixel.astype(np.float64)-target.astype(np.float64))**2))
            report['mode_outputs'][mode]={'geometry_and_finite':True,'numpy_float64_mse':mse,'torch_float32_mse':float(np.float32(mse)),
                'clipping':{'at_zero':0,'at_one':0,'outside_unit_interval':0}}
        report['loss']={**{k:report['mode_outputs']['zero'][k] for k in ('numpy_float64_mse','torch_float32_mse')},'scale':1024,'auxiliary_latent_audio_loss':False}
        for key in ('updated__reloaded','zero__updated','zero__reloaded','zero__merged','reloaded__merged'):
            a,b=key.split('__');report['comparisons'][key]={x:parity(loaded[a+'_'+x],loaded[b+'_'+x]) for x in ('video','audio','rgb')}
        result=A.outputs(report,loaded,{r:loaded[r] for r in ('guide','raw_warm_start','target','reference_video','reference_audio','reference_rgb','reference_decoder_input')})
        self.assertFalse(result['merge_equivalence_acceptance']);self.assertFalse(result['loss_descent_required'])
        bad=copy.deepcopy(report);bad['mode_outputs']['updated']['numpy_float64_mse']+=.001
        with self.assertRaisesRegex(ValueError,'RGB MSE'):A.outputs(bad,loaded,{})
        changed=loaded.copy();changed['merged_decoder_input']=loaded['merged_decoder_input']+np.float16(.1)
        with self.assertRaisesRegex(ValueError,'actual float16 cast'):A.outputs(report,changed,{})

    def test_same_mode_strict_parity_does_not_share_merge_tolerance(self):
        a=np.zeros(10,np.float32);b=a.copy();b[0]=1e-8
        with self.assertRaisesRegex(ValueError,'strict'):compare_outputs(b,a,exact=True)
        compare_outputs(b,a,exact=False)


class Boundaries(unittest.TestCase):
    def test_exact_budget_and_recompute_count(self):
        report=budget_report();self.assertEqual(A.budget(report,True)['explicit_NFE'],12)
        for mutate in (lambda r:r['actual']['NFE'].update(zero=4),lambda r:r['actual'].update(optimizer_steps=2),
            lambda r:r['block_forward_calls']['DiT']['backward'].update({'49':2})):
            bad=copy.deepcopy(report);mutate(bad)
            with self.assertRaises(ValueError):A.budget(bad,True)
        partial=budget_report();partial['actual']['NFE']['merged']=0
        A.budget(partial,False)
        with self.assertRaises(ValueError):A.budget(partial,True)

    def test_merge_200_allowed_writes_all_other_parameters_buffers_frozen(self):
        initial,final,report=inventories()
        result=A.frozen_merge(report,initial,final,initial);self.assertEqual(result['declared_weight_write_tensors'],200)
        for group,kind,index in [('DiT','parameters',200),('DiT','registered_buffers',0),('VAE','parameters',0),('VAE','registered_buffers',0)]:
            bad=copy.deepcopy(final);bad[group][kind][index]['sha256_bytes']='0'*64
            with self.subTest(group=group,kind=kind),self.assertRaisesRegex(ValueError,'unexpected'):A.frozen_merge(report,initial,bad,initial)
        bad=copy.deepcopy(report);name=sorted(A.MODULES)[0];bad['merge']['layers'][name]['actual_after_version']=3
        with self.assertRaisesRegex(ValueError,'declared merge'):A.frozen_merge(bad,initial,final,initial)
        bad=copy.deepcopy(report);bad['state_checks']['after_RGB_backward']['DiT'].pop('checks')
        with self.assertRaises(KeyError):A.frozen_merge(bad,initial,final,initial)

    def test_canonical_inventory_recomputed_not_just_reported(self):
        initial,_,_=inventories()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);row={'path':'frozen_initial.json','sha256':jwrite(p/'frozen_initial.json',initial)}
            self.assertEqual(len(A.inventory(row,p)['DiT']['parameters']),530)
            initial['VAE']['registered_buffers'][0]['sha256_bytes']='0'*64;row['sha256']=jwrite(p/'frozen_initial.json',initial)
            with self.assertRaisesRegex(ValueError,'canonical'):A.inventory(row,p)

    def test_contexts_all_930_block_entries_and_76_nested_restores(self):
        report=budget_report();state={'in_training':True,'jit_compile_false':True,'matmul_allow_hf32':False,'conv_allow_hf32':False,
            'deterministic_algorithms':False,'deterministic_warn_only':False,'rope':{'module':'dit','qualname':'rope'}}
        report.update(math_before=state,math_after=state,math_restored=True,session_math={m:state for m in A.MODES},math_observations=[])
        for g,phases in report['block_forward_calls'].items():
            for p,blocks in phases.items():
                for i,count in blocks.items():
                    observed=copy.deepcopy(state)
                    if g=='VAE':observed['rope']={'module':'comfy_kitchen.backends.ascend.rope','qualname':'rms_rope_split_half','id':1}
                    report['math_observations'] += [{'group':g,'phase':p,'block':int(i),'state':observed} for _ in range(count)]
        report['decoder_context_restorations']=[{'phase':p,'restored':True,'before':state,'after':state} for p,n in
            [('zero_decode',37),('backward',36),('updated_decode',1),('reloaded_decode',1),('merged_decode',1)] for _ in range(n)]
        keys=('session_state_initialized','session_shapes_restored','session_current_patcher_restored','whole_run_shapes_restored','whole_run_current_patcher_restored','inactive')
        report.update(session_restorations=[{'mode':m,**{k:True for k in keys}} for m in A.MODES],wrapper_attributes_restored=True,
            stale_backward_guard_rejected={m:True for m in A.MODES},patcher_restoration={k:True for k in keys[:-1]})
        self.assertEqual(A.contexts(report)['block_observations'],930)
        bad=copy.deepcopy(report);bad['decoder_context_restorations'].pop()
        with self.assertRaisesRegex(ValueError,'context count'):A.contexts(bad)
        bad=copy.deepcopy(report);bad['math_observations'][0]['state']={**state,'rope':{'module':'native decoder leaked'}}
        with self.assertRaisesRegex(ValueError,'leaked'):A.contexts(bad)

    def test_not_launched_and_failed_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'runs/x';run.mkdir(parents=True);jwrite(run/'queue_status.json',{'state':'not_launched'})
            result=A.audit(root,{}, {'run':run,'evidence':run/'eval'});self.assertFalse(result['complete'])
            self.assertEqual(result['numerical_acceptance'],'not_launched_not_accepted')
            jwrite(run/'queue_status.json',{'state':'failed'});jwrite(run/'eval/failure.json',{'schema':'openh3.nonzero_adapter_probe.v1','complete':True,'arrays':{}})
            with self.assertRaisesRegex(ValueError,'state mismatch'):A.audit(root,{}, {'run':run,'evidence':run/'eval'})

    def test_partial_failed_report_preserves_evidence_without_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();run=root/'runs/x';prep=root/'preparation/p';evidence=run/'eval'
            run.mkdir(parents=True);prep.mkdir(parents=True)
            source=Path(__file__).resolve().parents[2]/'configs/openh3_20261111/h3_nonzero_adapter_probe_contract_20261112.json'
            (root/'contract.json').write_bytes(source.read_bytes())
            config={'producer_sha256':A.PRODUCER_SHA,'trajectory_sha256':A.TRAJECTORY_SHA,'target_sha256':A.TARGET_SHA,
                'source_contract_sha256':A.CONTRACT_SHA,'execution_contract_sha256':'1'*64}
            contract={'schema':'openh3.nonzero_adapter_probe_execution.v1','root':str(root),'probe_sha256':A.PRODUCER_SHA,
                'source_contract':'contract.json','source_contract_sha256':A.CONTRACT_SHA,'implementation_sha256':{'trajectory.py':A.TRAJECTORY_SHA}}
            jwrite(prep/'execution_contract.json',contract);jwrite(run/'queue_status.json',{'state':'failed'})
            report=budget_report();report.update(schema='openh3.nonzero_adapter_probe.v1',complete=False,arrays={},failure='synthetic failure',
                contract={**contract,'execution_contract_sha256':'1'*64})
            jwrite(evidence/'failure.json',report)
            result=A.audit(root,config,{'run':run,'prep':prep,'evidence':evidence})
            self.assertTrue(result['audit_complete']);self.assertFalse(result['complete']);self.assertEqual(result['errors'],[])
            self.assertEqual(result['numerical_acceptance'],'producer_failed_not_accepted')

    def test_single_safe_output_component_and_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);own=root/'preparation/auditor';own.mkdir(parents=True)
            config={'schema':'openh3.nonzero_adapter_cpu_finalizer_admission.v1','run':'runs/model','prep':'preparation/model','out':'runs/audit',
                'producer_output_subdir':'eval','source_sha256':{n:'0'*64 for n in ('audit_terminal.py','audit_arrays.py','audit_checkpoint.py','finalize_once.py','small_archive.py')}}
            for key in ('producer','trajectory','execution_contract','queue','target','source_contract'):config[key+'_sha256']='0'*64
            path=own/'admission.json';F.admission(root,own,path,jwrite(path,config))
            for badvalue in ('/','../eval','a/b','a\\b','', '..'):
                bad={**config,'producer_output_subdir':badvalue}
                with self.subTest(value=badvalue),self.assertRaises(ValueError):F.admission(root,own,path,jwrite(path,bad))
            bad={**config,'out':config['run']}
            with self.assertRaisesRegex(ValueError,'overlap'):F.admission(root,own,path,jwrite(path,bad))

    def test_bounded_wait_has_no_dispatch_or_retry(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'queue.json';jwrite(p,{'state':'running'});now=[0.];observations=[]
            def sleep(seconds):now[0]+=seconds
            result=F.wait_terminal(p,lambda *x:observations.append(x),max_wait=5,poll=2,clock=lambda:now[0],sleep=sleep)
            self.assertFalse(result['terminal']);self.assertEqual(now[0],5);self.assertEqual(len(observations),4)
            jwrite(p,{'state':'completed'});self.assertTrue(F.wait_terminal(p,lambda *x:None,max_wait=0)['terminal'])
            # Default wait must include producer termination/grace, using a fake clock.
            jwrite(p,{'state':'running'});now[0]=0.
            def producer_finishes_after_grace(seconds):
                now[0]+=seconds
                if now[0]>=3630:jwrite(p,{'state':'failed'})
            result=F.wait_terminal(p,lambda *x:None,clock=lambda:now[0],sleep=producer_finishes_after_grace)
            self.assertTrue(result['terminal']);self.assertEqual(now[0],3630)
            jwrite(p,{'state':'running'});now[0]=0.
            result=F.wait_terminal(p,lambda *x:None,clock=lambda:now[0],sleep=sleep)
            self.assertFalse(result['terminal']);self.assertEqual(now[0],4200)

    def test_archive_allowlist_no_big_parameters_and_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prep=root/'preparation/p';own=root/'preparation/a';run=root/'runs/r';out=root/'runs/f';evidence=run/'eval'
            for d in (prep,own,run,out,evidence/'arrays'):d.mkdir(parents=True,exist_ok=True)
            jwrite(out/'finalize_receipt.json',{'complete':False});jwrite(run/'queue_status.json',{'state':'failed'});jwrite(evidence/'failure.json',{'complete':False})
            small=evidence/'arrays/zero_video.npy';np.save(small,np.zeros(10,np.float32));np.save(evidence/'arrays/gradient_raw__x.npy',np.ones(10,np.float32))
            (evidence/'diagnostic.safetensors').write_bytes(b'not archived')
            audit={'archive_arrays':{'zero_video':{'path':str(small),'file_sha256':file_sha(small)}}}
            result=S.package(root,prep,run,own,out,True,audit,evidence)
            with tarfile.open(result['path']) as tar:
                names=tar.getnames();self.assertFalse(any('gradient' in x or x.endswith('.safetensors') for x in names))
                manifest=json.load(tar.extractfile('MANIFEST.json'));self.assertFalse(manifest['parameter_arrays_included'])
            with self.assertRaises(FileExistsError):S.package(root,prep,run,own,out,True,audit,evidence)
            np.save(small,np.ones(10,np.float32))
            with self.assertRaisesRegex(ValueError,'changed after CPU audit'):S.package(root,prep,run,own,out,True,audit,evidence)


if __name__=='__main__':unittest.main(verbosity=2)
