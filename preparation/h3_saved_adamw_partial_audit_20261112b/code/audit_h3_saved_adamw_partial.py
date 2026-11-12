"""CPU-only audit of saved artifacts from a failed one-step H3 AdamW run.

Default inspection reads only metadata/source pins. --execute reads actual NPY and
safetensors bytes. A verified saved subset never accepts the original failed run,
strict reload, merge, deployed functional effect, quality or formal training.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

SCHEMA='openh3.saved_adamw_partial_audit_execution.v1'
REPORT_SCHEMA='openh3.saved_adamw_partial_audit.v1'
HELPER_PINS={'audit_terminal.py': 'bddea2c2f93c5ef9db97f1e1f009cadf73669db559763fcc626a403859a14b55', 'audit_arrays.py': '3690c70226993dbb4f88d613c7fc236b2831763bc3beaacc0a9c3e9dfeb9f1c4', 'audit_checkpoint.py': '4f947400caff009c69a7817ead63fe09d81ac7afbd5fcf402ad19b3ef9f436b8', 'adamw_equations.py': '973f97ef6b8f9f4751be84f6331ad1e74aa29e6617c2b385680ad61acdad3ef8', 'finalize_once.py': 'f71e9b6cfaab08cc38ef1c0573dbf9932e5b8a83d05e467cea06f332b3a311c0', 'small_archive.py': '9e2d0ff6d047ee4aa718dfe6e4ea85e8efc463217e34bab7c4e573f8d768bd6c', 'owned_process.py': '6e4e3beae0351f21cf81ff8559776419e7d4b61a8bd2ab88f0fd5fc9db60d60f'}
ROOT='/cache/zihaohe/omni_openh3_20261111'
EXPECTED_PARTIAL='ff278a84665a7cae78246dc017cbbfe0b0ddb2e93ac9f7d08c6fa31c0309b646'
EXPECTED_QUEUE='28299a42e3e76ce00b5e8fe7dae9bd471ebfa7fc7950051199857280de9d9edb'
EXPECTED_REVIEW='b6fba2f6b379bd40b22bbce21c4f7c91ab01f25548cebf2751ab2ca6d006bc14'
EXPECTED_EXECUTION='f81de1f4129085bd42579c6d7be15301fd4852749933d8cd452a73fd1cabe6d1'
EXPECTED_PRODUCER='27fb9145790635a34398f52ae437f6a3f3f698f097c9a22ccc74ebcd4c6827b3'
CONTRACT_SHA='febe204693b74c1ceae303492cb2100ec176b651cd45fc30bc229e4c33cf26d5'
TARGET_SHA='e15655f1da67e8000ba2fa78e7a3fcbf4407b758afc59cbf3c0760ed3d6def5d'
VIDEO=[1,24,7,30,52]; AUDIO=[1,32,2,37]; RGB=[22,480,832,3]
NAMES={f'blocks.{i}.{module}.lora_{letter}' for i in range(50)
       for module in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2') for letter in ('a','b')}
SMALL={k:(s,'float32') for k,s in [('guide',VIDEO),('raw_warm_start',VIDEO),('target',RGB)]}
for _m in ('reference','candidate','zero','updated'):
    for _s,_shape in (('video',VIDEO),('audio',AUDIO),('rgb',RGB)): SMALL[_m+'_'+_s]=(_shape,'float32')
    if _m!='candidate': SMALL[_m+'_decoder_input']=(VIDEO,'float16')
for _m in ('zero','updated'): SMALL[_m+'_stage_decoder_input']=(VIDEO,'float16')
EXPECTED_ARRAYS=set(SMALL)|{'video_gradient_raw','video_gradient_unscaled'}|{
    p+n for p in ('gradient_raw__','gradient_unscaled__','gradient_clipped__','parameters_before_update__','parameters_after_update__') for n in NAMES}|{
    'optimizer_state_after_update__'+n+'::'+k for n in NAMES for k in ('step','exp_avg','exp_avg_sq')}

def require(value,message):
    if not value: raise ValueError(message)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def json_value_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def bound(root,row):
    raw=Path(row['path']); path=(root/raw).resolve()
    require(not raw.is_absolute() and path.is_relative_to(root),'bound path escaped root')
    require(file_sha(path)==row['sha256'],'bound SHA differs: '+str(path))
    return path

def load_numeric_helpers(directory):
    """All imported helper sources are pinned before any of them is executed."""
    directory=Path(directory).resolve()
    for name,digest in HELPER_PINS.items(): require(file_sha(directory/name)==digest,'numeric helper source differs: '+name)
    sys.path.insert(0,str(directory))
    for name in ('audit_arrays','audit_checkpoint','adamw_equations','owned_process','small_archive','finalize_once','audit_terminal'):
        if name in sys.modules: require(Path(sys.modules[name].__file__).resolve()==directory/(name+'.py'),'shadowed numeric helper')
    global np,A,array_sha,verify_unscale,verify_reduction,SafeAdapter,grad_stats
    np=importlib.import_module('numpy'); A=importlib.import_module('audit_terminal')
    for name in ('array_sha','verify_unscale','verify_reduction','SafeAdapter','grad_stats'): globals()[name]=getattr(A,name)
    return A

def not_nonzero(value): return not bool(np.count_nonzero(value))

def saved_update(report,reader,shapes,contract,*,expected_device='npu:0',video_shape=VIDEO):
    from adamw_equations import first_step, LR, TOLERANCES
    initial=report['adapter_initial'];summary=report['adapter_gradients']
    require(initial['modules']==200 and initial['trainable_tensors']==400 and initial['rank']==8 and
        set(initial['tensors'])==NAMES and set(summary['tensors'])==NAMES,'adapter/gradient set differs')
    for key in ('parameters_before_update','parameters_after_update','gradient_clipped'):
        require(set(report[key])==NAMES,'parameter/clip set differs')
    require(set(report['update']['tensors'])==NAMES,'update set differs')
    recipe={'class':'torch.optim.AdamW','base_lr':1e-4,'warmup_steps':100,'global_step':1,'actual_lr':LR,
        'lr_rule':'base_lr * min(1.0, global_step / warmup_steps)','betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
        'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False,'foreach':False,'fused':False,
        'clip_grad_norm':{'max_norm':1.,'norm_type':2.,'error_if_nonfinite':True,'foreach':False},'loss_scale':1024.}
    require(report['optimizer']==recipe and report['gradient_clip']['recipe']==recipe['clip_grad_norm']
        and report['gradient_clip']['applied_once'] is True,'optimizer/clip recipe differs')
    norm=report['gradient_clip']['pre_clip_total_norm']
    require(type(norm) in (float,int) and math.isfinite(norm) and norm>=0,'invalid norm')
    coefficient=np.minimum(np.float32(1)/(np.float32(norm)+np.float32(1e-6)),np.float32(1))
    checkpoint=report['checkpoint'];path=(reader.run/checkpoint['path']).resolve()
    require(path.parent==reader.run and path.name=='real_train_AdamW_step1.safetensors','adapter checkpoint path differs')
    require(checkpoint['bytes']==path.stat().st_size and checkpoint['expected_shapes']==shapes and
        set(checkpoint['exact_saved_live_tensors'])==NAMES and all(checkpoint['exact_saved_live_tensors'].values()),'checkpoint coverage differs')
    meta=checkpoint['metadata'];training=meta['training']
    require(meta['rank']==8 and meta['pretrained_fingerprint']==report['pretrained']['openh3_pretrained']['fingerprint'],'saved pretrained identity')
    require(training['role']==checkpoint['role']=='one_real_TRAIN_AdamW_engineering_step_not_quality_checkpoint'
        and training['execution_contract_sha256']==report['contract']['execution_contract_sha256']
        and training['protocol_sha256']==CONTRACT_SHA and training['manifest_sha256']==contract['manifest_sha256']
        and training['implementation_sha256']==contract['implementation_sha256']
        and training['input_identity']==report['input_inspection']['identity']
        and training['input_arrays']=={k:report['arrays'][k] for k in ('guide','raw_warm_start','target')}
        and training['optimizer']==recipe and training['steps']==1 and training['loss_scale']==1024
        and training['rank']==training['alpha']==8 and training['objective']=='original clean proxy RGB MSE only',
        'saved training provenance differs')
    require(training['noise']==report['noise_observations'][0] and training['sigmas']==[.909375,6/7,2/3,0.] and
        training['sigmas_float32_sha256']==array_sha(np.asarray(training['sigmas'],np.float32)),'noise/sigmas metadata differs')
    saved=SafeAdapter(path,checkpoint['sha256'],shapes,meta)
    optim=checkpoint['optimizer'];op=(reader.run/optim['path']).resolve()
    require(op.parent==reader.run and op.name=='real_train_AdamW_step1_optimizer.safetensors'
        and op.stat().st_size==optim['bytes'] and optim['state_tensors']==1200,'optimizer file binding differs')
    order=optim['parameter_order'];require(len(order)==400 and set(order)==NAMES,'optimizer parameter order differs')
    ometa={'schema':'openh3.named_AdamW_state.v1','adapter_sha256':checkpoint['sha256'],
        'parameter_order':json.dumps(order),'optimizer_recipe':json.dumps(recipe,sort_keys=True),
        'training_contract':json.dumps(training,sort_keys=True)}
    require(optim['metadata']==ometa,'optimizer metadata differs')
    oshapes={n+'::'+k:([] if k=='step' else shapes[n]) for n in order for k in ('step','exp_avg','exp_avg_sq')}
    saved_optimizer=SafeAdapter(op,optim['sha256'],oshapes,ometa,provenance=False)
    inventories={'optimizer_state_after_update':report['optimizer_after_update']}
    for label,inv in inventories.items():
        require(inv['named_entries']==400 and inv['state_tensors']==1200 and set(inv['inventory'])==NAMES
            and inv['parameter_order']==order and inv['arrays']==report[label] and set(inv['arrays'])==set(oshapes),'optimizer inventory differs')
        group=inv['parameter_group'];wanted=dict(lr=LR,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,
            maximize=False,capturable=False,differentiable=False,foreach=False,fused=False)
        require(all(group[k]==v for k,v in wanted.items()),'actual saved parameter group differs')
    b2=0.;b_nonzero=0;changed=0;sum_unscaled2=0.;equations={}
    for name in sorted(NAMES):
        shape=shapes[name];init=initial['tensors'][name];row=summary['tensors'][name]
        require(init['shape']==shape and init['dtype']=='torch.float32' and row['present'] is True,'adapter metadata')
        raw=reader.load('gradient_raw__'+name,shape,record=row['raw']);unscaled=reader.load('gradient_unscaled__'+name,shape,record=row['unscaled'])
        verify_unscale(raw,unscaled);n,nz=grad_stats(raw,row['statistics_raw']);grad_stats(unscaled,row['statistics_unscaled'])
        sum_unscaled2+=float(np.sum(unscaled.astype(np.float64)**2))
        clipped=reader.load('gradient_clipped__'+name,shape,record=report['gradient_clipped'][name])
        # foreach=False norm's saved FP32 value defines the exact scalar multiply.
        require(array_sha(clipped)==array_sha(unscaled*coefficient),'actual clipped bytes differ')
        before=reader.load('parameters_before_update__'+name,shape,record=report['parameters_before_update'][name])
        after=reader.load('parameters_after_update__'+name,shape,record=report['parameters_after_update'][name])
        require(array_sha(before)==init['sha256_bytes'] and init['all_zero']==not_nonzero(before),'initial/before bytes differ')
        if name.endswith('.lora_a'):require(array_sha(raw)==array_sha(np.zeros_like(raw)),'fresh zeroB must yield bytezero A gradient')
        else:
            require(array_sha(before)==array_sha(np.zeros_like(before)) and init['all_zero'] is True,'fresh B not bytezero')
            b2+=n*n;b_nonzero+=nz>0;changed+=int(np.count_nonzero(after!=before))
        actual_states={}
        for key in ('step','exp_avg','exp_avg_sq'):
            namekey=name+'::'+key;ss=oshapes[namekey]
            first=reader.load('optimizer_state_after_update__'+namekey,ss,record=report['optimizer_state_after_update'][namekey])
            require(array_sha(first)==array_sha(saved_optimizer.load(namekey)),'optimizer saved bytes differ')
            for inv in inventories.values():
                rec=inv['inventory'][name][key]
                require(rec['sha256']==array_sha(first) and rec['shape']==ss and rec['dtype']=='torch.float32','optimizer inventory content')
                if key!='step':require(rec['device']==init['device']==expected_device,'moment residency')
                else:require(rec['device']=='cpu','capturableFalse/fusedFalse step must remain CPU')
            if key=='step':require(float(first)==1.,'actual optimizer step not1')
            actual_states[key]=first
        equations[name]=first_step(before,clipped,after,actual_states['exp_avg'],actual_states['exp_avg_sq'],report['update']['tensors'][name])
        require(array_sha(saved.load(name))==array_sha(after),'adapter file differs from actual update')
    require(math.isclose(norm,math.sqrt(sum_unscaled2),rel_tol=3e-6,abs_tol=1e-20),'saved global norm differs from arrays')
    require(summary['expected']==summary['present_tensors']==400 and summary['all_finite'] is True and summary['passed'] is True
        and summary['B_nonzero_tensors']==b_nonzero and b_nonzero>0 and b2>0,'gradient aggregate differs')
    verify_reduction(summary['B_l2_raw'],math.sqrt(b2),'Braw');verify_reduction(summary['B_l2_unscaled'],math.sqrt(b2)/1024.,'Bunscaled')
    require(report['update']['B_changed_coordinates']==changed and changed>0,'actual update differs')
    row=report['retained_video_gradient'];raw=reader.load('video_gradient_raw',video_shape,record=row['raw']);unscaled=reader.load('video_gradient_unscaled',video_shape,record=row['unscaled'])
    video=verify_unscale(raw,unscaled,nonzero=True);grad_stats(raw,row['statistics_raw']);grad_stats(unscaled,row['statistics_unscaled']);require(row['passed'] is True,'video gradient failed')
    return {'adapter_gradient_pairs':400,'updated_adapter_tensors':400,'optimizer_named_entries':400,
        'optimizer_after_update_arrays':1200,'optimizer_safetensors_keys':1200,'equations':equations,
        'B_changed_coordinates':changed,'independent_video_gradient':video,'tolerances':TOLERANCES,
        'clip_rule':'saved FP32 norm coefficient; exact multiply, norm independently within3e-6/1e-20',
        'zero_and_serialization_byte_exact':True,'resume_trajectory_claim':False,'actual_arrays_verified':True,'passed':True,
        'strict_reload_accepted':False,'merge_accepted':False,
        'checkpoint_file_sha256':checkpoint['sha256'],'optimizer_file_sha256':optim['sha256']}


def verify_clipping_summary(row, measured):
    numeric=('at_zero','at_one','outside_unit_interval')
    require(set(row)==set(numeric)|{'semantics'} and
        row['semantics']=='native internally clipped; no extra clamp','clipping semantics/fields differ')
    require(all(type(row[k]) is int and row[k]==measured[k] for k in numeric) and
        measured['outside_unit_interval']==0,'clipping summary differs')
    return {**measured,'semantics':row['semantics']}


def saved_outputs(report,loaded,prior_arrays):
    for role,value in prior_arrays.items():
        require(array_sha(loaded[role])==array_sha(value),'accepted reference/input bytes differ: '+role)
    for stream in ('video','audio','rgb'):
        require(array_sha(loaded['candidate_'+stream])==array_sha(loaded['zero_'+stream]),'zero alias differs')
    zero={}
    for stream in ('video','audio'):
        zero[stream]=A.compare_outputs(loaded['zero_'+stream],loaded['reference_'+stream],exact=True)
        A.parity_record(report['latent_parity'][stream],loaded['reference_'+stream],loaded['zero_'+stream],exact=True)
    zero['rgb']=A.compare_outputs(loaded['zero_rgb'],loaded['reference_rgb'],exact=False)
    A.parity_record(report['rgb_parity'],loaded['reference_rgb'],loaded['zero_rgb'])
    require(report['rgb_parity']['allclose'] is True and report['rgb_parity']['atol']==.0002 and report['rgb_parity']['rtol']==.0001,'zero RGB tolerance changed')
    for mode in ('reference','zero','updated'): A.verify_decoder_cast(loaded[mode+'_video'],loaded[mode+'_decoder_input'])
    require(array_sha(loaded['reference_decoder_input'])==array_sha(loaded['zero_decoder_input']),'zero decoder input differs')
    require(set(report['graph'])=={'video_requires_grad','video_has_grad_fn','rgb_requires_grad','rgb_has_grad_fn'} and all(v is True for v in report['graph'].values()),'graph attestation differs')
    require(array_sha(loaded['target'])==TARGET_SHA,'TRAIN target bytes differ')
    losses={}
    require(set(report['mode_outputs'])=={'zero','updated'},'partial mode outputs differ')
    for mode in ('zero','updated'):
        pixel=loaded[mode+'_rgb']; row=report['mode_outputs'][mode]
        require(row['geometry_and_finite'] is True,'geometry attestation')
        mse=A.recompute_rgb_mse(pixel,loaded['target'],TARGET_SHA)
        verify_reduction(row['numpy_float64_mse'],mse,mode+' RGB MSE')
        require(type(row['torch_float32_mse']) in (int,float) and math.isfinite(row['torch_float32_mse']) and row['torch_float32_mse']>=0,'invalid Torch reduction')
        clipping={'at_zero':int((pixel==0).sum()),'at_one':int((pixel==1).sum()),'outside_unit_interval':int(((pixel<0)|(pixel>1)).sum())}
        clipping=verify_clipping_summary(row['clipping'],clipping)
        losses[mode]={'independent_numpy_float64_mse':mse,'torch_float32_reduction_attestation':row['torch_float32_mse'],'clipping':clipping}
        require(array_sha(loaded[mode+'_stage_decoder_input'])==array_sha(loaded[mode+'_decoder_input']) and report['stage_decoder_inputs'][mode]==report['arrays'][mode+'_stage_decoder_input'],'actual stage cast differs')
    loss=report['loss']; verify_reduction(loss['numpy_float64_mse'],losses['zero']['independent_numpy_float64_mse'],'backward loss')
    require(loss['scale']==1024. and loss['auxiliary_latent_audio_loss'] is False and loss['torch_float32_mse']==report['mode_outputs']['zero']['torch_float32_mse'],'loss/scale differs')
    require(set(report['comparisons'])=={'reference__zero','zero__updated'},'unexpected partial comparisons')
    paired={}
    for key,rows in report['comparisons'].items():
        a,b=key.split('__'); require(set(rows)=={'video','audio','decoder_input','rgb'},'comparison streams')
        paired[key]={s:A.parity_record(r,loaded[a+'_'+s],loaded[b+'_'+s]) for s,r in rows.items()}
    return {'passed':True,'actual_arrays_verified':True,'zero_reference':zero,'objectives':losses,'paired_outputs':paired,
        'updated_unmerged_RGB_changed':not paired['zero__updated']['rgb']['exact_equal'],
        'zero_reference_RGB_byte_equal':paired['reference__zero']['rgb']['byte_equal'],
        'functional_deployment_ready':False,'quality_claim':False,
        'scope':'Actual saved zero/updated outputs only. Updated is unmerged. No reload/merged arrays exist; deployment attribution not accepted.'}


def continuation_readiness(complete, output_facts):
    pairs=output_facts.get('paired_outputs',{}).get('reference__zero',{})
    streams=('video','audio','decoder_input','rgb')
    byte_exact={stream:pairs.get(stream,{}).get('byte_equal') is True for stream in streams}
    return {'ready':bool(complete and all(byte_exact.values())),
        'zero_reference_byte_exact':byte_exact,
        'scope':'Eligibility for a separately admitted reload/merge continuation only; not original-run, deployment, quality or training acceptance.'}


def partial_scope(report,queue):
    require(queue['state']=='failed' and queue['complete'] is False and queue['pins_unchanged'] is True and queue['remaining_owned_live_processes']==[],'failed producer terminal scope differs')
    require(report['schema']=='openh3.real_train_adamw_single_update.v1' and report['complete'] is False and report['phase']=='clear_and_strict_reload','partial phase/schema differs')
    a=report['actual']; require(a['NFE']==dict(zero=3,updated=3,reloaded=0,merged=0) and a['VAE_decode_calls']==dict(zero=1,updated=1,reloaded=0,merged=0),'saved mode budget differs')
    for k in ('RGB_loss_expressions','backward_attempts','completed_backward_calls','optimizer_steps','adapter_updates','checkpoint_saves','optimizer_state_saves','unscale_calls','clip_calls','clear_reload_cycles','optimizer_state_clear_reload'): require(a[k]==1,'actual single operation differs: '+k)
    require(a['merge_calls']==a['VAE_encode_calls']==a['Rphi_calls']==0 and a['forbidden_calls']==[],'forbidden actual operation')
    require(a['stage_decode_calls']==dict(zero_decode=1,updated_decode=1),'stage budget')
    require(report['reload']=={'cleared':{'parameters_all_zero':True,'optimizer_tensors_all_zero':True,'optimizer_mapping_empty':True}},'unexpected reload scope')
    require('optimizer_state_after_reload' not in report and 'merge' not in report and 'final_frozen' not in report,'partial availability changed; new audit required')
    require(set(report['arrays'])==EXPECTED_ARRAYS,'saved subset exact array roles differ')
    A.budget(report,False)
    return {'passed':True,'actual':a,'original_failed_state_retained':True,'clear_reload_counter_is_not_acceptance':True,
        'strict_reload_accepted':False,'merge_accepted':False,'final_live_frozen_state_available':False}


def recorded_sources_and_lifecycle(report,original,prior,initial,prior_initial,loaded):
    # These are source-bound observations, not an independent live-weight reload.
    for group in ('DiT','VAE'):
        for kind in ('parameters','registered_buffers'):
            require(initial[group][kind+'_sha256']==prior_initial[group][kind+'_sha256'],'pristine canonical inventory differs')
    expected={str((Path(original['root'])/k).resolve()):v for k,v in original['implementation_sha256'].items()}
    require(report['verified_runtime_files']==expected,'producer runtime pin map differs')
    imports=report['actual_imports_before_sampling']
    require(imports['passed'] is True and imports['unbound']==[] and imports['modules'],'unbound recorded imports')
    require(all(expected.get(r['path'])==r['sha256'] for r in imports['modules'].values()),'recorded import source mismatch')
    require(set(imports['synthetic_namespaces'])=={'torch.ops','torch.classes'},'synthetic module definitions')
    for r in imports['synthetic_namespaces'].values(): require(expected.get(r['defining_source_path'])==r['defining_source_sha256'],'synthetic source differs')
    for row in imports['namespace_packages'].values():
        subset={p:h for p,h in expected.items() if Path(p).suffix=='.py' and any(Path(p).is_relative_to(Path(d)) for d in row['directories'])}
        require(row['loader_type']=='_frozen_importlib_external.NamespaceLoader' and row['source_files']==subset and subset,'namespace source scope')
    required={'after_injection','after_RGB_backward','after_update_and_save'}|{m+s for m in ('zero','updated') for s in ('_before_decode','_after_decode','_complete')}
    require(set(report['state_checks'])==required,'saved lifecycle phases differ')
    for phase,checks in report['state_checks'].items():
        require(set(checks)=={'DiT','VAE','adapters'},'lifecycle groups')
        for group,check in checks.items():
            require(check['passed'] is True and check['failures']==[] and check['tensors_checked']=={'DiT':532,'VAE':601,'adapters':400}[group],'recorded lifecycle failure')
            if group!='adapters' and phase in ('after_RGB_backward','after_update_and_save','updated_complete'):
                require(check['checks']==['canonical name','object','version','device','bytes','frozen','grad None'],'full recorded byte gate missing')
    keys=('openh3_pretrained','comfy_commit','dit','guide_helper_sha256','jit_compile_false','mm_bmm_format_nd','precision_used','torch','torch_npu','torch_npu_module_to_cast_weight_dropped')
    require(all(report['pretrained'][k]==prior['pretrained'][k] for k in keys),'prior pretrained identity')
    require(all(report['environment'][k]==prior['environment'][k] for k in ('torch','torch_npu','python')),'prior numeric environment')
    require(report['vae_native_plan']==prior['vae_native_plan'],'prior native padding/tiling')
    noise={k:v for k,v in prior['noise_observations'][0].items() if k!='phase'}
    require([r['phase'] for r in report['noise_observations']]==['zero','updated'] and all({k:v for k,v in r.items() if k!='phase'}==noise for r in report['noise_observations']),'noise seed/initial bytes')
    require(noise['seed']==101 and noise['joint_shapes']==[VIDEO,AUDIO],'native seed/shape')
    require(report['input_inspection']['identity']==prior['input_inspection']['identity'] and report['input_inspection']['nonzero_b_weights_loaded'] is False,'cache identity')
    inp=report['inputs']; require(inp['source_id']=='2BdYaGuVAKo' and inp['role']=='train' and inp['model_input_keys']==['guide','raw_warm_start'] and inp['seed']==101 and inp['audio_init']=='zeros' and inp['target_used_as_model_input'] is False,'TRAIN input role')
    require(set(report['trajectory_contracts'])==set(report['traces'])=={'zero','updated'},'partial trajectories')
    for mode,row in report['trajectory_contracts'].items():
        for k,v in {'guide_latent_sha256_float32':array_sha(loaded['guide']),'raw_warm_start_latent_sha256_float32':array_sha(loaded['raw_warm_start']),'packed_cpu_noise_sha256_float32':noise['packed_noise_sha256'],'packed_initial_latent_sha256_float32':noise['packed_initial_sha256']}.items(): require(row[k]==v,'trajectory input bytes')
        require(row['seed']==101 and row['audio_init']=='zeros' and row['actual_forward_NFE']==3 and row['target_used_as_model_input'] is False and row['checkpoint_blocks']==(mode=='zero') and row['checkpoint_decoder_blocks']==(mode=='zero'),'trajectory budget')
        require(np.array_equal(np.asarray(row['sigma_schedule_float32'],np.float32),np.asarray([.909375,6/7,2/3,0],np.float32)),'sigmas')
        require(row['cfg']==1 and row['mode']=='base' and row['refs']==[] and row['inner_dc'] is False and row['s_churn']==0 and row['merged'] is False and row['actual_live_adapter_tensors']==400 and row['adapter_parameter_tensors']==400 and row['adapter_B_all_zero_at_session_entry'] is (mode=='zero'),'trajectory conditioning/adapter count')
        require([r['step'] for r in report['traces'][mode]]==[0,1,2],'Euler trace')
    return {'passed':True,'initial_canonical_inventories_verified':True,'recorded_import_maps_checked':1,
        'live_model_weights_independently_reloaded':False,'final_live_state_available':False,
        'scope':'Source-bound producer lifecycle/import attestations through updated_complete; no claim of final live frozen state or successful reload.'}


def metadata_schema_check(report):
    """Validate real saved JSON structure only; never imply actual arrays passed."""
    require(set(report['arrays'])==EXPECTED_ARRAYS,'required partial array roles')
    require(set(report['adapter_initial']['tensors'])==NAMES and set(report['adapter_gradients']['tensors'])==NAMES,'metadata parameter coverage')
    for role,(shape,dtype) in SMALL.items():
        row=report['arrays'][role];require(row['role']==role and row['shape']==shape and row['dtype']==dtype,'small array metadata: '+role)
    for mode in ('zero','updated'):
        row=report['mode_outputs'][mode];require(row['geometry_and_finite'] is True,'mode geometry metadata')
        verify_clipping_summary(row['clipping'],{k:row['clipping'][k] for k in ('at_zero','at_one','outside_unit_interval')})
        require(report['stage_decoder_inputs'][mode]==report['arrays'][mode+'_stage_decoder_input'],'stage nested metadata')
    require(set(report['comparisons'])=={'reference__zero','zero__updated'},'partial comparison metadata')
    for pair in report['comparisons'].values():
        require(set(pair)=={'video','audio','decoder_input','rgb'},'comparison stream metadata')
        for row in pair.values():
            require({'torch_equal','same_shape','finite','reference_shape','candidate_shape','reference_sha256_float32','candidate_sha256_float32','max_abs','relative_l2','changed_coordinates'}<=set(row),'comparison record fields')
    for key in ('parameters_before_update','parameters_after_update','gradient_clipped'):
        require(set(report[key])==NAMES,'parameter mapping coverage')
        for n,row in report[key].items():require(row==report['arrays'][key+'__'+n],'nested parameter mapping')
    statekeys={n+'::'+key for n in NAMES for key in ('step','exp_avg','exp_avg_sq')}
    inv=report['optimizer_after_update']; require(set(report['optimizer_state_after_update'])==statekeys and inv['arrays']==report['optimizer_state_after_update'] and set(inv['inventory'])==NAMES,'postupdate metadata coverage')
    for name in NAMES:
        shape=report['adapter_initial']['tensors'][name]['shape'];g=report['adapter_gradients']['tensors'][name]
        for k in ('raw','unscaled'):
            row=g[k];require(row==report['arrays']['gradient_'+k+'__'+name] and row['shape']==shape and row['dtype']=='float32','gradient nested mapping')
            require({'present','elements','finite_count','all_finite','nonzero_finite_count','finite_l2','nan_count','inf_count'}<=set(g['statistics_'+k]),'gradient statistics fields')
        for key in ('step','exp_avg','exp_avg_sq'):
            nk=name+'::'+key;row=report['optimizer_state_after_update'][nk];expected=[] if key=='step' else shape
            require(row==report['arrays']['optimizer_state_after_update__'+nk] and row['shape']==expected and row['dtype']=='float32','optimizer nested array mapping')
            record=inv['inventory'][name][key]
            require(record['sha256']==row['array_sha256'] and record['shape']==expected and record['dtype']=='torch.float32','optimizer inventory metadata')
    require(report['loss']['scale']==1024 and report['loss']['auxiliary_latent_audio_loss'] is False,'loss metadata')
    require(set(report['update']['tensors'])==NAMES and set(report['checkpoint']['expected_shapes'])==NAMES and set(report['checkpoint']['exact_saved_live_tensors'])==NAMES,'saved checkpoint metadata coverage')
    return {'metadata_only':True,'actual_array_reads':0,'array_records':len(report['arrays']),
        'adapter_names':400,'postupdate_state_names':1200,'mode_names':['zero','updated'],'clipping_semantics_verified':True}


def verify_inspection(contract_path,expected_sha):
    path=Path(contract_path).resolve(); require(file_sha(path)==expected_sha,'execution contract SHA')
    c=read_json(path); require(c['schema']==SCHEMA and c['root']==ROOT,'execution root/schema')
    root=Path(c['root']).resolve()
    require(c['budget']=={'wall_seconds':900,'CPU_cores':2,'memory_bytes':4<<30,'model_calls':0,'NPU_calls':0,'Torch_imports':0},'CPU budget differs')
    bindings=c['files']; fixed={'partial':EXPECTED_PARTIAL,'queue':EXPECTED_QUEUE,'root_review':EXPECTED_REVIEW,'original_execution':EXPECTED_EXECUTION,'protocol':CONTRACT_SHA}
    require(set(bindings)==set(fixed),'metadata binding set differs')
    paths={k:bound(root,row) for k,row in bindings.items()}
    require(all(bindings[k]['sha256']==v for k,v in fixed.items()),'known failed run identity differs')
    partial=read_json(paths['partial']); original=read_json(paths['original_execution']); queue=read_json(paths['queue'])
    require(paths['partial'].name=='partial.json' and paths['partial'].parent.parent==paths['queue'].parent,'producer paths differ')
    require(original['probe_sha256']==EXPECTED_PRODUCER and original['protocol']['sha256']==CONTRACT_SHA,'original numerical source changed')
    require(partial['contract']=={**original,'execution_contract_sha256':EXPECTED_EXECUTION} and queue['execution_contract_sha256']==EXPECTED_EXECUTION,'embedded original contract differs')
    require(file_sha(Path(__file__).resolve())==c['entry_sha256'],'entry SHA')
    helper=(root/c['helper_directory']).resolve(); require(helper.is_relative_to(root) and c['helper_sha256']==HELPER_PINS,'helper binding')
    for n,h in HELPER_PINS.items(): require(file_sha(helper/n)==h,'helper SHA: '+n)
    verified={}
    # Original weights are not read; only original source/runtime code is independently pinned.
    for p,h in original['implementation_sha256'].items():
        p=(root/p).resolve()
        if p.suffix in ('.py','.so'):
            require(file_sha(p)==h,'original runtime source differs: '+str(p)); verified[str(p)]=h
    runtime=c['cpu_runtime']
    require(runtime['python_version']==sys.version and Path(runtime['executable']).resolve()==Path(sys.executable).resolve() and file_sha(sys.executable)==runtime['executable_sha256'],'CPU Python runtime differs')
    require(runtime['numpy_files'],'NumPy source/runtime pins missing')
    for p,h in runtime['numpy_files'].items(): require(file_sha(p)==h,'NumPy source/runtime differs: '+p)
    for p,h in c['additional_source_pins'].items(): require(file_sha(root/p)==h,'audit source pin differs: '+p)
    require(c['entry_sha256'] in c['additional_source_pins'].values(),'entry omitted from source pins')
    metadata_check=metadata_schema_check(partial)
    review=read_json(paths['root_review']); require(review.get('schema') and queue['state']=='failed' and partial['complete'] is False,'failed input evidence')
    return c,paths,original,partial,queue,{'complete':True,'inspection_only':True,'metadata_bindings_verified':bindings,
        'source_files_checked':len(verified),'source_manifest_value_sha256':json_value_sha(verified),
        'actual_partial_metadata_schema':metadata_check,'numeric_helpers_checked':HELPER_PINS,'cpu_runtime_pins_checked':True,'actual_NPY_or_safetensors_read':0,
        'original_weight_assets_rehashed':False,'model_calls':0,'Torch_imports':0,'NPU_calls':0}


def audit(c,paths,original,report,queue):
    root=Path(c['root']).resolve(); run=paths['partial'].parent
    A=load_numeric_helpers(root/c['helper_directory'])
    require(np.__version__==c['cpu_runtime']['numpy_version'],'NumPy version changed')
    for name,module in list(sys.modules.items()):
        if name=='numpy' or name.startswith('numpy.'):
            p=getattr(module,'__file__',None)
            if p: require(c['cpu_runtime']['numpy_files'].get(str(Path(p).resolve()))==file_sha(p),'unbound imported NumPy module: '+name)
    require(not any(n=='torch' or n.startswith(('torch.','torch_npu','comfy')) for n in sys.modules),'forbidden model/backend module imported')
    result={'schema':REPORT_SCHEMA,'audit_complete':False,'complete':False,'errors':[], 'components':{},
        'original_run_accepted':False,'strict_reload_accepted':False,'merge_accepted':False,'functional_deployment_ready':False,
        'formal_training_runs':0,'actions':{'model_calls':0,'NPU_calls':0,'Torch_imports':0,'producer_writes':0,'large_arrays_copied':0},
        'binding':{**{k+'_sha256':v['sha256'] for k,v in c['files'].items()},'producer_sha256':EXPECTED_PRODUCER,
                   'protocol_sha256':CONTRACT_SHA,'original_execution_contract_sha256':EXPECTED_EXECUTION,
                   'accepted_full22':original['accepted_full22']}}
    def check(name,fn):
        try:
            value=fn(); result['components'][name]={'status':'verified','facts':value}; return value
        except Exception as error:
            result['components'][name]={'status':'invalid','error':repr(error)}; result['errors'].append(name+': '+str(error));return None
    check('failed_producer_scope',lambda:partial_scope(report,queue))
    class OnceArrays(A.Arrays):
        def __init__(self,*args): super().__init__(*args);self.attempted=set()
        def load(self,role,*args,**kwargs):
            require(role not in self.attempted,'duplicate array read prohibited: '+role)
            self.attempted.add(role);return super().load(role,*args,**kwargs)
    reader=OnceArrays(run,report['arrays']); loaded={}
    for role,row in report['arrays'].items():
        try:
            require(role in EXPECTED_ARRAYS,'unexpected role')
            # Small is a semantic group, including full RGB. mmap avoids float copies.
            if role in SMALL: loaded[role]=reader.load(role,*SMALL[role])
        except Exception as error: result['errors'].append(role+': '+str(error))
    prior=check('accepted_full22',lambda:A.prior_evidence(root,original))
    if prior is not None: result['components']['accepted_full22']['facts']={'actual_reference_arrays_verified':7,'binding':original['accepted_full22']}
    initial=check('initial_inventory',lambda:A.inventory(report['frozen_initial'],run))
    if initial is not None: result['components']['initial_inventory']['facts']={g:{k:initial[g][k+'_sha256'] for k in ('parameters','registered_buffers')} for g in initial}
    if initial is not None: check('saved_update',lambda:saved_update(report,reader,A.parameter_shapes(initial),original))
    if prior is not None and set(loaded)==set(SMALL):
        check('outputs',lambda:saved_outputs(report,loaded,prior[1]))
        if initial is not None: check('recorded_frozen_sources_inputs',lambda:recorded_sources_and_lifecycle(report,original,prior[0],initial,prior[2],loaded))
    # Never silently omit existing artifacts after another section fails.
    for role,row in report['arrays'].items():
        if role not in reader.attempted:
            try:
                require(role in EXPECTED_ARRAYS,'unexpected array role')
                reader.load(role,row['shape'],row['dtype'])
            except Exception as error: result['errors'].append(role+': '+str(error))
    result['array_manifest']=reader.manifest
    required={'failed_producer_scope','accepted_full22','initial_inventory','saved_update','outputs','recorded_frozen_sources_inputs'}
    require(set(result['components'])<=required,'unrecognized component')
    missing=required-set(result['components'])
    if missing: result['errors'].append('dependent sections unavailable: '+str(sorted(missing)))
    if set(reader.manifest)!=EXPECTED_ARRAYS: result['errors'].append('actual array coverage incomplete')
    for k,row in c['files'].items(): require(file_sha(paths[k])==row['sha256'],'immutable metadata changed during audit')
    for name,digest in HELPER_PINS.items(): require(file_sha(root/c['helper_directory']/name)==digest,'helper changed during audit')
    for path,digest in c['additional_source_pins'].items(): require(file_sha(root/path)==digest,'audit source changed during audit')
    result['source_metadata_stable']=True
    result['binding']['checkpoint']={k:report['checkpoint'][k] for k in ('path','sha256','bytes')}
    result['binding']['optimizer_checkpoint']={k:report['checkpoint']['optimizer'][k] for k in ('path','sha256','bytes')}
    result.update(audit_complete=True,complete=not result['errors'],actual_array_count=len(reader.manifest),
        partial_artifact_acceptance='verified_saved_one_update_and_zero_updated_outputs_only' if not result['errors'] else 'invalid_saved_subset')
    readiness=continuation_readiness(result['complete'],result['components'].get('outputs',{}).get('facts',{}))
    result['continuation_readiness']=readiness;result['saved_subset_ready_for_continuation']=readiness['ready']
    result['limitations']=['Original producer remains failed; clear_reload counter1 does not establish strict reload.',
        'No reloaded/merged forward, final live freeze, deployment effect, restored quality, generalization or resumed-step claim.',
        'Only saved initial inventory and source-bound lifecycle attestations are checked; full live model weights were not independently loaded.',
        'AdamW equation tolerances are preregistered audit tolerances, not output quality thresholds or bitexact NPU kernel reenactment.',
        'Torch float32 loss is a recorded reduction; independent objective is NumPy float64 from actual saved pixels.']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--execution-contract-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    start=time.monotonic(); out=a.out.resolve();require(not out.exists() and not out.is_symlink(),'new output directory required')
    c,paths,original,report,queue,inspection=verify_inspection(a.execution_contract,a.execution_contract_sha256)
    require(not out.is_relative_to(paths['queue'].parent) and out.is_relative_to(Path(c['root'])),'audit output overlaps producer or escapes root')
    out.mkdir(parents=True,exist_ok=False)
    if not a.execute: result={'schema':REPORT_SCHEMA,**inspection,'audit_complete':False,'saved_subset_executed':False,'saved_subset_ready_for_continuation':False}
    else:
        admission=c['root_admission']; payload={k:v for k,v in c.items() if k!='root_admission'}
        require(admission and admission['approved'] is True and admission['scope']=='CPU_saved_partial_artifacts_only' and admission['payload_sha256']==json_value_sha(payload),'root CPU admission missing/mismatched')
        require(sys.platform=='linux' and len(os.sched_getaffinity(0))<=2,'execute requires external bounded 2CPU runner')
        require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),'numeric threads must be1')
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('fixed900s audit budget exhausted')))
        signal.setitimer(signal.ITIMER_REAL,max(.001,900-(time.monotonic()-start)))
        try: result=audit(c,paths,original,report,queue)
        except Exception as error: result={'schema':REPORT_SCHEMA,'audit_complete':False,'complete':False,'partial_artifact_acceptance':'invalid_or_incomplete_audit','saved_subset_ready_for_continuation':False,'errors':[repr(error)],'original_run_accepted':False,'strict_reload_accepted':False,'merge_accepted':False,'functional_deployment_ready':False}
        finally: signal.setitimer(signal.ITIMER_REAL,0)
    result['execution_contract_sha256']=a.execution_contract_sha256;result['elapsed_seconds']=time.monotonic()-start;result['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out/'audit.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':result['complete'],'inspection_only':not a.execute,'audit_sha256':file_sha(out/'audit.json'),'path':str(out/'audit.json')}))
    return 0 if result['complete'] else 2

if __name__=='__main__': raise SystemExit(main())
