"""Independent CPU terminal audit for the frozen one-update nonzero probe.

Reads actual NPY/safetensors bytes. Does not load Torch, models or live NPU
weights. Merge matrices were not saved: their arithmetic is not re-executed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_arrays import (require,file_sha,array_sha,verified_array,compare_outputs,
    verify_decoder_cast,recompute_rgb_mse,verify_unscale,verify_reduction)
from audit_checkpoint import SafeAdapter,update_tensor
from finalize_once import admission,verify

PRODUCER_SHA='ab710d9a365ac8f502aea177f8993354e2a17d086de97cc71a0e4b265ae99f6b'
TRAJECTORY_SHA='69bc9ca998b02cc9b40704743f63ac14806113732bb250f0c28d7b4bee2fce22'
CONTRACT_SHA='d3148504db9fbee2925448ea69e5f2ba83da5eba2127bf3229c9964f445bc08a'
TARGET_SHA='10efb5fae35949846d1853d3f2a5812f03b09c15ba0884f62e8e349170c1fef7'
VIDEO=[1,24,2,8,8];AUDIO=[1,32,2,8];RGB=[5,128,128,3]
MODES=('zero','updated','reloaded','merged')
TARGETS=('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')
MODULES={f'blocks.{i}.{target}' for i in range(50) for target in TARGETS}
NAMES={name+'.lora_'+letter for name in MODULES for letter in ('a','b')}
SMALL={'guide':(VIDEO,'float32'),'raw_warm_start':(VIDEO,'float32'),'target':([1,1,128,1],'float32')}
for _m in ('reference','candidate',*MODES):
    for _s,_shape in (('video',VIDEO),('audio',AUDIO),('rgb',RGB)):
        SMALL[_m+'_'+_s]=(_shape,'float32')
    if _m!='candidate':SMALL[_m+'_decoder_input']=(VIDEO,'float16')
ARCHIVE_SMALL=set(SMALL)-{'candidate_video','candidate_audio','candidate_rgb'}
EXPECTED_ARRAYS=set(SMALL)|{'video_gradient_raw','video_gradient_unscaled'}|{
    prefix+name for prefix in ('gradient_raw__','gradient_unscaled__','parameters_before_update__','parameters_after_update__') for name in NAMES}


def read_json(path):return json.loads(Path(path).read_text())


class Arrays:
    def __init__(self,run,rows):
        self.run,self.rows=Path(run).resolve(),rows
        self.manifest,self.archive={},{}
    def load(self,role,shape=None,dtype='float32',record=None):
        row=self.rows[role]
        require(row.get('role')==role,'array role mismatch')
        if record is not None:require(row==record,'nested array record differs: '+role)
        path=(self.run/row['path']).resolve()
        require(path.is_relative_to(self.run/'arrays'),'array escaped producer directory')
        require(row['dtype']==dtype and (shape is None or row['shape']==list(shape)),'array metadata shape/dtype differs')
        value=verified_array(path,row['file_sha256'],row['array_sha256'],shape or row['shape'],dtype)
        self.manifest[role]={**row,'bytes':path.stat().st_size}
        if role in ARCHIVE_SMALL:self.archive[role]={'path':str(path),'file_sha256':row['file_sha256']}
        return value


def budget(report,success):
    a=report['actual']
    for key,limit in (('NFE',3),('VAE_decode_calls',1)):
        require(set(a[key])==set(MODES),'mode budget keys differ')
        require(all(type(v) is int and 0<=v<=limit for v in a[key].values()),'operation budget exceeded')
        if success:require(a[key]=={m:limit for m in MODES},'successful operation count incomplete')
    singles=('RGB_loss_expressions','backward_attempts','completed_backward_calls','optimizer_steps',
        'adapter_updates','checkpoint_saves','clear_reload_cycles','merge_calls','unscale_calls')
    for key in singles:
        require(type(a[key]) is int and 0<=a[key]<=1,'single-operation budget exceeded: '+key)
        if success:require(a[key]==1,'successful single-operation incomplete: '+key)
    require(a['VAE_encode_calls']==0 and a['completed_backward_calls']<=a['backward_attempts'],'forbidden encode/backward budget')
    require(set(a['stage_decode_calls'])<={m+'_decode' for m in MODES} and
        all(type(v) is int and 0<=v<=1 for v in a['stage_decode_calls'].values()),'whole-stage decode retry')
    counts=report['block_forward_calls'];require(set(counts)=={'DiT','VAE'},'unknown block group')
    for group,phases,n,repeats in (('DiT',(*MODES,'backward'),50,3),
        ('VAE',(*(m+'_decode' for m in MODES),'backward'),36,1)):
        require(set(counts[group])<=set(phases),'unexpected block phase')
        for phase,rows in counts[group].items():
            require(set(rows)<={str(i) for i in range(n)} and all(type(v) is int and 0<=v<=repeats for v in rows.values()),'block count exceeded')
        if success:require(counts[group]=={p:{str(i):repeats for i in range(n)} for p in phases},'successful block counts incomplete')
    if success:
        require(not a['forbidden_calls'] and a['stage_decode_calls']=={m+'_decode':1 for m in MODES},'forbidden call/stage count')
        require(report['budget_passed'] is True and report['block_budget_passed'] is True,'producer budget verdict differs')
    return {'actual':a,'explicit_NFE':sum(a['NFE'].values()),'block_counts':counts,
        'checkpoint_recompute_is_additional_work':True}


def inventory(row,run):
    run=Path(run).resolve()
    path=(run/row['path']).resolve()
    require(path.parent==run and file_sha(path)==row['sha256'],'frozen inventory path/SHA differs')
    data=read_json(path);require(set(data)=={'DiT','VAE'},'frozen inventory model groups differ')
    for group,expected in (('DiT',(530,2)),('VAE',(559,42))):
        for kind,count in zip(('parameters','registered_buffers'),expected):
            records=data[group][kind]
            require(len(records)==count and len({r['name'] for r in records})==count,'frozen named tensor counts differ')
            content=[{k:r[k] for k in ('name','shape','dtype','sha256_bytes')} for r in records]
            digest=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            require(digest==data[group][kind+'_sha256'],'canonical frozen fingerprint differs')
    return data


def parameter_shapes(initial):
    weights={r['name']:r for r in initial['DiT']['parameters']}
    result={}
    for name in MODULES:
        row=weights[name+'.weight']
        require(row['dtype']=='torch.bfloat16' and len(row['shape'])==2,'merge target must be BF16 linear weight')
        result[name+'.lora_a']=[8,row['shape'][1]]
        result[name+'.lora_b']=[row['shape'][0],8]
    return result


def prior_evidence(root,contract):
    bound=contract['joint_prerequisite'];paths={}
    for key in ('report','audit','archive'):
        paths[key]=(root/bound[key]['path']).resolve()
        require(paths[key].is_relative_to(root) and file_sha(paths[key])==bound[key]['sha256'],'joint prerequisite SHA/path: '+key)
    report,audit=(read_json(paths[k]) for k in ('report','audit'))
    require(report['complete'] is True and audit['complete'] is True and audit['audit_complete'] is True and
        not audit['errors'] and audit['numerical_acceptance']=='passed_small_joint_RGB_only' and
        audit['report_sha256']==bound['report']['sha256'],'joint prerequisite not independently accepted')
    reader=Arrays(paths['report'].parent,report['arrays'])
    saved={r:reader.load(r,*SMALL[r]) for r in ('guide','raw_warm_start','target','reference_video',
        'reference_audio','reference_rgb','reference_decoder_input')}
    return report,saved,inventory(report['frozen_initial'],paths['report'].parent)


def noise_inputs(report,loaded,prior,contract):
    observed=report['noise_observations']
    require(len(observed)==4 and [r['phase'] for r in observed]==list(MODES),'actual noise observer count/order')
    identity_keys=('openh3_pretrained','comfy_commit','dit','guide_helper_sha256','jit_compile_false','mm_bmm_format_nd',
        'precision_used','torch','torch_npu','torch_npu_module_to_cast_weight_dropped')
    require(all(report['pretrained'][k]==prior['pretrained'][k] for k in identity_keys),'accepted joint pretrained identity differs')
    require(all(report['environment'][k]==prior['environment'][k] for k in ('torch','torch_npu','python')),
        'accepted joint environment versions differ')
    require(report['vae_native_plan']==prior['vae_native_plan'],'native VAE geometry/padding differs')
    reference={k:v for k,v in prior['noise_observations'][0].items() if k!='phase'}
    for row in observed:
        require({k:v for k,v in row.items() if k!='phase'}==reference,'noise/initial bytes differ from accepted joint')
    require(reference['seed']==101 and reference['joint_shapes']==[VIDEO,AUDIO],'noise seed/shapes differ')
    require(report['inputs']['seed']==101 and report['inputs']['audio_init']=='zeros' and
        report['inputs']['target_used_as_model_input'] is False,'synthetic input role changed')
    require(array_sha(loaded['guide'])!=array_sha(loaded['raw_warm_start']),'raw and guide unexpectedly same')
    require(set(report['trajectory_contracts'])==set(MODES) and set(report['traces'])==set(MODES),'missing mode trajectories')
    for mode,candidate in report['trajectory_contracts'].items():
        expected={'guide_latent_sha256_float32':array_sha(loaded['guide']),
            'raw_warm_start_latent_sha256_float32':array_sha(loaded['raw_warm_start']),
            'packed_cpu_noise_sha256_float32':reference['packed_noise_sha256'],
            'packed_initial_latent_sha256_float32':reference['packed_initial_sha256']}
        require(all(candidate[k]==v for k,v in expected.items()),'actual mode input hashes differ')
        require(candidate['seed']==101 and candidate['audio_init']=='zeros' and candidate['actual_forward_NFE']==3 and
            candidate['checkpoint_blocks']==(mode=='zero') and candidate['checkpoint_decoder_blocks']==(mode=='zero') and
            candidate['target_used_as_model_input'] is False,'mode checkpoint/seed/budget differs')
        require(np.array_equal(np.asarray(candidate['sigma_schedule_float32'],np.float32),np.asarray([.909375,6/7,2/3,0],np.float32)), 'native sigmas differ')
        require(candidate['cfg']==1 and candidate['mode']=='base' and candidate['refs']==[] and
            candidate['inner_dc'] is False and candidate['s_churn']==0,'conditioning changed')
        require(candidate['merged']==(mode=='merged') and candidate['actual_live_adapter_tensors']==(0 if mode=='merged' else 400)
            and candidate['adapter_parameter_tensors']==(0 if mode=='merged' else 400),'mode adapter structure differs')
        require(candidate['adapter_B_all_zero_at_session_entry'] is (None if mode=='merged' else mode=='zero'),'B initialization boundary differs')
        require([r['step'] for r in report['traces'][mode]]==[0,1,2],'Euler trace steps differ')
    require(contract['context_sha256']==prior['contract']['context_sha256'],'joint context differs')
    return {'all_four_noise_initial_hashes_match_accepted_joint':True,'input_arrays_checked':True,
            'target_not_input_attested':True,'mode_count':4}


def parity_record(row,a,b,exact=None):
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'invalid comparison arrays')
    same=bool(np.array_equal(a,b))
    require(row['torch_equal']==same and row['same_shape'] is True and row['finite'] is True,'comparison flags differ')
    require(row['reference_shape']==list(a.shape) and row['candidate_shape']==list(b.shape) and
        row['reference_sha256_float32']==array_sha(a) and row['candidate_sha256_float32']==array_sha(b),'comparison shapes/hashes differ')
    delta=(a-b).astype(np.float64)
    verify_reduction(row['max_abs'],float(np.max(np.abs(a-b))),'comparison maxabs')
    verify_reduction(row['relative_l2'],float(np.linalg.norm(delta)/max(float(np.linalg.norm(a.astype(np.float64))),1e-30)),'comparison relativeL2')
    if 'changed_coordinates' in row:require(row['changed_coordinates']==int(np.count_nonzero(a!=b)),'comparison changed count differs')
    if exact:require(same,'strict same-mode serialization parity failed')
    return {'exact_equal':same,'max_abs':row['max_abs'],'relative_l2':row['relative_l2'],
        'byte_equal':array_sha(a)==array_sha(b)}


def outputs(report,loaded,prior_arrays):
    for role,value in prior_arrays.items():
        require(array_sha(loaded[role])==array_sha(value),'saved reference/input differs from accepted joint: '+role)
    for stream in ('video','audio','rgb'):
        require(array_sha(loaded['candidate_'+stream])==array_sha(loaded['zero_'+stream]),'joint helper candidate alias differs from zero mode')
    facts={}
    for stream in ('video','audio'):
        facts[stream]=compare_outputs(loaded['zero_'+stream],loaded['reference_'+stream],exact=True)
        parity_record(report['latent_parity'][stream],loaded['reference_'+stream],loaded['zero_'+stream],exact=True)
    facts['zero_RGB']=compare_outputs(loaded['zero_rgb'],loaded['reference_rgb'],exact=False)
    parity_record(report['rgb_parity'],loaded['reference_rgb'],loaded['zero_rgb'])
    require(report['rgb_parity']['allclose'] is True and report['rgb_parity']['atol']==.0002 and report['rgb_parity']['rtol']==.0001,
        'original zero VAE tolerance changed')
    for mode in ('reference',*MODES):verify_decoder_cast(loaded[mode+'_video'],loaded[mode+'_decoder_input'])
    require(array_sha(loaded['reference_decoder_input'])==array_sha(loaded['zero_decoder_input']),'zero decoder cast differs')
    require(set(report['graph'])=={'video_requires_grad','video_has_grad_fn','rgb_requires_grad','rgb_has_grad_fn'} and
        all(v is True for v in report['graph'].values()),'same graph RGB connection missing')
    target=loaded['target'];require(array_sha(target)==TARGET_SHA,'actual Torch target bytes differ')
    losses={}
    for mode in MODES:
        pixel=loaded[mode+'_rgb'];row=report['mode_outputs'][mode]
        require(row['geometry_and_finite'] is True and np.isfinite(pixel).all(),'mode invalid geometry/finite')
        measured=recompute_rgb_mse(pixel,target,TARGET_SHA)
        verify_reduction(row['numpy_float64_mse'],measured,'independent '+mode+' RGB MSE')
        require(type(row['torch_float32_mse']) in (int,float) and math.isfinite(row['torch_float32_mse']) and row['torch_float32_mse']>=0,'mode torch MSE invalid')
        require(row['clipping']['at_zero']==int((pixel==0).sum()) and row['clipping']['at_one']==int((pixel==1).sum()) and
            row['clipping']['outside_unit_interval']==int(((pixel<0)|(pixel>1)).sum())==0,'clipping summary differs')
        losses[mode]={'independent_numpy_float64_mse':measured,'torch32_attestation':row['torch_float32_mse']}
    loss=report['loss'];verify_reduction(loss['numpy_float64_mse'],losses['zero']['independent_numpy_float64_mse'],'backward RGB MSE')
    require(loss['torch_float32_mse']==report['mode_outputs']['zero']['torch_float32_mse'] and loss['scale']==1024 and
        loss['auxiliary_latent_audio_loss'] is False,'backward objective/scale differs')
    required={'updated__reloaded','zero__updated','zero__reloaded','zero__merged','reloaded__merged'}
    require(set(report['comparisons'])==required,'missing output comparisons')
    comparisons={}
    for key in required:
        a,b=key.split('__');comparisons[key]={}
        require(set(report['comparisons'][key])=={'video','audio','rgb'},'comparison streams differ')
        for stream in ('video','audio','rgb'):
            comparisons[key][stream]=parity_record(report['comparisons'][key][stream],loaded[a+'_'+stream],loaded[b+'_'+stream],
                exact=key=='updated__reloaded')
    return {'zero_reference':facts,'objectives':losses,'paired_outputs':comparisons,
        'merge_equivalence_acceptance':False,'loss_descent_required':False}


def grad_stats(value,row):
    finite=np.isfinite(value);valid=value[finite]
    facts={'present':True,'elements':int(value.size),'finite_count':int(finite.sum()),'all_finite':bool(finite.all()),
        'nonzero_finite_count':int(np.count_nonzero(valid)),'nan_count':int(np.isnan(value).sum()),'inf_count':int(np.isinf(value).sum())}
    require(all(row[k]==v for k,v in facts.items()),'gradient statistics counts differ')
    norm=float(np.linalg.norm(valid.astype(np.float64)));verify_reduction(row['finite_l2'],norm,'gradient norm')
    return norm,facts['nonzero_finite_count']


def gradients_update_checkpoint(report,reader,shapes,contract):
    initial=report['adapter_initial'];summary=report['adapter_gradients']
    require(initial['modules']==200 and initial['trainable_tensors']==400 and initial['rank']==8 and
        set(initial['tensors'])==NAMES and set(summary['tensors'])==NAMES,'adapter initial/gradient set differs')
    for key in ('parameters_before_update','parameters_after_update'):
        require(set(report[key])==NAMES,'saved parameter set differs')
    require(set(report['update']['tensors'])==NAMES and report['update']['optimizer_state_entries_after']==0,'SGD update/state count differs')
    optimizer={'type':'torch.optim.SGD','lr':1e-4,'momentum':0.,'dampening':0.,'weight_decay':0.,'nesterov':False,
        'maximize':False,'foreach':False,'fused':False,'loss_scale':1024.,
        'expected_arithmetic':'device FP32 before.clone().add_(unscaled_grad, alpha=-1e-4)','loss_descent_required':False}
    require(report['optimizer']==optimizer,'optimizer contract changed')
    checkpoint=report['checkpoint'];path=(reader.run/checkpoint['path']).resolve()
    require(path.parent==reader.run and path.name=='diagnostic_synthetic_SGD_step1.safetensors','checkpoint path escaped/changed')
    require(checkpoint['bytes']==path.stat().st_size and checkpoint['expected_shapes']==shapes and
        set(checkpoint['exact_saved_live_tensors'])==NAMES and all(v is True for v in checkpoint['exact_saved_live_tensors'].values()),'checkpoint coverage/shape/live flags')
    meta=checkpoint['metadata'];pretrained=report['pretrained']['openh3_pretrained']
    require(meta['pretrained_fingerprint']==pretrained['fingerprint'] and meta['rank']==8,'checkpoint fingerprint/rank')
    training=meta['training']
    require(training['execution_contract_sha256']==report['contract']['execution_contract_sha256'] and
        training['source_contract_sha256']==CONTRACT_SHA and training['manifest_sha256']==contract['manifest_sha256'] and
        training['implementation_sha256']==contract['implementation_sha256'] and training['pretrained']==report['pretrained'] and
        training['input_arrays']=={r:report['arrays'][r] for r in ('guide','raw_warm_start','target')} and
        training['noise']==report['noise_observations'][0] and training['rank']==training['alpha']==8 and training['lr']==1e-4 and
        training['steps']==1 and training['loss_scale']==1024 and training['optimizer']==optimizer,'full diagnostic provenance differs')
    require(training['role']==checkpoint['role']=='diagnostic_synthetic_plain_SGD_step1_not_a_restoration_checkpoint' and
        training['objective']=='synthetic RGB MSE only','checkpoint role/objective differs')
    require(training['sigmas']==[.909375,6/7,2/3,0.] and training['sigmas_float32_sha256']==array_sha(np.asarray(training['sigmas'],np.float32)),'sigma metadata/hash differs')
    saved=SafeAdapter(path,checkpoint['sha256'],shapes,meta)
    reload=report['reload'];require(set(reload['cleared'])==set(reload['reloaded'])==NAMES and reload['loaded_count']==400 and
        reload['metadata']==meta and reload['file_sha_after']==checkpoint['sha256'],'strict clear/reload record differs')
    count,b2,b_nonzero,B_after=0,0.,0,0;arithmetic={}
    for name in sorted(NAMES):
        shape=shapes[name];init=initial['tensors'][name];row=summary['tensors'][name]
        require(init['shape']==shape and init['dtype']=='torch.float32' and row['present'] is True,'actual adapter shape/dtype/presence')
        raw=reader.load('gradient_raw__'+name,shape,record=row['raw'])
        unscaled=reader.load('gradient_unscaled__'+name,shape,record=row['unscaled'])
        verify_unscale(raw,unscaled);norm,nz=grad_stats(raw,row['statistics_raw']);grad_stats(unscaled,row['statistics_unscaled'])
        before=reader.load('parameters_before_update__'+name,shape,record=report['parameters_before_update'][name])
        after=reader.load('parameters_after_update__'+name,shape,record=report['parameters_after_update'][name])
        require(array_sha(before)==init['sha256_bytes'] and init['all_zero']==not_nonzero(before),'initial/before tensor differs')
        if name.endswith('.lora_b'):
            require(init['all_zero'] is True and not np.count_nonzero(before),'initial B not zero')
            b2+=norm*norm;b_nonzero+=nz>0;B_after+=int(np.count_nonzero(after))
        arithmetic[name]=update_tensor(before,unscaled,after,report['update']['tensors'][name])
        file_value=saved.load(name)
        require(array_sha(file_value)==array_sha(after),'saved safetensors != updated live array')
        clear,loaded=reload['cleared'][name],reload['reloaded'][name]
        require(clear['all_zero'] is True and clear['sha256']==array_sha(np.zeros(shape,np.float32)),'actual clear bytes not zero')
        require(loaded['equals_saved'] is True and loaded['sha256']==array_sha(file_value),'reloaded bytes differ from file')
        require(type(clear['version']) is int and loaded['version']==clear['version']+1 and
            clear['version']==report['update']['tensors'][name]['version']+1,'clear/reload parameter version writes differ')
        count+=1
    require(summary['expected']==summary['present_tensors']==count==400 and summary['all_finite'] is True and summary['passed'] is True and
        summary['B_nonzero_tensors']==b_nonzero and b_nonzero>0 and b2>0,'aggregate gradient gate differs')
    verify_reduction(summary['B_l2_raw'],math.sqrt(b2),'aggregate B raw norm')
    verify_reduction(summary['B_l2_unscaled'],math.sqrt(b2)/1024.,'aggregate B unscaled norm')
    require(report['update']['B_nonzero_coordinates']==B_after and B_after>0,'updated B not nonzero')
    row=report['retained_video_gradient'];raw=reader.load('video_gradient_raw',VIDEO,record=row['raw'])
    unscaled=reader.load('video_gradient_unscaled',VIDEO,record=row['unscaled'])
    video=verify_unscale(raw,unscaled,nonzero=True);grad_stats(raw,row['statistics_raw']);grad_stats(unscaled,row['statistics_unscaled'])
    require(row['passed'] is True,'retained video gradient failed')
    return {'adapter_pairs_checked':400,'checkpoint_keys_checked':400,'B_nonzero_gradient_tensors':b_nonzero,
        'B_after_nonzero_coordinates':B_after,'retained_video_gradient':video,'SGD_coordinate_checks':arithmetic,
        'checkpoint_file_sha256':checkpoint['sha256'],'checkpoint_bytes':checkpoint['bytes'],
        'checkpoint_or_parameter_arrays_archived':False}


def not_nonzero(value):return not bool(np.count_nonzero(value))


def frozen_merge(report,initial,final,prior_inventory):
    for group in ('DiT','VAE'):
        for kind in ('parameters','registered_buffers'):
            require(initial[group][kind+'_sha256']==prior_inventory[group][kind+'_sha256'],'initial base/VAE differs from accepted joint')
    merge=report['merge'];require(set(merge['layers'])==MODULES and merge['expected_writes']==200 and
        merge['remaining_adapter_tensors']==0 and merge['equivalence_claim'] is False,'merge target inventory differs')
    require(merge['runtime_receipt']['merged_modules']==200 and merge['runtime_receipt']['live_adapter_parameters']==0 and
        merge['runtime_receipt']['storage_dtypes']==['torch.bfloat16'],'actual runtime merge inventory differs')
    changed=[];allowed={name+'.weight' for name in MODULES}
    for group in ('DiT','VAE'):
        for kind in ('parameters','registered_buffers'):
            before={r['name']:r for r in initial[group][kind]};after={r['name']:r for r in final[group][kind]}
            require(before.keys()==after.keys(),'frozen final names differ')
            for name,row in before.items():
                new=after[name]
                require(new['shape']==row['shape'] and new['dtype']==row['dtype'] and new['requires_grad'] is False and
                    new['grad_is_none'] is True and new['device']==row['device']=='npu:0','frozen final metadata/residency')
                if group=='DiT' and kind=='parameters' and name in allowed:
                    layer=merge['layers'][name.removesuffix('.weight')]
                    require(layer['base_before_sha256']==row['sha256_bytes'] and layer['base_before_version']==row['version'] and
                        layer['actual_after_sha256']==layer['expected_after_sha256']==new['sha256_bytes'] and
                        layer['actual_after_version']==new['version']==row['version']+1 and
                        layer['exact_declared_write'] is True and layer['all_finite'] is True,'declared merge weight write differs')
                    require(layer['shape']==row['shape'] and layer['dtype']=='torch.bfloat16' and layer['scale']==1.,'merge layer shape/scale')
                    for key in ('ideal_delta_l2','storage_delta_l2','rounding_residual_l2'):
                        require(type(layer[key]) in (int,float) and math.isfinite(layer[key]) and layer[key]>=0,'invalid merge statistic')
                    size=math.prod(row['shape']);n=layer['changed_weight_count'];z=layer['nonzero_ideal_rounded_to_zero_count']
                    require(type(n) is int and type(z) is int and 0<=n<=size and 0<=z<=size,'merge statistic count bounds')
                    verify_reduction(layer['changed_weight_fraction'],n/size,'merge changed fraction')
                    require(layer['zero_denominator']==(layer['ideal_delta_l2']==0),'merge denominator flag')
                    if layer['zero_denominator']:require(layer['rounding_residual_relative_l2'] is None,'zero denominator not null')
                    else:verify_reduction(layer['rounding_residual_relative_l2'],layer['rounding_residual_l2']/layer['ideal_delta_l2'],'merge relative residual')
                    if new['sha256_bytes']!=row['sha256_bytes']:changed.append(name)
                else:
                    require(new['sha256_bytes']==row['sha256_bytes'] and new['version']==row['version'],'unexpected base/buffer/VAE mutation: '+name)
    stages={'after_injection','after_RGB_backward','after_update_and_save','after_reload','after_declared_merge','finally'}
    stages|={m+s for m in MODES for s in ('_before_decode','_after_decode','_complete')}
    require(set(report['state_checks'])==stages,'missing lifecycle/frozen gates')
    for label,checks in report['state_checks'].items():
        expected={'DiT','VAE'} if label in {'after_declared_merge','merged_before_decode','merged_after_decode','merged_complete','finally'} else {'DiT','VAE','adapters'}
        require(set(checks)==expected,'missing lifecycle group')
        for group,row in checks.items():
            require(row['passed'] is True and row['failures']==[] and row['tensors_checked']=={'DiT':532,'VAE':601,'adapters':400}[group],
                'frozen parameter/buffer/object/version gate failed')
            if group != 'adapters':
                full=label in {'after_RGB_backward','after_update_and_save','updated_complete','reloaded_complete','merged_complete','finally'}
                if full:require(row['checks']==['canonical name','object','version','device','bytes','frozen','grad None'],'full byte gate missing')
                else:require(row['byte_hash']=='separately checked at full stage gates, not recomputed at this boundary','lifecycle gate scope differs')
    require(merge['declared_mutation_check']['passed'] is True and not merge['declared_mutation_check']['failures'] and
        merge['declared_mutation_check']['tensors_checked']==532 and 'finally_state_error' not in report,'merge/final frozen gate failed')
    return {'declared_weight_write_tensors':200,'changed_weight_byte_tensors':len(changed),'changed_weight_names':changed,
        'untargeted_DiT_parameters_and_all_registered_buffers_unchanged':True,'entire_VAE_unchanged':True,
        'scope':'Canonical fingerprint JSONs independently recomputed and constrained. Object/version/live byte facts and expected merge hashes are source-bound producer attestations; weight matrices were not saved or reloaded, so BA/rounding statistics are not independently re-executed.'}


def contexts(report):
    require(report['math_before']==report['math_after'] and report['math_restored'] is True,'whole math not restored')
    require(set(report['session_math'])==set(MODES),'per-mode active math missing')
    counts=Counter();rope=None
    for row in report['math_observations']:
        group,phase,state=row['group'],row['phase'],row['state'];counts[(group,phase,str(row['block']))]+=1
        require(state['in_training'] is True and state['jit_compile_false'] is True and
            state['matmul_allow_hf32'] is False and state['conv_allow_hf32'] is False,'block runtime flags changed')
        require(all(state[k]==report['math_before'][k] for k in ('deterministic_algorithms','deterministic_warn_only')),'determinism changed')
        if group=='DiT':require(state==report['session_math']['zero' if phase=='backward' else phase],'native decoder leaked into DiT')
        else:
            require(state['rope']['module']=='comfy_kitchen.backends.ascend.rope' and state['rope']['qualname']=='rms_rope_split_half','VAE native rope differs')
            if rope is None:rope=state['rope']
            require(rope==state['rope'],'VAE rope object changed')
    expected=Counter({(g,p,i):v for g,phases in report['block_forward_calls'].items() for p,blocks in phases.items() for i,v in blocks.items()})
    require(counts==expected,'math observers do not cover actual block entries')
    restores=report['decoder_context_restorations']
    require(Counter(r['phase'] for r in restores)=={'zero_decode':37,'backward':36,'updated_decode':1,'reloaded_decode':1,'merged_decode':1},'native context count differs')
    require(all(r['restored'] is True and r['before']==r['after'] for r in restores),'decoder context failed restore')
    rows=report['session_restorations'];require([r['mode'] for r in rows]==list(MODES),'mode restoration sequence')
    keys=('session_state_initialized','session_shapes_restored','session_current_patcher_restored','whole_run_shapes_restored','whole_run_current_patcher_restored','inactive')
    require(all(all(r[k] is True for k in keys) for r in rows),'session lifecycle failed restore')
    require(report['wrapper_attributes_restored'] is True and report['stale_backward_guard_rejected']=={m:True for m in MODES},'wrappers/stale graph guard')
    require(all(report['patcher_restoration'][k] is True for k in keys[:-1]),'whole-run restoration failed')
    return {'block_observations':sum(counts.values()),'native_decoder_context_restorations':76,'session_restorations':4}


def imports(report,contract):
    expected={str((Path(contract['root'])/k).resolve()):v for k,v in contract['implementation_sha256'].items()}
    require(report['verified_runtime_files']==expected,'verified runtime map differs')
    for label in ('actual_imports_before_sampling','actual_imports'):
        row=report[label];require(row['passed'] is True and row['unbound']==[] and row['modules'],'unbound actual imports')
        require(all(expected.get(v['path'])==v['sha256'] for v in row['modules'].values()),'actual import SHA differs')
        require(set(row['synthetic_namespaces'])=={'torch.ops','torch.classes'},'Torch synthetic namespace definitions missing')
        for v in row['synthetic_namespaces'].values():require(expected.get(v['defining_source_path'])==v['defining_source_sha256'],'synthetic source not pinned')
        require('comfy_kitchen.backends' in row['namespace_packages'],'PEP420 kitchen evidence missing')
        for v in row['namespace_packages'].values():
            subset={p:h for p,h in expected.items() if Path(p).suffix=='.py' and any(Path(p).is_relative_to(Path(d)) for d in v['directories'])}
            require(v['loader_type']=='_frozen_importlib_external.NamespaceLoader' and v['source_files']==subset and subset,'PEP420 source tree differs')
    return {'actual_import_maps_verified':2,'scope':'Recorded actual source definitions matched admitted pins; no runtime or model imports by auditor.'}


def audit(root,config,paths):
    run=paths['evidence'];queue=read_json(paths['run']/'queue_status.json')
    require(queue['state'] in ('completed','failed','not_launched'),'producer not terminal')
    base={'schema':'openh3.nonzero_adapter_terminal_audit.v1','complete':False,'audit_complete':False,
        'producer_terminal_state':queue['state'],'errors':[],'checks':{},'archive_arrays':{},
        'actions':{'NPU_calls':0,'Torch_imports':0,'model_loads':0,'producer_writes':0,'large_arrays_copied':0}}
    if queue['state']=='not_launched':return {**base,'audit_complete':True,'numerical_acceptance':'not_launched_not_accepted'}
    success=queue['state']=='completed';chosen=run/('report.json' if success else 'failure.json')
    if not chosen.is_file():chosen=run/'partial.json'
    report=read_json(chosen)
    require(report['schema']=='openh3.nonzero_adapter_probe.v1' and report['complete'] is success,'producer/queue state mismatch')
    require(set(report['arrays'])<=EXPECTED_ARRAYS,'unexpected saved array role')
    if success:require(set(report['arrays'])==EXPECTED_ARRAYS,'successful saved array set incomplete')
    contract=read_json(paths['prep']/'execution_contract.json')
    require(contract['schema']=='openh3.nonzero_adapter_probe_execution.v1' and Path(contract['root']).resolve()==root,'execution schema/root')
    require(contract['probe_sha256']==config['producer_sha256']==PRODUCER_SHA and config['trajectory_sha256']==TRAJECTORY_SHA and
        config['target_sha256']==TARGET_SHA and contract['source_contract_sha256']==config['source_contract_sha256']==CONTRACT_SHA,'frozen numerical source pins changed')
    require(TRAJECTORY_SHA in contract['implementation_sha256'].values() and file_sha(root/contract['source_contract'])==CONTRACT_SHA,'trajectory/source contract differs')
    require(report['contract']=={**contract,'execution_contract_sha256':config['execution_contract_sha256']},'embedded execution contract differs')
    reader=Arrays(run,report['arrays']);loaded={}
    def check(label,fn,available=True):
        if not available:base['checks'][label]={'status':'unavailable'};return None
        try:
            result=fn();base['checks'][label]={'status':'verified','facts':result};return result
        except Exception as error:
            base['checks'][label]={'status':'invalid','error':repr(error)};base['errors'].append(label+': '+str(error));return None
    check('bounded_actual_work',lambda:budget(report,success))
    # Every actual persisted array is read even on failure; no pass is inferred
    # from a readable partial artifact. Exact scientific gates require success.
    for role,row in report['arrays'].items():
        try:
            shape,dtype=SMALL.get(role,(row['shape'],'float32'))
            arr=reader.load(role,shape,dtype)
            if role in SMALL:loaded[role]=arr
        except Exception as error:base['errors'].append(role+': '+str(error))
    if success:
        prior=check('accepted_joint_prerequisite',lambda:prior_evidence(root,contract))
        # Avoid embedding original giant report/arrays in this audit JSON.
        if prior is not None:
            base['checks']['accepted_joint_prerequisite']['facts']={'report_sha256':contract['joint_prerequisite']['report']['sha256'],
                'audit_sha256':contract['joint_prerequisite']['audit']['sha256'],'saved_reference_arrays_verified':7}
        initial=check('frozen_initial_inventory',lambda:inventory(report['frozen_initial'],run))
        final=check('frozen_final_inventory',lambda:inventory(report['final_frozen'],run))
        if initial is not None:base['checks']['frozen_initial_inventory']['facts']={g:{k:initial[g][k+'_sha256'] for k in ('parameters','registered_buffers')} for g in initial}
        if final is not None:base['checks']['frozen_final_inventory']['facts']={g:{k:final[g][k+'_sha256'] for k in ('parameters','registered_buffers')} for g in final}
        if prior is not None and set(loaded)==set(SMALL):
            check('actual_outputs_RGB_objectives',lambda:outputs(report,loaded,prior[1]))
            check('noise_input_schedule_identity',lambda:noise_inputs(report,loaded,prior[0],contract))
        if initial is not None:
            check('gradients_SGD_checkpoint_reload',lambda:gradients_update_checkpoint(report,reader,parameter_shapes(initial),contract))
        if initial is not None and final is not None and prior is not None:
            check('frozen_base_buffers_declared_merge',lambda:frozen_merge(report,initial,final,prior[2]))
        check('math_context_restoration',lambda:contexts(report))
        check('actual_imports',lambda:imports(report,contract))
        check('verified_asset_scope',lambda: require(report['asset_bytes_sha_verified'] is True and
            report['environment']['synthetic_no_dataset'] is True,'asset/data scope differs'))
    else:
        for label in ('frozen_initial','final_frozen'):
            check(label+'_partial',lambda label=label:inventory(report[label],run),label in report)
        base['checks']['failed_run_scope']={'status':'verified','facts':{'partial_arrays_read':len(reader.manifest),
            'no_success_or_optimizer_arithmetic_acceptance':True,'failure':report.get('failure')}}
    base.update(audit_complete=True,complete=success and not base['errors'],producer_phase=report.get('phase'),
        producer_error=report.get('failure'),report_sha256=file_sha(chosen),array_manifest=reader.manifest,archive_arrays=reader.archive)
    base['numerical_acceptance']='passed_small_single_SGD_strict_reload_merge_descriptive_only' if base['complete'] else (
        'producer_failed_not_accepted' if not success else 'invalid_evidence')
    base['limitations']=['No real restoration data, quality, convergence, AdamW state or full-resolution capacity acceptance.',
        'Nontrivial SGD fused/split CPU candidates are descriptive only; no NPU FMA reproduction or new tolerance.',
        'Complete saved fingerprint inventories are audited; live model matrices were not reloaded. BF16 BA/rounding statistics remain producer attestations.',
        'Merged versus unmerged endpoint differences have no equivalence threshold; no loss descent requirement.',
        'Large gradients/parameter arrays and adapter checkpoint are checked in place and excluded from the small archive.']
    return base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();own=Path(__file__).resolve().parent
    require(not args.out.exists() and not args.out.is_symlink(),'audit output exists')
    config,paths=admission(root,own,args.config,args.config_sha256);verify(config,paths,own)
    require(not args.out.resolve().is_relative_to(paths['run']),'auditor cannot modify producer output')
    try:result=audit(root,config,paths)
    except Exception as error:result={'schema':'openh3.nonzero_adapter_terminal_audit.v1','audit_complete':False,'complete':False,
        'numerical_acceptance':'invalid_or_missing_evidence','errors':[repr(error)],'archive_arrays':{},
        'actions':{'NPU_calls':0,'Torch_imports':0,'producer_writes':0}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':result['complete'],'numerical_acceptance':result['numerical_acceptance'],
        'errors':result['errors'],'audit_sha256':file_sha(args.out)}))
    return 0 if not result['errors'] else 2


if __name__=='__main__':raise SystemExit(main())
