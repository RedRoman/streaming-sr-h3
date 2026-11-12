"""Independent cache audit: saved K1 bytes, raw inputs/features and supervision.

No exporter import, restorer construction, checkpoint deserialization, NPU or
restorer replay. Torch is used only for pinned CPU raw-feature arithmetic.
Independent FP64 tile loops use the existing headroom 1e-12 metric tolerance;
normalization, feature bytes, selections and all provenance gates remain exact.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile

import numpy as np
import train_history_utility as T

SCHEMA = 'rphi.input_preservation_cache_audit_execution.v1'
PRODUCER_SHA = 'c1bfb152b99d62937abb88a23f8668f8193768fbcee59277cffbe3826af75bb4'
TEACHER_SHA = '79995694df154084af6e76020dfa05c0f64010600bba839521fee7a01d326e34'
HEADROOM_SHA = '6760dccb1e71cefae86ee23931b4b5b1491c88ba03f5c3a331e113183ed71578'
HEADROOM_AUDIT_SHA = '38d783deb2f41fc31da4657ba0329383bc192fd41ff943962d3041e5d4607807'
INPUT_MODULE_SHA = '7a52656a3119dcaad515dee0ec591bccbb21ee7756ba150b7f59827ad3b8100d'
TRAIN_NAMES = {'k1_rgb','raw_features','availability','input_mae','k1_mae','raw_benefit','normalized_benefit'}
DEV_NAMES = {'raw_features','availability','diagnostic_input_mae','diagnostic_K1_mae',
             'diagnostic_clipped_input_mae','diagnostic_clipped_K1_mae','oracle_unclamped_index','oracle_clipped_index'}
OLD_NAMES = {'current','target','all_history','current_only','remove_0','remove_1'}
EXTRA = {'observed_crop_sha256_float32','target_crop_sha256_float32','availability','all_history_mae',
         'current_only_mae','raw_utility','metrics','all_history_mae_harm_vs_input','current_only_mae_harm_vs_input'}
ATOL, PSNR_ATOL = 1e-12, 1e-10


def asha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)
def read_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def binding(p):return {'path':str(Path(p).resolve()),'sha256':T.file_sha(p)}


def compare(actual,expected,errors,path='root'):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),'dictionary keys differ: '+path)
        for k,v in expected.items():compare(actual[k],v,errors,path+'.'+k)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),'list differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,errors,path+'.'+str(i))
    elif isinstance(expected,float):
        need(type(actual) in (float,int) and math.isfinite(actual) and math.isfinite(expected),'nonfinite scalar: '+path)
        difference=abs(actual-expected);category='psnr' if 'psnr' in path else 'metric'
        errors[category]=max(errors.get(category,0.),difference)
        need(difference <= (PSNR_ATOL if category=='psnr' else ATOL),'metric differs: '+path)
    else:need(type(actual) is type(expected) and actual==expected,'exact value differs: '+path)


def raw_features_cpu(observed,availability):
    """Independent expression of the fixed raw-only feature math, no module."""
    import torch
    from torch.nn import functional as F
    x=torch.from_numpy(np.ascontiguousarray(observed));a=torch.from_numpy(np.ascontiguousarray(availability))
    need(x.dtype==a.dtype==torch.float32 and x.device.type=='cpu' and
         x.ndim==5 and tuple(x.shape[1:])==(3,3,96,96) and a.shape==(len(x),2), 'raw-feature shape/dtype')
    need(bool(torch.isfinite(x).all()) and bool(((x>=0)&(x<=1)).all()) and
         bool(((a==0)|(a==1)).all()), 'raw-feature input range')
    with torch.inference_mode():
        now=x[:,-1:].expand(-1,2,-1,-1,-1)
        past=torch.where(a[:,:,None,None,None].bool(),x[:,:2],now)
        evidence=torch.cat((now,past,torch.abs(now-past)),dim=2)
        pooled=F.avg_pool2d(evidence.reshape(len(x)*2,9,96,96),16,16)
        lag=x.new_tensor([1.,.5])[None].expand(len(x),-1).reshape(len(x)*2,1,1,1)
        mask=a.reshape(len(x)*2,1,1,1)
        result=torch.cat((pooled,lag.expand(-1,1,6,6),mask.expand(-1,1,6,6)),dim=1)
    return result.reshape(len(x),2,11,6,6).contiguous().numpy().copy()


def independent_tiles(prediction,target):
    need(prediction.dtype==target.dtype==np.float32 and prediction.shape==target.shape
         and prediction.ndim==4 and prediction.shape[1:]==(3,96,96)
         and np.isfinite(prediction).all() and np.isfinite(target).all(),'tile input shape/dtype/finite')
    pixels=np.abs(prediction.astype(np.float64)-target.astype(np.float64)).mean(axis=1)
    return np.asarray([[[[pixels[b,y:y+16,x:x+16].mean() for x in range(0,96,16)]
                         for y in range(0,96,16)]] for b in range(len(pixels))],np.float64)


def array_close(actual,expected,errors,name):
    need(actual.dtype==expected.dtype and actual.shape==expected.shape and np.isfinite(actual).all(),name+' shape/dtype/finite')
    error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
    errors[name]=max(errors.get(name,0.),error);need(error<=ATOL,name+' FP64 reduction differs')


def verify_labels(arrays,current,target,errors):
    k1=arrays['k1_rgb'];input_mae=independent_tiles(current,target);k1_mae=independent_tiles(k1,target)
    for name,expected in [('input_mae',input_mae),('k1_mae',k1_mae),('raw_benefit',input_mae-k1_mae)]:
        array_close(arrays[name],expected,errors,name)
    need(asha(arrays['raw_benefit'])==asha(arrays['input_mae']-arrays['k1_mae']), 'signed difference is not the actual saved MAE subtraction')
    normalized=np.clip(arrays['raw_benefit']/(.25/255),-4.,4.).astype(np.float32)
    need(asha(arrays['normalized_benefit'])==asha(normalized),'normalization bytes differ')


def expected_shapes(role,batch=16):
    result={'raw_features':((batch,2,11,6,6),np.dtype('float32')),
            'availability':((batch,2),np.dtype('float32'))}
    if role=='train':
        result['k1_rgb']=((batch,3,96,96),np.dtype('float32'))
        result.update({name:((batch,1,6,6),np.dtype('float32' if name=='normalized_benefit' else 'float64'))
                       for name in ('input_mae','k1_mae','raw_benefit','normalized_benefit')})
    else:
        result.update({name:((batch,1,6,6),np.dtype('uint8' if name.startswith('oracle_') else 'float64'))
                       for name in DEV_NAMES-set(result)})
    return result


def load_npz(entry,folder,shapes):
    p=Path(entry['path']);folder=Path(folder).resolve()
    need(set(entry['arrays'])==set(shapes),'NPZ declared array keys differ')
    need(not p.is_symlink() and p.resolve().parent==folder/'arrays' and p.stat().st_size==entry['bytes']
         and T.file_sha(p)==entry['sha256'],'actual NPZ file binding differs')
    with zipfile.ZipFile(p) as z:
        members=z.infolist()
        need(len(members)==len(shapes) and {m.filename for m in members}=={n+'.npy' for n in shapes}
             and all(m.compress_type==zipfile.ZIP_STORED and m.file_size<2<<20 for m in members),'NPZ membership/encoding differs')
    values={}
    with np.load(p,allow_pickle=False) as z:
        for name,(shape,dtype) in shapes.items():
            a=z[name]
            need(a.dtype==dtype and a.shape==shape and a.flags.c_contiguous and np.isfinite(a).all(),'actual cache shape/dtype/finite differs')
            need(entry['arrays'][name]=={'dtype':str(a.dtype),'shape':list(a.shape),'sha256':asha(a)},'actual cache array SHA differs')
            values[name]=a
    return values


def verify_record_records(entry,records,arrays,role):
    need(len(records)==len(next(iter(arrays.values()))),'batch row count differs')
    need(entry['row_indices']==[r['row_index'] for r in records] and entry['row_sha256']==[r['row_sha256'] for r in records], 'batch row bindings differ')
    for i,r in enumerate(records):
        need(r['cache_role']==role and r['usable_for_training'] is (role=='train')
             and r['teacher_checkpoint_sha256']==TEACHER_SHA and r['cache_file_sha256']==entry['sha256']
             and r['cache_record_sha256']==T.value_sha({k:v for k,v in r.items() if k!='cache_record_sha256'}), 'cache record identity/role differs')
        need(r['array_row_sha256']=={k:asha(v[i]) for k,v in arrays.items()},'actual per-row array hashes differ')
        need(r['feature_sha256_float32']==asha(arrays['raw_features'][i]) and
             r['availability']==arrays['availability'][i].tolist(),'feature/availability row binding differs')


def verify_identity_chain(pc,paths,teacher,terminal):
    """Independent metadata chain anchored in the accepted headroom report."""
    headroom=T.read(paths['headroom_report']);audit=T.read(paths['headroom_audit'])
    need(pc['files']['teacher_checkpoint']['sha256']==TEACHER_SHA
         and pc['files']['headroom_report']['sha256']==HEADROOM_SHA
         and pc['files']['headroom_audit']['sha256']==HEADROOM_AUDIT_SHA,
         'teacher/headroom fixed identities differ')
    need(headroom['complete'] is True and headroom['status']=='complete_headroom_fail'
         and headroom['assessment']['headroom_rule_passed'] is False and audit['complete'] is True
         and audit['report_sha256']==HEADROOM_SHA and audit['actual_cache_files']==180
         and audit['actual_arrays']==1080 and audit['actual_rows']==2880,'accepted headroom/audit differs')
    for name in ('teacher_contract','teacher_report','teacher_checkpoint','dataset_audit','headroom_contract'):
        need(pc['files'][name]==headroom['execution_contract'][name],'headroom identity link differs: '+name)
    need(terminal['complete'] is True and terminal['completed_step']==1000 and terminal['final_frozen'] is True
         and terminal['execution_contract']==teacher
         and terminal['execution_contract_file_sha256']==pc['files']['teacher_contract']['sha256']
         and terminal['final_model_sha256']==headroom['frozen_before_sha256']==headroom['frozen_after_sha256']
         and teacher['model_config']==T.CONFIG and teacher['optimizer']==T.OPTIMIZER
         and teacher['loss']==T.LOSS and teacher['phase']=='teacher' and teacher['steps']==1000,
         'accepted teacher final checkpoint contract differs')
    data=T.read(paths['dataset_audit'])
    need(data['complete'] is True and data['revised_pair_ready'] is True
         and data['pair']==terminal['data']['pair'] and data['pair']['train_sources']==19
         and data['pair']['dev_sources']==6,'accepted data role/isolation link differs')
    for name in ('schedule_manifest','student_schedule','dataset_audit'):
        need(pc['files'][name]==teacher['files'][name],'teacher schedule/data link differs: '+name)
    need(pc['files']['headroom_records']=={k:headroom['records'][k] for k in ('path','sha256')}
         and headroom['records']['rows']==2880 and len(headroom['actual_rgb_caches'])==180,'headroom records link differs')


def fresh_audit_location(out,producer_folder):
    out=Path(out).resolve();producer_folder=Path(producer_folder).resolve()
    need(out!=producer_folder and producer_folder not in out.parents,'audit output must not be inside producer cache')
    need(not out.exists(),'audit output already exists')
    return out


def inspection(path,*,execute=False,out=None):
    c=T.read(path);root=c['root'];role=c['role']
    need(c['schema']==SCHEMA and role in ('train','dev') and Path(root).is_absolute(),'cache audit contract')
    need(c['resources']=={'wall_seconds':900,'CPU_cores':2,'RSS_limit_bytes':(32 if role=='train' else 4)<<30,
                         'max_small_output_bytes':32<<20},'audit resource contract differs')
    manifest_path=T.bound_path(c['producer_manifest'],root);r=T.read(manifest_path)
    need(r['schema']=='rphi.input_preservation_cache.v1' and r['complete'] is True and r['role']==role
         and r['status']=='cache_complete_pending_independent_audit_no_training_admission'
         and r['head_training_admitted'] is False and r['original_history_headroom_remains_failed'] is True
         and r['usable_for_training'] is (role=='train') and r['completed_rows']==(16000 if role=='train' else 2880), 'incomplete/wrong-role producer')
    pc=r['execution_contract'];need(pc['root']==pc['evidence_root']==root and pc['role']==role
                                  and pc['schema']=='rphi.input_preservation_cache_execution.v1','producer contract root/role')
    ep=T.bound_path(c['producer_execution_contract'],root)
    need(c['producer_execution_contract']['sha256']==r['execution_contract_file_sha256'] and T.read(ep)==pc,'producer execution file differs')
    by_name={Path(p).name:h for p,h in pc['implementation_sha256'].items()}
    need(by_name.get('export_input_preservation_cache.py')==PRODUCER_SHA and
         by_name.get('rphi_input_preservation.py')==INPUT_MODULE_SHA,'producer/feature implementation differs')
    for source_path,h in pc['implementation_sha256'].items():need(T.file_sha(source_path)==h,'producer source changed')
    paths={k:T.bound_path(b,root) for k,b in pc['files'].items()}
    teacher=T.read(paths['teacher_contract']);terminal=T.read(paths['teacher_report'])
    verify_identity_chain(pc,paths,teacher,terminal)
    for name in T.DEPENDENCIES:
        local=Path(__file__).resolve().parent/name
        need(by_name.get(name)==T.file_sha(local),'auditor loader math differs from authenticated producer')
    for k in ('records','batches'):
        b=r[k];q=Path(b['path'])
        need(q.resolve().parent==manifest_path.parent and q.stat().st_size==b['bytes'] and T.file_sha(q)==b['sha256'],'producer index/records changed')
    count=r['counts'];need(count['backwards']==count['optimizer_updates']==0 and
                          count['raw_feature_rows']==r['completed_rows'] and count['raw_feature_batches']*16==r['completed_rows'],'operation counts differ')
    if role=='train':
        need(set(r['frozen_checks'])=={'state_dict_exact','versions_exact','eval_frozen_no_grads'}
             and all(value is True for value in r['frozen_checks'].values()),'frozen check schema/values differ')
        need(count['restorer_forward_attempts']==count['restorer_forwards_completed']==count['restorer_invocations_observed']==1000 and
             count['crop_forward_attempts']==count['crop_forwards_completed']==16000 and all(r['frozen_checks'].values()) and
             r['frozen_before_sha256']==r['frozen_after_sha256']==terminal['final_model_sha256'],'frozen K1 source-bound attestations differ')
    else:
        need(all(count[k]==0 for k in ('restorer_forward_attempts','restorer_forwards_completed','restorer_invocations_observed','crop_forward_attempts','crop_forwards_completed'))
             and count['old_cache_file_verification_attempts']==count['old_cache_files_verified']==180 and count['old_arrays_verified']==1080
             and count['raw_K3_exact_rows']==2880 and r['frozen_teacher_constructed'] is False,'cached dev budget differs')
    for q,h in c['implementation_sha256'].items():need(T.file_sha(q)==h,'audit source pin differs')
    need(c['implementation_sha256'].get(str(Path(__file__).resolve()))==T.file_sha(__file__),'auditor source not pinned')
    if execute:
        a=T.read(T.bound_path(c['admission'],root))
        need(a['schema']=='rphi.input_preservation_cache_audit_admission.v1' and a['approved'] is True and a['role']==role
             and a['execution_sha256']==T.execution_sha(c) and a['entry_sha256']==T.file_sha(__file__)
             and a['out']==str(Path(out).resolve()),'separate cache audit admission required')
    return c,r,pc,paths,teacher,terminal,manifest_path.parent


def verify_output_inventory(r,folder,entries):
    """Account the producer's complete existing files, without copying arrays."""
    folder=Path(folder).resolve()
    expected={'manifest.json','partial.json','records.jsonl','batches.jsonl'}
    for entry in entries:
        p=Path(entry['path'])
        need(p.is_absolute() and p.resolve().parent==folder/'arrays','cache array outside producer output')
        name=str(p.resolve().relative_to(folder));need(name not in expected,'duplicate cache file')
        expected.add(name)
    actual={str(p.relative_to(folder)):p for p in folder.rglob('*') if p.is_file() or p.is_symlink()}
    need(set(actual)==expected and all(not p.is_symlink() for p in actual.values()),'producer output membership changed')
    total=sum(p.stat().st_size for p in actual.values())
    need(total==r['actual_output_bytes'] and total<=3<<30,'producer actual output byte budget differs')
    return {'files':len(actual),'bytes':total,'array_bytes':sum(e['bytes'] for e in entries),
            'array_membership_sha256':T.value_sha([{'path':e['path'],'sha256':e['sha256'],'bytes':e['bytes']} for e in entries])}


def verify_train(c,r,pc,paths,teacher,terminal,folder,result,out):
    data,schedules,proof=T.load_inputs({**teacher,'data_cache_bytes':12<<30})
    need(proof==r['data_proof'],'actual train/dev pair/source proof differs')
    rows=schedules['student']['numeric_schedule']['rows'];records=read_lines(r['records']['path']);entries=read_lines(r['batches']['path'])
    need(len(rows)==len(records)==16000 and len(entries)==1000 and schedules['student']['schedule_sha256']==r['schedule_sha256'],'train schedule/cache coverage differs')
    # Actual loader validates receipt/NPZ metadata, pixels and original roles.
    need(data.source_ids==sorted(r['source_recompute_references']['sources']),'recompute source inventory differs')
    need(data.binding==r['source_recompute_references']['dataset_binding'],'recompute dataset binding differs')
    for sid,source in data._numeric.sources.items():
        expected=r['source_recompute_references']['sources'][sid]
        need(expected['source']==source['receipt']['source'] and expected['window_key']==source['receipt']['window_key']
             and expected['authority']==data.authorities[sid],'source recompute authority differs')
        for name,a in source['artifacts'].items():
            p=source['root']/(name+'.npz') if source['root'] else Path(a['path'])
            need(expected['artifacts'][name]=={**a,'path':str(p.resolve())},'source file/pixel reference differs')
    for j,entry in enumerate(entries):
        selected=rows[j*16:(j+1)*16];rr=records[j*16:(j+1)*16];batch=data.batch(selected)
        need(entry['batch_index']==j,'batch order differs');a=load_npz(entry,folder,expected_shapes('train'))
        verify_record_records(entry,rr,a,'train')
        x,y,availability=batch['model_inputs']['observed'],batch['target'],batch['model_inputs']['availability']
        need(asha(a['availability'])==asha(availability),'actual train availability differs')
        features=raw_features_cpu(x,availability);need(asha(features)==asha(a['raw_features']),'actual raw-only CPU features differ')
        verify_labels(a,x[:,-1],y,result['maximum_absolute_difference'])
        for i,(wanted,metadata,record) in enumerate(zip(selected,batch['metadata'],rr)):
            need(all(record.get(k)==v for k,v in metadata.items()) and wanted['split']=='train' and wanted['teacher_mask'] is None
                 and set(batch['model_inputs'])=={'observed','availability'} and 'teacher_weights' not in batch and record['batch_index']==j
                 and record['batch_row']==i and record['current_sha256_float32']==asha(x[i,-1])
                 and record['K1_sha256_float32']==asha(a['k1_rgb'][i]), 'actual train row/input/target/K1 identity differs')
            need(record['label_saturation_fraction']==float(np.mean(np.abs(a['normalized_benefit'][i])==4))
                 and record['K1_clip_fraction']==float(np.mean((a['k1_rgb'][i]<0)|(a['k1_rgb'][i]>1))), 'train label/clip statistics differ')
        result['actual_cache_files']+=1;result['actual_cache_arrays']+=len(a);result['actual_rows']+=16
        result['actual_cache_bytes']+=entry['bytes'];result['raw_feature_rows_recomputed']+=16
        T.write(out/'partial.json',{k:result[k] for k in ('complete','role','actual_cache_files','actual_rows','maximum_absolute_difference')})
    result['actual_pair']=proof['pair'];result['data_recompute_scope']='Original source19 train pixels; dev6 only source-isolation verification. No dev supervision used.'
    result['producer_output_inventory']=verify_output_inventory(r,folder,entries)


def metric(x,y):
    raw=x.astype(np.float64)-y.astype(np.float64);clipped=np.clip(x.astype(np.float64),0,1)-y.astype(np.float64)
    mse=float(np.square(clipped).mean())
    return {'unclamped_mae':float(np.abs(raw).mean()),'unclamped_mse':float(np.square(raw).mean()),
            'clipped_mae':float(np.abs(clipped).mean()),'clipped_mse':mse,
            'clipped_psnr':None if mse==0 else float(-10*np.log10(mse)),'clipped_exact_match':mse==0}


def dev_diagnostic(a,current,k1,target,errors):
    expected={'diagnostic_input_mae':independent_tiles(current,target),
              'diagnostic_K1_mae':independent_tiles(k1,target),
              'diagnostic_clipped_input_mae':independent_tiles(current,target),
              'diagnostic_clipped_K1_mae':independent_tiles(np.clip(k1,0,1),target)}
    for key,value in expected.items():array_close(a[key],value,errors,key)
    # Exact decision on the saved audited FP64 errors: both signed-zero ties
    # preserve input. No metric tolerance is applied to this threshold.
    choices=[]
    for name,first,second in [('oracle_unclamped_index','diagnostic_input_mae','diagnostic_K1_mae'),
                              ('oracle_clipped_index','diagnostic_clipped_input_mae','diagnostic_clipped_K1_mae')]:
        mask=(a[first]-a[second]>0).astype(np.uint8)
        need(asha(mask)==asha(a[name]),'oracle threshold/tie differs')
        choices.append(np.where(np.repeat(np.repeat(mask,16,axis=2),16,axis=3).astype(bool),k1,current))
    rows=[]
    for i in range(len(current)):
        rows.append({'input':metric(current[i],target[i]),'K1':metric(k1[i],target[i]),
                     'unclamped_tile_oracle':metric(choices[0][i],target[i]),'clipped_tile_oracle':metric(choices[1][i],target[i]),
                     'raw_oracle_K1_tile_fraction':float(a['oracle_unclamped_index'][i].mean()),
                     'clipped_oracle_K1_tile_fraction':float(a['oracle_clipped_index'][i].mean())})
    return rows


def summary(rows):
    modes=('input','K1','unclamped_tile_oracle','clipped_tile_oracle');keys=('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse')
    groups=defaultdict(list)
    for row in rows:
        for phase in set((row['severity_phase'],'all_frames')):groups[row['source_id'],row['profile_id'],phase].append(row['two_candidate_diagnostic'])
    detailed=[]
    for (sid,pid,phase),values in sorted(groups.items()):
        metrics={}
        for mode in modes:
            item={k:float(np.mean([v[mode][k] for v in values])) for k in keys};mse=item['clipped_mse']
            item['clipped_psnr_from_mean_mse']=None if mse==0 else float(-10*np.log10(mse));metrics[mode]=item
        detailed.append({'source_id':sid,'profile_id':pid,'severity_phase':phase,'frames':len(values),'metrics':metrics,
            **{k:float(np.mean([v[k] for v in values])) for k in ('raw_oracle_K1_tile_fraction','clipped_oracle_K1_tile_fraction')}})
    source=[]
    for sid in sorted({r['source_id'] for r in rows}):
        native=[r for r in detailed if r['source_id']==sid and r['severity_phase']=='all_frames' and r['profile_id'] not in ('identity','near_clean')]
        need(len(native)==3,'three native profiles required for source-balanced diagnostic')
        source.append({'source_id':sid,'native_equal_profile':{m:{k:float(np.mean([r['metrics'][m][k] for r in native])) for k in keys} for m in modes}})
    return {'per_source_profile_phase':detailed,'per_source_native':source,
            'native_equal_source_equal_profile':{m:{k:float(np.mean([r['native_equal_profile'][m][k] for r in source])) for k in keys} for m in modes},
            'scope':'GT-dependent hard tile oracle of input and K1 only, not a deployable head, not the prior five-candidate bound or a screen pass.'}


def verify_dev(c,r,pc,paths,teacher,terminal,folder,result,out):
    headroom=T.read(paths['headroom_report']);protocol=T.read(paths['headroom_contract'])
    original=read_lines(paths['headroom_records']);records=read_lines(r['records']['path']);entries=read_lines(r['batches']['path'])
    need(len(records)==len(original)==2880 and len(entries)==180,'dev coverage differs')
    base=[{k:v for k,v in row.items() if k not in EXTRA} for row in original]
    need(T.value_sha(base)==headroom['evaluation_rows_sha256'],'original dev schedule differs')
    expected=[(s,p,t) for s in protocol['dev_source_ids'] for p in protocol['profile_ids'] for t in range(96)]
    for i,(b,key) in enumerate(zip(base,expected)):
        need((b['source_id'],b['profile_id'],b['frame_index'])==key and b['row_index']==i and b['split']=='dev'
             and b['row_sha256']==T.value_sha({k:v for k,v in b.items() if k!='row_sha256'}),'dev row identity differs')
    oldroot=Path(headroom['records']['path']).resolve().parent;all_diagnostics=[]
    oldshapes={n:((16,3,96,96),np.dtype('float32')) for n in OLD_NAMES}
    for start in range(0,2880,96):
        old=[]
        for j in range(start//16,(start+96)//16):
            entry=headroom['actual_rgb_caches'][j]
            need(entry['row_indices']==list(range(j*16,(j+1)*16)) and entry['row_sha256']==[b['row_sha256'] for b in base[j*16:(j+1)*16]], 'old cache row bindings')
            arrays=load_npz(entry,oldroot,oldshapes)
            need(all(((arrays[k]>=0)&(arrays[k]<=1)).all() for k in ('current','target')),'original current/target unit range differs')
            old.append({k:arrays[k] for k in ('current','target','current_only')})
            result['original_cache_files_verified']+=1;result['original_cache_arrays_verified']+=6
            result['original_cache_bytes_verified']+=entry['bytes']
        current=np.concatenate([a['current'] for a in old]);target=np.concatenate([a['target'] for a in old]);k1=np.concatenate([a['current_only'] for a in old])
        observed=[]
        for t in range(96):
            row=original[start+t];h=[max(t-2,0),max(t-1,0),t];x=np.ascontiguousarray(current[h])
            need(row['crop_tlhw']==[192,368,96,96] and row['history_indices']==h and row['external_mask']==[1,1]
                 and row['teacher_mask'] is None and row['availability']==row['true_availability']==[int(t>=2),int(t>=1)], 'dev causal/crop policy differs')
            need(asha(x)==row['observed_crop_sha256_float32'] and asha(target[t])==row['target_crop_sha256_float32']
                 and asha(current[t])==row['metrics']['input']['prediction_float32_sha256']
                 and asha(k1[t])==row['metrics']['current_only']['prediction_float32_sha256'],'actual cached dev RGB/K3 differs')
            observed.append(x)
        observed=np.stack(observed)
        for offset in range(0,96,16):
            j=(start+offset)//16;rr=records[j*16:(j+1)*16];entry=entries[j]
            need(entry['batch_index']==j,'dev batch order differs');a=load_npz(entry,folder,expected_shapes('dev'));verify_record_records(entry,rr,a,'dev')
            availability=np.asarray([x['availability'] for x in original[j*16:(j+1)*16]],np.float32)
            need(asha(a['availability'])==asha(availability),'dev effective availability differs')
            features=raw_features_cpu(observed[offset:offset+16],availability)
            need(asha(features)==asha(a['raw_features']),'actual cached raw-only dev features differ')
            diagnostics=dev_diagnostic(a,current[offset:offset+16],k1[offset:offset+16],target[offset:offset+16],result['maximum_absolute_difference'])
            for i,(row,diag) in enumerate(zip(rr,diagnostics)):
                originalrow=original[j*16+i]
                need(all(row.get(k)==v for k,v in base[j*16+i].items()) and row['raw_K3_reconstructed_exact'] is True
                     and row['observed_crop_sha256_float32']==originalrow['observed_crop_sha256_float32']
                     and row['target_crop_sha256_float32']==originalrow['target_crop_sha256_float32']
                     and row['input_RGB_reference']=={'cache':headroom['actual_rgb_caches'][j],'batch_row':i},'dev record/original cache link differs')
                compare(row['two_candidate_diagnostic'],diag,result['maximum_absolute_difference'],'two_candidate')
                all_diagnostics.append({**row,'two_candidate_diagnostic':diag})
            result['actual_cache_files']+=1;result['actual_cache_arrays']+=len(a);result['actual_rows']+=16
            result['actual_cache_bytes']+=entry['bytes'];result['raw_feature_rows_recomputed']+=16
            T.write(out/'partial.json',{k:result[k] for k in ('complete','role','actual_cache_files','actual_rows','maximum_absolute_difference')})
    reduced=summary(all_diagnostics);compare(r['two_candidate_diagnostic'],reduced,result['maximum_absolute_difference'],'summary')
    result['two_candidate_diagnostic']=reduced
    need(result['original_cache_bytes_verified']==r['original_cache_bytes_verified']==headroom['array_bytes_on_disk'],
         'original dev cache bytes differ')
    result['producer_output_inventory']=verify_output_inventory(r,folder,entries)


def pack_small(out,contract,producer_manifest):
    files={'audit.json':out/'audit.json','partial.json':out/'partial.json','execution_contract.json':Path(contract),
           'producer_manifest.json':Path(producer_manifest),'source/audit_input_preservation_cache.py':Path(__file__)}
    inventory={name:{'sha256':T.file_sha(p),'bytes':p.stat().st_size} for name,p in files.items()}
    T.write(out/'MANIFEST.json',inventory);files['MANIFEST.json']=out/'MANIFEST.json'
    need(sum(p.stat().st_size for p in files.values())<16<<20,'small evidence would exceed16MiB uncompressed')
    archive=out/'evidence.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name,p in files.items():tar.add(p,arcname=name,recursive=False)
    T.write(out/'archive_receipt.json',{'path':str(archive),'sha256':T.file_sha(archive),'bytes':archive.stat().st_size,
        'NPZ_or_checkpoint_arrays_included':False,'payload_files':len(inventory)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    c,r,pc,paths,teacher,terminal,folder=inspection(a.execution_contract,execute=a.execute,out=a.out)
    out=fresh_audit_location(a.out,folder);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'schema':'rphi.input_preservation_cache_actual_audit.v1','complete':False,'role':c['role'],
        'producer_manifest':c['producer_manifest'],'producer_source_sha256':PRODUCER_SHA,
        'actual_rows':0,'actual_cache_files':0,'actual_cache_arrays':0,'actual_cache_bytes':0,'raw_feature_rows_recomputed':0,
        'original_cache_files_verified':0,'original_cache_arrays_verified':0,'original_cache_bytes_verified':0,'maximum_absolute_difference':{},
        'tolerances':{'FP64_metrics':ATOL,'PSNR':PSNR_ATOL,'features_normalization_thresholds_provenance':'exact bytes'},
        'restorer_calls':0,'restorer_constructed':False,'checkpoint_deserialized':False,'NPU_calls':0,
        'head_training_admitted':False,'original_history_headroom_remains_failed':True,
        'K1_forward_and_freeze_scope':'Source-bound producer counters and frozen SHA attestations; this audit does not replay the teacher.'}
    try:
        if a.execute:
            need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','disable backend autoload before importing Torch')
            import torch
            need('torch_npu' not in sys.modules,'audit must not import NPU backend');torch.set_num_threads(2)
            versions={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'torch':str(torch.__version__),'torch_npu':None}
            expected={**pc['runtime']['expected_versions'],'torch_npu':None};need(versions==expected,'same CPU feature backend required')
            result['runtime']={'versions':versions,'threads':2,'device':'cpu'}
            (verify_train if c['role']=='train' else verify_dev)(c,r,pc,paths,teacher,terminal,folder,result,out)
            rows,files,arrays=(16000,1000,7000) if c['role']=='train' else (2880,180,1440)
            need(result['actual_rows']==result['raw_feature_rows_recomputed']==rows and result['actual_cache_files']==files
                 and result['actual_cache_arrays']==arrays,'actual audit coverage incomplete')
            need('torch_npu' not in sys.modules and torch.get_num_threads()==2,'CPU-only audit runtime changed')
            result['runtime']['torch_npu_imported_at_end']=False
            result['status']='actual_cache_data_features_and_labels_verified_no_training_admission'
        else:result['status']='metadata_inspected_no_array_or_torch_execution'
        inspection(a.execution_contract,execute=a.execute,out=a.out)
        result['complete']=True
    except BaseException as error:result.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    result['elapsed_seconds']=time.monotonic()-started;T.write(out/'partial.json',result);T.write(out/'audit.json',result)
    try:pack_small(out,a.execution_contract,T.bound_path(c['producer_manifest'],c['root']))
    except BaseException as error:
        result.update(complete=False,status='failed_evidence_packaging',archive_error={'type':type(error).__name__,'message':str(error)})
        T.write(out/'audit.json',result);T.write(out/'partial.json',result)
    print(json.dumps({k:result[k] for k in ('complete','role','status','actual_rows','actual_cache_files')}))
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
