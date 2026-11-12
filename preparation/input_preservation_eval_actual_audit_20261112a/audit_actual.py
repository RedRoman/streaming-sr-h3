"""Independent NumPy audit of saved four-arm evaluation; no model imports/calls.

Reconstructs actual RGB from authenticated raw/K1 and saved threshold-zero masks.
LPIPS is reaggregated from authenticated saved distances, never recomputed.
The scientific screen is retained verbatim, not re-selected or re-admitted.
Wall/RSS limits require the separately authorized external bounded launcher.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

SCHEMA = 'rphi.input_preservation_eval_actual_audit_execution.v1'
PRODUCER_SHA = '25cdeec76d1472fe1a590d27ba0d65553a67ce9fada3d329210874d7bab28e39'
DEV_SHA = 'e509faed1b14840aecbb1a7e059d1e43e0b008dc4cee1bc9a88c2901f007e886'
DEV_AUDIT_SHA = '49290eb86666926f155a9578555e43c71de31c9cf8e7c0019a8208b1433fcff3'
ARMS = ('raw', 'K1', 'matched', 'shuffled')
OLD = {'current', 'target', 'all_history', 'current_only', 'remove_0', 'remove_1'}
EXTRA = {'observed_crop_sha256_float32','target_crop_sha256_float32','availability','all_history_mae',
         'current_only_mae','raw_utility','metrics','all_history_mae_harm_vs_input','current_only_mae_harm_vs_input'}
MEANS = ('clipped_mae','clipped_mse','unclamped_mae','unclamped_mse','lpips','corrected_tile_fraction',
         'changed_pixel_fraction_unclamped','changed_pixel_fraction_clipped','clipped_component_fraction',
         'clipped_pixel_fraction','clip_mean_absolute_change','residual_boundary_jump','residual_nonboundary_jump',
         'output_boundary_jump','correction_boundary_jump')
PAIR_KEYS = ('mask_flip_fraction','unwarped_output_change_mae','unwarped_error_change_mae')
RESOURCES = {'wall_seconds':900,'CPU_cores':2,'RSS_limit_bytes':4<<30}


def need(ok, message):
    if not ok: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def vsha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def asha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fsha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def bound(ref, root):
    p=Path(ref['path']);p=p if p.is_absolute() else Path(root)/p
    need(not p.is_symlink() and fsha(p)==ref['sha256'],'file SHA mismatch: '+str(p))
    if 'bytes' in ref: need(p.stat().st_size==ref['bytes'],'file size mismatch')
    return p.resolve()
def write(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def compare(actual, expected, errors, path='root'):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),'keys differ: '+path)
        for k in expected: compare(actual[k],expected[k],errors,path+'.'+k)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),'list differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)): compare(a,b,errors,path+'.'+str(i))
    elif type(expected) is float:
        need(type(actual) in (float,int) and math.isfinite(actual) and math.isfinite(expected),'nonfinite '+path)
        key='psnr' if 'psnr' in path.lower() else 'metric';d=abs(actual-expected)
        errors[key]=max(errors.get(key,0.),d)
        need(d<=(1e-10 if key=='psnr' else 1e-12),'numeric mismatch: '+path)
    else: need(type(actual) is type(expected) and actual==expected,'exact mismatch: '+path)


def load_npz(e, folder, shapes):
    p=bound(e,folder)
    need(p.parent==Path(folder).resolve()/'arrays' and set(e['arrays'])==set(shapes),'array path/key mismatch')
    with zipfile.ZipFile(p) as z:
        mm=z.infolist()
        need(len(mm)==len(shapes) and {x.filename for x in mm}=={k+'.npy' for k in shapes}
             and all(x.compress_type==zipfile.ZIP_STORED and x.file_size<(2<<20) for x in mm),'NPZ member/size mismatch')
    result={}
    with np.load(p,allow_pickle=False) as z:
        for k,(shape,dtype) in shapes.items():
            a=z[k]
            need(a.shape==shape and a.dtype==np.dtype(dtype) and a.flags.c_contiguous and np.isfinite(a).all(),'array geometry/type/finite '+k)
            need(e['arrays'][k]=={'dtype':str(a.dtype),'shape':list(a.shape),'sha256':asha(a)},'array bytes mismatch '+k)
            result[k]=a
    return result


def clipped(p):
    # Comparison-select preserves signed zero, as the pinned Torch clamp does.
    return np.ascontiguousarray(np.where(p<0,np.float32(0),np.where(p>1,np.float32(1),p)))
def compose(raw,k1,score,mask):
    need(score.dtype==np.float32 and score.shape==(1,6,6) and np.isfinite(score).all(),'invalid score')
    need(mask.dtype==np.uint8 and mask.shape==score.shape and np.array_equal(mask,score>0),'threshold/mask differs')
    selector=mask.repeat(16,axis=1).repeat(16,axis=2).astype(bool)
    return np.ascontiguousarray(np.where(selector,k1,raw))
def jumps(x):
    x=np.asarray(x,np.float64);on=[];off=[]
    for axis in (1,2):
        delta=np.abs(np.diff(x,axis=axis));ii=np.arange(95)
        on.append(np.take(delta,ii[(ii+1)%16==0],axis=axis).ravel())
        off.append(np.take(delta,ii[(ii+1)%16!=0],axis=axis).ravel())
    return float(np.concatenate(on).mean()),float(np.concatenate(off).mean())
def psnr(x): return None if x==0 else float(-10*math.log10(x))
def metrics(pred,raw,target,mask,distance):
    for a in (pred,raw,target):need(a.dtype==np.float32 and a.shape==(3,96,96) and np.isfinite(a).all(),'RGB invalid')
    need(((raw>=0)&(raw<=1)).all() and ((target>=0)&(target<=1)).all(),'raw/target out of range')
    c=clipped(pred);p=pred.astype(np.float64);q=c.astype(np.float64);y=target.astype(np.float64)
    residual=q-y;unclamped=p-y;mse=float(np.square(residual).mean());edge,nonedge=jumps(residual)
    changed=lambda a:float(np.any(np.ascontiguousarray(a).view(np.uint32)!=raw.view(np.uint32),axis=0).mean())
    return {'clipped_mae':float(np.abs(residual).mean()),'clipped_mse':mse,'clipped_psnr':psnr(mse),'clipped_exact_match':mse==0,
      'unclamped_mae':float(np.abs(unclamped).mean()),'unclamped_mse':float(np.square(unclamped).mean()),'lpips':float(distance),
      'corrected_tile_fraction':float(mask.mean()),'changed_pixel_fraction_unclamped':changed(pred),'changed_pixel_fraction_clipped':changed(c),
      'clipped_component_fraction':float(((pred<0)|(pred>1)).mean()),'clipped_pixel_fraction':float(np.any((pred<0)|(pred>1),axis=0).mean()),
      'clip_mean_absolute_change':float(np.abs(p-q).mean()),'residual_boundary_jump':edge,'residual_nonboundary_jump':nonedge,
      'output_boundary_jump':jumps(q)[0],'correction_boundary_jump':jumps(q-raw.astype(np.float64))[0],
      'unclamped_rgb_sha256':asha(pred),'clipped_rgb_sha256':asha(c)}


def summary_check(rows, summary, errors):
    """Verify each published grouping independently; no screen decision code."""
    indexed={(r['source_id'],r['profile_id'],r['frame_index']):r for r in rows}
    need(len(indexed)==2880,'duplicate row identity')
    ids=sorted({r['source_id'] for r in rows});profiles=sorted({r['profile_id'] for r in rows}-{'identity','near_clean'})
    need(len(ids)==6 and len(profiles)==3 and summary['source_ids']==ids and summary['native_profile_ids']==profiles,'source/profile scope differs')
    groups={}
    for sid in ids:
        for p in profiles+['identity','near_clean']:
            rr=[indexed[(sid,p,t)] for t in range(96)]
            for phase in {'all_frames'}|{r['severity_phase'] for r in rr}:
                vv=rr if phase=='all_frames' else [r for r in rr if r['severity_phase']==phase]
                aa={}
                for arm in ARMS:
                    m={k:float(np.mean([r['arms'][arm][k] for r in vv])) for k in MEANS}
                    m.update(psnr_from_mean_mse=psnr(m['clipped_mse']),exact_match=m['clipped_mse']==0);aa[arm]=m
                groups[sid,p,phase]={'source_id':sid,'profile_id':p,'phase':phase,'frames':len(vv),'arms':aa}
    compare([groups[k] for k in sorted(groups)],summary['per_source_profile_phase'],errors,'summary.groups')
    def macro(gg):
        rr={}
        for arm in ARMS:
            m={k:float(np.mean([g['arms'][arm][k] for g in gg])) for k in MEANS}
            pp=[g['arms'][arm]['psnr_from_mean_mse'] for g in gg];finite=[p for p in pp if p is not None]
            m.update(psnr_macro_finite=float(np.mean(finite)) if finite else None,psnr_finite_count=len(finite),psnr_exact_count=len(pp)-len(finite));rr[arm]=m
        return rr
    native=[groups[s,p,'all_frames'] for s in ids for p in profiles]
    compare(macro(native),summary['native_macro'],errors,'summary.native')
    compare([{'source_id':s,'arms':macro([groups[s,p,'all_frames'] for p in profiles])} for s in ids],summary['native_per_source'],errors,'summary.sources')
    compare({p:macro([groups[s,p,'all_frames'] for s in ids]) for p in profiles},summary['native_per_profile'],errors,'summary.profiles')
    for p in ('identity','near_clean'):
        gg=[groups[s,p,'all_frames'] for s in ids]
        compare({'arms':macro(gg),'per_source':gg},summary['controls'][p],errors,'summary.'+p)


def temporal_check(pairs, temporal, errors):
    groups=defaultdict(list)
    for r in pairs:groups[r['source_id'],r['profile_id']].append(r)
    need(len(groups)==30 and len(pairs)==2850,'temporal group/count mismatch')
    details=[]
    for (sid,profile),rows in sorted(groups.items()):
        need([(r['from_frame'],r['to_frame']) for r in rows]==[(t-1,t) for t in range(1,96)],'95 stride1 pairs required')
        for key in ('all','declared_transition','same_phase'):
            rr=rows if key=='all' else [r for r in rows if r['declared_phase_transition']==(key=='declared_transition')]
            details.append({'source_id':sid,'profile_id':profile,'support':key,'pairs':len(rr),
                'arms':{a:{k:float(np.mean([r['arms'][a][k] for r in rr])) if rr else None for k in PAIR_KEYS} for a in ARMS}})
    compare(details,temporal['per_source_profile'],errors,'temporal.groups')
    need(temporal['pairs']==2850 and temporal['declared_phase_transition_pairs']==sum(r['declared_phase_transition'] for r in pairs),'temporal totals differ')


def inspect_contract(path):
    c=read(path);root=c['root']
    need(c['schema']==SCHEMA and Path(root).is_absolute() and c['resources']==RESOURCES,'audit contract/budget differs')
    need(c['auditor_sha256']==fsha(__file__),'auditor SHA differs')
    rp=bound(c['producer_report'],root);ep=bound(c['producer_execution'],root);r=read(rp);e=read(ep)
    need(r['schema']=='rphi.input_preservation_heads_evaluation.v1' and r['complete'] is True
         and r['status']=='complete_fixed_development_screen' and r['completed_rows']==2880,'evaluation incomplete')
    need(r['execution_contract']==e and r['execution_contract_file_sha256']==fsha(ep),'execution/report identity differs')
    src=[p for p,h in e['implementation_sha256'].items() if Path(p).name=='eval_input_preservation_heads.py' and h==PRODUCER_SHA]
    need(len(src)==1 and fsha(src[0])==PRODUCER_SHA,'frozen producer source mismatch')
    need(e['threshold']==0 and e['batch_size']==16,'fixed threshold/batch differs')
    need(e['files']['dev_cache_manifest']['sha256']==DEV_SHA and e['files']['dev_cache_audit']['sha256']==DEV_AUDIT_SHA,'accepted dev cache differs')
    mp=bound(e['files']['dev_cache_manifest'],root);m=read(mp);a=read(bound(e['files']['dev_cache_audit'],root))
    need(m['complete'] is True and m['role']=='dev' and a['complete'] is True and a['role']=='dev'
         and a['actual_rows']==2880 and a['producer_manifest']==e['files']['dev_cache_manifest'],'dev acceptance differs')
    h=read(bound(m['execution_contract']['files']['headroom_report'],root))
    need(h['complete'] is True,'headroom parent incomplete')
    need(r['observed_model_invocations']=={'matched':180,'shuffled':180,'LPIPS':720},'observed model count differs')
    expected={'head_forward_attempts':360,'head_row_attempts':5760,'head_forwards_completed':360,'head_rows_completed':5760,
        'LPIPS_forward_attempts':720,'LPIPS_row_attempts':11520,'LPIPS_forwards_completed':720,'LPIPS_rows_completed':11520,
        'restorer_forwards':0,'backwards':0,'optimizer_updates':0}
    need(r['counts']==expected and r['LPIPS_frozen_exact'] is True,'operation/frozen metric attestation differs')
    need(set(r['frozen_checks'])=={'matched','shuffled'} and all(v=={'state_dict_exact':True,'versions_exact':True,'eval_frozen_no_grads':True} for v in r['frozen_checks'].values()),'head frozen attestation differs')
    return c,rp,r,m,h


def audit(c,rp,r,m,h,result,out):
    root=c['root'];errors=result['maximum_absolute_differences']
    records=lines(bound(r['records'],root));pairs=lines(bound(r['pairs'],root));entries=lines(bound(r['batches'],root))
    cached=lines(bound(m['records'],root));original=lines(bound(m['execution_contract']['files']['headroom_records'],root))
    need(len(records)==len(cached)==len(original)==2880 and len(pairs)==2850 and len(entries)==180,'row/batch coverage differs')
    base=[{k:v for k,v in rr.items() if k not in EXTRA} for rr in original]
    need(vsha(base)==h['evaluation_rows_sha256'] and len(h['actual_rgb_caches'])==180,'original evaluation row identity differs')
    oldroot=Path(h['records']['path']).parent;oldshape={k:((16,3,96,96),'float32') for k in OLD}
    shapes={k:((16,1,6,6),'float32' if k.endswith('score') else 'uint8') for k in ('matched_score','matched_mask','shuffled_score','shuffled_mask')}
    shapes['lpips']=((16,4),'float64');previous=None;pair_index=0;computed=[];actualpairs=[]
    for j in range(180):
        ee=entries[j];oldentry=h['actual_rgb_caches'][j];rr=records[j*16:(j+1)*16]
        need(ee['batch_index']==j and ee['row_sha256']==[v['row_sha256'] for v in rr],'evaluation batch row identity differs')
        aa=load_npz(ee,rp.parent,shapes);old=load_npz(oldentry,oldroot,oldshape)
        need(oldentry['row_indices']==list(range(j*16,(j+1)*16)) and oldentry['row_sha256']==[v['row_sha256'] for v in base[j*16:(j+1)*16]],'original batch rows differ')
        for i,saved in enumerate(rr):
            idx=j*16+i;cr=cached[idx];prior=original[idx]
            need(cr['cache_record_sha256']==vsha({k:v for k,v in cr.items() if k!='cache_record_sha256'}) and cr['cache_role']=='dev'
                 and cr['usable_for_training'] is False,'dev cached row identity differs')
            for key in ('row_index','row_sha256','source_id','source_sha256','source_group_id','profile_id','severity_phase','window_key','crop_tlhw','frame_index','observed_crop_sha256_float32','target_crop_sha256_float32'):
                need(saved[key]==cr[key]==prior[key],'row identity differs: '+key)
            need(saved['row_index']==idx and saved['batch_row']==i and saved['actual_array_file']==ee['sha256']
                 and saved['cache_record_sha256']==cr['cache_record_sha256'],'saved row/file binding differs')
            need(saved['input_RGB_reference']==cr['input_RGB_reference']=={'cache':oldentry,'batch_row':i},'raw/K1 reference differs')
            need(saved['array_row_sha256']=={k:asha(v[i]) for k,v in aa.items()},'saved array row SHA differs')
            raw,k1,target=old['current'][i],old['current_only'][i],old['target'][i]
            need(asha(target)==saved['target_crop_sha256_float32'] and asha(k1)==prior['metrics']['current_only']['prediction_float32_sha256'],'target/K1 identity differs')
            # Cached raw/K1/target bytes and record chain were independently accepted;
            # no new full K3 feature/model evaluation is part of this audit.
            pp={'raw':raw,'K1':k1};mm={'raw':np.zeros((1,6,6),np.uint8),'K1':np.ones((1,6,6),np.uint8)}
            for arm in ('matched','shuffled'):
                mm[arm]=aa[arm+'_mask'][i];pp[arm]=compose(raw,k1,aa[arm+'_score'][i],mm[arm])
            row={**saved,'arms':{a:metrics(pp[a],raw,target,mm[a],aa['lpips'][i,n]) for n,a in enumerate(ARMS)}}
            compare(row,saved,errors,'row.'+str(idx));computed.append(row)
            curr={**row,'pixels':{a:clipped(pp[a]) for a in ARMS},'masks':mm,'target':target}
            if row['frame_index']:
                need(previous is not None and previous['frame_index']+1==row['frame_index'] and
                     all(previous[k]==row[k] for k in ('source_id','profile_id','window_key','crop_tlhw')),'noncausal/mixed pair')
                q={'source_id':row['source_id'],'profile_id':row['profile_id'],'from_frame':previous['frame_index'],'to_frame':row['frame_index'],
                   'from_phase':previous['severity_phase'],'to_phase':row['severity_phase'],'declared_phase_transition':previous['severity_phase']!=row['severity_phase'],'arms':{}}
                for arm in ARMS:
                    delta=curr['pixels'][arm].astype(np.float64)-previous['pixels'][arm].astype(np.float64)
                    # Evaluate errors separately before subtraction to match the declared support/math.
                    nowerr=curr['pixels'][arm].astype(np.float64)-target.astype(np.float64)
                    olderr=previous['pixels'][arm].astype(np.float64)-previous['target'].astype(np.float64)
                    q['arms'][arm]={'mask_flip_fraction':float((mm[arm]!=previous['masks'][arm]).mean()),'unwarped_output_change_mae':float(np.abs(delta).mean()),'unwarped_error_change_mae':float(np.abs(nowerr-olderr).mean())}
                compare(q,pairs[pair_index],errors,'pair.'+str(pair_index));actualpairs.append(q);pair_index+=1
            previous=curr
        result['actual_batches']+=1;result['actual_rows']+=16;result['actual_saved_arrays']+=5;result['actual_original_RGB_arrays']+=6
        write(Path(out)/'progress.json',{k:result[k] for k in ('schema','complete','actual_batches','actual_rows','maximum_absolute_differences')})
    need(pair_index==2850,'pair total differs')
    summary_check(computed,r['summary'],errors);temporal_check(actualpairs,r['temporal'],errors)
    # Persist only small independently checked summaries, not duplicate images/arrays.
    result.update(actual_pairs=pair_index,actual_RGB_hashes_verified=2880*4*2,
        summary=r['summary'],temporal=r['temporal'],producer_scientific_screen=r['screen'],
        scientific_screen_rejudged=False,LPIPS_scope='Saved 11520 distances authenticated and independently reaggregated; no LPIPS forward or weight validation rerun.',
        model_predictions_recomputed=False,model_or_optimizer_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');args=p.parse_args()
    c,rp,r,m,h=inspect_contract(args.execution_contract);out=args.out.resolve()
    for parent in (rp.parent,Path(m['records']['path']).parent,Path(h['records']['path']).parent):
        need(out!=parent and parent not in out.parents,'audit output cannot be inside an original artifact')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'schema':'rphi.input_preservation_eval_actual_audit.v1','complete':False,'status':'started','producer_report':c['producer_report'],
      'execution_contract_sha256':fsha(args.execution_contract),'auditor_sha256':fsha(__file__),'actual_batches':0,'actual_rows':0,
      'actual_saved_arrays':0,'actual_original_RGB_arrays':0,'maximum_absolute_differences':{},'training_or_quality_admitted':False,
      'scope':'Independent NumPy saved-array/RGB/metrics audit only; frozen-model claims remain producer attestations.',
      'runtime':{'python':sys.version.split()[0],'numpy':np.__version__},'resources':RESOURCES}
    try:
        need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'model backend unexpectedly imported')
        if args.execute:
            audit(c,rp,r,m,h,result,out);inspect_contract(args.execution_contract)
            result['status']='actual_saved_arrays_RGB_metrics_and_aggregation_verified'
        else:result['status']='metadata_inspected_no_arrays_or_models'
        result['complete']=True
    except BaseException as error: result.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    result['elapsed_seconds']=time.monotonic()-started;write(out/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('complete','status','actual_batches','actual_rows','actual_saved_arrays')}))
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
