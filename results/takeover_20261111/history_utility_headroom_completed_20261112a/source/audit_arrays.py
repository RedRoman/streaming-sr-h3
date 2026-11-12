"""Independent streaming NumPy audit of saved P1 headroom RGB; no producer import."""
import argparse,hashlib,json,math,os,resource,signal,sys,time,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
PRODUCER_SHA='f20705d0b1851aec7b57264f7afb7b2fbe29cf359b5b4a0172a194b1225f5ed4'
PROTOCOL_SHA='d529c075ba5499d512b3728ff689cc104f1a80e9b985d5bb10bda7eaca690b03'
REDUCER_SHA='6cd463e1f93042e913cdcd8beb7bc8b48d5d88ac323949f97e5da17eee205518'
NAMES=('current','target','all_history','current_only','remove_0','remove_1')
MODES=('input','all_history','current_only','remove_0','remove_1')
ATOL=1e-12
PSNR_ATOL=1e-10
RULE={'neutral_margin':.05/255,'minimum_mean_degraded_mae_gain':.05/255,'minimum_sources_with_positive_degraded_gain':4,'minimum_positive_tile_fraction':.01,'minimum_negative_tile_fraction':.01,'minimum_sources_per_non_neutral_sign':3,'minimum_raw_utility_std':.05/255}
EXTRA={'observed_crop_sha256_float32','target_crop_sha256_float32','availability','all_history_mae','current_only_mae','raw_utility','metrics','all_history_mae_harm_vs_input','current_only_mae_harm_vs_input'}

def need(x,msg):
    if not x:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def asha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def vsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def bound(b,root):
    p=Path(b['path']);p=p if p.is_absolute() else Path(root)/p
    need(sha(p)==b['sha256'],'bound file differs: '+str(p));return p

def compare(actual,expected,errors,path='root'):
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),'dictionary keys differ: '+path)
        for k in expected:compare(actual[k],expected[k],errors,path+'.'+k)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),'list differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,errors,path+'.'+str(i))
    elif isinstance(expected,bool) or expected is None or isinstance(expected,(str,int)):
        need(type(actual)==type(expected) and actual==expected,'exact field differs: '+path)
    elif isinstance(expected,float):
        need(type(actual) in (float,int) and math.isfinite(actual) and math.isfinite(expected),'nonfinite scalar: '+path)
        error=abs(actual-expected);category='psnr' if 'psnr' in path else 'other'
        errors[category]=max(errors.get(category,0.),error)
        need(error<=(PSNR_ATOL if category=='psnr' else ATOL),'numerical mismatch: '+path)
    else:raise TypeError('unsupported comparison '+path)

def metric(pred,target,current):
    a=pred.astype(np.float64);y=target.astype(np.float64);clipped=np.clip(a,0.,1.)
    raw_error=a-y;clip_error=clipped-y;mse=float(np.square(clip_error).mean())
    return {'unclamped_mae':float(np.abs(raw_error).mean()),'unclamped_mse':float(np.square(raw_error).mean()),'clipped_mae':float(np.abs(clip_error).mean()),'clipped_mse':mse,'clipped_psnr':None if mse==0 else float(-10*np.log10(mse)),'clipped_exact_match':mse==0,'clip_fraction':float(np.logical_or(a<0,a>1).mean()),'changed_pixel_fraction':float(np.any(pred!=current,axis=0).mean()),'prediction_float32_sha256':asha(pred)}
def tiles(pred,target):
    # Deliberately distinct iteration from producer reshape reduction; both FP64.
    error=np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean(axis=0)
    return np.asarray([[error[y:y+16,x:x+16].mean() for x in range(0,96,16)] for y in range(0,96,16)],np.float64)
def recompute_row(arrays,i,row,errors):
    y=arrays['target'][i];current=arrays['current'][i]
    need(asha(y)==row['target_crop_sha256_float32'],'target crop differs from actual loader receipt')
    if row['profile_id']=='identity':need(np.array_equal(y,current),'identity input differs from target')
    result=dict(row);modes={m:metric(arrays['current' if m=='input' else m][i],y,current) for m in MODES}
    u=np.zeros((2,6,6),np.float64);base=tiles(arrays['all_history'][i],y)
    for slot,available in enumerate(row['availability']):
        if available:u[slot]=tiles(arrays['remove_'+str(slot)][i],y)-base
        else:
            need(np.array_equal(arrays['remove_'+str(slot)][i],arrays['all_history'][i]),'unavailable LOO cache must exactly reuse all-history')
            need(np.all(np.asarray(row['raw_utility'])[slot]==0),'unavailable raw utility must be exactly zero')
    result.update(metrics=modes,raw_utility=u.tolist(),all_history_mae=modes['all_history']['unclamped_mae'],current_only_mae=modes['current_only']['unclamped_mae'],all_history_mae_harm_vs_input=modes['all_history']['unclamped_mae']-modes['input']['unclamped_mae'],current_only_mae_harm_vs_input=modes['current_only']['unclamped_mae']-modes['input']['unclamped_mae'])
    for k in ('metrics','raw_utility','all_history_mae','current_only_mae','all_history_mae_harm_vs_input','current_only_mae_harm_vs_input'):compare(row[k],result[k],errors,k)
    return result

def phase(profile,t,profiles):
    if profile in ('identity','near_clean'):return profile
    if profile==profiles[0]:return 'static'
    if profile==profiles[2]:return 'abrupt-pre' if t<48 else 'abrupt-post'
    return ('low','ramp-up','high','ramp-down')[t//24]
def validate_rows(records,protocol):
    need(len(records)==2880,'exact2880 row coverage required');base_rows=[]
    expected=[(s,p,t) for s in protocol['dev_source_ids'] for p in protocol['profile_ids'] for t in range(96)]
    for i,(r,key) in enumerate(zip(records,expected)):
        t=key[2];a=[int(t>=2),int(t>=1)]
        need((r['source_id'],r['profile_id'],r['frame_index'])==key and r['row_index']==i and r['split']=='dev','chronology/profile/split differs')
        need(r['crop_tlhw']==[192,368,96,96] and r['history_indices']==[max(t-2,0),max(t-1,0),t] and r['availability']==r['true_availability']==a and r['external_mask']==[1,1] and r['teacher_mask'] is None,'causal/crop/mask differs')
        need(r['severity_phase']==phase(key[1],t,protocol['profile_ids']),'severity phase differs')
        b={k:v for k,v in r.items() if k not in EXTRA}
        need(b['row_sha256']==vsha({k:v for k,v in b.items() if k!='row_sha256'}),'actual row SHA differs')
        base_rows.append(b)
    return base_rows

def summary(records):
    groups=defaultdict(list)
    for r in records:
        for p in (r['severity_phase'],'all_frames'):groups[r['source_id'],r['profile_id'],p].append(r)
    results=[]
    for (sid,pid,phase_name),rs in sorted(groups.items()):
        modes={}
        for mode in MODES:
            values={k:float(np.mean([r['metrics'][mode][k] for r in rs])) for k in ('unclamped_mae','unclamped_mse','clipped_mae','clipped_mse','clip_fraction','changed_pixel_fraction')}
            mse=values['clipped_mse'];values['clipped_psnr_from_mean_mse']=None if mse==0 else float(-10*np.log10(mse));values['clipped_exact_match_frames']=sum(r['metrics'][mode]['clipped_exact_match'] for r in rs);modes[mode]=values
        results.append({'source_id':sid,'profile_id':pid,'severity_phase':phase_name,'frames':len(rs),'metrics':modes,'all_history_mae_harm_vs_input':float(np.mean([r['all_history_mae_harm_vs_input'] for r in rs])),'current_only_mae_harm_vs_input':float(np.mean([r['current_only_mae_harm_vs_input'] for r in rs]))})
    return results

def assessment(records,protocol):
    need(protocol['rule']==RULE,'preregistered rule differs');groups=defaultdict(list)
    for r in records:groups[r['source_id'],r['profile_id']].append(r)
    strata=[];source_rows=[];pooled=[];signs={'positive':[],'negative':[]};margin=RULE['neutral_margin']
    for sid in protocol['dev_source_ids']:
        gains=[];sv=[]
        for pid in protocol['profile_ids']:
            rs=groups[sid,pid];need(len(rs)==96,'partial stratum')
            e=np.asarray([(r['all_history_mae'],r['current_only_mae']) for r in rs],np.float64)
            u=np.concatenate([np.asarray(r['raw_utility'],np.float64)[np.asarray(r['availability'],bool)].reshape(-1) for r in rs])
            gain=float(np.mean(e[:,1]-e[:,0]));sv.append(u)
            if pid not in ('identity','near_clean'):gains.append(gain)
            strata.append({'source_id':sid,'profile_id':pid,'all_history_mae':float(e[:,0].mean()),'current_only_mae':float(e[:,1].mean()),'history_mae_gain':gain,'valid_tiles':int(u.size),'positive_fraction':float((u>margin).mean()),'negative_fraction':float((u< -margin).mean())})
        u=np.concatenate(sv);pooled.append(u)
        for name,test in (('positive',u>margin),('negative',u< -margin)):
            if np.any(test):signs[name].append(sid)
        source_rows.append({'source_id':sid,'degraded_mae_gain':float(np.mean(gains))})
    u=np.concatenate(pooled);g=np.asarray([r['degraded_mae_gain'] for r in source_rows],np.float64)
    mean=float(g.mean());support=int((g>0).sum());positive=float((u>margin).mean());negative=float((u< -margin).mean());std=float(u.std())
    gates={'source_balanced_degraded_gain':mean>RULE['minimum_mean_degraded_mae_gain'],'source_support':support>=RULE['minimum_sources_with_positive_degraded_gain'],'beneficial_tile_support':positive>=RULE['minimum_positive_tile_fraction'],'harmful_tile_support':negative>=RULE['minimum_negative_tile_fraction'],'beneficial_source_support':len(signs['positive'])>=RULE['minimum_sources_per_non_neutral_sign'],'harmful_source_support':len(signs['negative'])>=RULE['minimum_sources_per_non_neutral_sign'],'nondegenerate_raw_utility':std>RULE['minimum_raw_utility_std']}
    return {'schema':'rphi.history_utility_headroom_assessment.v1','coverage_rows':len(records),'valid_tiles':int(u.size),'source_balanced_degraded_mae_gain':mean,'sources_with_positive_degraded_gain':support,'positive_tile_fraction':positive,'negative_tile_fraction':negative,'neutral_tile_fraction':float((np.abs(u)<=margin).mean()),'raw_utility_std':std,'non_neutral_source_ids':signs,'per_source':source_rows,'per_source_profile':strata,'gates':gates,'headroom_rule_passed':all(gates.values()),'student_training_admitted':False,'interpretation':'Operational development headroom only; producer/source/freeze/cache/shuffle review remains required.'}

def load_batch(entry,out):
    p=Path(entry['path']);need(p.is_file() and not p.is_symlink() and p.resolve().parent==out/'arrays','cache outside fresh output')
    need(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'actual NPZ bytes/SHA differs')
    with zipfile.ZipFile(p) as z:
        members=z.infolist();need(len(members)==6 and {m.filename for m in members}=={n+'.npy' for n in NAMES} and all(m.compress_type==zipfile.ZIP_STORED and m.file_size<2<<20 for m in members),'unexpected NPZ member/size/compression')
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(NAMES),'NPZ keys differ');arrays={n:z[n] for n in NAMES}
    for n,a in arrays.items():
        need(a.dtype==np.float32 and a.shape==(16,3,96,96) and a.flags.c_contiguous and np.isfinite(a).all(),'nonfinite/dtype/shape cache differs')
        need(entry['arrays'][n]=={'dtype':'float32','shape':[16,3,96,96],'sha256':asha(a)},'actual array content SHA differs')
    for n in ('current','target'):need(np.logical_and(arrays[n]>=0,arrays[n]<=1).all(),'input/target not unit RGB')
    return arrays

def audit(report_path,report_sha):
    need(sha(report_path)==report_sha,'report file differs');r=read(report_path);out=Path(report_path).resolve().parent
    need(r['schema']=='rphi.history_utility_teacher_headroom_run.v1' and r['complete'] is True and r['status'] in ('complete_headroom_pass','complete_headroom_fail'),'not a complete headroom run')
    c=r['execution_contract'];root=c['evidence_root'];protocol=read(bound(c['headroom_contract'],root))
    need(c['schema']=='rphi.history_utility_teacher_headroom_execution.v1' and c['batch_size']==16 and c['max_crop_forwards']==11520 and c['max_output_bytes']==3<<30,'execution geometry/budget differs')
    need(c['headroom_contract']['sha256']==PROTOCOL_SHA and protocol['implementation']['sha256']==REDUCER_SHA and protocol['rule']==RULE,'protocol binding differs')
    pins=c['implementation_sha256'];by_name={Path(p).name:h for p,h in pins.items()}
    need(by_name.get('eval_history_utility_teacher.py')==PRODUCER_SHA and by_name.get('history_utility_headroom.py')==REDUCER_SHA,'producer/reducer missing from pins')
    for p,h in pins.items():need(sha(Path(p) if Path(p).is_absolute() else Path(root)/p)==h,'implementation pin differs')
    ad=read(bound(c['admission'],root));need(ad['approved'] is True and ad['execution_sha256']==vsha({k:v for k,v in c.items() if k!='admission'}) and ad['entry_sha256']==PRODUCER_SHA and ad['out']==str(out),'execution admission differs')
    teacher=read(bound(c['teacher_report'],root));tc=read(bound(c['teacher_contract'],root));data=read(bound(c['dataset_audit'],root))
    need(teacher['complete'] is True and teacher['completed_step']==1000 and teacher['final_frozen'] is True and teacher['execution_contract']==tc and teacher['execution_contract_file_sha256']==c['teacher_contract']['sha256'],'not frozen final1000 teacher')
    finals=[p for p in teacher['checkpoints'] if p['final']];need(len(finals)==1 and finals[0]['step']==1000 and finals[0]['sha256']==c['teacher_checkpoint']['sha256'],'final teacher identity differs')
    need(data['complete'] is True and data['actual_npz_files_pixel_metadata_verified']==100 and data['pair']==teacher['data']['pair'] and data['pair']['dev_sources']==6 and data['pair']['train_sources']==19 and r['dev_dataset_sha256']==data['datasets']['dev']['dataset_sha256'],'dataset proof differs')
    need(c['dataset_audit']['sha256']==tc['files']['dataset_audit']['sha256']==protocol['data_audit']['sha256']==teacher['data']['actual_data_audit_sha256'],'dataset audit chain differs')
    need(c['runtime']['dtype']=='float32' and c['runtime']['autocast'] is False and r['runtime']['versions']==c['runtime']['expected_versions'],'actual runtime/precision differs')
    need(r['frozen_before_sha256']==r['frozen_after_sha256']==teacher['final_model_sha256']==teacher['final_observed_model_sha256'] and all(r[k] is True for k in ('frozen_flags_preserved','parameter_versions_preserved','state_sha_preserved')),'producer frozen lifecycle differs')
    expected_counts={'crop_forward_attempts':11430,'crop_forwards_completed':11430,'model_invocation_attempts':720,'model_invocations_completed':720,'unavailable_LOO_crops_skipped':90}
    need(r['counts']==expected_counts and r['optimizer_updates']==r['backwards']==0 and r['student_training_admitted'] is False,'actual declared operation budget differs')
    need(r['completed_rows']==r['evaluation_rows']==r['records']['rows']==2880 and len(r['actual_rgb_caches'])==180 and r['output_budget_bytes']==3<<30,'coverage/output budget differs')
    rp=bound(r['records'],root);need(rp.resolve()==out/'records.jsonl','record path outside run');records=[json.loads(s) for s in rp.read_text().splitlines() if s.strip()]
    base_rows=validate_rows(records,protocol);need(vsha(base_rows)==r['evaluation_rows_sha256'] and base_rows[-1]['row_sha256']==r['last_completed_row_sha256'],'base evaluation rows hash differs')
    # Source/target identities refer to the already independently audited dataset.
    ip=bound(tc['files']['dev_receipt_index'],tc['evidence_root']);index=read(ip);sources={}
    need(tc['files']['dev_receipt_index']['sha256']==data['datasets']['dev']['receipt_index']['sha256'] and index['cohort']==tc['cohort'],'audited dev receipt index/cohort differs')
    for binding in index['receipts']:
        rec=read(bound(binding,tc['evidence_root']));sid=rec['source']['source_id'];need(sid not in sources,'duplicate source receipt');sources[sid]=rec
    need(sorted(sources)==protocol['dev_source_ids'] and len(index['receipts'])==6,'dev receipt roles differ')
    for row in base_rows:
        rec=sources[row['source_id']];need(all(row[k]==v for k,v in rec['source'].items()) and row['window_key']==rec['window_key'] and row['target_pixel_sha256']==rec['target']['pixel_sha256'],'row source/target identity differs')
    errors={};actual=[];total=0;seen=set()
    for batch_id,e in enumerate(r['actual_rgb_caches']):
        need(e['path'] not in seen,'duplicate array cache');seen.add(e['path']);start=batch_id*16;rs=records[start:start+16]
        need(e['row_indices']==list(range(start,start+16)) and e['row_sha256']==[x['row_sha256'] for x in rs] and set(e['arrays'])==set(NAMES),'batch row/array inventory differs')
        arrays=load_batch(e,out);total+=e['bytes'];need(total<=3<<30,'array disk budget exceeded')
        actual.extend(recompute_row(arrays,i,row,errors) for i,row in enumerate(rs));del arrays
    need(total==r['array_bytes_on_disk'] and total+rp.stat().st_size+Path(report_path).stat().st_size<=3<<30,'actual saved evidence budget differs')
    descriptive=summary(actual);assessed=assessment(actual,protocol)
    compare(r['descriptive_by_source_profile_phase'],descriptive,errors,'summary');compare(r['assessment'],assessed,errors,'assessment')
    need((r['status']=='complete_headroom_pass')==assessed['headroom_rule_passed'],'reported gate status differs')
    need(not any(n=='torch' or n.startswith('torch_npu') for n in sys.modules),'unexpected framework import')
    return {'schema':'rphi.history_utility_headroom_actual_arrays_audit.v1','complete':True,'status':'actual_arrays_metrics_and_preregistered_rule_verified','report_sha256':report_sha,'producer_source_sha256':PRODUCER_SHA,'protocol_sha256':PROTOCOL_SHA,'teacher_report_sha256':c['teacher_report']['sha256'],'teacher_checkpoint_sha256':c['teacher_checkpoint']['sha256'],'actual_cache_files':180,'actual_arrays':1080,'actual_rows':2880,'actual_cache_bytes':total,'maximum_absolute_difference':errors,'fixed_absolute_tolerances':{'metrics_utility_summary':ATOL,'psnr':PSNR_ATOL,'gate_booleans':'exact; no tolerance applied to thresholds'},'assessment':assessed,'descriptive_by_source_profile_phase':descriptive,'student_training_admitted':False,'model_forwards':0,'backwards':0,'optimizer_updates':0,'source_NPZ_reads':0,'teacher_checkpoint_bytes_rehashed':False,'framework_imported':False,'evidence_scope':'Actual saved FP32 RGB/target crops, array/file/row hashes and all numerical metrics independently verified. Full observed history and live teacher freeze/counts rely on authenticated producer/dataset receipts; no model replay or source NPZ reread. A passed operational headroom rule is not final quality/generalization/omni acceptance.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    need(not out.exists() and not out.is_symlink(),'fresh audit output only');out.parent.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    def timeout(*_):raise TimeoutError('independent headroom array audit600seconds exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600);start=time.monotonic()
    try:r=audit(a.report,a.report_sha256)
    except BaseException as e:r={'schema':'rphi.history_utility_headroom_actual_arrays_audit.v1','complete':False,'report_sha256':a.report_sha256,'error':{'type':type(e).__name__,'message':str(e)},'student_training_admitted':False}
    finally:signal.alarm(0)
    r.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,auditor_sha256=sha(__file__),resources={'CPU':2,'AS_bytes':4<<30,'wall_seconds':600})
    with out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:r.get(k) for k in ('complete','status','error','elapsed_seconds')}));raise SystemExit(0 if r['complete'] else 1)
if __name__=='__main__':main()
