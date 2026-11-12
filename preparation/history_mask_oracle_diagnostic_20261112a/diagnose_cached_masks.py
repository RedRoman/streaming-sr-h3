"""Offline GT diagnostic inside a fixed saved candidate set; no model or training."""
import argparse,hashlib,json,os,resource,signal,sys,time
from collections import defaultdict
from pathlib import Path
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_name]='1'
import numpy as np
import array_checks as C
HELPER_SHA='f5eecaf866cd62a1390de8713b2d3d58a5a308816ac69a16e1d5d6b31bae1696'
MASKS=('00','10','01','11')
CANDIDATE_KEYS=('current_only','remove_1','remove_0','all_history')
MARGIN=.05/255
PARAMETERS={'tile':16,'metric':'unclamped RGB MAE float64 of saved float32','four_tie_order':list(MASKS),'five_tie_order':['input',*MASKS],'primary_drop_threshold':0.,'secondary_drop_threshold':-MARGIN,'keep_on_threshold_equality':True,'interaction':'E00+E11-E01-E10; both history slots available only','GT_scope':'offline saved-candidate-set diagnostic only; never a deployable selector or student admission'}
RESOURCES={'CPU':2,'AS_bytes':4<<30,'wall_seconds':180}

def selector(errors,availability,threshold):
    """Bit0=oldest, bit1=recent; simultaneous removal based on all-history LOO."""
    e=np.asarray(errors,np.float64);a=np.asarray(availability,int)
    C.need(e.shape==(4,6,6) and a.shape==(2,) and np.isfinite(e).all() and ((a==0)|(a==1)).all(),'invalid tile errors/availability')
    keep0=(e[2]-e[3]>=threshold)&bool(a[0]);keep1=(e[1]-e[3]>=threshold)&bool(a[1])
    # Canonical order00,10,01,11; unavailable slots never become selectable.
    return keep0.astype(int)+2*keep1.astype(int)

def diagnostic_tiles(errors,input_error,availability):
    e=np.asarray(errors,np.float64);inp=np.asarray(input_error,np.float64);a=np.asarray(availability,int)
    C.need(e.shape==(4,6,6) and inp.shape==(6,6) and np.isfinite(e).all() and np.isfinite(inp).all() and np.all(e>=0) and np.all(inp>=0),'nonfinite/negative tile error')
    C.need(a.shape==(2,) and np.logical_or(a==0,a==1).all(),'invalid availability')
    feasible=np.asarray([all(int(b)<=v for b,v in zip(mask,a)) for mask in MASKS],bool)
    candidates=np.where(feasible[:,None,None],e,np.inf)
    oracle4=np.argmin(candidates,axis=0)  # exact tie order, no tolerance or outcome-based choice.
    oracle5=np.argmin(np.concatenate((inp[None],candidates),axis=0),axis=0)
    zero=selector(e,a,0.);margin=selector(e,a,-MARGIN)
    C.need(feasible[zero].all() and feasible[margin].all(),'LOO selection violates availability')
    chosen=lambda ids:np.take_along_axis(e,ids[None],axis=0)[0]
    e4=chosen(oracle4);e5=np.take_along_axis(np.concatenate((inp[None],e),axis=0),oracle5[None],axis=0)[0]
    values={'input':inp,'K1':e[0],'all':e[3],'loo_zero':chosen(zero),'loo_margin':chosen(margin),'oracle4':e4,'oracle5':e5}
    C.need(np.all(e4<=e[0]) and np.all(e5<=e4),'oracle nesting violated')
    fractions=lambda ids,names:{name:float(np.mean(ids==i)) for i,name in enumerate(names)}
    interaction=e[0]+e[3]-e[2]-e[1] if bool(a.all()) else None
    stats={'mae':{k:float(v.mean()) for k,v in values.items()},
           'gain_vs_K1':{k:float((e[0]-v).mean()) for k,v in values.items() if k not in ('input','K1')},
           'harm_vs_input':{k:float((v-inp).mean()) for k,v in values.items() if k!='input'},
           'regret_vs_oracle4':{k:float((values[k]-e4).mean()) for k in ('all','loo_zero','loo_margin')},
           'oracle5_gain_over_oracle4':float((e4-e5).mean()),
           'selection_fraction':{'loo_zero':fractions(zero,MASKS),'loo_margin':fractions(margin,MASKS),'oracle4':fractions(oracle4,MASKS),'oracle5':fractions(oracle5,('input',*MASKS))},
           'feasible_mask_fraction':{mask:float(feasible[i]) for i,mask in enumerate(MASKS)},
           'loo_zero_differs_from_oracle4_fraction':float(np.mean(zero!=oracle4)),
           'loo_zero_strict_positive_regret_fraction':float(np.mean(values['loo_zero']>e4)),
           'interaction':None if interaction is None else {'signed_mean':float(interaction.mean()),'mean_absolute':float(np.abs(interaction).mean()),'mean_frame_RMS':float(np.sqrt(np.square(interaction).mean())),'positive_fraction':float((interaction>0).mean()),'negative_fraction':float((interaction<0).mean()),'above_margin_fraction':float((interaction>MARGIN).mean()),'below_negative_margin_fraction':float((interaction< -MARGIN).mean())}}
    return stats

def mean_tree(items):
    items=[x for x in items if x is not None]
    if not items:return None
    if isinstance(items[0],dict):
        C.need(all(set(x)==set(items[0]) for x in items),'aggregate schema differs')
        return {k:mean_tree([x[k] for x in items]) for k in items[0]}
    C.need(all(isinstance(x,(float,int)) and np.isfinite(x) for x in items),'invalid scalar aggregation')
    return float(np.mean(items,dtype=np.float64))

def summarize(rows,protocol):
    groups=defaultdict(list)
    for r in rows:groups[r['source_id'],r['profile_id']].append(r['statistics'])
    by_pair=[];lookup={}
    for sid in protocol['dev_source_ids']:
        for pid in protocol['profile_ids']:
            values=groups[sid,pid];C.need(len(values)==96,'source/profile coverage differs')
            stats=mean_tree(values);lookup[sid,pid]=stats
            by_pair.append({'source_id':sid,'profile_id':pid,'frames':96,'tiles':3456,'interaction_frames':94,'interaction_tiles':3384,'statistics':stats})
    degraded=protocol['profile_ids'][:3]
    by_source=[{'source_id':sid,'degraded':mean_tree([lookup[sid,p] for p in degraded]),'identity':lookup[sid,'identity'],'near_clean':lookup[sid,'near_clean']} for sid in protocol['dev_source_ids']]
    profiles=[{'profile_id':pid,'statistics':mean_tree([lookup[sid,pid] for sid in protocol['dev_source_ids']])} for pid in protocol['profile_ids']]
    return {'primary_degraded_equal_source_equal_profile':mean_tree([r['degraded'] for r in by_source]),'identity_equal_source':mean_tree([r['identity'] for r in by_source]),'near_clean_equal_source':mean_tree([r['near_clean'] for r in by_source]),'per_source':by_source,'per_profile_equal_source':profiles,'per_source_profile':by_pair}

def report_view(value,secondary=False):
    """Keep the additional input oracle in a separately labelled control table."""
    if isinstance(value,list):return [report_view(x,secondary) for x in value]
    if not isinstance(value,dict):return value
    if secondary and 'mae' in value:
        return {'input_mae':value['mae']['input'],'oracle4_mae':value['mae']['oracle4'],
                'oracle5_mae':value['mae']['oracle5'],
                'oracle5_gain_over_oracle4':value['oracle5_gain_over_oracle4'],
                'oracle5_harm_vs_input':value['harm_vs_input']['oracle5'],
                'oracle5_selection_fraction':value['selection_fraction']['oracle5']}
    return {k:report_view(v,secondary) for k,v in value.items()
            if secondary or k not in ('oracle5','oracle5_gain_over_oracle4')}

def inspect(path,execute=False,out=None):
    c=C.read(path);C.need(c['schema']=='rphi.history_mask_oracle_diagnostic_execution.v1' and c['parameters']==PARAMETERS and c['resources']==RESOURCES,'fixed diagnostic contract differs')
    C.need(C.sha(__file__)==c['source_pins'].get(str(Path(__file__).resolve())) and C.sha(C.__file__)==HELPER_SHA==c['source_pins'].get(str(Path(C.__file__).resolve())),'actual source/helper identity differs')
    for p,h in c['source_pins'].items():C.need(C.sha(p)==h,'source pin differs')
    report=C.read(C.bound(c['report'],c['root']));audit=C.read(C.bound(c['prior_actual_array_audit'],c['root']));protocol=C.read(C.bound(c['protocol'],c['root']))
    C.need(report['complete'] is True and report['status']=='complete_headroom_fail' and report['assessment']['headroom_rule_passed'] is False,'preserve the completed negative headroom result')
    C.need(audit['complete'] is True and audit['report_sha256']==c['report']['sha256'] and audit['auditor_sha256']==HELPER_SHA and audit['actual_cache_files']==180 and audit['actual_arrays']==1080 and audit['actual_rows']==2880,'not the independently audited cache/report')
    # Independent FP64 tile reductions may differ by rounding; use the already
    # frozen cross-implementation tolerance and keep every Boolean gate exact.
    C.compare(audit['assessment'],report['assessment'],{},'prior_assessment')
    C.need(c['protocol']['sha256']==C.PROTOCOL_SHA==audit['protocol_sha256'] and protocol['rule']==C.RULE,'original headroom rule changed')
    C.need(c['records']==report['records'] and len(report['actual_rgb_caches'])==180,'records/cache binding differs')
    rp=C.bound(c['records'],c['root']);records=[json.loads(x) for x in rp.read_text().splitlines() if x.strip()];base=C.validate_rows(records,protocol)
    C.need(C.vsha(base)==report['evaluation_rows_sha256'],'row provenance differs')
    if execute:
        a=C.read(C.bound(c['admission'],c['root']))
        C.need(a['approved'] is True and a['scope']=='offline_cached_GT_diagnostic_only' and a['execution_payload_sha256']==C.vsha({k:v for k,v in c.items() if k!='admission'}) and a['entry_sha256']==C.sha(__file__) and a['out']==str(Path(out).resolve()),'separate diagnostic admission required')
    return c,report,audit,protocol,records

def run(c,report,prior,protocol,records):
    out=Path(c['report']['path']).resolve().parent;rows=[];actual_bytes=0;checks={};batch_hashes=[]
    for batch,e in enumerate(report['actual_rgb_caches']):
        start=batch*16;selected=records[start:start+16]
        C.need(e['row_indices']==list(range(start,start+16)) and e['row_sha256']==[r['row_sha256'] for r in selected],'cache row bindings differ')
        arrays=C.load_batch(e,out);actual_bytes+=e['bytes'];batch_hashes.append({'path':e['path'],'file_sha256':e['sha256'],'arrays':e['arrays']})
        for i,r in enumerate(selected):
            y=arrays['target'][i];C.need(C.asha(y)==r['target_crop_sha256_float32'],'target differs from actual loader receipt')
            if r['profile_id']=='identity':C.need(np.array_equal(arrays['current'][i],y),'identity input differs from target')
            errors=np.stack([C.tiles(arrays[key][i],y) for key in CANDIDATE_KEYS]);inp=C.tiles(arrays['current'][i],y)
            for slot,available in enumerate(r['availability']):
                if not available:C.need(np.array_equal(arrays['remove_'+str(slot)][i],arrays['all_history'][i]) and np.all(np.asarray(r['raw_utility'])[slot]==0),'unavailable LOO record differs')
            for key,computed in [('all_history_mae',float(errors[3].mean())),('current_only_mae',float(errors[0].mean()))]:C.compare(r[key],computed,checks,key)
            for slot,idx in ((0,2),(1,1)):
                raw=errors[idx]-errors[3] if r['availability'][slot] else np.zeros((6,6),np.float64)
                C.compare(r['raw_utility'][slot],raw.tolist(),checks,'raw_utility')
            rows.append({'source_id':r['source_id'],'profile_id':r['profile_id'],'statistics':diagnostic_tiles(errors,inp,r['availability'])})
        del arrays
    C.need(actual_bytes==report['array_bytes_on_disk']==prior['actual_cache_bytes'] and len(rows)==2880,'actual coverage/bytes differs')
    sums=summarize(rows,protocol)
    # This old result is an immutable cross-check, not a new acceptance rule.
    old_gain=sums['primary_degraded_equal_source_equal_profile']['gain_vs_K1']['all']
    C.compare(old_gain,report['assessment']['source_balanced_degraded_mae_gain'],checks,'original_gain')
    for p,h in c['source_pins'].items():C.need(C.sha(p)==h,'source changed during diagnostic')
    for name in ('report','prior_actual_array_audit','protocol','records'):C.bound(c[name],c['root'])
    C.need(not any(n=='torch' or n.startswith('torch_npu') for n in sys.modules),'framework import prohibited')
    return {'schema':'rphi.history_mask_oracle_diagnostic.v1','complete':True,'status':'offline_cached_candidate_diagnostic_complete','parameters':PARAMETERS,'report':c['report'],'prior_actual_array_audit':c['prior_actual_array_audit'],'protocol':c['protocol'],'actual_cache_files':180,'actual_arrays':1080,'actual_rows':2880,'actual_tiles':103680,'interaction_rows_both_available':2820,'interaction_tiles_both_available':101520,'actual_cache_bytes':actual_bytes,'actual_cache_inventory_sha256':C.vsha(batch_hashes),'maximum_recomputation_abs_error':checks,'source_and_bound_small_files_unchanged':True,'four_mask_primary':{'meaning':'GT minimum inside four saved whole-crop binary-mask predictions; per16tile error selection only','tie_order':list(MASKS),'summaries':report_view(sums)},'five_candidate_secondary_control':{'meaning':'Separate GT minimum of same four saved outputs plus raw current input; not a history gate or accepted method','tie_order':['input',*MASKS],'summaries':report_view(sums,True)},'original_headroom_rule_passed':False,'original_headroom_rule_unchanged':True,'student_training_admitted':False,'model_calls':0,'training_steps':0,'source_NPZ_reads':0,'image_composites_created':0,'limitations':['Post-failure descriptive GT diagnostic on the same selected development set; no independent test evidence.','Oracle and LOO-sign selectors both use target-derived errors and are not deployable raw-only decisions.','Each cached candidate came from a whole-crop binary mask. Tile-wise error selection does not execute a spatially varying input-mask gate; convolution neighborhoods mean its output may differ.','This is an upper bound on improvement only inside the saved candidate set, not arbitrary soft masks, retraining, or full-frame restoration.','Interaction is evaluated only where both history slots exist; boundary frames cannot supply all four physical states.','No model replay, student admission, original-gate change, threshold search, LPIPS or flow-temporal claim.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-contract',required=True);p.add_argument('--out',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    def timeout(*_):raise TimeoutError('cached diagnostic180seconds exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180);start=time.monotonic();out=Path(a.out)
    C.need(not out.exists() and not out.is_symlink(),'fresh result only');out.parent.mkdir(parents=True,exist_ok=True)
    try:
        c,r,prior,p,records=inspect(a.execution_contract,a.execute,out)
        result=run(c,r,prior,p,records) if a.execute else {'schema':'rphi.history_mask_oracle_diagnostic_inspection.v1','complete':True,'status':'bindings_verified_no_RGB_NPZ_read','actual_cache_files':0,'model_calls':0}
    except BaseException as e:result={'complete':False,'status':'failed','error':{'type':type(e).__name__,'message':str(e)},'student_training_admitted':False}
    finally:signal.alarm(0)
    result.update(source_sha256=C.sha(__file__),execution_contract_sha256=C.sha(a.execution_contract),elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,resources=RESOURCES)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('complete','status','error','elapsed_seconds')}));raise SystemExit(0 if result['complete'] else 1)
if __name__=='__main__':main()
