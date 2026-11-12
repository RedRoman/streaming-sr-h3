"""One bounded stdlib-only review of persisted JSON; no NPZ/model/regeneration."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,os,resource,signal,time
from pathlib import Path

AUDIT_SHA='6456d81b20392b6ba8fb16f58324b649c26ad93ad6131bca88c992bdae226abd'

def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def unit(r,slot):return {'row_index':r['row_index'],'slot':slot,'lag':2-slot,'row_sha256':r['row_sha256'],'source_id':r['source_id'],'source_group_id':r['source_group_id']}
def key(r,slot):return json.dumps([r['profile_id'],r['severity_phase'],2-slot,r['external_mask']],separators=(',',':'))

def audit(path,expected):
    need(sha(path)==expected,'manifest file hash differs');m=read(path)
    need(m['schema']=='openh3.history_utility_persistent_schedules.v1' and m['complete'] is True,'manifest not complete')
    for k in ('training_steps','model_calls','NPU_calls','actual_batch_constructions'):need(m[k]==0,'unexpected '+k)
    need(m['training_admitted'] is False and m['legacy_research_ready'] is False and m['revised_pair_ready'] is True,'readiness misrepresented')
    need(m['dataset_audit']['sha256']==AUDIT_SHA==sha(m['dataset_audit']['path']),'prior actual dataset audit differs');prior=read(m['dataset_audit']['path'])
    need(m['pair']==prior['pair'] and m['dataset_sha256']=={s:d['dataset_sha256'] for s,d in prior['datasets'].items()},'dataset/pair mismatch')
    need(m['read_accounting']['fresh_unique_NPZ_files_verified']==100 and m['read_accounting']['additional_train_target_NPZ_reads_for_near_clean']==19 and m['read_accounting']['near_clean_fullframes_generated_and_hashed']==1824,'fresh read accounting differs')
    sources={r['source']['source_id']:r['source'] for r in prior['datasets']['train']['sources']}
    need(set(m['near_clean_control_SHA'])==set(sources),'control hashes missing')
    digests={};student=None
    for role,seed in [('teacher',20261110),('student',20261111)]:
        ref=m[role+'_schedule'];need(sha(ref['path'])==ref['file_sha256'],'schedule file hash differs');s=read(ref['path'])
        need(s['schema']=='openh3.history_utility_revised_schedule.v1' and s['role']==role and s['seed']==seed and s['steps']==1000 and s['batch_size']==16,'schedule header mismatch')
        need(s['schedule_sha256']==vsha({k:v for k,v in s.items() if k!='schedule_sha256'})==ref['schedule_sha256']==prior['schedules'][role]['schedule_sha256'],'schedule semantic hash differs')
        n=s['numeric_schedule'];need(n['schedule_sha256']==vsha({k:v for k,v in n.items() if k!='schedule_sha256'})==ref['numeric_schedule_sha256']==prior['schedules'][role]['numeric_schedule_sha256'],'numeric schedule hash differs')
        need(len(n['rows'])==16000,'row count differs')
        for i,r in enumerate(n['rows']):
            need(r['row_index']==i and r['row_sha256']==vsha({k:v for k,v in r.items() if k!='row_sha256'}),'row digest/index differs')
            need(r['source_id'] in sources and all(r[k]==v for k,v in sources[r['source_id']].items()),'source ID/hash/group/role differs')
            t=r['frame_index'];need(type(t)is int and 0<=t<96,'frame invalid')
            need(r['true_availability']==[int(t>=2),int(t>=1)] and r['history_indices']==[max(t-2,0),max(t-1,0),t],'padded availability differs')
            need(len(r['external_mask'])==2 and all(type(v)is int and v in (0,1) for v in r['external_mask']),'mask invalid')
            if role=='student':need(r['teacher_mask'] is None,'student contains teacher mask')
            if r['profile_id']=='near_clean':need(r['observation_pixel_sha256']==m['near_clean_control_SHA'][r['source_id']],'near clean binding differs')
        digests[role]={'file_sha256':ref['file_sha256'],'schedule_sha256':s['schedule_sha256'],'numeric_schedule_sha256':n['schedule_sha256'],'actual_rows_checked':16000}
        if role=='student':student=n['rows']
        del s,n
    ref=m['donor_manifest'];need(sha(ref['path'])==ref['file_sha256'],'donor file digest differs');d=read(ref['path'])
    need(d['schema']=='openh3.history_utility_whole_map_donors.v1' and d['seed']==20261112,'donor schema/seed differs')
    need(d['donor_map_sha256']==vsha({k:v for k,v in d.items() if k!='donor_map_sha256'})==ref['donor_map_sha256'],'donor semantic hash differs')
    need(d['student_schedule_sha256']==m['student_schedule']['schedule_sha256'],'donors bind wrong rows')
    need(d['stratum_fields']==['profile_id','severity_phase','history_lag','raw_external_mask'],'donor strata differ')
    expected_units={};groups=defaultdict(Counter);bykey=defaultdict(set)
    for r in student:
        for slot in (0,1):
            if r['true_availability'][slot] and r['external_mask'][slot]:
                identity=(r['row_index'],slot);k=key(r,slot);expected_units[identity]=(unit(r,slot),k);bykey[k].add(identity);groups[k][r['source_group_id']]+=1
    impossible={k for k,c in groups.items() if 2*max(c.values())>sum(c.values())}
    declared={json.dumps(x['stratum'],separators=(',',':')):x for x in d['strata']}
    need(set(declared)==set(groups),'stratum census differs')
    for k,c in groups.items():
        s=declared[k];need(s['valid_slots']==sum(c.values()) and s['source_groups']==len(c) and s['maximum_group_slots']==max(c.values()) and s['feasible']==(k not in impossible),'stratum capacity/count differs')
    bad={json.dumps(x['stratum'],separators=(',',':')):x for x in d['impossible_strata']};need(set(bad)==impossible,'incorrect impossibility claim')
    for k,x in bad.items():
        ids={(u['row_index'],u['slot']) for u in x['units']};need(ids==bykey[k] and len(x['units'])==len(ids),'impossible units omitted/duplicated')
        need(all(u==expected_units[(u['row_index'],u['slot'])][0] for u in x['units']),'impossible unit identity differs')
    rec=set();don=set()
    for e in d['entries']:
        a,b=e['recipient'],e['donor'];ri=(a['row_index'],a['slot']);di=(b['row_index'],b['slot'])
        need(ri in expected_units and di in expected_units,'invalid/padded donor support')
        need(a==expected_units[ri][0] and b==expected_units[di][0],'donor row identity differs')
        k=expected_units[ri][1]
        need(k==expected_units[di][1]==json.dumps(e['stratum'],separators=(',',':')) and k not in impossible,'cross-stratum mapping')
        need(a['source_id']!=b['source_id'] and a['source_group_id']!=b['source_group_id'],'same-source/group donor')
        need(ri not in rec and di not in don,'duplicate recipient/donor');rec.add(ri);don.add(di)
    feasible=set().union(*(v for k,v in bykey.items() if k not in impossible)) if any(k not in impossible for k in bykey) else set()
    need(rec==don==feasible,'permutation is not onto all feasible valid supports')
    need(d['valid_slots']==len(expected_units) and d['mapped_slots']==len(rec),'donor counters differ')
    ready=not impossible;need(m['donor_ready']==d['donor_ready']==ready and m['student_training_blocked']==d['student_training_blocked']==(not ready),'donor readiness differs')
    need(d['label_values_read'] is False and d['no_resampling_or_cross_stratum_pooling'] is True,'donor scope differs')
    return {'schema':'openh3.history_utility_persisted_schedule_independent_audit.v1','complete':True,'manifest_sha256':expected,'schedule_files':digests,'donor_file_sha256':ref['file_sha256'],'donor_semantic_sha256':ref['donor_map_sha256'],'actual_existing_rows_checked':32000,'valid_slots':len(expected_units),'mapped_slots':len(rec),'strata':len(bykey),'impossible_strata':len(impossible),'donor_ready':ready,'teacher_schedule_ready':True,'student_training_blocked':not ready,'actual_NPZ_reads':0,'model_calls':0,'schedule_regenerations':0,'label_values_read':False,'source_group_exclusion_and_stratum_bijection_verified':True,'scope':'Persisted JSON bytes/semantic hashes and donor properties only; not label-cache or training acceptance.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);need(not out.exists() and not out.is_symlink(),'fresh output only');out.parent.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    def timeout(*_):raise TimeoutError('independent audit120seconds exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);start=time.monotonic()
    try:r=audit(a.manifest,a.manifest_sha256)
    except BaseException as e:r={'complete':False,'error':{'type':type(e).__name__,'message':str(e)},'manifest_sha256':a.manifest_sha256}
    finally:signal.alarm(0)
    r.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_sha256=sha(__file__),resources={'CPU':1,'wall_seconds':120,'AS_bytes':2<<30})
    with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r));raise SystemExit(0 if r['complete'] else 1)
if __name__=='__main__':main()
