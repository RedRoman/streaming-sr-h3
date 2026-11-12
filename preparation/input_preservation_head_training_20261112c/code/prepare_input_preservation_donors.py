"""Prepare one metadata-only cross-source/group donor per frozen student row.

The old slot-wise map is not used. Default inspection reads only the fixed
proposal and persistent manifest; --execute additionally reads the existing
student JSON, never labels, GT/pixels, NPZ, models, or a regenerated schedule.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,json,os,resource,signal,sys,time
from pathlib import Path

PROPOSAL_SHA='d1dd2afd1b9daddea8340fe1edff7c71ed6215fbf6f3ef038adaa4b01aa262de'
MANIFEST_SHA='5c001aaa3335e335a16311f9d42e400f29274d65049f967859b7969c9d75ee0b'
SCHEDULE_FILE_SHA='b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'
SCHEDULE_SHA='adda5689d88417b4141ca8f632a7b03515a829b18266057312569dea3298a100'
NUMERIC_SHA='17c7ca57110c4006b2a0f41986dac58daa9945bff7e113265bb8b66ffb3be5cc'
COHORT_SHA='c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
DATASET_SHA='3e075085a2e846e10bb9edf5775b2df00fa32fad3f04d4135e49713b2d6f3e87'
SEED=20261112
ROWS=16000
SCHEDULE_BYTES=14924552
PROFILES=('native_static_blur_noise_h264','native_smooth_blur_noise_h264','native_abrupt_blur_noise_h264')
MAP_SCHEMA='rphi.input_preservation_whole_benefit_map_donors.v1'
EXECUTION_SCHEMA='rphi.input_preservation_donor_execution.v1'
RESOURCES={'CPU_cores':1,'address_space_bytes':2<<30,'wall_seconds':120,'maximum_output_bytes':32<<20}
FORBIDDEN={'torch','torch_npu','numpy','h3_comfy','cv2','av'}

def need(x,msg):
    if not x:raise ValueError(msg)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def value_sha(v):return hashlib.sha256(canonical(v).encode('utf-8')).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n')
    return {'path':str(Path(path).resolve()),'file_sha256':file_sha(path),'bytes':Path(path).stat().st_size}
def bound(b,root):
    need(set(b)=={'path','sha256'},'exact file binding required');p=Path(b['path']);p=p if p.is_absolute() else Path(root)/p
    need(file_sha(p)==b['sha256'],'file SHA differs: '+str(p));return p.resolve()
def payload_sha(c):return value_sha({k:v for k,v in c.items() if k!='admission'})
def phase(profile,t):
    if profile in ('identity','near_clean'):return profile
    need(profile in PROFILES,'unknown profile')
    if profile==PROFILES[0]:return 'static'
    if profile==PROFILES[2]:return 'abrupt-pre' if t<48 else 'abrupt-post'
    return ('low','ramp-up','high','ramp-down')[t//24]

def units_from_rows(rows,expected_rows=ROWS):
    need(len(rows)==expected_rows and expected_rows>0,'full declared row coverage required')
    groups={};hashes={};units=[]
    for i,r in enumerate(rows):
        need(r['row_index']==i and r['row_sha256']==value_sha({k:v for k,v in r.items() if k!='row_sha256'}),'row index/content SHA differs')
        sid,gid=r['source_id'],r['source_group_id']
        need(r['split']=='train' and isinstance(sid,str) and sid and isinstance(gid,str) and gid,'train source/group required')
        need(sid not in groups or groups[sid]==gid,'source assigned multiple groups');groups[sid]=gid
        source_sha=r['source_sha256'];need(isinstance(source_sha,str) and len(source_sha)==64 and all(c in '0123456789abcdef' for c in source_sha),'source SHA invalid')
        need(sid not in hashes or hashes[sid]==source_sha,'source bytes identity differs');hashes[sid]=source_sha
        t=r['frame_index'];need(type(t) is int and 0<=t<96,'frame outside frozen window')
        true=[int(t>=2),int(t>=1)];ext=r['external_mask']
        need(r['true_availability']==true and r['history_indices']==[max(t-2,0),max(t-1,0),t],'causal boundary differs')
        need(type(ext) is list and len(ext)==2 and all(type(x) is int and x in (0,1) for x in ext),'external availability invalid')
        need(r['teacher_mask'] is None and r['severity_phase']==phase(r['profile_id'],t),'not original student metadata')
        effective=[true[j]*ext[j] for j in (0,1)]
        units.append({'row_index':i,'row_sha256':r['row_sha256'],'source_id':sid,'source_group_id':gid,
                      'stratum':[r['profile_id'],r['severity_phase'],effective]})
    return units

def summarize_strata(units):
    strata=defaultdict(list)
    for u in units:strata[canonical(u['stratum'])].append(u)
    summaries=[]
    for key,items in sorted(strata.items()):
        counts=Counter(u['source_group_id'] for u in items);maximum=max(counts.values())
        summaries.append({'stratum':json.loads(key),'rows':len(items),'source_groups':len(counts),'maximum_group_rows':maximum,
                          'group_row_counts':dict(sorted(counts.items())),'feasible':2*maximum<=len(items)})
    return strata,summaries

def build_donor_map(rows,seed=SEED,expected_rows=ROWS):
    """Seeded contiguous group rotation, one whole map for every row incl00."""
    need(seed==SEED,'fixed seed must not be changed');units=units_from_rows(rows,expected_rows)
    strata,summaries=summarize_strata(units);impossible=[]
    for summary in summaries:
        if not summary['feasible']:
            key=canonical(summary['stratum']);impossible.append({**summary,'reason':'largest source-group count exceeds all other groups; no source/group-disjoint bijection',
                'recipient_row_indices':[u['row_index'] for u in strata[key]]})
    entries=[]
    # No partial usable mapping if any stratum is infeasible.
    if not impossible:
        for key,items in sorted(strata.items()):
            groups=defaultdict(list)
            for u in items:groups[u['source_group_id']].append(u)
            maximum=max(map(len,groups.values()));ordered=[]
            for group in sorted(groups,key=lambda g:(value_sha([seed,key,'group',g]),g)):
                ordered.extend(sorted(groups[group],key=lambda u:(value_sha([seed,key,'unit',u['row_index']]),u['row_index'])))
            for i,recipient in enumerate(ordered):entries.append({'recipient':recipient,'donor':ordered[(i+maximum)%len(ordered)]})
        entries.sort(key=lambda e:e['recipient']['row_index'])
    result={'schema':MAP_SCHEMA,'seed':seed,'unit':'one entire1x6x6 benefit map per row; no tile permutation',
        'algorithm':'SHA256[seed,canonical-stratum,group,group_id] order; SHA256[seed,canonical-stratum,unit,row_index] within group; cyclic offset largest group count',
        'stratum_fields':['profile_id','severity_phase','effective_availability[2]'],
        'effective_availability_rule':'true_availability * external_mask; all16000rows including[0,0]',
        'total_rows':len(rows),'mapped_rows':len(entries),'includes_both_absent_rows':True,
        'both_absent_rows':sum(u['stratum'][2]==[0,0] for u in units),'strata':summaries,'impossible_strata':impossible,
        'donor_ready':not impossible,'both_head_training_arms_blocked':bool(impossible),
        'label_values_read':False,'GT_values_read':False,'model_calls':0,'no_resampling_or_coarsening':True,'entries':entries}
    result['donor_map_sha256']=value_sha(result);validate_donor_map(rows,result,expected_rows)
    return result

def validate_donor_map(rows,mapping,expected_rows=ROWS):
    """Validate invariants independently of ordering/rotation construction."""
    units=units_from_rows(rows,expected_rows);_,summaries=summarize_strata(units)
    need(mapping['schema']==MAP_SCHEMA and mapping['seed']==SEED and mapping['total_rows']==expected_rows,'donor schema/seed/coverage differs')
    need(mapping['donor_map_sha256']==value_sha({k:v for k,v in mapping.items() if k!='donor_map_sha256'}),'donor semantic SHA differs')
    need(mapping['strata']==summaries,'stratum counts/capacity differ')
    bad=[s for s in summaries if not s['feasible']]
    need(mapping['both_absent_rows']==sum(u['stratum'][2]==[0,0] for u in units) and mapping['includes_both_absent_rows'] is True,'both-absent rows were discarded')
    if bad:
        expected_bad=[]
        for s in bad:expected_bad.append({**s,'reason':'largest source-group count exceeds all other groups; no source/group-disjoint bijection',
            'recipient_row_indices':[u['row_index'] for u in units if u['stratum']==s['stratum']]})
        need(mapping['impossible_strata']==expected_bad and mapping['donor_ready'] is False and mapping['both_head_training_arms_blocked'] is True and mapping['mapped_rows']==0 and mapping['entries']==[],'infeasible stratum must block the whole mapping')
        return {'valid_infeasibility_evidence':True,'bijection_verified':False,'rows_accounted':expected_rows,'mapped_rows':0,'impossible_strata':len(bad)}
    need(mapping['impossible_strata']==[] and mapping['donor_ready'] is True and mapping['both_head_training_arms_blocked'] is False and mapping['mapped_rows']==expected_rows and len(mapping['entries'])==expected_rows,'complete bijection required')
    recipients=[];donors=[]
    for e in mapping['entries']:
        ri=e['recipient']['row_index'];di=e['donor']['row_index']
        need(type(ri) is int and type(di) is int and 0<=ri<expected_rows and 0<=di<expected_rows,'donor row outside frozen schedule')
        r,d=units[ri],units[di]
        need(e['recipient']==r and e['donor']==d,'donor metadata differs from authenticated rows')
        need(ri!=di and r['source_id']!=d['source_id'] and r['source_group_id']!=d['source_group_id'],'self/same-source/same-group donor')
        need(r['stratum']==d['stratum'],'cross-stratum donor')
        recipients.append(ri);donors.append(di)
    need(recipients==list(range(expected_rows)) and sorted(donors)==list(range(expected_rows)),'not an ordered full bijection; donor reuse/missing row')
    return {'bijection_verified':True,'rows_accounted':expected_rows,'mapped_rows':expected_rows,'source_and_group_exclusion_verified':True,'exact_stratum_verified':True,'impossible_strata':0}

def validate_schedule(s,proposal):
    fixed=proposal['fixed_train_rows'];expected=fixed['schedule'];n=s['numeric_schedule']
    need(s['schema']=='openh3.history_utility_revised_schedule.v1' and s['role']=='student' and s['seed']==20261111 and s['steps']==1000 and s['batch_size']==16,'not frozen student schedule')
    need(s['schedule_sha256']==expected['schedule_sha256']==value_sha({k:v for k,v in s.items() if k!='schedule_sha256'}),'outer schedule semantic SHA differs')
    need(n['schema']=='openh3.history_utility_schedule.v1' and n['role']=='student' and n['seed']==20261111 and n['steps']==1000 and n['batch_size']==16 and n['schedule_sha256']==expected['numeric_schedule_sha256']==value_sha({k:v for k,v in n.items() if k!='schedule_sha256'}),'numeric schedule identity differs')
    need(s['dataset_sha256']==fixed['dataset_sha256'] and s['cohort_sha256']==fixed['cohort_sha256'] and s['legacy_research_ready'] is False and s['revised_cohort_ready'] is True and s['training_admitted'] is False,'dataset/readiness differs')
    rows=n['rows'];units_from_rows(rows,ROWS);need(len({r['source_id'] for r in rows})==19,'exact19train sources required')
    expected_batch={**{p:4 for p in PROFILES},'identity':2,'near_clean':2}
    for start in range(0,ROWS,16):need(Counter(r['profile_id'] for r in rows[start:start+16])==expected_batch,'frozen12+2+2 batch composition differs')
    return rows

def inspect_contract(c):
    need(c['schema']==EXECUTION_SCHEMA and c['resources']==RESOURCES and set(c['files'])=={'proposal','persistent_schedule_manifest','student_schedule'},'execution schema/resources differ')
    need(c['files']['proposal']['sha256']==PROPOSAL_SHA and c['files']['persistent_schedule_manifest']['sha256']==MANIFEST_SHA and c['files']['student_schedule']['sha256']==SCHEDULE_FILE_SHA,'frozen evidence differs')
    proposal=read(bound(c['files']['proposal'],c['root']));manifest=read(bound(c['files']['persistent_schedule_manifest'],c['root']))
    need(proposal['schema']=='rphi.input_preservation_interface_proposal.v1' and proposal['original_history_headroom_remains_failed'] is True and proposal['student_training_admitted'] is False,'proposal does not preserve prior failure/admission boundary')
    expected=proposal['fixed_train_rows'];schedule=expected['schedule']
    need(schedule['rows']==ROWS and schedule['bytes']==SCHEDULE_BYTES and schedule['file_sha256']==SCHEDULE_FILE_SHA and schedule['schedule_sha256']==SCHEDULE_SHA and schedule['numeric_schedule_sha256']==NUMERIC_SHA and expected['cohort_sha256']==COHORT_SHA and expected['dataset_sha256']==DATASET_SHA,'fixed row contract differs')
    need(manifest['schema']=='openh3.history_utility_persistent_schedules.v1' and manifest['complete'] is True and manifest['student_schedule']==schedule and manifest['cohort_sha256']==COHORT_SHA and manifest['dataset_sha256']['train']==DATASET_SHA,'persistent manifest differs from proposal')
    pins=c['implementation_sha256'];need(pins.get(str(Path(__file__).resolve()))==file_sha(__file__) and all(file_sha(p)==h for p,h in pins.items()),'source pin differs')
    need(not FORBIDDEN.intersection(x.split('.')[0] for x in sys.modules),'forbidden model/pixel framework imported')
    return proposal,manifest

def execute(c,proposal,manifest,out):
    out=Path(out);need(not out.exists() and not out.is_symlink(),'fresh output directory only')
    a=read(bound(c['admission'],c['root']))
    need(a['approved'] is True and a['scope']=='input_preservation_donor_metadata_only' and a['execution_payload_sha256']==payload_sha(c) and a['entry_sha256']==file_sha(__file__) and a['out']==str(out.resolve()),'separate metadata admission required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();report={'schema':'rphi.input_preservation_donor_manifest.v1','complete':False,'donor_ready':False,'both_head_training_arms_blocked':True,'student_training_admitted':False,'original_history_headroom_remains_failed':True,'model_calls':0,'label_values_read':False,'GT_values_read':False,'source_NPZ_reads':0,'schedule_regenerated':False,'input_files':c['files']}
    try:
        p=Path(c['files']['student_schedule']['path']);p=p if p.is_absolute() else Path(c['root'])/p
        need(p.stat().st_size==SCHEDULE_BYTES,'actual student schedule byte size differs')
        raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==SCHEDULE_FILE_SHA,'actual student schedule file SHA differs');s=json.loads(raw);del raw
        rows=validate_schedule(s,proposal);mapping=build_donor_map(rows)
        mapping['student_schedule_sha256']=SCHEDULE_SHA;mapping['student_schedule_file_sha256']=SCHEDULE_FILE_SHA
        mapping['donor_map_sha256']=value_sha({k:v for k,v in mapping.items() if k!='donor_map_sha256'})
        validation=validate_donor_map(rows,mapping)
        artifact=write_new(out/'input_preservation_donor_map.json',mapping)
        need(artifact['bytes']<RESOURCES['maximum_output_bytes']-(1<<20),'output budget exceeded')
        inspect_contract(c);need(file_sha(p)==SCHEDULE_FILE_SHA,'student schedule changed during metadata execution')
        report.update(complete=mapping['donor_ready'],donor_ready=mapping['donor_ready'],both_head_training_arms_blocked=not mapping['donor_ready'],status='metadata_bijection_ready_no_training_admission' if mapping['donor_ready'] else 'infeasible_strata_no_global_mapping',
            rows=ROWS,train_sources=19,cohort_sha256=COHORT_SHA,dataset_sha256=DATASET_SHA,student_schedule_sha256=SCHEDULE_SHA,numeric_schedule_sha256=NUMERIC_SHA,
            donor_map={**artifact,'donor_map_sha256':mapping['donor_map_sha256']},independent_invariant_validation=validation,
            strata=len(mapping['strata']),impossible_strata=mapping['impossible_strata'],both_absent_rows=mapping['both_absent_rows'])
    except BaseException as e:report.update(status='failed_preserve_evidence_no_retry',error={'type':type(e).__name__,'message':str(e)})
    report.update(source_sha256=file_sha(__file__),execution_payload_sha256=payload_sha(c),elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,resources=RESOURCES)
    write_new(out/'manifest.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--out',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c=read(a.execution_contract);proposal,manifest=inspect_contract(c)
    if not a.execute:
        out=Path(a.out);need(not out.exists() and not out.is_symlink(),'fresh inspection directory only');out.mkdir(parents=True)
        r={'complete':True,'status':'small_metadata_bindings_verified_student_JSON_not_read','student_schedule_bytes_read':0,'model_calls':0,'label_values_read':False,'execution_payload_sha256':payload_sha(c)};write_new(out/'inspection.json',r);print(canonical(r));return
    need(sys.platform.startswith('linux'),'actual execution requires Linux resource limits');os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    def timeout(*_):raise TimeoutError('donor metadata120seconds budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:r=execute(c,proposal,manifest,a.out)
    finally:signal.alarm(0)
    print(canonical({k:r.get(k) for k in ('complete','status','rows','strata','donor_ready','error')}));raise SystemExit(0 if r['complete'] else 1)
if __name__=='__main__':main()
