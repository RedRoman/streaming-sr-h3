"""Persist accepted P1 schedules and a metadata-only constrained donor permutation.

Default inspection hashes small evidence only. --execute requires a separate root
admission and performs a fresh, real 100-NPZ verification; no model is imported.
Donor labels do not exist here: the map addresses future whole6x6 utility maps.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

SCHEMA='openh3.history_utility_schedule_preparation_execution.v1'
MANIFEST_SCHEMA='openh3.history_utility_persistent_schedules.v1'
DONOR_SCHEMA='openh3.history_utility_whole_map_donors.v1'
COHORT_SHA='c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
AUDIT_SHA='6456d81b20392b6ba8fb16f58324b649c26ad93ad6131bca88c992bdae226abd'
ROOT_ACCEPTANCE_SHA='43edaf6d84b80e271b8d078b195c7d25dc47aa49fefa6333075b59f3b42238c8'
LOADER_SHA='87ad6a83965869846fdeb4dc3abcea800ed0e16ce1819074d6753c8ff481a1d7'
WRAPPER_SHA='3f48214e3df626f988a92e78349a8929dea01414b831566fda0dd793ac1937d1'
SEEDS={'teacher':20261110,'student':20261111}
DONOR_SEED=20261112
RESOURCES={'CPU_cores':2,'threads':1,'wall_seconds':900,'RSS_limit_bytes':32<<30,
           'address_space_limit_bytes':32<<30,'schedule_cache_bytes':12<<30,'maximum_output_bytes':256<<20}
FILES={'dataset_audit','root_acceptance','materialization_complete','producer_contract','train_receipt_index',
       'dev_receipt_index','cohort','plan','train_manifest','dev_manifest','source_contract'}
PROFILES=('native_static_blur_noise_h264','native_smooth_blur_noise_h264','native_abrupt_blur_noise_h264')
FORBIDDEN={'torch','torch_npu','h3_comfy','rphi_model','history_utility_model','cv2','av'}


def require(value,message):
    if not value: raise ValueError(message)
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def value_sha(v): return hashlib.sha256(canonical(v).encode('utf-8')).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save_new(p,v):
    p=Path(p)
    with p.open('x',encoding='utf-8') as f: f.write(canonical(v)+'\n')
    return {'path':str(p.resolve()),'file_sha256':file_sha(p),'bytes':p.stat().st_size}
def binding_path(binding,root):
    require(set(binding)=={'path','sha256'},'exact path/SHA binding required')
    p=Path(binding['path']); p=p if p.is_absolute() else Path(root)/p
    require(file_sha(p)==binding['sha256'],'file SHA differs: '+str(p)); return p.resolve()
def payload_sha(c): return value_sha({k:v for k,v in c.items() if k!='admission'})
def phase_for(profile,t):
    if profile in ('identity','near_clean'): return profile
    require(profile in PROFILES,'unknown profile')
    if profile==PROFILES[0]: return 'static'
    if profile==PROFILES[2]: return 'abrupt-pre' if t<48 else 'abrupt-post'
    return 'low' if t<24 else 'ramp-up' if t<48 else 'high' if t<72 else 'ramp-down'


def donor_map(rows,seed=DONOR_SEED):
    """One permutation per exact stratum, never pixel shuffle or donor reuse.

    A source belongs to exactly one group. Therefore different groups guarantee
    different source IDs; max group size <= half of a stratum is necessary and
    sufficient. Seed-keyed contiguous groups rotated by the maximum group size
    construct a valid bijection without search, retries or a label-dependent RNG.
    """
    require(seed==DONOR_SEED,'donor seed is frozen')
    strata=defaultdict(list); source_groups={}; valid_slots=0
    for i,r in enumerate(rows):
        require(r['row_index']==i and r['row_sha256']==value_sha({k:v for k,v in r.items() if k!='row_sha256'}),'row identity/hash differs')
        require(r['split']=='train' and isinstance(r['source_group_id'],str) and bool(r['source_group_id']),'invalid donor source role/group')
        sid,gid=r['source_id'],r['source_group_id']
        require(sid not in source_groups or source_groups[sid]==gid,'one source maps to multiple groups');source_groups[sid]=gid
        t=r['frame_index']; require(type(t) is int and 0<=t<96,'invalid frame')
        true=[int(t>=2),int(t>=1)]
        require(r['true_availability']==true and r['history_indices']==[max(t-2,0),max(t-1,0),t],'history/boundary differs')
        ext=r['external_mask']; require(len(ext)==2 and all(type(x) is int and x in (0,1) for x in ext),'invalid raw external mask')
        require(r['severity_phase']==phase_for(r['profile_id'],t),'severity phase differs')
        require(r['teacher_mask'] is None,'teacher row cannot provide student donor')
        for slot,lag in enumerate((2,1)):
            if not(true[slot] and ext[slot]): continue
            key=canonical([r['profile_id'],r['severity_phase'],lag,ext])
            strata[key].append({'row_index':i,'slot':slot,'lag':lag,'row_sha256':r['row_sha256'],'source_id':sid,'source_group_id':gid})
            valid_slots+=1
    entries=[]; summaries=[]; impossible=[]
    for key,units in sorted(strata.items()):
        groups=defaultdict(list)
        for u in units: groups[u['source_group_id']].append(u)
        n=len(units); maximum=max(map(len,groups.values()))
        descriptor=json.loads(key)
        summary={'stratum':descriptor,'valid_slots':n,'source_groups':len(groups),'maximum_group_slots':maximum,'feasible':maximum*2<=n}
        summaries.append(summary)
        if maximum*2>n:
            impossible.append({**summary,'reason':'group capacity exceeds all other groups; source/group-disjoint bijection impossible','units':units});continue
        order=sorted(groups,key=lambda g:(value_sha([seed,key,'group',g]),g))
        ordered=[]
        for g in order: ordered.extend(sorted(groups[g],key=lambda u:(value_sha([seed,key,'unit',u['row_index'],u['slot']]),u['row_index'],u['slot'])))
        pairs=[]
        for i,u in enumerate(ordered):
            d=ordered[(i+maximum)%n]
            require(u['source_id']!=d['source_id'] and u['source_group_id']!=d['source_group_id'],'internal donor derangement error')
            pairs.append({'recipient':u,'donor':d,'stratum':descriptor})
        require(len({(e['donor']['row_index'],e['donor']['slot']) for e in pairs})==n,'donor reuse')
        entries.extend(pairs)
    entries.sort(key=lambda x:(x['recipient']['row_index'],x['recipient']['slot']))
    result={'schema':DONOR_SCHEMA,'seed':seed,'algorithm':'SHA256-seeded group/unit ordering; contiguous-group rotation by maximum group size',
      'unit':'whole6x6 utility map for one valid predecessor slot','stratum_fields':['profile_id','severity_phase','history_lag','raw_external_mask'],
      'boundary_rule':'include iff true_availability[slot]*external_mask[slot]==1; never merge raw mask patterns',
      'label_values_read':False,'model_calls':0,'valid_slots':valid_slots,'mapped_slots':len(entries),'strata':summaries,
      'impossible_strata':impossible,'donor_ready':not impossible,'student_training_blocked':bool(impossible),
      'no_resampling_or_cross_stratum_pooling':True,'entries':entries}
    result['donor_map_sha256']=value_sha(result);return result


def validate_schedule(s,audit,role):
    require(s['schema']=='openh3.history_utility_revised_schedule.v1' and s['role']==role and s['seed']==SEEDS[role], 'schedule role/seed differs')
    require(s['steps']==1000 and s['batch_size']==16 and len(s['numeric_schedule']['rows'])==16000,'schedule budget differs')
    require(s['schedule_sha256']==value_sha({k:v for k,v in s.items() if k!='schedule_sha256'}),'schedule digest differs')
    n=s['numeric_schedule']; require(n['schedule_sha256']==value_sha({k:v for k,v in n.items() if k!='schedule_sha256'}),'numeric schedule digest differs')
    expected=audit['schedules'][role]
    require(s['schedule_sha256']==expected['schedule_sha256'] and n['schedule_sha256']==expected['numeric_schedule_sha256'],'new schedule differs from accepted audit')
    require(s['dataset_sha256']==audit['datasets']['train']['dataset_sha256'] and s['cohort_sha256']==COHORT_SHA,'schedule dataset/cohort differs')
    require(s['legacy_research_ready'] is False and s['training_admitted'] is False and s['revised_cohort_ready'] is True,'schedule readiness changed')
    return {'schedule_sha256':s['schedule_sha256'],'numeric_schedule_sha256':n['schedule_sha256'],'rows':16000}


def inspect_contract(c):
    require(c['schema']==SCHEMA and c['resources']==RESOURCES and set(c['files'])==FILES,'execution contract/budget differs')
    paths={k:binding_path(v,c['root']) for k,v in c['files'].items()}
    require(c['files']['dataset_audit']['sha256']==AUDIT_SHA and c['files']['cohort']['sha256']==COHORT_SHA and c['files']['root_acceptance']['sha256']==ROOT_ACCEPTANCE_SHA,'accepted dataset pins differ')
    a=read(paths['dataset_audit']); ac=read(paths['root_acceptance']); comp=read(paths['materialization_complete']);pc=read(paths['producer_contract'])
    require(a['complete'] is True and a['revised_pair_ready'] is True and a['actual_npz_files_pixel_metadata_verified']==100 and a['training_steps']==0,'dataset audit is not accepted')
    require(ac['accepted'] is True and ac['revised_pair_ready'] is True and ac['training_admitted'] is False and AUDIT_SHA in ac['evidence'].values(),'root acceptance differs')
    require(a['producer']['complete']['sha256']==c['files']['materialization_complete']['sha256'] and a['producer']['contract']['sha256']==c['files']['producer_contract']['sha256'],'materialization/audit chain differs')
    require(comp['complete'] is True and comp['new_codec_passes_completed']==132 and comp['new_target_decodes']==0,'producer count differs')
    for k in ('plan','train_manifest','dev_manifest','cohort'): require(pc['files'][k]==c['files'][k],'producer data binding changed')
    for split,count in [('train',19),('dev',6)]:
        ix=read(paths[split+'_receipt_index']); require(ix['receipts']==comp['receipt_indices'][split] and len(ix['receipts'])==count,'receipt index differs')
        require(a['datasets'][split]['receipt_index']['sha256']==c['files'][split+'_receipt_index']['sha256'],'index/audit digest differs')
    impl={str((Path(k) if Path(k).is_absolute() else Path(c['root'])/k).resolve()):v for k,v in c['implementation_sha256'].items()}
    require(impl and all(file_sha(k)==v for k,v in impl.items()),'implementation source differs')
    for name,expected in [('prepare_history_utility_schedules.py',file_sha(__file__)),('history_utility_data.py',LOADER_SHA),('history_utility_revised_data.py',WRAPPER_SHA)]:
        path=Path(__file__).resolve().parent/name;require(impl.get(str(path))==expected==file_sha(path),'loaded source not pinned: '+name)
    require(not FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules),'forbidden module already imported')
    return paths,a,{'status':'inspected_small_evidence_only','actual_NPZ_reads':0,'schedule_generation':False,'model_calls':0,'training_admitted':False,'execution_payload_sha256':payload_sha(c),'input_files':c['files'],'implementation_sha256':c['implementation_sha256']}


@contextmanager
def bounded_cpu():
    """One process, no subprocesses; AS hard cap also bounds resident memory."""
    require(sys.platform.startswith('linux'),'execution requires Linux resource limits')
    old_aff=os.sched_getaffinity(0); old_as=resource.getrlimit(resource.RLIMIT_AS); old_alarm=signal.getsignal(signal.SIGALRM)
    require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing process timer cannot be replaced')
    def timeout(*_): raise TimeoutError('fixed schedule preparation wall budget exceeded')
    try:
        os.sched_setaffinity(0,set(sorted(old_aff)[:RESOURCES['CPU_cores']]))
        cap=RESOURCES['address_space_limit_bytes']; hard=old_as[1]
        resource.setrlimit(resource.RLIMIT_AS,(min(cap,hard) if hard>=0 else cap,hard))
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,RESOURCES['wall_seconds'])
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        resource.setrlimit(resource.RLIMIT_AS,old_as);os.sched_setaffinity(0,old_aff)


def execute(c,paths,audit,out):
    out=Path(out);require(not out.exists() and not out.is_symlink(),'fresh output only; no overwrite/resume')
    admission=read(binding_path(c['admission'],c['root']))
    require(admission['approved'] is True and admission['scope']=='history_utility_schedule_preparation_only' and admission['execution_payload_sha256']==payload_sha(c) and admission['entry_sha256']==file_sha(__file__) and admission['out']==str(out.resolve()),'separate exact root admission required')
    out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); report={'schema':MANIFEST_SCHEMA,'complete':False,'training_admitted':False,'model_calls':0,'training_steps':0,'NPU_calls':0,'actual_batch_constructions':0,'phase':'starting'}
    try:
        with bounded_cpu():
            import history_utility_data as D
            import history_utility_revised_data as R
            require(file_sha(D.__file__)==LOADER_SHA and file_sha(R.__file__)==WRAPPER_SHA,'import resolution differs')
            require(D.np.__version__==audit['numpy_version'],'NumPy version differs from accepted schedule generation')
            report['numpy_version']=D.np.__version__
            ds={}
            for split in ('train','dev'):
                report['phase']='verify_actual_'+split+'_NPZ';print(report['phase'],flush=True)
                ix=read(paths[split+'_receipt_index'])
                ds[split]=R.RevisedHistoryUtilityDataset(ix['receipts'],cohort=c['files']['cohort'],plan=paths['plan'],train_manifest=paths['train_manifest'],dev_manifest=paths['dev_manifest'],split=split,evidence_root=c['root'],cache_bytes=0)
                require(ds[split].dataset_sha256==audit['datasets'][split]['dataset_sha256'],'fresh dataset differs from accepted audit')
            pair=R.assert_revised_pair(ds['train'],ds['dev']);require(pair==audit['pair'],'fresh exact pair differs')
            ds['train']._numeric.cache_bytes=RESOURCES['schedule_cache_bytes']
            report.update(dataset_audit=c['files']['dataset_audit'],root_acceptance=c['files']['root_acceptance'],cohort_sha256=COHORT_SHA,pair=pair,
                dataset_sha256={split:d.dataset_sha256 for split,d in ds.items()},
                train_dataset_sha256=ds['train'].dataset_sha256,dev_dataset_sha256=ds['dev'].dataset_sha256,
                legacy_research_ready=False,revised_pair_ready=True,
                read_accounting={'fresh_unique_NPZ_files_verified':100,'constructor_cache_bytes':0,'additional_train_target_NPZ_reads_for_near_clean':19,'near_clean_fullframes_generated_and_hashed':1824,'metadata_only_NPZ_opens_additional':88,'schedule_cache_budget_bytes':RESOURCES['schedule_cache_bytes'],'source_MP4_reads':0})
            student=None
            for role in ('teacher','student'):
                report['phase']='persist_'+role+'_schedule';print(report['phase'],flush=True);s=ds['train'].schedule(role,1000);summary=validate_schedule(s,audit,role)
                record=save_new(out/(role+'_schedule.json'),s);report[role+'_schedule']={**record,**summary}
                if role=='student':student=s
                del s
            require(len(ds['train']._numeric._control_sha)==19,'near-clean hash generation incomplete')
            report['near_clean_control_SHA']=dict(sorted(ds['train']._numeric._control_sha.items()))
            report['phase']='freeze_donor_map';print(report['phase'],flush=True);donors=donor_map(student['numeric_schedule']['rows'])
            donors['student_schedule_sha256']=report['student_schedule']['schedule_sha256']
            # Include the schedule binding in the final semantic digest.
            donors['donor_map_sha256']=value_sha({k:v for k,v in donors.items() if k!='donor_map_sha256'})
            report['donor_manifest']={**save_new(out/'donor_map.json',donors),'donor_map_sha256':donors['donor_map_sha256']}
            report.update(donor_ready=donors['donor_ready'],student_training_blocked=not donors['donor_ready'],donor_summary={'valid_slots':donors['valid_slots'],'mapped_slots':donors['mapped_slots'],'strata':len(donors['strata']),'impossible_strata':len(donors['impossible_strata'])},teacher_schedule_ready=True)
            inspect_contract(c)
            require(not FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules),'model/backend imported')
            require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<RESOURCES['maximum_output_bytes'],'output exceeds fixed cap')
            report.update(complete=True,phase='complete',status='schedules_persisted_donor_ready' if donors['donor_ready'] else 'schedules_persisted_students_blocked_impossible_strata')
    except BaseException as exc:
        report.update(complete=False,status='failed_preserve_partial_no_retry',error={'type':type(exc).__name__,'message':str(exc)})
    report.update(elapsed_seconds=time.monotonic()-started,peak_process_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,execution_payload_sha256=payload_sha(c),entry_sha256=file_sha(__file__),resources=RESOURCES)
    save_new(out/'manifest.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution-contract','--contract',dest='contract',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    c=read(args.contract);paths,audit,inspection=inspect_contract(c)
    if not args.execute:
        require(not args.out.exists() and not args.out.is_symlink(),'fresh inspection output only');args.out.mkdir(parents=True);save_new(args.out/'inspection.json',inspection);print(canonical(inspection));return
    r=execute(c,paths,audit,args.out);print(canonical({k:r.get(k) for k in ('complete','status','donor_ready','teacher_schedule_ready','error')}))
    if not r['complete']:raise SystemExit(1)

if __name__=='__main__':main()
