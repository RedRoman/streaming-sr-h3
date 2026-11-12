"""Five-fold TRAIN-source/group-held-out linear Huber diagnostic; no dev/model replay."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import zipfile

SCHEMA='rphi.input_preservation_train_feature_probe_execution.v1'
SEED=20261112
FOLDS=5
ITERATIONS=200
RIDGE=.001
ZERO_STD=1e-12
RESOURCES={'wall_seconds':180,'CPU_cores':2,'address_space_bytes':4 << 30,'maximum_output_bytes':8 << 20}
PINS={'train_cache_manifest':'4fc49311e3c03b1e089e9eafbd33207965b806ddc5b250fb4fbe936297dc70b9',
      'train_cache_audit':'ce53dccdd3c8a10c52a01944bb6829d13de492e133850ebf077fb8e5fc7c6893',
      'student_schedule':'b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'}
PROTOCOL={'folds':5,'split_seed':SEED,'split':'SHA256 canonical [seed,source_group_id] sorted then round-robin folds',
    'feature':'FP64 arithmetic mean over the two saved raw-feature slots;11 channels per tile; no source/profile input',
    'standardization':'all11 channels using fold-training tiles only; std<=1e-12 becomes zero channel; intercept retained',
    'target':'original saved normalized_benefit FP32 converted to FP64; no scale or clip change',
    'loss':'tile-mean SmoothL1 beta1 + .5*.001*sum(slope**2); intercept unregularized',
    'solver':'zero coefficients; exactly200 full-batch gradient steps, fixed step1/lambda_max(Ztrain.T@Ztrain/n+ridge_diagonal); no line search or early stopping',
    'constant':'fold-training unregularized Huber minimizer interval using60 bisections per endpoint; closest-to-zero minimizer',
    'threshold':0,'iterations':ITERATIONS,'ridge':RIDGE,'std_zero_threshold':ZERO_STD,
    'primary_summary':'held-out source macro mean; tile-micro summary separately',
    'hyperparameter_search':False,'dev_allowed':False}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def value_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4 << 20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def write_new(p,v):
    body=json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    need(len(body.encode())<=RESOURCES['maximum_output_bytes'],'small JSON output budget exceeded')
    with Path(p).open('x') as f:f.write(body)

class Files:
    def __init__(self,root):self.root=Path(root).resolve();self.used={}
    def bound(self,b,limit=32 << 20):
        p=Path(b['path']);p=p if p.is_absolute() else self.root/p
        need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(self.root),'invalid/nonowned input')
        need(0<p.stat().st_size<=limit,'input size budget')
        if 'bytes' in b:need(p.stat().st_size==b['bytes'],'input size mismatch')
        need(file_sha(p)==b['sha256'],'input file SHA mismatch: '+str(p));self.used[str(p)]={'sha256':b['sha256'],'bytes':p.stat().st_size}
        return p

def split_rows(rows,expected_rows=16000,expected_sources=19):
    """Uses authenticated metadata only, before any feature/label member loads."""
    need(len(rows)==expected_rows,'row coverage differs');source_groups={}
    for i,r in enumerate(rows):
        need(r['row_index']==i and r['split']=='train' and r['row_sha256']==value_sha({k:v for k,v in r.items() if k!='row_sha256'}),'not authenticated TRAIN row')
        s,g=r['source_id'],r['source_group_id'];need(isinstance(s,str) and s and isinstance(g,str) and g,'source/group missing')
        need(s not in source_groups or source_groups[s]==g,'source crosses groups');source_groups[s]=g
    need(len(source_groups)==expected_sources,'source count differs')
    groups=sorted(set(source_groups.values()),key=lambda g:(value_sha([SEED,g]),g));need(len(groups)>=FOLDS,'fewer than5 groups; do not coarsen')
    fold_by_group={g:i%FOLDS for i,g in enumerate(groups)}
    folds=[fold_by_group[r['source_group_id']] for r in rows]
    byfold=[]
    for f in range(FOLDS):
        held_sources=sorted(s for s,g in source_groups.items() if fold_by_group[g]==f)
        byfold.append({'fold':f,'held_out_sources':held_sources,'held_out_groups':[g for g in groups if fold_by_group[g]==f],
            'training_sources':sorted(set(source_groups)-set(held_sources)),'held_out_rows':sum(x==f for x in folds)})
    need(all(x['held_out_rows']>0 and x['training_sources'] for x in byfold),'empty fold')
    return {'schema':'rphi.input_preservation_train_group_split.v1','seed':SEED,'folds':byfold,'group_order':groups,
        'source_to_group':source_groups,'source_to_fold':{s:fold_by_group[g] for s,g in source_groups.items()},
        'row_folds':folds,'feature_members_loaded_before_split':0,'label_members_loaded_before_split':0,
        'score_values_seen_before_split':0,'dev_rows':0,'scope':'TRAIN internal diagnostic; not independent test'}

def metadata(c):
    need(c['schema']==SCHEMA and c['resources']==RESOURCES and c['protocol']==PROTOCOL and set(c['files'])==set(PINS),'scope/resources/protocol changed')
    need(c['runtime']=={'python':'3.12.14','numpy':'1.26.4','BLAS_threads':2,'Torch_imported':False},'frozen CPU runtime differs')
    need(c['source_sha256']==file_sha(__file__),'diagnostic source changed')
    fs=Files(c['root']);docs={}
    for k,h in PINS.items():need(c['files'][k]['sha256']==h,'frozen TRAIN binding changed');docs[k]=read(fs.bound(c['files'][k]))
    manifest,audit,schedule=(docs[k] for k in ('train_cache_manifest','train_cache_audit','student_schedule'))
    need(manifest['complete'] is True and manifest['role']=='train' and manifest['completed_rows']==16000,'train cache incomplete')
    need(audit['complete'] is True and audit['role']=='train' and audit['actual_rows']==16000
         and audit['raw_feature_rows_recomputed']==16000 and audit['producer_manifest']==c['files']['train_cache_manifest'],'actual TRAIN audit not bound')
    need(schedule['role']=='student' and schedule['steps']==1000 and schedule['batch_size']==16
         and schedule['schedule_sha256']==value_sha({k:v for k,v in schedule.items() if k!='schedule_sha256'}),'student schedule changed')
    numeric=schedule['numeric_schedule'];need(numeric['schedule_sha256']==value_sha({k:v for k,v in numeric.items() if k!='schedule_sha256'}),'numeric schedule changed')
    rows=numeric['rows'];split=split_rows(rows)
    return fs,manifest,rows,split

def read_training_arrays(fs,manifest,rows,split_path,split_sha):
    import numpy as np
    need(Path(split_path).is_file() and file_sha(split_path)==split_sha,'persisted split required before array reads')
    batches=[json.loads(s) for s in fs.bound(manifest['batches']).read_text().splitlines()]
    need(len(batches)==1000,'batch coverage differs')
    x=np.empty((576000,11),dtype=np.float64);y=np.empty(576000,dtype=np.float64);bytes_hashed=0
    allowed={'k1_rgb','raw_features','availability','input_mae','k1_mae','raw_benefit','normalized_benefit'}
    for bi,b in enumerate(batches):
        selected=rows[bi*16:(bi+1)*16]
        need(b['batch_index']==bi and b['row_indices']==list(range(bi*16,(bi+1)*16))
             and b['row_sha256']==[r['row_sha256'] for r in selected],'batch/row mismatch')
        path=fs.bound(b,4 << 20);bytes_hashed+=path.stat().st_size
        with zipfile.ZipFile(path) as z:
            members=z.infolist();need(len(members)==7 and {v.filename for v in members}=={k+'.npy' for k in allowed}
                and all(v.compress_type==zipfile.ZIP_STORED and v.file_size<2 << 20 for v in members),'NPZ format changed')
        with np.load(path,allow_pickle=False) as z:
            f=z['raw_features'];v=z['normalized_benefit']
            for name,a,shape in [('raw_features',f,(16,2,11,6,6)),('normalized_benefit',v,(16,1,6,6))]:
                need(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all()
                     and b['arrays'][name]=={'dtype':'float32','shape':list(shape),'sha256':array_sha(a)},'actual training array differs')
            need(np.abs(v).max()<=4,'label range changed')
            x[bi*576:(bi+1)*576]=f.astype(np.float64).mean(axis=1).transpose(0,2,3,1).reshape(576,11)
            y[bi*576:(bi+1)*576]=v.reshape(-1)
    return x,y,{'unique_NPZ_files':1000,'file_bytes_hashed':bytes_hashed,'feature_members_decompressed':1000,
        'normalized_label_members_decompressed':1000,'RGB_raw_benefit_or_target_members_decompressed':0,'dev_paths_opened':0,
        'rows':16000,'tiles':576000,'feature_mean_array_sha256':array_sha(x),'label_array_sha256':array_sha(y)}

def huber(r):
    import numpy as np
    a=np.abs(r);return np.where(a<=1,.5*r*r,a-.5)

def constant_huber(y):
    import numpy as np
    y=np.asarray(y,dtype=np.float64);need(y.ndim==1 and len(y)>0 and np.isfinite(y).all() and np.abs(y).max()<=4,'constant target invalid')
    sorted_y=np.sort(y);prefix=np.concatenate(([0.],np.cumsum(sorted_y,dtype=np.float64)));n=len(y)
    def grad(c):
        left=int(np.searchsorted(sorted_y,c-1,'left'));right=int(np.searchsorted(sorted_y,c+1,'right'))
        return (left-(n-right)+(right-left)*c-(prefix[right]-prefix[left]))/n
    ends=[]
    for upper in (False,True):
        lo,hi=-4.,4.
        for _ in range(60):
            mid=(lo+hi)/2;g=grad(mid)
            if g<0 or (upper and g==0):lo=mid
            else:hi=mid
        ends.append((lo+hi)/2)
    lo,hi=ends;need(lo<=hi+1e-12,'Huber minimum interval invalid')
    c=0. if lo<=0<=hi else (lo if abs(lo)<=abs(hi) else hi)
    return {'minimizer_interval':ends,'selected_nearest_zero':c,'gradient_at_selected':float(np.mean(np.clip(c-y,-1,1))),
        'gradient_at_zero':float(np.mean(np.clip(-y,-1,1))),'loss':float(huber(c-y).mean()),'root_iterations_per_endpoint':60}

def standardizer(train):
    import numpy as np
    need(train.ndim==2 and np.isfinite(train).all(),'training design invalid')
    mean=train.mean(axis=0);std=train.std(axis=0);zero=std<=ZERO_STD
    scale=np.where(zero,1.,std)
    return mean,scale,zero

def transform(x,stats):
    import numpy as np
    mean,scale,zero=stats;z=(x-mean)/scale;z[:,zero]=0
    return np.ascontiguousarray(np.column_stack((z,np.ones(len(z)))),dtype=np.float64)

def linear_probe(z,y,iterations=ITERATIONS,ridge=RIDGE):
    import numpy as np
    need(iterations==ITERATIONS and ridge==RIDGE,'no iteration/regularization search')
    w=np.zeros(z.shape[1],dtype=np.float64);diag=np.full(len(w),ridge);diag[-1]=0.
    curvature=z.T@z/len(z)+np.diag(diag);L=float(np.linalg.eigvalsh(curvature)[-1]);need(math.isfinite(L) and L>0,'invalid fixed Lipschitz bound')
    initial=float(huber(-y).mean())
    for _ in range(iterations):
        grad=z.T@np.clip(z@w-y,-1.,1.)/len(z)+diag*w
        w-=grad/L
        need(np.isfinite(w).all(),'nonfinite probe update')
    residual=z@w-y;objective=float(huber(residual).mean()+.5*ridge*np.dot(w[:-1],w[:-1]))
    need(objective<=initial+1e-10,'fixed-step convex objective increased')
    return w,{'iterations':iterations,'step_size':1/L,'Lipschitz_upper_bound':L,'initial_objective':initial,
        'final_objective':objective,'final_gradient_norm':float(np.linalg.norm(z.T@np.clip(residual,-1.,1.)/len(z)+diag*w)),
        'convergence_claimed':False,'coefficients':w.tolist()}

def score_summary(y,s):
    import numpy as np
    need(y.shape==s.shape and np.isfinite(s).all(),'score shape/finite differs')
    pos=y>0;neg=y<0;action=s>0
    def rate(v,where):return float(v[where].mean()) if where.any() else None
    return {'tiles':len(y),'Huber_loss':float(huber(s-y).mean()),'K1_tile_fraction':float(action.mean()),
        'positive_label_fraction':float(pos.mean()),'negative_label_fraction':float(neg.mean()),
        'negative_label_false_K1_fraction':rate(action,neg),'positive_label_K1_retained_fraction':rate(action,pos),
        'zero_label_false_K1_fraction':rate(action,~(pos|neg)),
        'score_quantiles_0_25_50_75_100':np.quantile(s,[0,.25,.5,.75,1]).tolist(),
        'physical_MAE_gain_or_harm':None}

def fit_folds(x,y,rows,split):
    import numpy as np
    rowfold=np.asarray(split['row_folds']);tilefold=np.repeat(rowfold,36)
    predicted=np.empty(len(y));baseline=np.empty(len(y));fit_reports=[]
    for f in range(FOLDS):
        train=tilefold!=f;test=~train;stats=standardizer(x[train]);zt=transform(x[train],stats)
        yt=y[train];constant=constant_huber(yt);weights,fit=linear_probe(zt,yt);del zt,yt
        predicted[test]=transform(x[test],stats)@weights;baseline[test]=constant['selected_nearest_zero']
        fit_reports.append({'fold':f,'training_tiles':int(train.sum()),'held_out_tiles':int(test.sum()),
            'training_mean':stats[0].tolist(),'training_scale':stats[1].tolist(),'zero_variance_channel_indices':np.where(stats[2])[0].tolist(),
            'constant':constant,'probe_fit':fit,'held_out':{'constant':score_summary(y[test],baseline[test]),'linear':score_summary(y[test],predicted[test])}})
    per_source={};per_source_profile={};row_sources=[r['source_id'] for r in rows];row_profiles=[r['profile_id'] for r in rows]
    for sid in sorted(set(row_sources)):
        mask=np.repeat(np.asarray([s==sid for s in row_sources]),36)
        per_source[sid]={'fold':split['source_to_fold'][sid],'constant':score_summary(y[mask],baseline[mask]),'linear':score_summary(y[mask],predicted[mask])}
        per_source[sid]['Huber_gain_constant_minus_linear']=per_source[sid]['constant']['Huber_loss']-per_source[sid]['linear']['Huber_loss']
        for profile in sorted(set(row_profiles)):
            m=np.repeat(np.asarray([s==sid and p==profile for s,p in zip(row_sources,row_profiles)]),36)
            if m.any():per_source_profile[sid+'|'+profile]={'constant':score_summary(y[m],baseline[m]),'linear':score_summary(y[m],predicted[m])}
    def macro(key,arm,items):
        values=[v[arm][key] for v in items if v[arm][key] is not None];return sum(values)/len(values) if values else None
    keys=['Huber_loss','K1_tile_fraction','negative_label_false_K1_fraction','positive_label_K1_retained_fraction']
    macro_summary={a:{k:macro(k,a,per_source.values()) for k in keys} for a in ('constant','linear')}
    macro_counts={a:{k:sum(v[a][k] is not None for v in per_source.values()) for k in keys} for a in ('constant','linear')}
    profile_macro={p:{a:{k:macro(k,a,[v for n,v in per_source_profile.items() if n.rsplit('|',1)[1]==p]) for k in keys}
        for a in ('constant','linear')} for p in sorted(set(row_profiles))}
    profile_counts={p:{a:{k:sum(v[a][k] is not None for n,v in per_source_profile.items() if n.rsplit('|',1)[1]==p) for k in keys}
        for a in ('constant','linear')} for p in sorted(set(row_profiles))}
    return {'folds':fit_reports,'source_macro_primary':macro_summary,'source_macro_valid_sources':macro_counts,
        'profile_source_macro':profile_macro,'profile_source_macro_valid_sources':profile_counts,
        'sources_with_positive_Huber_gain':sum(v['Huber_gain_constant_minus_linear']>0 for v in per_source.values()),
        'source_count':len(per_source),'per_source':per_source,'per_source_profile':per_source_profile,
        'tile_micro_secondary':{'constant':score_summary(y,baseline),'linear':score_summary(y,predicted)},
        'held_out_score_sha256':{'constant':array_sha(baseline),'linear':array_sha(predicted)},
        'actual_probe_fits':5,'actual_gradient_updates':5*ITERATIONS,'held_out_tiles_predicted':len(y)}

def set_limits():
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    need(hasattr(os,'sched_getaffinity'),'actual execution requires Linux CPU affinity')
    cpus=sorted(os.sched_getaffinity(0));need(len(cpus)>=2,'two CPUs unavailable');os.sched_setaffinity(0,cpus[:2])
    resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
    def stop(signum,frame):raise TimeoutError('fixed180s wall budget or termination; no retry')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(180)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True);p.add_argument('--execution-contract-sha256',required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    need(file_sha(a.execution_contract)==a.execution_contract_sha256,'execution file SHA differs');c=read(a.execution_contract)
    need(c['out']==str(a.out.resolve()) and not a.out.exists() and not a.out.is_symlink(),'fresh exact output required')
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    r={'schema':'rphi.input_preservation_train_feature_probe.v1','complete':False,'status':'preparing',
        'source_sha256':file_sha(__file__),'execution_contract_sha256':a.execution_contract_sha256,'protocol':PROTOCOL,
        'resources':RESOURCES,'research_head_training_updates':0,'restorer_calls':0,'dev_paths_opened':0,
        'previous_development_screen_remains_failed':True,'deployment_or_further_training_admitted':False,
        'limitations':['Five-fold TRAIN internal source/group diagnostic only; not an independent test.',
            'Only this linear head is fitted on fold-training groups. The fixed K1 teacher previously trained on all19 sources, so these are not teacher/restorer-held-out sources.',
            'Frames and tiles remain correlated. Primary comparison is source macro, with every source reported; no significance claim.',
            'Conditional positive/negative-label rates use valid-source macro; sources with no applicable label are null, with exact macro denominators reported.',
            'Labels remain clipped normalized utility; this is not physical MAE loss or proof of deployed benefit.',
            'Linear probe uses tile-local slot-mean features, without original3x3 spatial context or nonlinear slot interactions. Failure is not proof of no raw-observation information. Fixed200 steps need not reach the optimum.',
            'No threshold, fold, feature, regularization or iteration search; no automatic follow-up after any outcome.']}
    try:
        if a.execute:set_limits()
        fs,manifest,rows,split=metadata(c)
        split_path=a.out/'split.json';write_new(split_path,split);split_sha=file_sha(split_path)
        r['split']={'path':str(split_path),'sha256':split_sha,'written_before_feature_or_label_load':True}
        if a.execute:
            import numpy as np
            versions={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'BLAS_threads':2,'Torch_imported':False}
            need(versions==c['runtime'],'actual CPU runtime differs');r['runtime']=versions
            x,y,proof=read_training_arrays(fs,manifest,rows,split_path,split_sha);r['array_load']=proof
            r['result']=fit_folds(x,y,rows,split)
        need(file_sha(split_path)==split_sha and file_sha(__file__)==c['source_sha256'],'split/source changed')
        # All1000 NPZ were hashed when read. Rehash only small contract-bound JSON,
        # never re-scan2GB of cache after fitting.
        for ref in c['files'].values():fs.bound(ref)
        need(file_sha(a.execution_contract)==a.execution_contract_sha256,'execution changed')
        need(not any(x.split('.')[0] in {'torch','torch_npu','h3_comfy'} for x in sys.modules),'model framework imported')
        r.update(complete=True,status='TRAIN_internal_probe_complete_no_quality_or_training_admission' if a.execute else 'metadata_split_only_no_feature_or_label_load',inputs=fs.used)
    except BaseException as e:r.update(status='failed_fixed_budget_no_retry',error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        write_new(a.out/'report.json',r)
    print(json.dumps({k:r.get(k) for k in ('complete','status','error','elapsed_seconds')}));return 0 if r['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
