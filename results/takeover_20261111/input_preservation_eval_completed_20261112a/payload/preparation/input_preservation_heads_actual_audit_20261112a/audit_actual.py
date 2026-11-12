"""Read-only saved-head audit: no model/optimizer construction, NPZ or inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time

SCHEMA = 'rphi.input_preservation_heads_actual_audit_config.v1'
TRAINER_SHA = '2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd'
DONOR_SHA = 'fdb872a4e1b20c07b67af2c7b40a322f6b968ea7efcac9b2a96239c8898a2d6a'
STUDENT_SHA = 'b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'
ARMS = ('matched', 'shuffled')
SHAPES = {'gate.net.0.weight': (16,11,3,3), 'gate.net.0.bias': (16,),
          'gate.net.2.weight': (1,16,3,3), 'gate.net.2.bias': (1,)}
COUNTS = ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')
RESOURCE = {'wall_seconds':120, 'CPU_cores':2, 'address_space_bytes':4 << 30}

def need(ok, message):
    if not ok: raise ValueError(message)

def value_sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1 << 20),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def execution_sha(c): return value_sha({k:v for k,v in c.items() if k!='admission'})
def identity(c): return value_sha({k:v for k,v in c.items() if k not in ('arm','admission','mode','backend_smokes')})
def lr(step):
    warm=min(1.,step/50)
    decay=.2+.8*.5*(1+math.cos(math.pi*max(0,step-50)/950))
    return 1e-4*warm*decay

def typed(v, expected, message):
    need(all(k in v and type(v[k]) is type(x) and v[k]==x for k,x in expected.items()),message)

class Files:
    def __init__(self, root): self.root=Path(root).resolve(); self.used={}
    def get(self,b,max_bytes=64 << 20):
        p=Path(b['path']);p=p if p.is_absolute() else self.root/p
        need(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(self.root),'file outside root or symlink: '+str(p))
        n=p.stat().st_size;need(0<n<=max_bytes,'file byte limit: '+str(p))
        if 'bytes' in b: need(n==b['bytes'],'bound byte size differs')
        h=sha(p);need(h==b['sha256'],'file SHA differs: '+str(p))
        self.used[str(p.resolve())]={'sha256':h,'bytes':n}
        return p
    def doc(self,b): return read(self.get(b))
    def unchanged(self):
        for p,b in self.used.items():need(Path(p).stat().st_size==b['bytes'] and sha(p)==b['sha256'],'input changed during audit')

def report_header(r,c,arm,steps):
    mode='research' if steps==1000 else 'backend_smoke'
    typed(r,{'schema':'rphi.input_preservation_head_training.v1','complete':True,'mode':mode,'arm':arm,
        'status':'final_head_ready_for_bound_evaluation' if steps==1000 else 'CPU_backend_smoke_complete_not_research_resume',
        'completed_step':steps,'row_cursor':steps*16,'strict_checkpoint_reload':True,'NPU_calls':0,
        'new_restorer_forwards':0,'dev_label_arrays_read':0,'source_NPZ_reads':0,'resumed':False,'quality_accepted':False},'incomplete or wrong training report')
    need(set(r['counts'])==set(COUNTS),'count fields differ');typed(r['counts'],{k:steps for k in COUNTS},'operation counts differ')
    need(r['comparison_identity_sha256']==identity(c),'comparison identity differs')

def metadata(config):
    need(config['schema']==SCHEMA and config['resources']==RESOURCE and set(config['arms'])==set(ARMS),'audit config differs')
    need(config['auditor_sha256']==sha(__file__),'auditor source not bound')
    files=Files(config['root']);reports={};contracts={}
    for arm in ARMS:
        a=config['arms'][arm];need(set(a)=={'report','execution_contract'},'arm bindings differ')
        c=files.doc(a['execution_contract']);r=files.doc(a['report']);report_header(r,c,arm,1000)
        typed(c,{'schema':'rphi.input_preservation_head_training_execution.v1','mode':'research','arm':arm,'steps':1000,'batch_size':16,'seed':20261111},'research contract differs')
        need(c['root']==str(files.root) and r['execution_payload_sha256']==execution_sha(c),'root/execution binding differs')
        pins=c['implementation_sha256'];need({Path(k).name:v for k,v in pins.items()}['train_input_preservation_heads.py']==TRAINER_SHA,'trainer source differs')
        for p,h in pins.items():files.get({'path':p,'sha256':h})
        expected_binding={'files':c['files'],'backend_smokes':c['backend_smokes'],'comparison_identity_sha256':identity(c)}
        need(r['binding']==expected_binding,'report input binding differs')
        typed(r['runtime'],{'device':'cpu','threads':2,'dtype':'float32','autocast':False,'grad_scaler':False},'recorded training runtime differs')
        need(r['runtime']['versions']==c['runtime']['expected_versions'] and r['runtime']['versions']['torch_npu'] is None,'recorded runtime versions differ')
        report_dir=files.get(a['report']).parent
        for key,name in (('initial_state','initial_state.pt'),('checkpoint','head_step1000.pt'),('metrics','metrics.jsonl')):
            q=files.get(r[key],2 << 20 if key!='metrics' else 8 << 20)
            need(q.parent==report_dir and q.name==name,'saved artifact outside exact report directory')
        permission=files.doc(c['admission'])
        typed(permission,{'schema':'rphi.input_preservation_head_training_admission.v1','approved':True,'arm':arm,'mode':'research',
            'execution_payload_sha256':execution_sha(c),'entry_sha256':TRAINER_SHA,'out':str(files.get(a['report']).parent)},'root research admission differs')
        need(set(c['backend_smokes'])==set(ARMS),'both smoke reports required')
        for smoke_arm, ref in c['backend_smokes'].items():
            smoke=files.doc(ref);report_header(smoke,c,smoke_arm,2)
            need(smoke['binding']['files']==c['files'] and smoke['initial_state_sha256']==r['initial_state_sha256']
                 and smoke['runtime']==r['runtime'],'smoke input/initial/runtime differs')
        reports[arm]=r;contracts[arm]=c
    a,b=(contracts[x] for x in ARMS)
    need({k:v for k,v in a.items() if k not in ('arm','admission')}=={k:v for k,v in b.items() if k not in ('arm','admission')},'arms differ beyond arm/admission')
    need(reports['matched']['cache_load']==reports['shuffled']['cache_load'],'actual loaded input proof differs')
    typed(reports['matched']['cache_load'],{'rows':16000,'train_NPZ_files_verified':1000,
        'feature_and_label_arrays_loaded':2000,'K1_or_GT_arrays_loaded':0,'dev_NPZ_files_opened':0},'recorded input coverage differs')
    need(a['files']['donor_map']['sha256']==DONOR_SHA and a['files']['student_schedule']['sha256']==STUDENT_SHA,'fixed donor/schedule changed')
    # JSON and source bytes only. Metric weight files and cache NPZ are not opened.
    for ref in a['files'].values():files.get(ref)
    return files,reports,contracts

def donor_indices(mapping,rows):
    n=len(rows);need(n==16000 and mapping['total_rows']==mapping['mapped_rows']==n and mapping['donor_ready'] is True
        and mapping['impossible_strata']==[],'donor incomplete')
    need(mapping['donor_map_sha256']==value_sha({k:v for k,v in mapping.items() if k!='donor_map_sha256'}),'donor semantic SHA differs')
    need(len(mapping['entries'])==n,'donor length differs');donors=[]
    for i,e in enumerate(mapping['entries']):
        r,d=e['recipient'],e['donor'];j=d['row_index']
        need(type(j) is int and 0<=j<n and r['row_index']==i and i!=j,'donor row invalid')
        for u,index in ((r,i),(d,j)):
            row=rows[index];need(row['row_index']==index and row['split']=='train'
                and row['row_sha256']==value_sha({k:v for k,v in row.items() if k!='row_sha256'}),'student row invalid')
            need(all(u[k]==row[k] for k in ('row_sha256','source_id','source_group_id')),'donor actual metadata differs')
            effective=[x*y for x,y in zip(row['true_availability'],row['external_mask'])]
            need(u['stratum']==[row['profile_id'],row['severity_phase'],effective],'donor stratum differs')
        need(r['source_id']!=d['source_id'] and r['source_group_id']!=d['source_group_id'] and r['stratum']==d['stratum'],'source/group/stratum donor isolation differs')
        donors.append(j)
    need(sorted(donors)==list(range(n)),'donor is not a full bijection');return donors

def check_metrics(pair,donors,batches,row_hashes):
    need(len(pair['matched'])==len(pair['shuffled'])==len(batches)==1000 and len(donors)==len(row_hashes)==16000,'metric/batch coverage differs')
    hashes={arm:[] for arm in ARMS};losses={arm:[] for arm in ARMS}
    for i,b in enumerate(batches):
        indices=list(range(i*16,(i+1)*16));feature_sha=b['arrays']['raw_features']['sha256']
        need(b['batch_index']==i and b['row_indices']==indices and b['row_sha256']==row_hashes[i*16:(i+1)*16],'cache index row binding differs')
        for arm in ARMS:
            m=pair[arm][i];step=i+1
            need(m['step']==step and m['row_end_exclusive']==step*16 and m['feature_rows']==indices
                 and m['feature_batch_sha256']==feature_sha,'metric feature sequence/SHA differs')
            need(type(m['lr']) is float and m['lr']==lr(step) and m['lr_hex']==lr(step).hex(),'actual LR differs')
            need(m['label_rows']==(indices if arm=='matched' else donors[i*16:(i+1)*16]),'label rows differ from fixed donor')
            if arm=='matched':need(m['label_batch_sha256']==b['arrays']['normalized_benefit']['sha256'],'matched label batch hash differs')
            need(isinstance(m['label_batch_sha256'],str) and len(m['label_batch_sha256'])==64,'missing label batch hash')
            values=[m['loss'],m['gradient_norm_before_clip'],m['correction_tile_fraction'],*m['tail_gradient_norms'].values()]
            need(all(type(v) in (int,float) and math.isfinite(v) for v in values)
                 and m['loss']>=0 and m['gradient_norm_before_clip']>=0 and 0<=m['correction_tile_fraction']<=1
                 and set(m['tail_gradient_norms'])=={'gate.net.2.weight','gate.net.2.bias'}
                 and all(v>=0 for v in m['tail_gradient_norms'].values()),'invalid training diagnostics')
            losses[arm].append(m['loss']);hashes[arm].append(m['label_batch_sha256'])
    return {'paired_steps':1000,'paired_feature_rows':16000,'feature_batch_hashes_match_cache_index':True,
        'matched_label_batch_hashes_match_cache_index':True,'shuffled_label_indices_match_donor':True,
        'shuffled_label_batch_hashes_independently_recomputed':False,'LR_values_and_hex_exact':True,
        'losses':{a:{'first':v[0],'last':v[-1],'mean':sum(v)/len(v),'min':min(v),'max':max(v)} for a,v in losses.items()},
        'tail_nonzero_gradient_steps':{a:sum(any(x>0 for x in m['tail_gradient_norms'].values()) for m in pair[a]) for a in ARMS},
        'logged_label_batch_sequence_sha256':{a:value_sha(v) for a,v in hashes.items()}}

def tensor_sha(v):
    a=v.detach().cpu().contiguous()
    h=hashlib.sha256(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes());return h.hexdigest()

def state_sha(v): return value_sha({k:tensor_sha(x) for k,x in sorted(v.items())})

def state_check(v,torch):
    need(isinstance(v,dict) and set(v)==set(SHAPES),'head state keys differ')
    for k,shape in SHAPES.items():
        t=v[k];need(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32
            and tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'nonfinite/wrong head tensor '+k)
    need(sum(x.numel() for x in v.values())==1745,'head parameter count differs');return state_sha(v)

def optimizer_check(opt,state,torch):
    need(set(opt)=={'state','param_groups'} and len(opt['param_groups'])==1,'optimizer schema differs')
    group=opt['param_groups'][0]
    expected={'lr':lr(1000),'betas':(.9,.999),'eps':1e-8,'weight_decay':0.,'amsgrad':False,'foreach':False,'fused':False,
              'maximize':False,'capturable':False,'differentiable':False,'params':[0,1,2,3]}
    need(all(k in group and type(group[k]) is type(v) and group[k]==v for k,v in expected.items()),'optimizer recipe/parameter order differs')
    need(set(group)-set(expected)<= {'decoupled_weight_decay'} and group.get('decoupled_weight_decay',True) is True,'unknown optimizer group fields')
    need(set(opt['state'])=={0,1,2,3},'optimizer state inventory differs')
    result={}
    for pid,name in enumerate(SHAPES):
        s=opt['state'][pid];need(set(s)=={'step','exp_avg','exp_avg_sq'},'optimizer tensor fields differ')
        for k,t in s.items():
            need(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and t.device.type=='cpu'
                 and tuple(t.shape)==(() if k=='step' else SHAPES[name]) and bool(torch.isfinite(t).all()),'invalid optimizer tensor')
        need(float(s['step'])==1000. and bool((s['exp_avg_sq']>=0).all()),'AdamW step/variance differs')
        result[name]={k:tensor_sha(t) for k,t in s.items()}
    return {'parameter_states':4,'tensor_states':12,'completed_step':1000,'state_tensor_sha256':result}

def tree_fingerprint(v,torch):
    if isinstance(v,torch.Tensor):return {'tensor':tensor_sha(v),'dtype':str(v.dtype),'shape':list(v.shape)}
    if isinstance(v,dict):return {str(k):tree_fingerprint(x,torch) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return {'container':type(v).__name__,'items':[tree_fingerprint(x,torch) for x in v]}
    need(v is None or type(v) in (str,int,float,bool),'unexpected saved non-tensor value')
    if isinstance(v,float):need(math.isfinite(v),'nonfinite saved scalar')
    return v

def rng_check(r,torch):
    need(set(r)=={'python','numpy','torch_cpu','torch_device'} and r['torch_device'] is None,'RNG scope differs')
    t=r['torch_cpu'];need(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0,'invalid CPU RNG tensor')
    need(type(r['python']) is tuple and len(r['python'])==3 and r['python'][0]==3,'Python RNG state differs')
    n=r['numpy'];need(type(n) is tuple and len(n)==5 and n[0]=='MT19937' and len(n[1])==624
        and all(type(x) is int and 0<=x<2**32 for x in n[1]) and 0<=n[2]<=624,'NumPy RNG state differs')
    return value_sha(tree_fingerprint(r,torch))

def actual(files,reports,contracts):
    import torch
    need('torch_npu' not in sys.modules,'NPU backend imported');torch.set_num_threads(2);torch.set_num_interop_threads(1)
    c=contracts['matched'];schedule=files.doc(c['files']['student_schedule'])
    need(schedule['schedule_sha256']==value_sha({k:v for k,v in schedule.items() if k!='schedule_sha256'}),'schedule semantic SHA differs')
    numeric=schedule['numeric_schedule'];need(numeric['schedule_sha256']==value_sha({k:v for k,v in numeric.items() if k!='schedule_sha256'}),'numeric schedule SHA differs')
    rows=numeric['rows'];mapping=files.doc(c['files']['donor_map']);donors=donor_indices(mapping,rows)
    manifest=files.doc(c['files']['train_cache_manifest']);batches=[json.loads(s) for s in files.get(manifest['batches']).read_text().splitlines()]
    metrics={a:[json.loads(s) for s in files.get(reports[a]['metrics']).read_text().splitlines()] for a in ARMS}
    metric_proof=check_metrics(metrics,donors,batches,[r['row_sha256'] for r in rows])
    initial={};final={};states={}
    for arm in ARMS:
        r=reports[arm];contract=contracts[arm]
        need(r['metrics']['rows']==1000 and r['last_metric']==metrics[arm][-1],'terminal metric differs')
        ip=files.get(r['initial_state'],2 << 20);fp=files.get(r['checkpoint'],2 << 20)
        need(ip.name=='initial_state.pt' and fp.name=='head_step1000.pt' and ip.parent==fp.parent,'terminal checkpoint locations differ')
        init=torch.load(ip,map_location='cpu',weights_only=True);saved=torch.load(fp,map_location='cpu',weights_only=True)
        typed(init,{'schema':'rphi.input_preservation_head_initial_state.v1','artifact_role':'initial_state_not_resume'},'initial artifact differs')
        typed(saved,{'schema':'rphi.input_preservation_head_checkpoint.v1','artifact_role':'research_final_head','arm':arm,
            'completed_step':1000,'row_cursor':16000,'quality_accepted':False},'final artifact differs')
        ih=state_check(init['state_dict'],torch);fh=state_check(saved['state_dict'],torch)
        need(ih==init['model_sha256']==saved['initial_state_sha256']==r['initial_state_sha256']==r['initial_state']['model_sha256'],'initial state hash differs')
        need(fh==saved['final_state_sha256']==r['final_state_sha256']==r['checkpoint']['model_sha256'] and fh!=ih
            and r['final_parameters_changed'] is True,'final state unchanged/hash differs')
        need(saved['training_contract']==contract and saved['execution_payload_sha256']==execution_sha(contract)
             and saved['binding']==r['binding'] and saved['runtime']==r['runtime']
             and saved['optimizer_specification']==contract['optimizer']
             and saved['model_config']==init['model_config']==contract['model_config'],'checkpoint contract/runtime/input binding differs')
        expected_versions=contract['runtime']['expected_versions']
        need(expected_versions['torch']==str(torch.__version__) and expected_versions['torch_npu'] is None,'saved/current Torch backend differs')
        for k in ('gate.net.2.weight','gate.net.2.bias'):need(bool((init['state_dict'][k]==0).all()),'initial output layer is not zero')
        initial[arm]={'state':ih,'rng':rng_check(init['rng_state'],torch)}
        final[arm]={'state':fh,'rng':rng_check(saved['rng_state'],torch)}
        changed=[k for k in SHAPES if tensor_sha(init['state_dict'][k])!=tensor_sha(saved['state_dict'][k])]
        states[arm]={'initial_state_sha256':ih,'final_state_sha256':fh,'parameters':1745,'FP32_parameter_tensors':4,
            'changed_parameter_tensor_names':changed,'optimizer':optimizer_check(saved['optimizer_state_dict'],saved['state_dict'],torch),
            'initial_rng_sha256':initial[arm]['rng'],'final_rng_sha256':final[arm]['rng']}
    need(initial['matched']==initial['shuffled'],'initial model/RNG bytes differ between arms')
    need(final['matched']['rng']==final['shuffled']['rng'],'final CPU RNG bytes differ between arms')
    need(len({r['source_id'] for r in rows})==19 and sum(e['recipient']['stratum'][2]==[0,0] for e in mapping['entries'])==1815,'source/00 coverage differs')
    need(reports['matched']['cache_load']['student_row_SHA']==value_sha([r['row_sha256'] for r in rows]),'loaded schedule row hash differs')
    need(sys.byteorder=='little' and reports['matched']['cache_load']['donor_index_array_sha256']==
        hashlib.sha256(struct.pack('<16000q',*donors)).hexdigest(),'loaded donor int64 index SHA differs')
    need('torch_npu' not in sys.modules,'NPU backend imported during audit')
    return {'actual_saved_states':states,'metrics':metric_proof,'initial_model_and_rng_exact_between_arms':True,
            'donor_rows_verified':16000,'source_count':len({r['source_id'] for r in rows}),
            'final_states_identical_between_arms':final['matched']['state']==final['shuffled']['state']}

def apply_limits():
    os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',ASCEND_RT_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    need(hasattr(os,'sched_getaffinity'),'execution requires Linux affinity support')
    cpus=sorted(os.sched_getaffinity(0));need(len(cpus)>=2,'two CPU cores unavailable');os.sched_setaffinity(0,cpus[:2])
    resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
    def stop(signum,frame):raise TimeoutError('audit wall timeout or termination signal')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.alarm(120)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    need(sha(a.config)==a.config_sha256,'config SHA differs');config=read(a.config)
    need(config['out']==str(a.out.resolve()) and not a.out.exists() and not a.out.is_symlink(),'fresh exact output required')
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'schema':'rphi.input_preservation_heads_actual_audit.v1','complete':False,'config_sha256':a.config_sha256,
        'source_sha256':sha(__file__),'scope':'saved_CPU_states_and_paired_metadata_only','model_calls':0,'optimizer_calls':0,
        'NPZ_reads':0,'quality_accepted':False,'resources':RESOURCE,
        'limitations':['No forward, optimizer replay, gradients, or source/cache NPZ were recomputed.',
            'Saved tensors, metric metadata and actual file hashes are independently read; operation counts and strict live reload are producer attestations.',
            'Shuffled label row indices are checked against the bound whole-map permutation; their numeric label batch SHA is logged, not recomputed without NPZ.',
            'Same saved initial states are byte verified; no seeded model is re-created. This is not a quality screen or training admission.']}
    try:
        if a.execute:apply_limits()
        files,reports,contracts=metadata(config)
        if a.execute:result['verification']=actual(files,reports,contracts)
        files.unchanged();need(sha(a.config)==a.config_sha256,'audit config changed')
        result.update(complete=True,status='saved_heads_and_paired_metadata_verified_no_quality_admission' if a.execute else 'metadata_only_no_checkpoint_loaded',
            inputs=files.used,checkpoint_loads=4 if a.execute else 0)
    except BaseException as e:result.update(status='failed_preserved_no_retry',error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.alarm(0);result['elapsed_seconds']=time.monotonic()-start
        result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        with (a.out/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('complete','status','error','elapsed_seconds')}));return 0 if result['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
