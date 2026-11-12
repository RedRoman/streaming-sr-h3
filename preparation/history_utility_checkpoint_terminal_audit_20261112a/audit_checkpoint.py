"""CPU-only saved P1 checkpoint audit; no model construction/forward or optimizer."""
import argparse,hashlib,json,math,os,resource,signal,sys,time
from pathlib import Path
BASE_SHA='61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617'
TRAINER_SHA='3cc874cf5dbdb9859f504ca24ef591eaa4a06e46ea17494563d327748cfb76ed'
BACKEND_SHA='0a48f122891e4ec898b4998c94424976729e6948c481f2f63a88eb1fb9f5945f'

def need(x,msg):
    if not x:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def vsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def bound(b,root):
    p=Path(b['path']);p=p if p.is_absolute() else Path(root)/p
    need(sha(p)==b['sha256'],'bound file differs: '+str(p));return p

def tensor_sha(t):
    t=t.detach().cpu().contiguous();h=hashlib.sha256(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes());return h.hexdigest()
def state_sha(state):return vsha({k:tensor_sha(v) for k,v in sorted(state.items())})
def lr(step):return 1e-4*min(1.,step/50)*(.2+.8*.5*(1+math.cos(math.pi*max(0,step-50)/950)))

def check_tensor_state(state,baseline,step,spec):
    """Validate actual serialized tensors; only CPU Tensor operations."""
    import torch
    model=state['model_state'];need(set(model)==set(baseline),'strict model key mismatch')
    names=state['parameter_names'];need(len(names)==len(model) and set(names)==set(model) and len(names)==len(set(names)),'parameter inventory differs')
    need(state['model_sha256']==state_sha(model),'model content SHA differs')
    changed=[];delta2=0.;elements=0
    for name,t in model.items():
        b=baseline[name]
        need(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and t.shape==b.shape and bool(torch.isfinite(t).all()),'model shape/dtype/device/finite differs: '+name)
        elements+=t.numel()
        if not torch.equal(t,b):changed.append(name);delta2+=float((t.double()-b.double()).square().sum())
    need(changed,'actual saved model is unchanged from fresh inherited baseline')
    opt=state['optimizer_state'];need(set(opt)=={'state','param_groups'} and len(opt['param_groups'])==1,'optimizer schema differs')
    g=opt['param_groups'][0];ids=g['params'];need(len(ids)==len(names) and len(set(ids))==len(ids) and set(opt['state'])==set(ids),'optimizer state/parameter count differs')
    need(g['lr']==lr(step),'completed-step learning rate differs')
    for k in ('eps','weight_decay','amsgrad','foreach','fused'):need(g[k]==spec[k],'optimizer option differs: '+k)
    need(tuple(g['betas'])==tuple(spec['betas']) and not g.get('maximize',False) and not g.get('capturable',False) and not g.get('differentiable',False),'optimizer math differs')
    moment_nonzero=0
    for name,pid in zip(names,ids):
        s=opt['state'][pid];need(set(s)=={'step','exp_avg','exp_avg_sq'},'optimizer state keys differ')
        st=s['step'];need(isinstance(st,torch.Tensor) and st.device.type=='cpu' and st.dtype==torch.float32 and st.numel()==1 and bool(torch.isfinite(st).all()) and float(st)==step,'optimizer state step differs')
        for k in ('exp_avg','exp_avg_sq'):
            t=s[k];need(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and t.shape==model[name].shape and bool(torch.isfinite(t).all()),'moment shape/dtype/finite differs')
            if k=='exp_avg_sq':need(bool((t>=0).all()),'negative second moment')
            moment_nonzero+=int(bool((t!=0).any()))
    need(moment_nonzero>0,'all actual optimizer moments are zero')
    return {'model_parameters':len(names),'model_elements':elements,'optimizer_parameter_states':len(ids),'optimizer_tensors':3*len(ids),'step':step,'all_model_and_optimizer_tensors_finite_FP32':True,'saved_model_sha256':state['model_sha256'],'changed_model_tensors':len(changed),'baseline_delta_L2':math.sqrt(delta2),'nonzero_moment_tensors':moment_nonzero}


def audit(report_path,expected,kind):
    import torch
    need(sha(report_path)==expected,'report bytes differ');report=read(report_path)
    step=2 if kind=='backend2' else 1000
    need(report['complete'] is True,'producer failed or incomplete')
    out=Path(report_path).parent; execution=report['execution_contract']
    if kind=='backend2':
        need(report['schema']=='rphi.history_utility_backend_smoke.v1' and report['status']=='passed_two_step_backend_only_not_research_initialization','backend scope differs')
        c=read(bound(execution['teacher_contract'],execution['evidence_root']))
        need(execution['entry']['sha256']==BACKEND_SHA and execution['trainer']['sha256']==TRAINER_SHA,'backend/trainer source differs')
        bound(execution['entry'],execution['evidence_root']);bound(execution['trainer'],execution['evidence_root'])
        need(report['actual']=={'teacher_forwards':2,'teacher_backwards':2,'teacher_optimizer_updates':2,'save':1,'clear_reload':1},'actual two-step budget differs')
        need(report['operation_attempts']=={'forward':2,'backward':2,'optimizer':2},'backend attempted budget differs')
        need(report['strict_model_optimizer_RNG_reload'] is True and report['actual_final_step']==2 and report['research_updates']==0 and report['checkpoint_eligible_as_research_warmstart'] is False,'strict reload or research isolation missing')
        cp=report['checkpoint'];metrics=report['step_metrics'];needed_rows=32
    else:
        need(report['schema']=='rphi.history_utility_teacher_run.v1' and report['status']=='teacher1000_complete_pending_headroom','teacher terminal scope differs')
        c=execution;need(report['start_step']==0 and c.get('resume') is None,'this auditor contract accepts one fresh1000-step run only')
        need(report['completed_step']==1000 and report['optimizer_updates_this_run']==1000 and report['actual_completed_optimizer_updates_this_run']==1000,'teacher update budget differs')
        need(report['operation_attempts']==report['operation_completions']=={'forward':1000,'backward':1000,'optimizer':1000},'teacher operation counts differ')
        need(report['final_frozen'] is True and report['headroom_accepted'] is False and report['student_training_admitted'] is False,'final teacher scope/freeze differs')
        cp=report['checkpoints'][-1];need(cp['step']==1000 and cp['final'] is True,'final checkpoint boundary differs')
        metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines() if line.strip()];needed_rows=16000
    root=c['evidence_root'];need(c['steps']==1000 and c['batch_size']==16 and c['baseline_checkpoint']['sha256']==BASE_SHA,'teacher numerical contract differs')
    for p,digest in c['implementation_sha256'].items():need(sha(Path(p) if Path(p).is_absolute() else Path(root)/p)==digest,'implementation pin differs')
    need(TRAINER_SHA in c['implementation_sha256'].values(),'trainer source unpinned')
    need(report['baseline_file_unchanged'] is True,'producer reports changed baseline')
    runtime=report['runtime'];need(runtime['dtype']=='float32' and runtime['autocast'] is False,'producer precision differs')
    normalize=lambda x:x.split('+')[0] if isinstance(x,str) and c['runtime']['version_match']=='ignore_local_build_suffix' else x
    need({k:normalize(v) for k,v in runtime['versions'].items()}=={k:normalize(v) for k,v in c['runtime']['expected_versions'].items()},'runtime version receipt differs')
    schedule_path=bound(c['files']['teacher_schedule'],root);s=read(schedule_path)
    need(s['schedule_sha256']==vsha({k:v for k,v in s.items() if k!='schedule_sha256'}),'teacher schedule digest differs')
    rows=s['numeric_schedule']['rows'];need(len(rows)==16000,'teacher plan row count differs')
    for i,r in enumerate(rows[:needed_rows]):need(r['row_index']==i and r['row_sha256']==vsha({k:v for k,v in r.items() if k!='row_sha256'}) and r['split']=='train','scheduled row differs')
    if kind=='backend2':need(report['row_sha256']==[r['row_sha256'] for r in rows[:32]],'actual first32 input row bindings differ')
    need(len(metrics)==step,'step metric count differs')
    for i,m in enumerate(metrics,1):
        need(m['step']==i and m['row_end_exclusive']==16*i and m['lr']==lr(i) and m['lr_hex']==lr(i).hex(),'step/LR/cursor differs')
        need(all(not isinstance(v,(int,float)) or math.isfinite(v) for v in m.values()),'nonfinite metric')
    need(metrics[-1]['row_end_exclusive']==needed_rows,'final row cursor differs')
    checkpoint_path=Path(cp['path']);need(checkpoint_path.resolve().parent==out.resolve() and sha(checkpoint_path)==cp['sha256'],'checkpoint path/bytes differ')
    baseline_path=bound(c['baseline_checkpoint'],root);base=torch.load(baseline_path,map_location='cpu',weights_only=True)
    need(base['schema']=='rphi.checkpoint.v1' and base['step']==3000 and base['model_config']==c['model_config'],'actual baseline config differs')
    initial={k:v.detach().clone() for k,v in base['state_dict'].items()};w=initial['backbone.head.weight']
    need(w.dtype==torch.float32 and w.shape[1]==9,'old head geometry differs');initial['backbone.head.weight']=torch.cat((w,w.new_zeros(w.shape[0],2,3,3)),1)
    initial_sha=state_sha(initial);state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    need(state['schema']=='rphi.history_utility_teacher_checkpoint.v1' and state['phase']=='teacher' and state['step']==step and state['next_row_index']==step*16 and state['final']==(step==1000),'saved checkpoint cursor/scope differs')
    need(state['optimizer_specification']==c['optimizer'] and state['runtime']==runtime,'saved optimizer/runtime contract differs')
    expected_binding={'training_identity_sha256':vsha({k:v for k,v in c.items() if k not in ('admission','resume')}),'baseline_checkpoint_sha256':BASE_SHA,**report['data'],'initial_teacher_sha256':initial_sha}
    if kind=='backend2':expected_binding['artifact_role']='backend_two_step_smoke_never_research_resume'
    else:expected_binding['cohort_sha256']=c['cohort']['sha256']
    need(state['binding']==expected_binding,'checkpoint identity/data binding differs')
    if kind=='teacher1000':need(report['binding']==expected_binding,'teacher live binding differs')
    checks=check_tensor_state(state,initial,step,c['optimizer'])
    expected_model_sha=report['updated_model_sha256'] if kind=='backend2' else report['final_model_sha256']
    need(checks['saved_model_sha256']==expected_model_sha==report['final_observed_model_sha256']==cp['model_sha256'],'report/saved/reloaded model fingerprints differ')
    if kind=='backend2':need(report['initial_model_sha256']==initial_sha,'initial actual warmstart differs')
    rng=state['rng_state'];need(set(rng)=={'python','numpy','torch_cpu','torch_device'},'RNG record incomplete')
    need(rng['torch_cpu'].device.type=='cpu' and rng['torch_cpu'].dtype==torch.uint8 and rng['torch_cpu'].numel()>0,'CPU RNG record malformed')
    need(not any(n.startswith('torch_npu') for n in sys.modules),'NPU extension was imported')
    return {'schema':'rphi.history_utility_saved_checkpoint_independent_audit.v1','complete':True,'kind':kind,'report_sha256':expected,'checkpoint_sha256':cp['sha256'],'baseline_sha256':BASE_SHA,'initial_expanded_baseline_state_sha256':initial_sha,'actual_checkpoint_tensor_checks':checks,'actual_schedule_row_hashes_checked':needed_rows,'operation_counts_match_fixed_budget':True,'strict_reload':'producer live clear/reload attestation plus independent saved tensor/optimizer consistency; no forward replay','frozen_scope':'For1000-step teacher, eval/requires_grad/grad-clear flags are producer lifecycle attestations, not inferred from detached checkpoint tensors.','actual_NPZ_reads':0,'model_constructions':0,'model_forwards':0,'optimizer_steps':0,'NPU_imported':False,'headroom_or_quality_accepted':False,'scope':'Actual CPU serialized tensor/state proof only; does not prove exact future resumed trajectory or restoration headroom.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--kind',choices=['backend2','teacher1000'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);need(not out.exists() and not out.is_symlink(),'fresh audit only');out.parent.mkdir(parents=True,exist_ok=True)
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    def timeout(*_):raise TimeoutError('CPU checkpoint audit120seconds exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120);start=time.monotonic()
    try:
        import torch
        torch.set_num_threads(1);r=audit(a.report,a.report_sha256,a.kind)
    except BaseException as e:r={'complete':False,'kind':a.kind,'report_sha256':a.report_sha256,'error':{'type':type(e).__name__,'message':str(e)}}
    finally:signal.alarm(0)
    r.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_sha256=sha(__file__),resources={'CPU':2,'AS_bytes':4<<30,'wall_seconds':120})
    with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r));raise SystemExit(0 if r['complete'] else 1)
if __name__=='__main__':main()
