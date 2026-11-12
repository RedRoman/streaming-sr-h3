"""One CPU development evaluation; execution and quality verdicts are separate."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PRIMITIVES_SHA='f8958a817af333ac9a8332a556abc2d0f8a63d0ee17d7c3e0a17a1a965bad0ec'
_base=Path(__file__).resolve().parent/'run_input_preservation_heads_once.py'
if hashlib.sha256(_base.read_bytes()).hexdigest()!=PRIMITIVES_SHA:raise RuntimeError('frozen CPU runner primitives changed')
_spec=importlib.util.spec_from_file_location('frozen_head_runner_primitives',_base)
B=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(B)
need,read,sha,typed,semantic=B.need,B.read,B.sha,B.typed,B.semantic

SCHEMA='rphi.input_preservation_evaluation_once.v1'
ENTRY_SHA='25cdeec76d1472fe1a590d27ba0d65553a67ce9fada3d329210874d7bab28e39'
TRAINER_SHA='2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd'
RESOURCE={'wall_seconds':1800,'CPU_threads':2,'rss_bytes':4<<30,'grace_seconds':30}
ARMS=('matched','shuffled')
COUNTS={'head_forward_attempts':360,'head_forwards_completed':360,'head_row_attempts':5760,'head_rows_completed':5760,
        'LPIPS_forward_attempts':720,'LPIPS_forwards_completed':720,'LPIPS_row_attempts':11520,'LPIPS_rows_completed':11520,
        'restorer_forwards':0,'backwards':0,'optimizer_updates':0}


def bound(item,root,pins):
    q=Path(item['path']);q=q if q.is_absolute() else root/q;q=q.resolve()
    need(pins.get(str(q))==item['sha256']==sha(q),'evidence/permission not exactly pinned')
    return q


def completed_heads(p,c):
    root=Path(p['root']);common=None;initial=None
    need(set(p['training_reports'])==set(p['training_execution_contracts'])==set(ARMS),'both completed head arms required')
    for arm in ARMS:
        report=read(bound(p['training_reports'][arm],root,p['pins']))
        tc=read(bound(p['training_execution_contracts'][arm],root,p['pins']))
        typed(report,{'schema':'rphi.input_preservation_head_training.v1','complete':True,'mode':'research','arm':arm,
            'status':'final_head_ready_for_bound_evaluation','completed_step':1000,'row_cursor':16000,
            'strict_checkpoint_reload':True,'NPU_calls':0,'new_restorer_forwards':0,'dev_label_arrays_read':0,
            'source_NPZ_reads':0,'resumed':False,'quality_accepted':False},'not a complete1000 CPU training report')
        typed(report['counts'],{k:1000 for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')},'training counts incomplete')
        need(tc['schema']=='rphi.input_preservation_head_training_execution.v1' and tc['mode']=='research'
             and tc['arm']==arm and tc['steps']==1000 and tc['batch_size']==16
             and report['execution_payload_sha256']==semantic({k:v for k,v in tc.items() if k!='admission'}), 'training execution binding differs')
        need({Path(k).name:v for k,v in tc['implementation_sha256'].items()}.get('train_input_preservation_heads.py')==TRAINER_SHA,
             'training used another source')
        numerical={k:v for k,v in tc.items() if k not in ('arm','admission')}
        if common is None:common=numerical;initial=report['initial_state_sha256']
        need(numerical==common and report['initial_state_sha256']==initial and bool(initial),'head arms not matched')
        for key in ('proposal','screen','metric_readiness','train_cache_manifest','train_cache_audit','dev_cache_manifest','dev_cache_audit'):
            need(tc['files'][key]==report['binding']['files'][key]==c['files'][key],'training/eval input binding differs')
        ck=report['checkpoint'];bound(c['files'][arm+'_checkpoint'],root,p['pins'])
        need({k:ck[k] for k in ('path','sha256')}==c['files'][arm+'_checkpoint']
             and ck['artifact_role']=='research_final_head' and ck['model_sha256']==report['final_state_sha256'],
             'evaluation checkpoint not the completed final1000 artifact')


def validate(path):
    p=read(path);root=Path(p['root']);run=Path(p['run']);out=Path(p['out']);pins=p['pins']
    need(p['schema']==SCHEMA and root.is_absolute() and root==root.resolve() and root.is_dir()
         and p['resource']==RESOURCE and p['physical_card'] is None,'CPU evaluation plan/resource differs')
    need(run.parent==root/'runs' and run==run.absolute() and not run.is_symlink() and out==run/'output','fresh isolated eval run required')
    need(pins and all(Path(k).is_absolute() and sha(k)==v for k,v in pins.items()),'source/input pins differ')
    for q in (str(Path(__file__).resolve()),str(_base),p['entry'],p['supervision_source'],p['execution_contract']):need(q in pins,'required source pin missing')
    need(Path(p['entry']).name=='eval_input_preservation_heads.py' and pins[p['entry']]==ENTRY_SHA
         and sha(p['supervision_source'])==B.SUPERVISOR_SHA and sha(_base)==PRIMITIVES_SHA,'frozen eval/owned supervisor differs')
    need(Path(p['python']).is_absolute() and os.access(p['python'],os.X_OK),'absolute executable Python required')
    need(p['argv']==[p['python'],p['entry'],'--execution-contract',p['execution_contract'],'--out',p['out'],'--execute'],'exact eval-only argv required')
    c=read(p['execution_contract']);need(c['schema']=='rphi.input_preservation_heads_evaluation_execution.v1' and c['root']==str(root)
         and c['resources']=={'wall_seconds':1800,'CPU_cores':2,'RSS_limit_bytes':4<<30,'max_output_bytes':256<<20},'numerical eval contract differs')
    need(all(pins.get(k)==v for k,v in c['implementation_sha256'].items()) and c['implementation_sha256'].get(p['entry'])==ENTRY_SHA,'eval dependencies not pinned')
    for item in c['files'].values():bound(item,root,pins)
    ready=read(bound(c['files']['metric_readiness'],root,pins))
    typed(ready,{'schema':'rphi.input_preservation_metric_readiness.v1','complete':True,'frozen':True},'metric readiness incomplete')
    need(ready['evaluator']=={'path':p['entry'],'sha256':ENTRY_SHA} and all(pins.get(k)==v for k,v in ready['implementation_sha256'].items()),'metric source pins differ')
    for item in ready['LPIPS']['weights']:bound(item,root,pins)
    typed(ready['LPIPS'],{'available':True,'network':'alex','version':'0.1','downloads':False},'complete local LPIPS required')
    completed_heads(p,c)
    numerical=bound(c['admission'],root,pins)
    typed(read(numerical),{'schema':'rphi.input_preservation_heads_evaluation_admission.v1','approved':True,
        'execution_sha256':semantic({k:v for k,v in c.items() if k!='admission'}),'entry_sha256':ENTRY_SHA,'out':str(out)},'numerical eval permission differs')
    need(Path(p['admission']).resolve()!=numerical,'launch/numerical admission must differ')
    typed(read(p['admission']),{'schema':'rphi.input_preservation_evaluation_launch_admission.v1','approved':True,
        'scope':'one_CPU_fixed_development_evaluation_no_training','plan_sha256':sha(path),'runner_sha256':sha(__file__),
        'execution_contract_sha256':sha(p['execution_contract']),'physical_card':None},'exact evaluation launch permission differs')
    return p


def terminal(p):
    out=Path(p['out']);r=read(out/'report.json');c=read(p['execution_contract'])
    typed(r,{'schema':'rphi.input_preservation_heads_evaluation.v1','complete':True,'status':'complete_fixed_development_screen',
        'completed_rows':2880,'original_history_headroom_remains_failed':True,'head_or_restorer_training_admitted':False,
        'quality_proven':False,'LPIPS_frozen_exact':True},'evaluation incomplete/wrong scope')
    typed(r['counts'],COUNTS,'actual evaluation budget incomplete or exceeded')
    typed(r['observed_model_invocations'],{'matched':180,'shuffled':180,'LPIPS':720},'observed model-call count differs')
    need(r['execution_contract']==c and r['execution_contract_file_sha256']==sha(p['execution_contract']),'terminal execution differs')
    ready_path=Path(c['files']['metric_readiness']['path'])
    ready=read(ready_path if ready_path.is_absolute() else Path(p['root'])/ready_path)
    need(r['runtime']==ready['runtime']['expected_versions'] and r['runtime']['torch_npu'] is None,'actual CPU eval runtime differs')
    typed(r['lpips'],{'available':True,'network':'alex','device':'cpu','downloads':False},'actual LPIPS missing')
    need(set(r['frozen_checks'])==set(ARMS),'missing frozen head check')
    for v in r['frozen_checks'].values():typed(v,{'state_dict_exact':True,'versions_exact':True,'eval_frozen_no_grads':True},'head changed during eval')
    for arm in ARMS:
        need(r['checkpoints'][arm]['file']==c['files'][arm+'_checkpoint'] and r['checkpoints'][arm]['completed_step']==1000,'terminal evaluated other checkpoint')
    for name,n in [('records',2880),('pairs',2850),('batches',180)]:
        b=r[name];q=Path(b['path']);need(q.resolve().parent==out.resolve() and not q.is_symlink()
             and q.stat().st_size==b['bytes'] and sha(q)==b['sha256'],'eval index artifact differs')
        with q.open() as f:need(sum(1 for line in f if line.strip())==n,'actual output index row count differs')
    need(r['temporal']['pairs']==2850,'temporal support incomplete')
    screen=r['screen'];typed(screen,{'required_metric_coverage_complete':True,'quality_proven':False,'independent_test':False,
        'automatic_training_or_fullframe_admission':False},'quality interpretation/metric coverage differs')
    need(screen['status'] in ('failed_dev_screen','pending_review','eligible_for_next_bounded_fullframe_validation'), 'unknown scientific verdict')
    need(sum(q.stat().st_size for q in out.rglob('*') if q.is_file())<=256<<20,'output byte budget exceeded')
    return {'report_sha256':sha(out/'report.json'),'scientific_verdict':screen['status'],
        'quality_gate_pass_not_required_for_execution_completion':True,'actual_RGB_arrays_independently_recomputed':False}


def run(path):
    p=validate(path);plan_sha=sha(path);root=Path(p['root']);runpath=Path(p['run']);runpath.mkdir(exist_ok=False)
    s=B.load_supervisor(p['supervision_source']);old=os.sched_getaffinity(0);sp=runpath/'status.json'
    status={'schema':'rphi.input_preservation_evaluation_once_status.v1','state':'checking','complete':False,'producer_started':False,
        'plan_sha256':plan_sha,'resource':RESOURCE,'physical_card':None,'automatic_retry':False,'quality_proven':False,
        'training_or_fullframe_launched':False,'RSS_scope':'sampled current owned-PG sum, not kernel quota'}
    try:
        with ExitStack() as stack,s.signal_guard():
            lock=stack.enter_context((runpath/'status.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['resources']=B.resources(root);affinity=sorted(old)[:2];need(len(affinity)==2,'two CPU cores required')
            os.sched_setaffinity(0,set(affinity));status['CPU_affinity']=affinity;os.nice(10)
            env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'',
                 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
            s.supervise(p['argv'],root,env,runpath/'worker.log',status,sp,wall=1800,rss_limit=4<<30,grace=30)
            if status['state']=='worker_completed':status.update(terminal(p),state='completed',complete=True)
    except BaseException as e:status.update(state='failed' if status['producer_started'] else 'not_launched',complete=False,error={'type':type(e).__name__,'message':str(e)})
    finally:
        os.sched_setaffinity(0,old)
        try:need(sha(path)==plan_sha,'plan mutated');validate(path);status['pins_unchanged']=True
        except BaseException as e:status.update(state='failed',complete=False,pins_unchanged=False,error={'type':type(e).__name__,'message':str(e)})
        s.write(sp,status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan');p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
    if a.inspect_only:validate(a.plan);print(json.dumps({'valid':True,'worker_started':False}));raise SystemExit(0)
    r=run(a.plan);print(json.dumps({k:r.get(k) for k in ('complete','state','scientific_verdict','error')}));raise SystemExit(0 if r['complete'] else 2)
