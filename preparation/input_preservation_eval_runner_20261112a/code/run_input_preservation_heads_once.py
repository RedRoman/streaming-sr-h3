"""One admitted CPU head arm; bounded owned PG, no eval/retry/device launch."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

SCHEMA='rphi.input_preservation_head_once.v1'
ENTRY='train_input_preservation_heads.py'
ENTRY_SHA='2df11a8563940aea729e2b3e4252f38515c0a66889caa2635debef72239e54cd'
SUPERVISOR_SHA='efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388'
RESOURCE={'wall_seconds':900,'CPU_threads':2,'rss_bytes':4<<30,'grace_seconds':30}


def need(ok,message):
    if not ok:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def typed(d,expected,message):need(all(type(d.get(k)) is type(v) and d[k]==v for k,v in expected.items()),message)


def validate(path):
    p=read(path);root=Path(p['root']);run=Path(p['run']);out=Path(p['out'])
    need(p['schema']==SCHEMA and root.is_absolute() and root==root.resolve() and root.is_dir()
         and p['mode'] in ('backend_smoke','research') and p['arm'] in ('matched','shuffled')
         and p['resource']==RESOURCE and p['physical_card'] is None,'invalid CPU head plan/resource')
    need(run.parent==root/'runs' and run==run.absolute() and not run.is_symlink() and out==run/'output','new isolated run required')
    pins=p['pins'];need(pins and all(Path(k).is_absolute() and sha(k)==v for k,v in pins.items()),'source/input pins differ')
    for k in (str(Path(__file__).resolve()),p['entry'],p['supervision_source'],p['execution_contract']):need(k in pins,'required launch pin missing')
    need(Path(p['entry']).name==ENTRY and pins[p['entry']]==ENTRY_SHA and sha(p['supervision_source'])==SUPERVISOR_SHA,'frozen entry/supervisor differs')
    need(Path(p['python']).is_absolute() and os.access(p['python'],os.X_OK),'absolute executable Python required')
    need(p['argv']==[p['python'],p['entry'],'--execution-contract',p['execution_contract'],'--out',p['out'],'--execute'],'exact head-only argv required')
    c=read(p['execution_contract']);need(c['schema']=='rphi.input_preservation_head_training_execution.v1'
        and c['root']==str(root) and c['mode']==p['mode'] and c['arm']==p['arm'],'training identity differs')
    need(c['resources']=={'wall_seconds':900,'CPU_cores':2,'RSS_limit_bytes':4<<30,'maximum_output_bytes':32<<20},'training resources differ')
    need(all(pins.get(k)==v for k,v in c['implementation_sha256'].items()) and c['implementation_sha256'].get(p['entry'])==ENTRY_SHA,'training dependencies not pinned')
    for item in [*c['files'].values(),*c['backend_smokes'].values(),c['admission']]:
        q=Path(item['path']);q=q if q.is_absolute() else root/q
        need(pins.get(str(q.resolve()))==item['sha256'],'training evidence/admission pin missing')
    numerical=Path(c['admission']['path']);numerical=numerical if numerical.is_absolute() else root/numerical
    a=read(numerical)
    typed(a,{'schema':'rphi.input_preservation_head_training_admission.v1','approved':True,'arm':p['arm'],'mode':p['mode'],
        'execution_payload_sha256':semantic({k:v for k,v in c.items() if k!='admission'}),'entry_sha256':ENTRY_SHA,'out':str(out)},'numerical training permission differs')
    need(Path(p['admission']).resolve()!=numerical.resolve(),'launch/numerical admissions must differ')
    a=read(p['admission']);typed(a,{'schema':'rphi.input_preservation_head_launch_admission.v1','approved':True,
        'scope':'one_CPU_input_preservation_head_arm_no_evaluation','arm':p['arm'],'mode':p['mode'],
        'plan_sha256':sha(path),'runner_sha256':sha(__file__),'execution_contract_sha256':sha(p['execution_contract']),
        'physical_card':None},'exact separate launch permission differs')
    return p


def terminal(p):
    out=Path(p['out']);r=read(out/'report.json');steps=1000 if p['mode']=='research' else 2
    typed(r,{'schema':'rphi.input_preservation_head_training.v1','complete':True,'mode':p['mode'],'arm':p['arm'],
        'status':'final_head_ready_for_bound_evaluation' if steps==1000 else 'CPU_backend_smoke_complete_not_research_resume',
        'completed_step':steps,'row_cursor':steps*16,'strict_checkpoint_reload':True,'NPU_calls':0,
        'new_restorer_forwards':0,'dev_label_arrays_read':0,'source_NPZ_reads':0,'resumed':False,
        'quality_accepted':False,'original_history_headroom_remains_failed':True},'head terminal incomplete/wrong scope')
    typed(r['counts'],{k:steps for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')},'head operation counts differ')
    c=read(p['execution_contract']);need(r['execution_payload_sha256']==semantic({k:v for k,v in c.items() if k!='admission'}),'terminal execution differs')
    typed(r['runtime'],{'device':'cpu','threads':2,'dtype':'float32','autocast':False,'grad_scaler':False},'wrong head runtime')
    need(r['runtime']['versions']==c['runtime']['expected_versions'] and r['runtime']['versions']['torch_npu'] is None,'head runtime versions differ')
    for name in ('checkpoint','initial_state','metrics'):
        b=r[name];q=Path(b['path']);need(q.resolve().parent==out.resolve() and not q.is_symlink() and sha(q)==b['sha256'],'terminal artifact differs')
    need(r['checkpoint']['artifact_role']==('research_final_head' if steps==1000 else 'backend_smoke_not_research_resume')
         and r['metrics']['rows']==steps and r['checkpoint']['model_sha256']==r['final_state_sha256'],'checkpoint role/state differs')
    need(sum(f.stat().st_size for f in out.iterdir() if f.is_file())<=32<<20,'actual small output exceeds32MiB')
    return {'report_sha256':sha(out/'report.json'),'checkpoint_sha256':r['checkpoint']['sha256'],
            'completed_step':steps,'actual_checkpoint_CPU_inventory_independently_recomputed':False}


def load_supervisor(path):
    spec=importlib.util.spec_from_file_location('owned_head_CPU_supervisor',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def resources(root):
    mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    disk=shutil.disk_usage(root).free;need(mem>=8<<30 and disk>=1<<30,'insufficient CPU head resources')
    return {'MemAvailable_bytes':mem,'disk_free_bytes':disk}


def run(path):
    p=validate(path);plan_sha=sha(path);root=Path(p['root']);runpath=Path(p['run']);runpath.mkdir(exist_ok=False)
    s=load_supervisor(p['supervision_source']);old=os.sched_getaffinity(0);sp=runpath/'status.json'
    status={'schema':'rphi.input_preservation_head_once_status.v1','state':'checking','complete':False,'producer_started':False,
        'mode':p['mode'],'arm':p['arm'],'plan_sha256':plan_sha,'resource':RESOURCE,'physical_card':None,'automatic_retry':False,
        'quality_accepted':False,'evaluation_launched':False,'RSS_scope':'sampled current owned-PG sum; not a kernel quota'}
    try:
        with ExitStack() as stack,s.signal_guard():
            lock=stack.enter_context((runpath/'status.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['resources']=resources(root);affinity=sorted(old)[:2];need(len(affinity)==2,'two CPU cores required')
            os.sched_setaffinity(0,set(affinity));status['CPU_affinity']=affinity;os.nice(10)
            env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'',
                 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
            s.supervise(p['argv'],root,env,runpath/'worker.log',status,sp,wall=900,rss_limit=4<<30,grace=30)
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
    r=run(a.plan);print(json.dumps({k:r.get(k) for k in ('complete','state','mode','arm','error')}));raise SystemExit(0 if r['complete'] else 2)
