"""One bounded CPU audit after producer terminal; no retry/model actions."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from owned_process import supervise
from small_archive import package

SOURCES={'audit_terminal.py','audit_arrays.py','audit_checkpoint.py','adamw_equations.py',
         'finalize_once.py','small_archive.py','owned_process.py'}
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write_json(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def admission(root,own,path,expected_sha):
    root,own,path=(Path(p).resolve() for p in (root,own,path))
    if not own.is_relative_to(root) or path.parent!=own or sha_file(path)!=expected_sha:raise ValueError('admission SHA/path')
    c=json.loads(path.read_text())
    fields={'schema','run','prep','out','producer_sha256','trajectory_sha256','execution_contract_sha256',
            'queue_sha256','target_sha256','source_sha256','source_contract_sha256','producer_output_subdir'}
    if set(c)!=fields or c['schema']!='openh3.real_adamw_cpu_finalizer_admission.v1':raise ValueError('admission fields/schema')
    if c['producer_output_subdir']!='eval' or set(c['source_sha256'])!=SOURCES:raise ValueError('source/evidence scope')
    paths={}
    for k,parent in (('run','runs'),('prep','preparation'),('out','runs')):
        p=Path(c[k]);q=(root/p).resolve()
        if p.is_absolute() or len(p.parts)!=2 or p.parts[0]!=parent or not q.is_relative_to(root):raise ValueError('bounded paths')
        paths[k]=q
    if paths['run']==paths['out'] or paths['prep']==own:raise ValueError('output overlap')
    paths['evidence']=paths['run']/'eval'
    for h in [c[k] for k in fields if k.endswith('sha256') and k!='source_sha256']+list(c['source_sha256'].values()):
        if not isinstance(h,str) or len(h)!=64 or set(h)-set('0123456789abcdef'):raise ValueError('invalid SHA')
    return c,paths
def verify(c,p,own):
    pins={own/n:h for n,h in c['source_sha256'].items()}
    pins.update({p['prep']/'execution_contract.json':c['execution_contract_sha256'],p['prep']/'queue.json':c['queue_sha256'],
        p['prep']/'code/check_h3_real_adamw_update.py':c['producer_sha256'],
        p['prep']/'code/openh3_train_trajectory.py':c['trajectory_sha256']})
    for path,h in pins.items():
        if sha_file(path)!=h:raise ValueError('changed frozen source: '+str(path))
def wait_terminal(path,notify,*,max_wait=4200,poll=30,clock=time.monotonic,sleep=time.sleep):
    start=clock();polls=0
    while True:
        polls+=1
        try:
            raw=Path(path).read_bytes();r={'record':json.loads(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        except (FileNotFoundError,json.JSONDecodeError) as e:r={'record':None,'read_error':repr(e)}
        elapsed=clock()-start;notify(polls,elapsed,r)
        if r['record'] and r['record'].get('state') in ('completed','failed','not_launched'):return {'terminal':True,**r}
        if elapsed>=max_wait:return {'terminal':False,**r}
        sleep(min(poll,max_wait-elapsed))
def remaining(start,clock=time.monotonic):
    value=900-(clock()-start)
    if value<=0:raise TimeoutError('audit+archive total900s exhausted; no retry')
    return value
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    a=p.parse_args();root=a.root.resolve();own=Path(__file__).resolve().parent;c,paths=admission(root,own,a.config,a.config_sha256)
    out=paths['out'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,affinity)
    status={'schema':'openh3.real_adamw_cpu_finalizer.v1','state':'starting','complete':False,'pid':os.getpid(),
        'admission':c,'admission_sha256':a.config_sha256,'wait_limit_seconds':4200,'poll_seconds':30,
        'audit_archive_total_seconds':900,'audit_worker_wall_seconds':880,'termination_grace_seconds':10,
        'owned_group_RSS_limit_bytes':4<<30,'CPU_affinity':affinity,'audit_invocations':0,'NPU_calls':0,'model_job_interventions':0}
    def save():
        status['elapsed_seconds']=time.monotonic()-start;status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write_json(out/'finalize_status.json',status)
    terminal=False;audited=None;active_started=None;save()
    try:
        verify(c,paths,own);status['state']='waiting';save()
        def notify(n,e,r):status.update(polls=n,wait_seconds=e,last_queue_observation=r);save()
        obs=wait_terminal(paths['run']/'queue_status.json',notify);terminal=obs['terminal'];status['terminal_observation']=obs
        if not terminal:raise TimeoutError('wait4200 exhausted; producer untouched')
        active_started=time.monotonic();verify(c,paths,own)
        status.update(state='auditing',audit_invocations=1);save()
        env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        child={'complete':False,'producer_started':False}
        supervise([sys.executable,str(own/'audit_terminal.py'),'--root',str(root),'--config',str(a.config.resolve()),
            '--config-sha256',a.config_sha256,'--out',str(out/'terminal_audit.json')],root,env,out/'auditor.log',child,out/'audit_worker_status.json',
            wall=min(880,remaining(active_started)-10),rss_limit=4<<30,grace=10)
        status['worker']=child
        if child['state']!='worker_completed':raise RuntimeError('single CPU audit failed: '+str(child.get('error')))
        audited=json.loads((out/'terminal_audit.json').read_text());status['audit_sha256']=sha_file(out/'terminal_audit.json')
        status['numerical_acceptance']=audited['numerical_acceptance'];status['functional_deployment_ready']=audited.get('functional_deployment_ready',False)
        status['complete']=audited['complete'] is True and obs['record']['state']=='completed'
        if not status['complete']:status['error']='Producer/audit unsuccessful; no retry'
    except BaseException as e:status.update(complete=False,error=repr(e))
    status['state']='completed' if status['complete'] else 'failed';save();write_json(out/'finalize_receipt.json',status)
    # No active worker PG remains here. Alarm only interrupts bounded packaging.
    if active_started is None:active_started=time.monotonic()
    old=signal.getsignal(signal.SIGALRM)
    def deadline(*_):raise TimeoutError('total900s audit+archive deadline')
    try:
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,remaining(active_started))
        status['archive']=package(root,paths['prep'],paths['run'],own,out,terminal,audited,evidence=paths['evidence'])
    except BaseException as e:status.update(complete=False,state='failed',archive_error=repr(e))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    save();print(json.dumps(status));return 0 if status['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
