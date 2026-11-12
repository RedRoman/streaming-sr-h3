"""Explicitly admitted one-shot teacher1000 terminal CPU audit, never training."""
import argparse,hashlib,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
from contextlib import contextmanager

TERMINAL={'completed','failed','not_launched'}

def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def value_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def bound(b):need(sha(b['path'])==b['sha256'],'bound bytes differ: '+b['path']);return Path(b['path'])

def validate(c):
    need(c['schema']=='rphi.history_utility_teacher_terminal_watch.v1' and c['kind']=='teacher1000','only explicitly bound teacher1000 supported')
    need(c['resources']=={'wait_seconds':7500,'poll_seconds':30,'audit_seconds':120,'cleanup_seconds':10,'CPU':2,'AS_bytes':4<<30},'terminal budget changed')
    need(c['source_pins'].get(str(Path(__file__).resolve()))==sha(__file__),'finalizer self pin missing')
    for p,h in c['source_pins'].items():need(sha(p)==h,'source pin changed')
    auditor=bound(c['auditor']);need(c['source_pins'].get(str(auditor.resolve()))==c['auditor']['sha256'],'auditor unpinned')
    plan=read(bound(c['producer_plan']));contract=read(bound(c['producer_execution_contract']))
    need(plan['kind']=='teacher1000' and plan['execution_contract']==c['producer_execution_contract']['path'],'wrong producer scope')
    need(Path(c['producer_status'])==Path(plan['run'])/'status.json' and Path(c['report'])==Path(plan['out'])/'report.json','producer terminal paths differ')
    need(contract['steps']==1000 and contract['phase']=='teacher' and contract.get('resume') is None,'fresh1000-only audit contract')
    need(plan['pins'].get(c['producer_execution_contract']['path'])==c['producer_execution_contract']['sha256'],'execution file not producer-pinned')
    return plan

def wait_terminal(path,seconds=7500,poll=30,clock=time.monotonic,sleep=time.sleep):
    start=clock();reads=0
    while True:
        if Path(path).exists():
            s=read(path);reads+=1
            if s.get('state') in TERMINAL:return s,clock()-start,reads
        left=seconds-(clock()-start)
        if left<=0:raise TimeoutError('terminal wait exhausted; no audit/model replay')
        sleep(min(poll,left))

@contextmanager
def signals():
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(signum,_):raise InterruptedError('terminal finalizer signal '+str(signum))
    try:
        for s in previous:signal.signal(s,stop)
        yield
    finally:
        for s,h in previous.items():signal.signal(s,h)

def kill_owned(proc,grace=10):
    if proc is None or proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=2)

def pack(out,paths):
    files={};total=0
    need(len(paths)<=100 and 'MANIFEST.json' not in paths,'archive inventory exceeds bound or shadows manifest')
    for name,p in paths.items():
        p=Path(p)
        if not p.exists():continue
        need(p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log','.py','.md'),'unsafe/non-small evidence')
        need('..' not in Path(name).parts and not Path(name).is_absolute(),'unsafe archive member')
        total+=p.stat().st_size;need(total<=16<<20,'small archive cap exceeded')
        files[name]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    manifest={'schema':'rphi.teacher_checkpoint_small_archive.v1','files':[{k:v for k,v in r.items() if k!='path'}|{'path':n} for n,r in files.items()],'checkpoint_or_arrays_included':False}
    mp=out/'MANIFEST.json'
    with mp.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    ap=out/'evidence.tar.gz'
    with ap.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            t.add(mp,arcname='MANIFEST.json',recursive=False)
            for n,r in files.items():need(sha(r['path'])==r['sha256'],'small file changed during pack');t.add(r['path'],arcname=n,recursive=False)
    with tarfile.open(ap) as t:
        need(len(t.getmembers())==len(files)+1,'member count differs')
        for n,r in files.items():
            m=t.getmember(n);need(m.isfile() and m.size==r['bytes'] and hashlib.sha256(t.extractfile(m).read()).hexdigest()==r['sha256'],'archive member differs')
    return {'path':str(ap),'bytes':ap.stat().st_size,'sha256':sha(ap),'manifest_sha256':sha(mp),'payloads':len(files)}

def finalize(config_path,c):
    plan=validate(c);ad=read(bound(c['admission']));payload=value_sha({k:v for k,v in c.items() if k!='admission'})
    need(ad['approved'] is True and ad['scope']=='teacher_final_checkpoint_audit_only' and ad['execution_payload_sha256']==payload and ad['finalizer_sha256']==sha(__file__),'separate finalizer admission required')
    out=Path(c['out']);need(not out.exists() and not out.is_symlink(),'one-shot finalizer output exists');out.mkdir(parents=True,exist_ok=False)
    status={'schema':'rphi.teacher_checkpoint_terminal_supervision.v1','state':'waiting','complete':False,'audit_invocations':0,'expected_step':1000,'automatic_training_or_evaluation':False,'config_sha256':sha(config_path),'execution_payload_sha256':payload};write(out/'status.json',status)
    proc=None
    try:
        with signals():
            terminal,waited,reads=wait_terminal(c['producer_status']);status.update(wait_seconds=waited,status_reads=reads)
            (out/'producer_status.json').write_bytes(Path(c['producer_status']).read_bytes())
            need(terminal.get('state')=='completed' and terminal.get('complete') is True and terminal.get('pins_unchanged') is True,'producer failed/not_launched/incomplete; no checkpoint acceptance')
            need(terminal['plan_sha256']==c['producer_plan']['sha256'],'producer used wrong plan')
            report_path=Path(c['report']);need(sha(report_path)==terminal['report_sha256'],'producer report SHA differs')
            validate(c);status.update(state='auditing',audit_invocations=1,report_sha256=terminal['report_sha256']);write(out/'status.json',status)
            env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
            argv=[c['python'],c['auditor']['path'],'--report',str(report_path),'--report-sha256',terminal['report_sha256'],'--kind','teacher1000','--out',str(out/'audit.json')]
            with (out/'audit.log').open('x') as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                status['owned_audit_PID']=proc.pid;write(out/'status.json',status)
                try:rc=proc.wait(timeout=120)
                finally:kill_owned(proc,10)
                proc=None
            need(rc==0 and read(out/'audit.json')['complete'] is True,'CPU checkpoint audit failed')
            validate(c);status.update(state='completed',complete=True,audit_sha256=sha(out/'audit.json'),scope='saved teacher1000 CPU audit only; no headroom/quality acceptance');write(out/'status.json',status)
    except BaseException as e:
        kill_owned(proc,10);proc=None;status.update(state='failed',complete=False,error={'type':type(e).__name__,'message':str(e)});write(out/'status.json',status)
    paths={'status.json':out/'status.json','producer_status.json':out/'producer_status.json','audit.json':out/'audit.json','audit.log':out/'audit.log','config.json':config_path,'admission.json':c['admission']['path'],'producer/report.json':c['report'],'producer/metrics.jsonl':Path(c['report']).parent/'metrics.jsonl','producer/failure.json':Path(c['report']).parent/'failure.json','producer_plan.json':c['producer_plan']['path'],'producer_execution_contract.json':c['producer_execution_contract']['path']}
    for p in c['source_pins']:
        name='source/'+Path(p).name
        need(name not in paths,'colliding source basename')
        paths[name]=p
    try:
        receipt=pack(out,paths);write(out/'archive_receipt.json',receipt)
    except BaseException as e:status.update(state='failed',complete=False,archive_error={'type':type(e).__name__,'message':str(e)});write(out/'status.json',status)
    return status

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();c=read(a.config)
    if not a.execute:validate(c);print(json.dumps({'prepared':True,'producer_not_polled':True,'audit_invocations':0}));return
    r=finalize(a.config,c);print(json.dumps(r));raise SystemExit(0 if r['complete'] else 1)
if __name__=='__main__':main()
