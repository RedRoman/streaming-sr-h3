"""Linux owned-process CPU audit guard. No waits for or replays of the producer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RESOURCE={'CPU_threads':2,'total_wall_seconds':120,'worker_timeout_seconds':118,
          'termination_cleanup_seconds':2,'owned_process_group_RSS_limit_bytes':1<<30}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,row):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(p)
def members(pgid):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=pgid:continue
            rss=next((int(s.split()[1])*1024 for s in (p/'status').read_text().splitlines() if s.startswith('VmRSS:')),0)
            rows.append({'pid':int(p.name),'rss_bytes':rss})
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows
def stop_owned(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    end=time.monotonic()+2
    while time.monotonic()<end:
        proc.poll()
        if not members(proc.pid):return
        time.sleep(.05)
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait(timeout=2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True);ap.add_argument('--freeze-sha256',required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--expected-report-sha256',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();own=Path(__file__).resolve().parent
    if sys.platform!='linux' or not Path('/proc').is_dir():raise ValueError('actual guarded run requires Linux procfs')
    if a.freeze.resolve()!=own/'freeze.json':raise ValueError('freeze must be in this isolated snapshot')
    if sha(a.freeze)!=a.freeze_sha256:raise ValueError('freeze SHA differs')
    freeze=json.loads(a.freeze.read_text())
    if freeze['resource']!=RESOURCE:raise ValueError('resource budget differs')
    if set(freeze['files'])!={'audit_arrays.py','run_audit_once.py','small_archive.py','README.md','test_audit_arrays.py'}:raise ValueError('freeze inventory differs')
    for name,want in freeze['files'].items():
        if sha(own/name)!=want:raise ValueError('source pin differs: '+name)
    if sha(a.report)!=a.expected_report_sha256:raise ValueError('producer report SHA differs')
    a.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,str(own/'audit_arrays.py'),'--report',str(a.report.resolve()),
        '--expected-report-sha256',a.expected_report_sha256,'--out',str(a.out.resolve()/'arrays_audit')]
    affinity=sorted(os.sched_getaffinity(0))[:2]
    if not affinity:raise ValueError('no CPUs available')
    os.sched_setaffinity(0,set(affinity))
    env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    write(a.out/'invocation.json',{'command':command,'freeze_sha256':a.freeze_sha256,'report_sha256':a.expected_report_sha256,
        'source':str(own),'resource':RESOURCE,'cpu_affinity':affinity,'no_producer_replay':True})
    status={'schema':'rphi.history_utility_array_audit_supervision.v1','state':'starting','complete':False,
        'resource':RESOURCE,'peak_sampled_owned_group_RSS_bytes':0,'quality_verdict':'requires_review'}
    started=time.monotonic();proc=None
    def terminate(signum,*_):
        if signum==signal.SIGALRM and proc is not None and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        raise InterruptedError('audit supervisor interrupted or total120s deadline')
    old={s:signal.signal(s,terminate) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    signal.alarm(120)
    try:
        with (a.out/'worker.log').open('xb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            status.update(state='running',owned_pgid=proc.pid);write(a.out/'supervision.json',status)
            while proc.poll() is None:
                rss=sum(r['rss_bytes'] for r in members(proc.pid));status['peak_sampled_owned_group_RSS_bytes']=max(status['peak_sampled_owned_group_RSS_bytes'],rss)
                if rss>1<<30:raise MemoryError('owned audit group exceeded1GiB')
                if time.monotonic()-started>=118:raise TimeoutError('worker118s leaves2s cleanup within120s wall')
                time.sleep(.1)
            if members(proc.pid):raise RuntimeError('audit left owned descendants')
            if proc.returncode!=0:raise RuntimeError('array auditor failed; preserve its audit.json')
            audit=json.loads((a.out/'arrays_audit/audit.json').read_text())
            if not audit['audit_complete']:raise RuntimeError('incomplete array audit')
            status.update(state='completed',complete=True,producer_complete=audit['producer_complete'],audit_status=audit['status'])
    except BaseException as error:
        if proc is not None:stop_owned(proc)
        status.update(state='failed',error=repr(error))
    finally:
        for s,handler in old.items():
            if s!=signal.SIGALRM:signal.signal(s,handler)
        status.update(elapsed_seconds=time.monotonic()-started,returncode=proc.poll() if proc else None)
        write(a.out/'supervision.json',status)
    from small_archive import pack
    try:archive=pack(a.out,a.report,own)
    except BaseException as error:
        write(a.out/'archive_failure.json',{'error':repr(error),'producer_replayed':False});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old[signal.SIGALRM])
    print(json.dumps({'state':status['state'],'producer_complete':status.get('producer_complete'),'archive':archive}))
    return 0 if status['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
