"""Owned bounded CPU validation and one real-checkpoint integration; no retries."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def rss(pgid):
    total=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid:total+=next((int(x.split()[1])*1024 for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return total

def bounded(cmd,log,env,seconds,memory):
    with log.open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();peak=0;reason=None
    try:
        while p.poll() is None:
            used=rss(p.pid);peak=max(peak,used)
            if used>memory:reason='owned_group_RSS_exceeded';break
            if time.monotonic()-start>seconds:reason='wall_timeout';break
            time.sleep(.1)
    finally:
        if p.poll() is None or rss(p.pid):
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            stop=time.monotonic()+5
            while (p.poll() is None or rss(p.pid)) and time.monotonic()<stop:time.sleep(.1)
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=5)
    return {'command':cmd,'pid':p.pid,'returncode':p.poll(),'stop_reason':reason,'elapsed_seconds':time.monotonic()-start,
        'peak_owned_group_RSS_bytes':peak,'wall_seconds':seconds,'grace_seconds':5,'rss_limit_bytes':memory,'log_sha256':sha(log)}

def main():
    config_path=Path(sys.argv[1]);expected=sys.argv[2]
    assert sha(config_path)==expected,'freeze changed'
    c=json.loads(config_path.read_text());assert c['schema']=='rphi.history_utility_integration_cpu_freeze.v1'
    assert all(sha(p)==h for p,h in c['pins'].items()),'pin mismatch'
    assert c['pins'][str(Path(__file__).resolve())]==sha(__file__),'runner not pinned'
    out=Path(c['supervisor_out']);out.mkdir(parents=True,exist_ok=False)
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,set(affinity))
    env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    r={'schema':c['schema'],'complete':False,'freeze_sha256':expected,'cpu_affinity':affinity,'status':'running',
        'pins_before':c['pins'],'checkpoint_path':c['checkpoint_path'],'actual_checkpoint_sha256_before':sha(c['checkpoint_path'])}
    write(out/'receipt.json',r)
    try:
        r['tests']=bounded(c['tests_command'],out/'cpu_tests.log',env,60,2<<30);write(out/'receipt.json',r)
        assert r['tests']['returncode']==0 and r['tests']['stop_reason'] is None,'CPU tests failed'
        r['actual_probe']=bounded(c['probe_command'],out/'actual_probe.log',env,180,2<<30);write(out/'receipt.json',r)
        assert r['actual_probe']['returncode']==0 and r['actual_probe']['stop_reason'] is None,'real-checkpoint forward probe failed'
        report=Path(c['probe_out'])/'report.json';r['report_sha256']=sha(report)
        r['complete']=json.loads(report.read_text())['complete'];r['status']='completed'
    except BaseException as error:r.update(status='failed',failure=repr(error),complete=False)
    finally:
        r['pins_after']={p:sha(p) for p in c['pins']};r['pins_unchanged']=r['pins_before']==r['pins_after']
        r['actual_checkpoint_sha256_after']=sha(c['checkpoint_path'])
        if not r['pins_unchanged'] or r['actual_checkpoint_sha256_before']!=r['actual_checkpoint_sha256_after']:r.update(complete=False,status='failed',failure='source/checkpoint changed')
        write(out/'receipt.json',r)
    print(json.dumps(r),flush=True);return 0 if r['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
