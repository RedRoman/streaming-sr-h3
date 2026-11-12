"""One bounded synthetic CPU unit-test run; no real checkpoint/data reads."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group_rss(pgid):
    total=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if int((p/'stat').read_text().rsplit(')',1)[1].split()[2])==pgid:
                total+=next((int(x.split()[1])*1024 for x in (p/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return total

def main():
    root=Path(__file__).resolve().parent;config=json.loads((root/'freeze.json').read_text())
    assert sha(root/'freeze.json')==sys.argv[1]
    for rel,h in config['pins'].items():assert sha(root/rel)==h,rel
    out=root/'validation';out.mkdir(exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,set(cpus))
    env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[sys.executable,str(root/'code/tests/test_history_utility_one_step.py'),'-v']
    with (out/'tests.log').open('xb') as log:p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();peak=0;error=None
    try:
        while p.poll() is None:
            used=group_rss(p.pid);peak=max(peak,used)
            if used>2<<30:raise MemoryError('test owned RSS exceeds2GiB')
            if time.monotonic()-start>90:raise TimeoutError('test wall exceeds90seconds')
            time.sleep(.1)
    except BaseException as e:
        error=repr(e)
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
    after={rel:sha(root/rel) for rel in config['pins']}
    receipt={'schema':'rphi.history_utility_one_step_cpu_tests.v1','complete':p.returncode==0 and error is None and after==config['pins'],
        'scope':'synthetic small-Torch unit fixtures only; realcheckpoint and actualdata not read; no NPU or research training',
        'source_sha256':after,'pins_unchanged':after==config['pins'],'freeze_sha256':sys.argv[1],
        'command':command,'exit_code':p.returncode,'error':error,'elapsed_seconds':time.monotonic()-start,
        'peak_owned_group_RSS_bytes':peak,'CPU_affinity':cpus,'wall_limit_seconds':90,'RSS_limit_bytes':2<<30,
        'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','log_sha256':sha(out/'tests.log')}
    (out/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));return 0 if receipt['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
