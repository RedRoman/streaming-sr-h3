import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=json.loads((root/'source_manifest.json').read_text())
assert all(sha(root/n)==v for n,v in files.items())
assert not (root/'cpu_receipt.json').exists()
affinity=sorted(os.sched_getaffinity(0))[:2]
assert len(affinity)==2
os.sched_setaffinity(0,set(affinity))
def child_limits(): resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONNOUSERSITE':'1'}
started=time.monotonic();error=None;code=None
try:
 with (root/'tests.log').open('xb') as f:
  code=subprocess.run([sys.executable,'-m','unittest','discover','-s','code/tests','-p','test_history_utility_headroom_once.py','-v'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60,preexec_fn=child_limits).returncode
except BaseException as e: error={'type':type(e).__name__,'message':str(e)}
unchanged=all(sha(root/n)==v for n,v in files.items())
receipt={'schema':'rphi.headroom_runner_CPU_validation.v1','complete':code==0 and error is None and unchanged,'returncode':code,'error':error,'elapsed_seconds':time.monotonic()-started,'CPU_affinity':affinity,'child_address_space_cap_bytes':512<<20,'child_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'wall_cap_seconds':60,'source':files,'source_unchanged':unchanged,'python':sys.version,'tests_log_sha256':sha(root/'tests.log'),'scope':'Six stdlib Linux runner fixtures. Mock card inspector and supervision; no actual card query, NPU/model import, evaluation, admission, or actual job launch.'}
(root/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(0 if receipt['complete'] else 1)
