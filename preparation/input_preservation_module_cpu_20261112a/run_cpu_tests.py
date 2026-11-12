import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((root/'source_manifest.json').read_text())
assert all(sha(root/n)==v for n,v in source.items())
assert not (root/'cpu_receipt.json').exists()
aff=sorted(os.sched_getaffinity(0))[:2];assert len(aff)==2
os.sched_setaffinity(0,set(aff))
env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONNOUSERSITE':'1'}
def cap():resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
started=time.monotonic();error=None;code=None;process=None
try:
 with (root/'tests.log').open('xb') as log:
  process=subprocess.Popen([sys.executable,'-m','unittest','discover','-s','code/tests','-p','test_input_preservation.py','-v'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=cap)
  code=process.wait(timeout=60)
except BaseException as e:
 error={'type':type(e).__name__,'message':str(e)}
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
finally:
 if process is not None:code=process.poll()
unchanged=all(sha(root/n)==v for n,v in source.items())
r={'schema':'rphi.input_preservation_module_CPU_validation.v1','complete':code==0 and error is None and unchanged,'returncode':code,'error':error,'source':source,'source_unchanged':unchanged,'elapsed_seconds':time.monotonic()-started,'CPU_affinity':aff,'threads':2,'wall_seconds':60,'owned_cleanup_grace_seconds':5,'child_AS_limit_bytes':4<<30,'child_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'tests_log_sha256':sha(root/'tests.log'),'python':sys.version,'scope':'CPU synthetic fixtures only;1745parameter head, raw-only interfaces and tiny randomly initialized K1 fixture.0actual train/dev data,0pretrained checkpoints,0NPU,0optimizer updates. These tests do not demonstrate learnability or restoration quality.'}
(root/'cpu_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['complete'] else 1)
