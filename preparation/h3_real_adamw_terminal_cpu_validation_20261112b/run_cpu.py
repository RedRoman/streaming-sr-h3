from pathlib import Path
import os,sys,subprocess,json,time,hashlib,signal
p=Path(__file__).resolve().parent
source=json.loads((p/'source_manifest.json').read_text())
assert all(hashlib.sha256((p/f).read_bytes()).hexdigest()==v for f,v in source.items())
assert not(p/'cpu_receipt.json').exists() and not(p/'tests.log').exists()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(p/'code'),OPENH3_COMFY_REFERENCE_ROOT='/cache/zihaohe/omni_openh3_20261111/code/ComfyUI',ADAMW_CPU_PRODUCER_FIXTURE='/cache/zihaohe/omni_openh3_20261111/preparation/h3_real_adamw_cpu_validation_20261112a')
started=time.monotonic();peak=0;reason=None
with (p/'tests.log').open('xb') as f:
 proc=subprocess.Popen([sys.executable,'-m','unittest','test_audit','-v'],cwd=p,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
while proc.poll() is None:
 rss=0
 for d in Path('/proc').iterdir():
  if not d.name.isdecimal():continue
  try:
   stat=(d/'stat').read_text().rsplit(')',1)[1].split()
   if int(stat[2])==proc.pid:rss+=int(stat[21])*os.sysconf('SC_PAGE_SIZE')
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 peak=max(peak,rss)
 if rss>4*(1<<30) or time.monotonic()-started>180:
  reason='RSS' if rss>4*(1<<30) else 'wall';os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  break
 time.sleep(.1)
text=(p/'tests.log').read_text();passed=proc.returncode==0 and '\nOK\n' in text and 'skipped=' not in text
r={'schema':'openh3.real_AdamW_terminal_CPU_validation.v1','complete':passed,'returncode':proc.returncode,'limit_failure':reason,'elapsed_seconds':time.monotonic()-started,'peak_owned_group_RSS_bytes':peak,'wall_cap_seconds':180,'RSS_cap_bytes':4*(1<<30),'CPU_affinity':sorted(os.sched_getaffinity(0)),'threads':1,'source':source,'tests_log_sha256':hashlib.sha256((p/'tests.log').read_bytes()).hexdigest(),'scope':'NumPy independent AdamW equations/checkpoint/actual400Torchstate fixture. No model forward, assets or NPU. Linux runner tests separate.'}
with (p/'cpu_receipt.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r));print(text[-4500:]);sys.exit(0 if passed else 2)
