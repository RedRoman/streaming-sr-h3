from pathlib import Path
import os,sys,json,hashlib,importlib.util,time
r=Path('/cache/zihaohe/omni_openh3_20261111');p=Path(__file__).resolve().parent;prep=r/'preparation/h3_real_adamw_update_20261112a'
assert not (p/'status.json').exists() and not (p/'input').exists()
runner=prep/'run_once.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388'
s=importlib.util.spec_from_file_location('owned_cpu_guard',runner);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(prep/'code')+':'+str(r/'preparation/vae_roundtrip_runtime_20261111_v2/code')}
command=[str(r/'venv/bin/python'),str(prep/'code/check_h3_real_adamw_update.py'),'--pretrained-manifest',str(r/'configs/openh3_20261111/refiner_turbo_30794.json'),'--execution-contract',str(prep/'execution_contract.json'),'--out',str(p/'input')]
status={'schema':'openh3.real_AdamW_actual_CPU_inspection.v1','complete':False,'NPU_calls':0,'model_loads':0,'CPU_affinity':sorted(os.sched_getaffinity(0)),'wall_limit':180,'RSS_limit':4<<30}
g.supervise(command,r,env,p/'inspect.log',status,p/'status.json',wall=180,rss_limit=4<<30,grace=10)
if status['state']=='worker_completed':
 q=p/'input/inspection.json';d=json.loads(q.read_text());assert d['inspection_only'] and d['planned']['NFE']==12 and d['formal_training_runs']==0
 status.update(complete=True,inspection_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),runtime_files_verified=d['runtime_files_verified'],execution_contract_sha256=hashlib.sha256((prep/'execution_contract.json').read_bytes()).hexdigest())
g.write(p/'status.json',status);print(json.dumps(status));sys.exit(0 if status['complete'] else 2)
