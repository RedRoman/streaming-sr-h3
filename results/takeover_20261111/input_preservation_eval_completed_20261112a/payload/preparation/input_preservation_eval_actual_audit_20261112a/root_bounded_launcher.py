import pathlib,json,hashlib,importlib.util,os
R=pathlib.Path('/cache/zihaohe/omni_openh3_20261111');P=R/'preparation/input_preservation_eval_actual_audit_20261112a';run=R/'runs/input_preservation_eval_actual_audit_20261112a';run.mkdir(exist_ok=False)
helper=R/'preparation/input_preservation_eval_runner_20261112a/owned_process.py'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(helper)=='efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388'
assert sha(P/'audit_actual.py')=='80803d349c7dbd66d3765d557bf7fb945b553d82e0bcdbe057edcf91d4441cc2'
contract_sha=sha(P/'execution_contract.json')
spec=importlib.util.spec_from_file_location('owned_eval_audit_supervisor',helper);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
status={'state':'checking','complete':False,'producer_started':False,'automatic_retry':False,'NPU_calls':0,'contract_sha256':contract_sha};sp=run/'status.json';old=os.sched_getaffinity(0)
try:
 os.sched_setaffinity(0,set(sorted(old)[:2]));os.nice(10)
 env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','ASCEND_RT_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
 with s.signal_guard():
  s.supervise([str(R/'venv/bin/python'),str(P/'audit_actual.py'),'--execution-contract',str(P/'execution_contract.json'),'--out',str(run/'output'),'--execute'],R,env,run/'worker.log',status,sp,wall=900,rss_limit=4<<30,grace=30)
 if status['state']=='worker_completed':
  a=json.loads((run/'output/audit.json').read_text());assert a['complete'] and a['actual_rows']==2880 and a['actual_pairs']==2850
  status.update(state='completed',complete=True,audit_sha256=sha(run/'output/audit.json'))
except BaseException as e:status.update(state='failed',complete=False,error={'type':type(e).__name__,'message':str(e)})
finally:
 os.sched_setaffinity(0,old)
 if sha(P/'execution_contract.json')!=contract_sha or sha(P/'audit_actual.py')!='80803d349c7dbd66d3765d557bf7fb945b553d82e0bcdbe057edcf91d4441cc2':status.update(state='failed',complete=False,pins_unchanged=False)
 else:status['pins_unchanged']=True
 s.write(sp,status)
print(json.dumps(status))
