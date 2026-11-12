"""Prepare immutable inputs/pins only; never remote execute or materialize."""
from pathlib import Path
import hashlib,json,shutil,platform,datetime
ROOT=Path(__file__).resolve().parents[2]
PREP=Path(__file__).resolve().parent
REMOTE=Path('/cache/zihaohe/omni_openh3_20261111')
RPREP=REMOTE/'preparation/history_utility_revised_materializer_20261112a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def ref(local,remote):return {'path':str(remote),'sha256':sha(local)}
cohort_path=ROOT/'configs/openh3_20261111/history_utility_revised_cohort_20261112.json'
cohort=read(cohort_path);assert sha(cohort_path)=='c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
assert sha(ROOT/'code/history_utility_revised_data.py')=='3f48214e3df626f988a92e78349a8929dea01414b831566fda0dd793ac1937d1'
for name in ['materialize_history_utility_revised.py','history_utility_revised_data.py','tests/test_history_utility_revised_materializer.py']:
 dst=PREP/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'code'/name,dst)
old=ROOT/'preparation/history_utility_materializer_20261112a'; rold=REMOTE/'preparation/history_utility_materializer_20261112a'
screen=ROOT/'results/takeover_20261111/history_utility_target_screen_completed_20261112b'
material=ROOT/'results/takeover_20261111/history_utility_materializer_completed_20261112b'
def local(remote):
 rel=Path(remote).relative_to(REMOTE)
 for base in (ROOT,screen,material):
  p=base/rel
  if p.is_file():return p
 raise FileNotFoundError(str(remote))
files={
 'entry_source':ref(PREP/'code/materialize_history_utility_revised.py',RPREP/'code/materialize_history_utility_revised.py'),
 'revised_loader_source':ref(PREP/'code/history_utility_revised_data.py',RPREP/'code/history_utility_revised_data.py'),
 'cohort':ref(cohort_path,REMOTE/cohort_path.relative_to(ROOT)),
 'source_inventory':ref(screen/'evidence/history_utility_full_cohort_inventory_20261112a.json',REMOTE/'evidence/history_utility_full_cohort_inventory_20261112a.json')}
for key,name in [('loader_source','code/history_utility_data.py'),('numeric_source','code/materialize_mixed_temporal_pilot.py'),('numerical_window_contract','execution_contract.json'),('plan','plan.json'),('train_manifest','train_materialized.jsonl'),('dev_manifest','dev_materialized.jsonl')]:files[key]=ref(old/name,rold/name)
run=material/'runs/history_utility_engineering_2x2_20261112b';reuse={}
for split in ('train','dev'):
 for entry in read(run/(split+'_receipt_index.json'))['receipts']:
  r=read(local(entry['path']));sid=r['source']['source_id']
  if sid in cohort['profile_reuse_candidates']:reuse[sid]=entry
assert set(reuse)=={'4EKWTiWjPTQ','6UZVovkR3IU','6rJrrfRO3cA'}
software={read(local(x['path']))['software_sha256'] for x in reuse.values()};assert len(software)==1
contract={'schema':'openh3.history_utility_revised_materialization_execution.v1','scope':'revised_primary_CPU_materialization_only_no_training',
 'files':files,'evidence_root':str(REMOTE),'selected_source_ids':cohort['primary_source_ids'],'reuse_receipts':reuse,
 'new_sources':sorted(set(sum(cohort['primary_source_ids'].values(),[]))-set(reuse)),
 'software_sha256':software.pop(),'source_MP4_decodes':0,'codec_passes':132,'replays':2,
 'profiles':['native_static_blur_noise_h264','native_smooth_blur_noise_h264','native_abrupt_blur_noise_h264'],
 'budget':{'wall_seconds':3600,'termination_grace_seconds':30,'memory_mib':768,'worker_processes':1,'threads':1,'minimum_free_bytes':6<<30,'maximum_new_output_bytes':30<<30,'source_MP4_decodes':0,'model_calls':0,'NPU_calls':0},
 'numerical_namespace':'Original13b6 spec/window/seed contract unchanged; this separate contract authorizes selected25 current sources.',
 'disk_boundary':'Free space and new output are measured before/after each source; no filesystem quota claimed.22x4 raw uint8 arrays plus temporary buffers fit below30GiB by geometry.',
 'readiness':'Materialized complete does not mean revised dataset/training admitted; wrapper validation follows separately.',
 'retry_policy':'No automatic retries; original and partial outputs are never overwritten.'}
put(PREP/'execution_contract.json',contract)
log=(PREP/'cpu_tests.log').read_text();assert 'Ran 11 tests in 3.287s' in log and log.rstrip().endswith('OK')
put(PREP/'cpu_tests.json',{'schema':'openh3.history_utility_revised_materializer_CPU_tests.v1','complete':True,'tests_passed':11,'skips':0,'seconds':3.287,'fixture_scope':'Actual synthetic fullshape NPZ/file/pixel+v1 loader; observation numeric outputs stubbed,0realcodec/0MP4decode/0model. Supervisor pending-TERM/ownedcleanup and zombie gates are mocked CPU lifecycle fixtures.',
 'python':platform.python_version(),'log_sha256':sha(PREP/'cpu_tests.log'),'entry_sha256':files['entry_source']['sha256'],'tests_sha256':sha(PREP/'code/tests/test_history_utility_revised_materializer.py'),'supervisor_sha256':sha(PREP/'run_once.py')})
pins={x['path']:x['sha256'] for x in files.values()}
for name in ['execution_contract.json','run_once.py','cpu_tests.json','cpu_tests.log','code/tests/test_history_utility_revised_materializer.py']:
 pins[str(RPREP/name)]=sha(PREP/name)
for key in ['target_terminal_audit','target_engineering_review','local_target_archive_verification']:
 b=cohort[key]; p=Path(b['path']); p=p if p.is_absolute() else REMOTE/p;pins[str(p)]=b['sha256'];assert sha(local(p))==b['sha256']
for sid in sum(cohort['primary_source_ids'].values(),[]):
 row=cohort['decisions'][sid]
 for key in ('target_receipt','visual_review'):
  b=row[key];p=Path(b['path']);p=p if p.is_absolute() else REMOTE/p;pins[str(p)]=b['sha256'];assert sha(local(p))==b['sha256']
 r=read(local(row['target_receipt']['path']));t=r['target'];pins[t['path']]=t['file_sha256']
 for a in r['diagnostics']['artifacts'].values():
  for image in (a if isinstance(a,list) else [a]):
   if isinstance(image,dict) and {'path','sha256'}<=set(image):pins[image['path']]=image['sha256'];assert sha(local(image['path']))==image['sha256']
for entry in reuse.values():
 pins[entry['path']]=entry['sha256'];r=read(local(entry['path']))
 for artifact in [r['target']]+[r['observations'][pid]['artifact'] for pid in contract['profiles']]:pins[artifact['path']]=artifact['file_sha256']
inv=read(local(files['source_inventory']['path']))
for row in inv['records']:
 if row['source_id'] in sum(cohort['primary_source_ids'].values(),[]):pins[row['path']]=row['source_sha256']
freeze={'schema':'openh3.history_utility_cpu_freeze.v1','host':'31984','root':str(REMOTE),'wall_seconds':3600,'grace_seconds':30,'memory_mib':768,
 'run':str(REMOTE/'runs/history_utility_revised_materialization_20261112a'),'supervisor':str(REMOTE/'runs/history_utility_revised_materialization_20261112a_supervisor'),
 'inspection':str(RPREP/'inspection'),'execution_contract':str(RPREP/'execution_contract.json'),
 'command':[str(REMOTE/'venv/bin/python'),files['entry_source']['path'],'--contract',str(RPREP/'execution_contract.json'),'--out',str(REMOTE/'runs/history_utility_revised_materialization_20261112a')],
 'pins':dict(sorted(pins.items()))}
put(PREP/'freeze.json',freeze)
put(PREP/'preparation_receipt.json',{'schema':'openh3.history_utility_revised_materialization_prepared.v1','status':'prepared_not_admitted_not_executed','freeze_sha256':sha(PREP/'freeze.json'),'contract_sha256':sha(PREP/'execution_contract.json'),'pins':len(pins),'budget':contract['budget'],'primary':{'train':19,'dev':6},'reuse_profiles':3,'new_sources':22,'planned_codec_passes':132,'new_video_decodes':0,'prior_engineering_seconds_per_source':[75.87879840936512,75.70231740828604,71.41287328954786],'prior_timing_scope':'Old three engineering sources include1target decode and6codec passes each; not new cohort measurement,quality evidence,or guaranteed runtime.','new_materialization_started':False})
print(json.dumps({'freeze':sha(PREP/'freeze.json'),'contract':sha(PREP/'execution_contract.json'),'entry':files['entry_source']['sha256'],'supervisor':sha(PREP/'run_once.py'),'pins':len(pins)}))
