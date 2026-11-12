"""Target-only recovery: immutable17 reuse,1 prior ineligible,22 single attempts.

No timestamp guard or numerical transform is changed. Only the exact frozen
source-count rejection is classified as ineligible; other exceptions fail closed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SCHEMA='openh3.history_utility_target_screen_recovery.v1'
COUNT_ERROR='fewer than 96 actual source frames reached the filter; metadata is insufficient'
BUDGET={'wall_seconds':1800,'grace_seconds':30,'memory_mib':768,'CPU_threads':1,
 'simultaneous_compute_workers':1,'output_disk_bytes':6*(1<<30),'new_target_decode_attempts':22,
 'reused_complete_targets':17,'prior_ineligible_sources':1,'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def value_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def require(x,message):
 if not x:raise ValueError(message)
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def bound(p):return {'path':str(p),'sha256':sha(p)}
def check_bound(r):require(sha(r['path'])==r['sha256'],'bound file changed: '+r['path'])


def load_screen(c):
 p=Path(c['files']['original_screen']['path']);spec=importlib.util.spec_from_file_location('_frozen_target_screen_a',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def artifact_bindings(obj):
 if isinstance(obj,dict):
  if set(obj)=={'path','sha256'}:yield obj
  else:
   for v in obj.values():yield from artifact_bindings(v)
 elif isinstance(obj,list):
  for v in obj:yield from artifact_bindings(v)


def validate(c):
 require(c['schema']==SCHEMA and c['budget']==BUDGET,'wrong recovery scope/budget')
 require(c['scientific_continuity_approved'] is False,'no continuity approval')
 for r in c['files'].values():check_bound(r)
 require(Path(c['files']['entry']['path']).resolve()==Path(__file__).resolve(),'unbound recovery entry')
 a=read(c['files']['original_contract']['path']);screen=load_screen(c);old,rows=screen.validate(a)
 failure=read(c['files']['original_failure']['path'])
 require(failure['complete'] is False and failure['new_target_decodes']==13 and failure['reused_targets']==4,'prior failure differs')
 inherited={r['source_id']:r['receipt'] for r in failure['sources']}
 require(c['reused_receipts']==inherited and len(inherited)==17,'17 completed receipt bindings differ')
 require(c['prior_ineligible_source']=='DtgdD48804M','wrong prior ineligible source')
 rejected=read(c['files']['prior_source_failure']['path'])
 require(rejected['source']['source_id']=='DtgdD48804M' and rejected['complete'] is False
         and rejected['error']==repr(ValueError(COUNT_ERROR)),'prior actual qualification failure differs')
 expected=sorted(set(a['source_order'])-set(inherited)-{'DtgdD48804M'})
 require(c['attempt_order']==expected and len(expected)==22,'remaining22 scope changed')
 for sid,r in inherited.items():
  check_bound(r);p=read(r['path']);require(p['complete'] is True and p['status']=='target_only_complete_not_observation_dataset','prior target incomplete')
  require(p['source']['source_id']==sid and p['source']['source_sha256']==rows[sid]['source_sha256'] and p['source']['split']==rows[sid]['split'],'reuse source role differs')
  require(p['execution_contract_sha256']==value_sha(a),'reuse execution namespace differs')
  t=p['target'];require(sha(t['path'])==t['file_sha256'] and Path(t['path']).stat().st_size==t['bytes_on_disk'],'reuse target file changed')
  for artifact in artifact_bindings(p['diagnostics']['artifacts']):check_bound(artifact)
 return screen,a,old,rows


def inspect(c):
 screen,a,old,rows=validate(c)
 return {'schema':SCHEMA,'status':'inspected_not_materialized','source_file_SHA_verified':40,
  'reused_target_file_and_artifact_SHA_verified':17,'actual_pixels_verified_in_this_inspection':False,
  'remaining_single_attempts':22,'prior_ineligible':1,'budget':BUDGET,'model_calls':0,'NPU_calls':0,
  'scientific_continuity_approved':False,'import_guard':screen.load_original(a).check_no_models()}


@contextmanager
def preserve_decode_log(module,destination):
 """Calls the original decoder unchanged; copy diagnostics even if parser rejects."""
 original=module.decode_target_owned
 def wrapped(old,source,spec,scratch,ffmpeg,budget):
  try:return original(old,source,spec,scratch,ffmpeg,budget)
  finally:
   log=Path(scratch)/'target_decode.log'
   if log.is_file():shutil.copyfile(log,destination)
 module.decode_target_owned=wrapped
 try:yield
 finally:module.decode_target_owned=original


def source_count_rejection(error,log_path,numeric):
 if type(error) is not ValueError or str(error)!=COUNT_ERROR:return None
 require(Path(log_path).is_file(),'source-count rejection missing persistent original log')
 rows={'source':[],'target':[]};bases={}
 for line in Path(log_path).read_text(errors='replace').splitlines():
  b=numeric.TIMEBASE.search(line)
  if b:bases[b[1]]=b[2]
  m=numeric.SHOWINFO.search(line)
  if m:
   tag,n,pts,seconds=m.groups();rows[tag].append({'index':int(n),'pts':int(pts),'pts_time_seconds':float(seconds)})
 require(set(bases)=={'source','target'},'missing rejection timebases')
 require(len(rows['target'])==96 and [r['index'] for r in rows['target']]==list(range(96)),'not solely source-count rejection')
 require(0<len(rows['source'])<96 and [r['index'] for r in rows['source']]==list(range(len(rows['source']))),'source-count evidence malformed')
 require(all((b['pts']-a['pts'])*Fraction(bases['target'])==Fraction(1,24) for a,b in zip(rows['target'],rows['target'][1:])),'target log also violates frozen24fps grid')
 require(all(b['pts_time_seconds']>a['pts_time_seconds'] for a,b in zip(rows['source'],rows['source'][1:])),'source log also has nonmonotone timestamps')
 return {'reason':'source_frame_count_below_frozen96_gate','exception':str(error),'decoded_source_count':len(rows['source']),
  'decoded_target_count':96,'timebases':bases,'timestamps':rows,'original_log':bound(log_path),
  'interpretation':'Protocol-ineligible only; no target accepted, no interpolation rule changed, no claim video is corrupt.'}


def worker(c,out,sid,mode):
 screen,a,old,rows=validate(c);require((mode=='reuse' and sid in c['reused_receipts']) or (mode=='attempt' and sid in c['attempt_order']),'invalid worker role')
 dest=out/'sources'/sid
 report={'schema':'openh3.history_utility_target_screen_recovery_source.v1','complete':False,'source_id':sid,
  'recovery_contract_sha256':value_sha(c),
  'scientific_continuity_approved':False,'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0,'pid':os.getpid()}
 if mode=='reuse':
  import numpy as np
  m=screen.load_original(a);numeric=m.load_numeric(old['files']['numeric_source']['path']);budget=numeric.Budget(1800,768)
  prior=read(c['reused_receipts'][sid]['path']);t=prior['target']
  with np.load(t['path'],allow_pickle=False) as z:frames=z['frames']
  require(frames.dtype==np.uint8 and frames.shape==m.D.SHAPE and numeric.pixel_sha(frames)==t['pixel_sha256'],'reused actual pixels changed')
  require(prior['window_key']==numeric.value_sha(prior['target_specification']) and prior['target_specification']['execution_contract_sha256']==numeric.value_sha(old),'reuse numerical window changed')
  del frames;budget.check();dest.mkdir(parents=True,exist_ok=False)
  report.update(complete=True,status='reused_actual_pixels_verified',original_receipt=c['reused_receipts'][sid],target=t,
   target_eligible=True,new_decode_attempts=0,actual_pixel_SHA_verified=True,import_guard=m.check_no_models())
 else:
  original_loader=screen.load_original;original_decoder=[]
  def observed_loader(contract):
   m=original_loader(contract);original_decoder.append(m);return m
  # Scope the decoder hook on each instance returned by the frozen entry loader.
  contexts=[]
  def scoped_loader(contract):
   m=observed_loader(contract);ctx=preserve_decode_log(m,dest/'target_decode.log');ctx.__enter__();contexts.append(ctx);return m
  screen.load_original=scoped_loader
  try:
   value=screen.worker(a,out,sid)
   report.update(complete=True,status='new_target_completed',target_eligible=True,new_decode_attempts=1,
    numerical_receipt=bound(dest/'complete.json'),target=value['target'],actual_pixel_SHA_verified=True,
    import_guard=value['import_guard'])
  except BaseException as e:
   m=original_decoder[-1] if original_decoder else original_loader(a)
   numeric=m.load_numeric(old['files']['numeric_source']['path'])
   diagnosis=source_count_rejection(e,dest/'target_decode.log',numeric)
   if diagnosis is None:raise
   require((dest/'failure.json').is_file() and not (dest/'complete.json').exists(),'invalid qualification-failure artifacts')
   report.update(complete=True,status='ineligible_source_count',target_eligible=False,new_decode_attempts=1,
    diagnosis=diagnosis,numerical_failure=bound(dest/'failure.json'),import_guard=m.check_no_models())
  finally:
   for ctx in reversed(contexts):ctx.__exit__(None,None,None)
   screen.load_original=original_loader
 write(dest/'recovery_receipt.json',report);return report


def execute(path,c,out):
 require(out.is_absolute(),'absolute new namespace required')
 if out.exists():raise FileExistsError('output exists; no overwrite')
 validate(c);require(shutil.disk_usage(out.parent).free>=6<<30,'need6GiB free disk');out.mkdir(parents=True,exist_ok=False)
 r={'schema':SCHEMA,'complete':False,'status':'running','sources':[],'reused_complete_targets':0,'new_target_decode_attempts':0,
  'new_targets_completed':0,'new_ineligible_sources':0,'prior_ineligible_sources':1,'prior_ineligible_source':'DtgdD48804M',
  'prior_failure':c['files']['prior_source_failure'],'degradation_codec_passes':0,'model_calls':0,'NPU_calls':0,'scientific_continuity_approved':False}
 r['source_outcomes']={sid:{'state':'noattempt'} for sid in list(c['reused_receipts'])+c['attempt_order']}
 r['source_outcomes']['DtgdD48804M']={'state':'ineligible','basis':'prior actual frozen sourcecount rejection; no new attempt'}
 r['attempt_workers_started']=0;r['reuse_workers_started']=0
 started=time.monotonic();write(out/'partial.json',r);prior=signal.getsignal(signal.SIGTERM)
 def terminate(signum,frame):raise TimeoutError('owned supervisor requested termination')
 signal.signal(signal.SIGTERM,terminate)
 try:
  order=[(sid,'reuse') for sid in sorted(c['reused_receipts'])]+[(sid,'attempt') for sid in c['attempt_order']]
  for sid,mode in order:
   command=[sys.executable,str(Path(__file__).resolve()),'--recovery-contract',str(path),'--out',str(out),'--source-id',sid,'--mode',mode]
   r['attempt_workers_started']+=int(mode=='attempt');r['reuse_workers_started']+=int(mode=='reuse')
   write(out/'partial.json',r)
   with (out/(sid+'.log')).open('xb') as f:done=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=False)
   if done.returncode!=0:
    r['source_outcomes'][sid]={'state':'error','returncode':done.returncode,'decode_attempt_count_unknown_for_this_failed_worker':mode=='attempt'}
    raise ValueError('fresh recovery worker failed: '+sid)
   p=out/'sources'/sid/'recovery_receipt.json';x=read(p);require(x['complete'],'worker incomplete')
   require(x['source_id']==sid and x['new_decode_attempts']==int(mode=='attempt'),'worker role/count changed')
   r['sources'].append({'source_id':sid,'receipt':bound(p),'pid':x['pid'],'status':x['status']})
   r['source_outcomes'][sid]={'state':'eligible' if x['target_eligible'] else 'ineligible','receipt':bound(p),'new_decode_attempts':x['new_decode_attempts']}
   r['reused_complete_targets']+=int(mode=='reuse');r['new_target_decode_attempts']+=x['new_decode_attempts']
   r['new_targets_completed']+=int(x['status']=='new_target_completed');r['new_ineligible_sources']+=int(x['status']=='ineligible_source_count')
   r['output_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());require(r['output_bytes']<=6<<30,'output disk budget exceeded');write(out/'partial.json',r)
  require(len(r['sources'])==39 and r['reused_complete_targets']==17 and r['new_target_decode_attempts']==22,'recovery budget/accounting differs')
  require(r['new_targets_completed']+r['new_ineligible_sources']==22,'unaccounted attempt')
  require(len({x['pid'] for x in r['sources']})==39,'fresh worker PID reused');validate(c)
  r.update(complete=True,status='all40_sources_accounted_targets_pending_visual_review',accounted_source_count=40,
   target_eligible_count=17+r['new_targets_completed'],ineligible_source_count=1+r['new_ineligible_sources'],error_source_count=0,noattempt_source_count=0,elapsed_seconds=time.monotonic()-started)
  write(out/'complete.json',r);(out/'partial.json').unlink();return r
 except BaseException as e:r.update(complete=False,status='failed',error=repr(e));write(out/'failure.json',r);raise
 finally:signal.signal(signal.SIGTERM,prior)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery-contract',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 p.add_argument('--execute',action='store_true');p.add_argument('--source-id');p.add_argument('--mode',choices=['reuse','attempt']);a=p.parse_args();c=read(a.recovery_contract)
 if a.source_id:worker(c,a.out,a.source_id,a.mode)
 elif a.execute:execute(a.recovery_contract,c,a.out)
 else:
  if a.out.exists():raise FileExistsError('inspection exists')
  r=inspect(c);a.out.parent.mkdir(parents=True,exist_ok=True)
  with a.out.open('x') as f:json.dump(r,f,indent=2)

if __name__=='__main__':main()
