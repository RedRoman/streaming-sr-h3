"""Read-only NumPy target audit and bounded existing-artifact packaging; no decode."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse,hashlib,io,json,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/cache/zihaohe/omni_openh3_20261111')
RUN=ROOT/'runs/history_utility_target_screen_20261112b'
OUT=ROOT/'evidence/history_utility_target_screen_completed_20261112b'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def check(r,key='sha256'):require(sha(r['path'])==r[key],'file SHA differs: '+r['path'])
def artifacts(x):
 if isinstance(x,dict):
  if set(x)=={'path','sha256'}:yield x
  else:
   for y in x.values():yield from artifacts(y)
 elif isinstance(x,list):
  for y in x:yield from artifacts(y)
def relative(p):return str(Path(p).relative_to(ROOT))

def pack(name,files):
 rows=[{'path':relative(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 manifest={'schema':'openh3.safe_artifact_archive.v1','payloads':rows,'target_NPZ_included':False,'model_calls':0,'decoded_frames':0}
 path=OUT/name
 require(not path.exists(),'archive already exists')
 with tarfile.open(path,'w:gz') as archive:
  for p in sorted(files):archive.add(p,arcname=relative(p),recursive=False)
  data=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();entry=tarfile.TarInfo('MANIFEST.json');entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
 require(path.stat().st_size<=8<<20,'archive exceeds8MiB')
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payload_count':len(rows)}

def worker():
 import gc
 import numpy as np
 started=time.monotonic();r=read(RUN/'complete.json');status=read(Path(str(RUN)+'_supervisor')/'status.json')
 require(r['complete'] is True and status['complete'] is True and status['state']=='completed','run incomplete')
 require(r['accounted_source_count']==40 and r['target_eligible_count']==38 and r['ineligible_source_count']==2,'unexpected terminal accounting')
 require(r['new_target_decode_attempts']==22 and r['reused_complete_targets']==17 and r['new_targets_completed']==21,'attempt/reuse accounting differs')
 require(not Path('/proc',str(status['worker_pid'])).exists(),'old worker still exists')
 check({'path':str(RUN/'complete.json'),'sha256':status['report_sha256']})
 inv=read(ROOT/'evidence/history_utility_full_cohort_inventory_20261112a.json');source={x['source_id']:x for x in inv['records']}
 for x in source.values():require(sha(x['path'])==x['source_sha256'],'source video SHA differs')
 rows=[];images=set();small=set()
 for row in r['sources']:
  check(row['receipt']);p=Path(row['receipt']['path']);small.add(p);x=read(p);sid=x['source_id'];require(sid==row['source_id'] and x['complete'],'source incomplete')
  if not x['target_eligible']:
   require(sid=='Q0-CI3XJvxw' and x['status']=='ineligible_source_count','unexpected rejected source');check(x['diagnosis']['original_log']);small.add(Path(x['diagnosis']['original_log']['path']))
   rows.append({'source_id':sid,'split':source[sid]['split'],'state':'ineligible','source_sha256':source[sid]['source_sha256'],'receipt':row['receipt'],'actual_source_frames':x['diagnosis']['decoded_source_count'],'actual_target_frames':x['diagnosis']['decoded_target_count'],'images':[]});continue
  ref=x.get('original_receipt',x.get('numerical_receipt'));check(ref);small.add(Path(ref['path']));original=read(ref['path']);t=x['target'];require(t==original['target'],'target binding differs')
  check(t,'file_sha256');require(Path(t['path']).stat().st_size==t['bytes_on_disk'],'target byte count differs')
  with np.load(t['path'],allow_pickle=False) as z:frames=z['frames']
  require(frames.dtype==np.uint8 and frames.shape==(96,480,832,3),'target shape/dtype differs')
  digest=hashlib.sha256(memoryview(np.ascontiguousarray(frames)).cast('B')).hexdigest();require(digest==t['pixel_sha256'],'actual target pixelSHA differs')
  del frames;gc.collect()
  bound_images=[]
  for art in artifacts(original['diagnostics']['artifacts']):
   check(art);image=Path(art['path']);require(image.suffix=='.png','unexpected image format');images.add(image);bound_images.append(art)
  require(original['source']['source_sha256']==source[sid]['source_sha256'] and original['source']['split']==source[sid]['split'],'source role differs')
  rows.append({'source_id':sid,'split':source[sid]['split'],'state':'target_protocol_eligible_continuity_unreviewed','source_sha256':source[sid]['source_sha256'],'receipt':row['receipt'],'numerical_receipt':ref,'target_file_sha256':t['file_sha256'],'target_pixel_sha256':digest,'shape':[96,480,832,3],'dtype':'uint8','window_key':original['window_key'],'source_frames':original['decode']['decoded_source_count'],'target_frames':original['decode']['decoded_target_count'],'images':bound_images,'all_frame_sheet':original['diagnostics']['artifacts']['all_frame_sheet'],'ranked_pairs':original['diagnostics']['artifacts']['ranked_candidate_pairs'],'full_resolution_pairs':original['diagnostics']['artifacts']['full_resolution_pairs']})
  put(OUT/'partial.json',{'complete':False,'verified_sources':len(rows),'rows':rows})
 check(r['prior_failure']);small.add(Path(r['prior_failure']['path']))
 rows.append({'source_id':'DtgdD48804M','split':source['DtgdD48804M']['split'],'state':'prior_protocol_ineligible_not_retried','source_sha256':source['DtgdD48804M']['source_sha256'],'receipt':r['prior_failure'],'actual_source_frames':'less_than96_exact_count_unavailable','images':[]})
 require(len(rows)==40 and len(images)==78,'wrong source/image inventory')
 require(not any(m.split('.')[0] in {'torch','torch_npu','comfy','h3_comfy'} for m in sys.modules),'model/backend imported')
 report={'schema':'openh3.history_utility_target_screen_actual_terminal_audit.v1','complete':True,'source_files_actual_SHA_verified':40,'target_files_actual_SHA_and_numpy_pixel_SHA_verified':38,'source_receipts_verified':39,'prior_ineligible_reference_verified':1,'images_actual_SHA_verified':78,'source_rows':sorted(rows,key=lambda x:x['source_id']),'source_count':40,'target_protocol_eligible_count':38,'continuity_approved_count':0,'ineligible_count':2,'worker_elapsed_seconds':time.monotonic()-started,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'model_calls':0,'NPU_calls':0,'decoded_frames':0,'target_NPZ_downloaded':False,'scientific_continuity_approved':False}
 put(OUT/'actual_audit.json',report);small.add(OUT/'actual_audit.json')
 for sub in ['runs/history_utility_target_screen_20261112a','runs/history_utility_target_screen_20261112a_supervisor','runs/history_utility_target_screen_20261112b','runs/history_utility_target_screen_20261112b_supervisor','preparation/history_utility_target_screen_20261112a','preparation/history_utility_target_screen_20261112b','preparation/history_utility_materializer_20261112a']:
  small.update(p for p in (ROOT/sub).rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.py','.log'})
 small.add(ROOT/'evidence/history_utility_full_cohort_inventory_20261112a.json')
 packs=[pack('small_evidence.tar.gz',small)];group=[];size=0;index=1
 for p in sorted(images):
  if group and size+p.stat().st_size>7*(1<<20):packs.append(pack('images_%02d.tar.gz'%index,group));group=[];size=0;index+=1
  group.append(p);size+=p.stat().st_size
 if group:packs.append(pack('images_%02d.tar.gz'%index,group))
 put(OUT/'archive_receipt.json',{'schema':'openh3.history_utility_target_screen_archive_receipt.v1','complete':True,'archives':packs,'image_files':78,'target_sets':38,'ineligible_without_images':['DtgdD48804M','Q0-CI3XJvxw'],'target_NPZ_included':False,'actual_audit_sha256':sha(OUT/'actual_audit.json')})
 (OUT/'partial.json').unlink(missing_ok=True)


def group_rss(pgid):
 total=0
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[2])==pgid:total+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return total

def main():
 if '--worker' in sys.argv:worker();return
 OUT.mkdir(exist_ok=False);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});started=time.monotonic();peak=0
 with (OUT/'worker.log').open('xb') as f:
  proc=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  report={'complete':False,'worker_pid':proc.pid,'CPU_threads':1,'memory_limit_bytes':768<<20,'wall_limit_seconds':300,'model_calls':0,'decoded_frames':0}
  try:
   while proc.poll() is None:
    peak=max(peak,group_rss(proc.pid));require(peak<=768<<20,'ownedPG memory exceeded');require(time.monotonic()-started<300,'read-only audit wall exceeded');time.sleep(.2)
   require(proc.returncode==0,'audit worker failed');report.update(complete=True,state='completed')
  except BaseException as e:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   report.update(state='failed',error=repr(e));raise
  finally:report.update(returncode=proc.poll(),peak_ownedPG_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started);put(OUT/'audit_supervision.json',report)
 print(json.dumps({'complete':True,'receipt':str(OUT/'archive_receipt.json'),'receipt_sha256':sha(OUT/'archive_receipt.json'),'peak_ownedPG_RSS_bytes':peak,'elapsed_seconds':time.monotonic()-started}))

if __name__=='__main__':main()
