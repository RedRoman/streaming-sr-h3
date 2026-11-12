#!/usr/bin/env python3
"""Read-only source validation, then fresh bounded missing-range materialization."""
import datetime,hashlib,json,os,pathlib,signal,stat
BASE=pathlib.Path(__file__).resolve().parent
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 second CPU deadline')))
signal.alarm(180)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=p.stat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink();return (s.st_size,s.st_mtime_ns,s.st_ino)
request=json.loads((BASE/'prefix_receipt.json').read_text())
claim=BASE/'materialize_claim.json'
with claim.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid()},f)
result={'schema':'openh3.turbo_suffix_source.v1','complete':False,'request_sha256':sha(BASE/'prefix_receipt.json'),'segments':[],'part_checks':[]}
try:
 expected_source_root=pathlib.Path('/cache/zihaohe/omni_openh3_20261111/preparation/turbo_transfer_31112_20261112/parts')
 for row in request['parts']:
  p=pathlib.Path(row['source_part']);assert p==expected_source_root/row['name'];assert row['name']==f"part_{int(row['name'][5:]):03d}"
  before=stamp(p);assert before[0]==row['expected_bytes'];assert sha(p)==row['expected_sha256']
  h=hashlib.sha256();remaining=row['prefix_bytes']
  with p.open('rb') as f:
   while remaining:
    b=f.read(min(4*1024*1024,remaining));assert b;h.update(b);remaining-=len(b)
  assert h.hexdigest()==row['prefix_sha256'],f"partial prefix mismatch: {row['name']}"
  assert stamp(p)==before
  result['part_checks'].append({'name':row['name'],'full_sha256':row['expected_sha256'],'prefix_bytes':row['prefix_bytes'],'prefix_sha256':h.hexdigest(),'prefix_exact':True,'source_stamp':before})
 out=BASE/'segments';out.mkdir(exist_ok=False)
 for row in request['parts']:
  p=pathlib.Path(row['source_part']);before=stamp(p);offset=row['prefix_bytes'];index=0
  with p.open('rb') as f:
   f.seek(offset)
   while offset<row['expected_bytes']:
    length=min(request['segment_max_bytes'],row['expected_bytes']-offset)
    name=f"{row['name']}_suffix_{index:03d}_{offset}_{length}.bin";dest=out/name
    data=f.read(length);assert len(data)==length
    with dest.open('xb') as g:g.write(data);g.flush();os.fsync(g.fileno())
    digest=hashlib.sha256(data).hexdigest();assert sha(dest)==digest
    result['segments'].append({'name':name,'part':row['name'],'offset':offset,'bytes':length,'sha256':digest,'parent_bytes':row['expected_bytes'],'parent_sha256':row['expected_sha256'],'remote_path':str(dest)})
    offset+=length;index+=1
  assert stamp(p)==before
 assert sum(x['bytes'] for x in result['segments'])==request['missing_bytes']
 result.update(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),missing_bytes=request['missing_bytes'],retained_bytes=request['retained_bytes'],segment_count=len(result['segments']))
except BaseException as e:
 result.update(error=repr(e),utc=datetime.datetime.now(datetime.timezone.utc).isoformat());raise
finally:
 with (BASE/'source_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('segments','part_checks')}))
