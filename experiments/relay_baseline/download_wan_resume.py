"""Foreground public official Wan continuation with full canonical SHA256."""
import fcntl,hashlib,json,os,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
weights=ROOT/'external/RelayVSR/weights';weights.mkdir(exist_ok=True)
out=ROOT/'results/relay_baseline';out.mkdir(parents=True,exist_ok=True)
canonical=weights/'wan.safetensors';part=weights/'wan.safetensors.partial'
size=10066399024;digest='638acb6a27c17ba87b2e66a1770f17e61a99ad87cfff6fa3d3e05c2d57f2eb98'
revision='d43dc5545f9a3bd1c7730b68ebff4ab1f9199bd1'
url=f'https://huggingface.co/kopper/RelayVSR/resolve/{revision}/wan.safetensors'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(16*1024**2):h.update(block)
 return h.hexdigest()
def live(**values):
 state.update(values,epoch=time.time());p=out/'wan_download_live.json';q=p.with_suffix('.json.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p);print(json.dumps(state),flush=True)
state={'status':'starting','pid':os.getpid(),'url':url,'revision':revision,'expected_bytes':size,'expected_sha256':digest,'part':str(part),'canonical':str(canonical),'foreground_only':True}
with (weights/'wan_resume.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  if canonical.exists():
   if canonical.stat().st_size==size:
    assert sha(canonical)==digest;live(status='complete',bytes=size,sha256=digest,existing_canonical_verified=True);raise SystemExit(0)
   assert not part.exists(),'Ambiguous partial files; inspect rather than overwrite'
   canonical.replace(part);state['incomplete_canonical_moved_to_partial']=True
  start=part.stat().st_size if part.exists() else 0;assert 0<=start<=size
  began=last=time.time();mark=start;live(status='running',resume_offset=start,bytes=start)
  if start<size:
   request=urllib.request.Request(url,headers={'Range':f'bytes={start}-','User-Agent':'streaming-sr-public-baseline/1.0'})
   with urllib.request.urlopen(request,timeout=60) as response:
    assert response.status==206,(response.status,'Refuse append of non-range response')
    content_range=response.headers.get('Content-Range','');assert content_range==f'bytes {start}-{size-1}/{size}',content_range
    live(http_status=response.status,content_range=content_range)
    with part.open('ab') as stream:
     while block:=response.read(4*1024**2):
      stream.write(block);current=stream.tell();assert current<=size
      now=time.time()
      if now-last>=20 or current-mark>=64*1024**2:
       stream.flush();live(bytes=current,downloaded_this_execution=current-start,bytes_per_second_since_launch=(current-start)/(now-began));last=now;mark=current
  assert part.stat().st_size==size;live(status='verifying_full_sha256',bytes=size)
  actual=sha(part);assert actual==digest,(actual,digest)
  part.replace(canonical);live(status='complete',bytes=size,sha256=actual,seconds=time.time()-began)
 except Exception as e:live(status='failed',exception=repr(e),bytes=part.stat().st_size if part.exists() else None);raise
