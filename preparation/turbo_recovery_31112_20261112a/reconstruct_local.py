#!/usr/bin/env python3
"""Reconstruct a verified original part into a fresh path; never alter retained partials."""
import argparse,datetime,hashlib,json,os,pathlib,stat

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stable(path):
 s=path.stat();assert stat.S_ISREG(s.st_mode) and not path.is_symlink();return s.st_size,s.st_mtime_ns,s.st_ino

def reconstruct(prefix, suffixes, dest, expected_bytes, expected_sha256):
 """suffixes is ordered explicit (path, offset, bytes, SHA); strict no-overwrite output."""
 assert not dest.exists() and not dest.is_symlink(),f'output exists: {dest}'
 expected_offset=prefix[1];pieces=[(prefix[0],0,prefix[1],prefix[2])]+list(suffixes)
 for p,off,size,sha in pieces:
  before=stable(p);assert before[0]==size;assert digest(p)==sha;assert stable(p)==before
 for p,off,size,sha in suffixes:
  assert off==expected_offset,'gap/overlap/reordered suffix';expected_offset+=size
 assert expected_offset==expected_bytes
 partial=dest.with_suffix('.partial');checks=[];full=hashlib.sha256()
 with partial.open('xb') as out:
  for p,off,size,sha in pieces:
   before=stable(p);h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(4*1024*1024),b''):out.write(b);h.update(b);full.update(b)
   assert stable(p)==before and h.hexdigest()==sha
   checks.append({'path':str(p),'offset':off,'bytes':size,'sha256':sha})
  out.flush();os.fsync(out.fileno())
 assert partial.stat().st_size==expected_bytes and full.hexdigest()==expected_sha256
 assert digest(partial)==expected_sha256
 os.link(partial,dest);partial.unlink()
 return {'path':str(dest),'bytes':expected_bytes,'sha256':expected_sha256,'inputs':checks}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--part',required=True);a=ap.parse_args();base=pathlib.Path(__file__).resolve().parent
 request=json.loads((base/'prefix_receipt.json').read_text());source=json.loads((base/'source_receipt.json').read_text());jobs=json.loads((base/'recovery_jobs.json').read_text())
 assert source['complete'] and source['request_sha256']==digest(base/'prefix_receipt.json')
 row=next(x for x in request['parts'] if x['name']==a.part)
 ss=sorted([x for x in source['segments'] if x['part']==a.part],key=lambda x:x['offset']);mapped={x['name']:x for x in jobs}
 for x in ss:assert mapped[x['name']]['download_verified'] is True
 outdir=base/'reconstructed';outdir.mkdir(exist_ok=True)
 receipt=reconstruct((pathlib.Path(row['local_partial']),row['prefix_bytes'],row['prefix_sha256']),[(base/'segments'/s['name'],s['offset'],s['bytes'],s['sha256']) for s in ss],outdir/a.part,row['expected_bytes'],row['expected_sha256'])
 # The complete, verified part is split for bounded uploads. These are new files.
 dest=base/'upload_segments';dest.mkdir(exist_ok=True);upload=[]
 with (outdir/a.part).open('rb') as f:
  off=0;i=0
  while off<row['expected_bytes']:
   b=f.read(min(16*1024*1024,row['expected_bytes']-off));name=f'{a.part}_chunk_{i:03d}.bin';p=dest/name
   with p.open('xb') as g:g.write(b);g.flush();os.fsync(g.fileno())
   sha=hashlib.sha256(b).hexdigest();assert digest(p)==sha
   upload.append({'name':name,'part':a.part,'local_path':str(p),'offset':off,'bytes':len(b),'sha256':sha,'parent_bytes':row['expected_bytes'],'parent_sha256':row['expected_sha256'],'upload_started':False,'upload_verified':False,'upload_session':f'omni-tup-{a.part[5:]}-{i:03d}-20261112a'})
   off+=len(b);i+=1
 receipt.update(schema='openh3.turbo_part_reconstruction.v1',complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),upload_segments=upload)
 with (outdir/(a.part+'.receipt.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps({'part':a.part,'complete':True,'bytes':row['expected_bytes'],'sha256':row['expected_sha256'],'upload_segments':len(upload)}))
if __name__=='__main__':main()
