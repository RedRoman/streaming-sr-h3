"""Verify exact transferred parts and add files without replacing any existing bytes."""
import datetime, hashlib, json, pathlib, tarfile, shutil
parts=pathlib.Path('/cache/zihaohe/omni_migration_parts_20261112')
root=pathlib.Path('/cache/zihaohe/omni_openh3_20261111')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
m=json.loads((parts/'manifest.json').read_text())
assert m['archive_sha256']=='57e4c935fea095073355676c74548a8a7367666bd61020b13a37041288c4d383'
for r in m['parts']:
 p=parts/r['name'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
archive=parts/'payload.verified.tar'
with archive.open('xb') as out:
 for r in m['parts']:
  with (parts/r['name']).open('rb') as f:shutil.copyfileobj(f,out,1<<20)
assert archive.stat().st_size==m['total_bytes'] and sha(archive)==m['archive_sha256']
created=[];identical=[]
with tarfile.open(archive) as tar:
 members=tar.getmembers();seen=set()
 for member in members:
  rel=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and member.name not in seen
  seen.add(member.name);out=root/member.name
  assert out.resolve().is_relative_to(root.resolve()),out
  assert not any(p.is_symlink() for p in [out,*out.parents] if p!=root.parent),out
  if out.exists():
   expected=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
   assert out.is_file() and out.stat().st_size==member.size and sha(out)==expected,out
   identical.append(member.name)
 for member in members:
  if member.name in identical:continue
  out=root/member.name;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:shutil.copyfileobj(tar.extractfile(member),f,1<<20)
  created.append(member.name)
manifest=root/'preparation/migration_31112_20261112/transfer_manifest.json'
files=json.loads(manifest.read_text())['files']
for rel,meta in files.items():
 p=root/rel;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],p
qpath=root/'preparation/train_vae_cache_portable_20261112d_31112/queue.json';q=json.loads(qpath.read_text())
assert sha(qpath)=='554d456c8849a700e7c13fa37bbece984e42d66b51efc65ba28957479bc1e5fe'
for rel,digest in q['implementation_sha256'].items():
 p=root/rel;assert p.is_file() and sha(p)==digest,p
result=dict(schema='openh3.host_migration_verified.v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_host='30794',destination_host='31112',archive_sha256=sha(archive),verified_files=len(files),new_files=len(created),existing_identical=identical,queue_sha256=sha(qpath),queue_pins_verified=len(q['implementation_sha256']),NPU_calls=0,overwrites=0)
out=root/'preparation/migration_31112_20261112/target_verification.json'
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
