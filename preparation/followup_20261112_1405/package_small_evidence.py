"""Snapshot current project-owned small source and evidence without publishing."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,io,datetime
root=Path.cwd();dest=root/'preparation/followup_20261112_1405';archive=dest/'backup.tar.gz'
names=set(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],text=True).split('\0'))
names.update(subprocess.check_output(['git','diff','--name-only','-z','HEAD'],text=True).split('\0'))
files=[]; omitted=[]
for name in sorted(names):
 if not name or name.startswith('preparation/followup_20261112_1405/'):continue
 p=root/name
 if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
 if name.endswith('.tar.gz') or p.suffix=='.png':
  omitted.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='Local transport/image copy; original remote evidence retained and SHA-bound metadata included'));continue
 if p.stat().st_size>8<<20:raise ValueError(('unexpected large file',name))
 if p.suffix in ('.npy','.npz','.safetensors','.pt','.pem'):raise ValueError(('unexpected asset/key',name))
 files.append(p)
manifest=dict(schema='openh3.followup_small_backup.v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='New owned source, immutable snapshots and small scientific evidence; no model weights/large arrays/transport chunks',files={},omitted_local_transport_or_images=omitted)
with tarfile.open(archive,'x:gz') as tar:
 for p in files:
  data=p.read_bytes();rel=str(p.relative_to(root));manifest['files'][rel]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());info=tarfile.TarInfo(rel);info.size=len(data);tar.addfile(info,io.BytesIO(data))
 blob=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode();info=tarfile.TarInfo('MANIFEST.json');info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
for rel,m in manifest['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==m['sha256'],rel
(dest/'manifest.json').write_bytes(blob)
receipt=dict(files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(blob).hexdigest(),destination_host='31984',destination='/cache/zihaohe/omni_openh3_20261111/preparation/followup_20261112_1405')
(dest/'pack_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
