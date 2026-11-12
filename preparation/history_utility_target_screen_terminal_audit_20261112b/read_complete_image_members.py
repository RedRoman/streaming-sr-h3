"""Read completed PNG tar members while a single archive transfer is ongoing.
Each complete payload must match its already verified actual-audit SHA before
writing. Full archive SHA/manifest validation still happens after transfer.
"""
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile
BASE=Path(__file__).resolve().parents[2]/'results/takeover_20261111/history_utility_target_screen_completed_20261112b'
a=json.loads((BASE/'evidence/history_utility_target_screen_completed_20261112b/actual_audit.json').read_text());expected={str(Path(x['path']).relative_to('/cache/zihaohe/omni_openh3_20261111')):x['sha256'] for r in a['source_rows'] for x in r['images']};done=[]
for p in sorted(BASE.glob('images_*.tar.gz')):
 try:
  with tarfile.open(p,mode='r|gz') as t:
   for m in t:
    if m.name not in expected:continue
    rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
    b=t.extractfile(m).read();assert len(b)==m.size and hashlib.sha256(b).hexdigest()==expected[m.name]
    dest=BASE/m.name
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected[m.name]
    else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    done.append({'path':m.name,'sha256':expected[m.name]})
 except (EOFError,tarfile.ReadError):pass
(BASE/'progressive_image_payload_verification.json').write_text(json.dumps({'status':'payloads_verified_full_archive_gate_separate','files':done},indent=2)+'\n');print('complete_png_payloads_verified',len(done))
