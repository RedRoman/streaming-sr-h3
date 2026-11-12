from pathlib import Path,PurePosixPath
import hashlib,json,tarfile
BASE=Path(__file__).resolve().parents[2]/'results/takeover_20261111/history_utility_target_screen_completed_20261112b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((BASE/'archive_receipt.json').read_text());done=[]
for pack in r['archives']:
 p=BASE/Path(pack['path']).name
 if not p.exists() or p.stat().st_size<pack["bytes"]:continue
 assert p.stat().st_size==pack['bytes'] and sha(p)==pack['sha256']
 with tarfile.open(p) as t:
  members=t.getmembers();assert len(members)==pack['payload_count']+1
  for m in members:
   rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
  manifest=json.loads(t.extractfile('MANIFEST.json').read());assert len(manifest['payloads'])==pack['payload_count']
  wanted={x['path']:x for x in manifest['payloads']};assert set(wanted)=={m.name for m in members if m.name!='MANIFEST.json'}
  for name,row in wanted.items():
   b=t.extractfile(name).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];dest=BASE/name
   if dest.exists():assert dest.read_bytes()==b
   else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  q=BASE/'archive_manifests'/('manifest_'+p.name+'.json');q.parent.mkdir(exist_ok=True);q.write_text(json.dumps(manifest,indent=2)+'\n')
 done.append({'name':p.name,'sha256':pack['sha256'],'payload_count':pack['payload_count']})
audit=BASE/'evidence/history_utility_target_screen_completed_20261112b/actual_audit.json'
if audit.exists():
 assert sha(audit)==r['actual_audit_sha256'];a=json.loads(audit.read_text());index=[]
 for x in a['source_rows']:
  y=dict(x)
  for key in ['all_frame_sheet','ranked_pairs']:
   if key in y:
    target=BASE/Path(y[key]['path']).relative_to('/cache/zihaohe/omni_openh3_20261111');y[key]={**y[key],'local_path':str(target),'downloaded':target.exists()}
  if 'full_resolution_pairs' in y:y['full_resolution_pairs']=[{**v,'local_path':str(BASE/Path(v['path']).relative_to('/cache/zihaohe/omni_openh3_20261111'))} for v in y['full_resolution_pairs']]
  index.append(y)
 eligible=[x for x in index if 'all_frame_sheet' in x]
 receipt={'schema':'openh3.history_utility_target_screen_local_inventory.v1','rows':index,'visual_shards':{'inspect_architecture':[x['source_id'] for x in eligible[:19]],'root':[x['source_id'] for x in eligible[19:]]},'archives_verified':done,'all_archives_downloaded':len(done)==len(r['archives']),'local_pixels_recomputed':False,'remote_actual_numpy_pixel_SHA_verified':38,'target_NPZ_downloaded':False}
 (BASE/'image_inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'archives_verified':len(done),'payloads_verified':sum(x['payload_count'] for x in done),'all_archives':len(done)==len(r['archives'])}))
