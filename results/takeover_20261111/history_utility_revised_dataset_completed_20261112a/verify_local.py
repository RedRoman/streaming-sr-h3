"""Verify only archived small evidence; no arrays, model, or remote calls."""
import json,hashlib,datetime
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent; B=P/'payload'; repo=P.parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def vsha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
a=read(B/'audit/dataset_audit/audit.json'); c=read(B/'producer/complete.json'); status=read(B/'audit/status.json'); ps=read(B/'audit/producer_status.json'); contract=read(B/'input/producer_contract.json'); plan=read(B/'source/execution_plan.json'); freeze=read(B/'source/freeze.json')
assert status['complete'] and status['state']=='completed' and status['audit_invocations']==1 and status['returncode']==0
assert status['audit_sha256']==sha(B/'audit/dataset_audit/audit.json')=='6456d81b20392b6ba8fb16f58324b649c26ad93ad6131bca88c992bdae226abd'
assert status['producer_status_sha256']==sha(P/'original_producer_status.json')==a['producer']['status']['sha256']
assert read(P/'original_producer_status.json')==ps
assert ps['complete'] and ps['returncode']==0 and ps['report_sha256']==sha(B/'producer/complete.json')==a['producer']['complete']['sha256']
assert status['freeze_sha256']==sha(B/'source/freeze.json')=='ceb2f41aab2315e6b3f111cbcb590baa98b6e05814f02273400f10e137f0191e'
original_admission=repo/'preparation/history_utility_revised_terminal_audit_20261112a/admission.json'
assert status['admission_sha256']==sha(original_admission)
assert read(original_admission)==read(B/'audit/admission.json')
source_files=[]; historical_files=[]
for name,h in freeze['files'].items():
 p=B/'source'/name
 if p.exists(): source_files.append(name)
 else: p=repo/'preparation/history_utility_revised_terminal_audit_20261112a'/name;historical_files.append(name)
 assert sha(p)==h
for k in ['producer_contract','producer_freeze','producer_admission']:
 assert sha(B/'input'/f'{k}.json')==plan[k]['sha256']
assert sha(B/'source/execution_plan.json')=='1bee29dd7a1ab6f4e389ef7765ecd1cc80856873dd7503a5a84821a761f0faee'
assert c['execution_contract']==plan['producer_contract']
cohort_path=repo/'configs/openh3_20261111/history_utility_revised_cohort_20261112.json'; co=read(cohort_path)
assert sha(cohort_path)==c['cohort']['sha256']=='c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
assert co['primary_source_ids']==contract['selected_source_ids']
assert a['complete'] and a['revised_pair_ready'] and not a['legacy_research_ready'] and not a['training_admitted']
for k in ['training_steps','model_calls','NPU_calls','source_MP4_reads','decode_calls','actual_batch_constructions']: assert a[k]==0
assert a['actual_npz_files_pixel_metadata_verified']==100
assert c['new_codec_passes_completed']==132 and c['new_target_decodes']==0 and c['model_calls']==0 and c['NPU_calls']==0
assert not c['legacy_research_ready'] and not c['training_admitted']
assert len(c['source_results'])==25 and Counter(r['mode'] for r in c['source_results'])=={'new_materialization':22,'reused_complete':3}
assert sum(r['new_codec_passes'] for r in c['source_results'])==132
resultmap={r['source_id']:r for r in c['source_results']}; profiles=set(contract['profiles'])
ids={}; hashes={}; groups={}; rows=[]; artifact_paths=set(); replays=0
for split,n in [('train',19),('dev',6)]:
 d=a['datasets'][split]; assert d['source_count']==n and d['actual_npz_count']==4*n and d['split']==split
 assert d['dataset_sha256']==vsha(d['binding'])
 assert d['binding']['legacy_research_ready'] is False and d['binding']['revised_cohort_ready'] is True
 assert d['binding']['frozen_test_sources']==0 and d['binding']['development_selection_used'] is True
 assert d['binding']['source_ids']==co['primary_source_ids'][split]
 assert sha(B/'producer'/f'{split}_receipt_index.json')==d['receipt_index']['sha256']
 assert read(B/'producer'/f'{split}_receipt_index.json')['receipts']==c['receipt_indices'][split]
 ids[split]=set();hashes[split]=set();groups[split]=set()
 for row in d['sources']:
  s=row['source']; sid=s['source_id']; dec=co['decisions'][sid]; r=resultmap[sid]; rp=B/'receipts'/sid/'complete.json'; receipt=read(rp)
  assert s['split']==split==dec['split'] and s['source_sha256']==dec['source_sha256'] and dec['disposition']=='primary'
  assert row['window_key']==dec['window_key']==receipt['window_key']
  assert sha(rp)==row['receipt_sha256']==r['entry']['sha256']
  assert row['mode']==r['mode'] and r['new_target_decodes']==0
  assert r['verification']['old_loader_actual_files_pixels_metadata_verified']==4
  assert row['artifacts']['target']['pixel_sha256']==dec['target_pixel_sha256']
  au=row['source_authority']; assert au['kind']==row['mode'] and au['receipt_sha256']==sha(rp)
  ea=au['execution_authority']; assert ea['cohort']==c['cohort'] and ea['execution_contract']==c['execution_contract']
  assert ea['execution_contract_value_sha256']==vsha(contract)
  assert ea['input_target']['file_sha256']==dec['target_file_sha256'] and ea['input_target']['pixel_sha256']==dec['target_pixel_sha256']
  if row['mode']=='reused_complete':
   assert r['new_codec_passes']==0 and sha(B/'receipts'/sid/'reuse_admission.json')==r['entry']['revised_authority']['sha256']==au['sidecar']['sha256']
  else:
   assert r['new_codec_passes']==6 and receipt['new_codec_passes_completed']==6 and receipt['new_target_decodes']==0
   assert receipt['execution_authority']==ea
  assert set(row['artifacts'])==profiles|{'target'} and set(receipt['observations'])==profiles
  for key,artifact in row['artifacts'].items():
   assert artifact['actual_file_pixel_metadata_verified_by_frozen_loader'] is True
   assert artifact['shape']==[96,480,832,3] and artifact['dtype']=='uint8'
   rr=receipt['target'] if key=='target' else receipt['observations'][key]['artifact']
   for k in ['file_sha256','pixel_sha256','shape','dtype','bytes_on_disk']:assert artifact[k]==rr[k]
   assert artifact['actual_path'] not in artifact_paths;artifact_paths.add(artifact['actual_path'])
  for name,ob in receipt['observations'].items():
   assert ob['pixel_replay_equal'] and ob['replay_count']==2 and len(ob['replays'])==2
   assert all(q['observation_pixel_sha256']==ob['artifact']['pixel_sha256'] for q in ob['replays'])
   if row['mode']=='new_materialization': replays+=2
  ids[split].add(sid);hashes[split].add(s['source_sha256']);groups[split].add(s['source_group_id'])
  rows.append({'source_id':sid,'split':split,'source_sha256':s['source_sha256'],'mode':row['mode'],'receipt_sha256':sha(rp),'NPZ_records':4,'window_key':row['window_key']})
 assert ids[split]==set(co['primary_source_ids'][split])
assert not(ids['train']&ids['dev']) and not(hashes['train']&hashes['dev']) and not(groups['train']&groups['dev'])
assert len(artifact_paths)==100 and replays==132
assert a['read_accounting']['unique_NPZ_files_verified']==100 and a['read_accounting']['target_NPZ_additional_reads_for_near_clean_hash']==19 and a['read_accounting']['near_clean_full_frames_generated_and_hashed']==1824
for role,seed in [('teacher',20261110),('student',20261111)]:
 s=a['schedules'][role]; assert s['steps']==1000 and s['batch_size']==16 and s['rows']==16000 and s['seed']==seed
 assert set(s['source_counts'])==ids['train'] and sum(s['source_counts'].values())==16000 and sum(s['external_mask_counts'].values())==16000
 assert not s['legacy_research_ready'] and not s['training_admitted']
assert a['schedules']['teacher']['external_mask_counts']=={'[1, 1]':16000}
assert a['runtime_imports']['forbidden_loaded']==[]
assert ps['elapsed_seconds']<3600 and ps['peak_owned_group_rss_bytes']<768*1024**2
assert status['audit_elapsed_before_archive_seconds']<900 and status['peak_owned_group_rss_bytes']<4*1024**3
out={'schema':'openh3.revised_dataset_local_verification.v1','complete':True,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':read(P/'extraction_receipt.json'),'terminal_audit_sha256':sha(B/'audit/dataset_audit/audit.json'),'producer_complete_sha256':sha(B/'producer/complete.json'),'producer_original_status_sha256':sha(P/'original_producer_status.json'),'producer_archive_reserialized_status_sha256':sha(B/'audit/producer_status.json'),'producer_status_semantic_equality_despite_sorted_key_serialization':True,'admission_semantic_equality_to_original_local_pinned_file':True,'finalizer_status_sha256':sha(B/'audit/status.json'),'auditor_source_files_SHA_checked':source_files,'prior_local_cpu_log_SHA_checked':historical_files,'source_receipts_checked':rows,'original_source_role_hash_group_disjoint':True,'source_authorities_and_frozen_cohort_match':True,'recorded_target_pixel_window_and_input_file_SHA_match_frozen_cohort':True,'recorded_100_NPZ_file_pixel_metadata_rows_match_receipts':True,'new_codec_replay_records':replays,'NPZ_read_scope':'100 unique actual NPZ verified once by the frozen remote terminal loader;19 additional target reads generate1824 near-clean frames;additional metadata-only opens. This local review checks receipts only and performs0 NPZ reads.','schedule_scope':'Remote audited1000x16 rows per teacher/student. Locally validates summary counts/source boundaries/digest records, not full rows (not archived). No model/batch/training run.','local_model_calls':0,'local_NPZ_reads':0,'remote_rescans':0,'freeze_and_admission_chain_checked':True}
with (P/'local_verification.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print('PASS',sha(P/'local_verification.json'))
