from pathlib import Path
import datetime,hashlib,json,re,tarfile,sys
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads((root/'source_sha256.json').read_text())
assert all(sha(root/p)==v for p,v in pins.items())
assert pins['code/export_train_vae_cache.py']=='5e2e18f3325ed2473bba088ce4fb1d846b0fa356319a1ff64e0c60958f7714a1'
log=(root/'cpu_tests.log').read_text();m=re.search(r"Ran (\d+) tests in ([0-9.]+)s",log)
assert m and int(m[1])==9 and log.rstrip().endswith('OK') and 'skipped=' not in log
import torch
receipt={'schema':'openh3.train_vae_cache_portable_cpu_validation.v1','complete':True,
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'huawei-modelarts-30794',
 'source_sha256':pins,'tests':{'passed':9,'skipped':0,'seconds':float(m[2]),'log':'cpu_tests.log','log_sha256':sha(root/'cpu_tests.log')},
 'environment':{'python':sys.executable,'torch':torch.__version__,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1'},
 'scope':'CPU synthetic file-admission/delegation tests. Actual immutable VAE-only loader function bodies executed via AST with fake torch/H3, not a real device/model. No new numerical VAE path: input inspection, preprocessing, encode_pair, execute and loader remain from unchanged pinned legacy sources. No model weights loaded, no actual VAE encode/decode, no NPU, no launch configuration.',
 'verified_boundaries':['Complete parent manifest SHA/schema/component order/size-and-digest metadata preserved',
 'New fingerprint exactly equals old full-manifest algorithm with the same metadata',
 'Unused base/refiner/turbo paths may be absent and are never resolved, stat-ed or opened',
 'Used VAE/context size and actual SHA mandatory; same-size corruption fails',
 'Partial full-recipe verification never marked true or admitted as DiT preload',
 'Original loader AST only verifies VAE/context and calls H3(load_dit=False,verify_sha=True); scoped binding restores',
 'Original execute receives identical raw/Rphi input objects; no GT/degradation changes',
 'Partial, success and failure reports accurately mark unused assets metadata-only; complete resets on failure'],
 'not_executed':['actual31112 portable main input inspection','real VAE2encode1decode','NPU admission'],
 'prior_numeric_CPU_evidence':{'receipt':'results/takeover_20261111/train_vae_cache_cpu_validation_20261112/cpu_receipt.json',
 'sha256':'935b1668c21eb7f399c8abf8217b40a906dd7bcd0f7c31b5f71c346e16573310','repeated':False},
 'execution_schema':'openh3.train_vae_cache_portable_execution.v1',
 'additional_execution_fields':{'legacy_entry_sha256':pins['code/export_train_vae_cache.py'],'asset_verification_scope':'only_used_components_checked; unused_components_metadata_only'}}
(root/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
with tarfile.open(root/'cpu_evidence.tar.gz','w:gz') as tar:
 for rel in list(pins)+['source_sha256.json','cpu_receipt.json','cpu_tests.log','write_cpu_receipt.py']:
  tar.add(root/rel,arcname=rel)
print(json.dumps({'receipt_sha256':sha(root/'cpu_receipt.json'),'archive_sha256':sha(root/'cpu_evidence.tar.gz'),'archive_bytes':(root/'cpu_evidence.tar.gz').stat().st_size}))
