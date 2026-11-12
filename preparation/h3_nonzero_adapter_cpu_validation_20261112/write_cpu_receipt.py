import datetime,hashlib,json,re,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads((root/'source_sha256.json').read_text())
assert all(sha(root/p)==v for p,v in pins.items())
text=(root/'cpu_tests_failure_safe.log').read_text()
match=re.search(r"Ran (\d+) tests in ([0-9.]+)s",text)
assert match and int(match[1])==10 and text.rstrip().endswith('OK') and 'skipped=' not in text
toy=next(json.loads(line)['CPU_tiny_joint'] for line in text.splitlines() if line.startswith('{"CPU_tiny_joint"'))
assert toy['adapter_tensors']==400 and toy['NFE']==12 and toy['backward']==toy['SGD_steps']==1
assert toy['actual_VAE_native_context_entries']==toy['actual_context_restorations']==76
import torch
receipt={'schema':'openh3.nonzero_adapter_cpu_validation.v1','complete':True,
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'host':'huawei-modelarts-30794','source_contract_sha256':'d3148504db9fbee2925448ea69e5f2ba83da5eba2127bf3229c9964f445bc08a',
 'source_sha256':pins,'tests':{'passed':10,'skipped':0,'seconds':float(match[2]),'log':'cpu_tests_failure_safe.log','log_sha256':sha(root/'cpu_tests_failure_safe.log')},
 'actual_CPU_tiny_path':toy,
 'environment':{'python':sys.executable,'torch':torch.__version__,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','OPENH3_COMFY_REFERENCE_ROOT':'/cache/zihaohe/omni_openh3_20261111/code/ComfyUI'},
 'scope':'Real Torch CPU arithmetic with tiny 50-block/200-linear/400-adapter model and pinned official sampler AST. No real H3/VAE weights loaded, no Comfy device imports, no NPU call, no training dataset, no production checkpoint or launch configuration. Actual arrays and safetensors are exercised in temporary test fixtures; this is not real-model numerical acceptance.',
 'fixture_IO':'Only full 400-adapter CPU test batches partial.json metadata; every actual NPY/safetensors/array hash is still written and checked. Small tests use production atomic partial writes. Production code save behavior is unchanged.',
 'development_history':[{'log':'cpu_tests_initial.log','sha256':sha(root/'cpu_tests_initial.log'),'status':'first8_passed_full400_fixture_reached_reloaded_decode_then_110s_timeout_due_repeated_JSON_IO; not a numeric pass'},
 {'log':'cpu_tests_final.log','sha256':sha(root/'cpu_tests_final.log'),'status':'9_passed_before_failure_safe_statistics_patch; not used as final_source_acceptance'},
 {'status':'Final source adds JSON-safe null for nonfinite observation statistics; tenth test proves failed merge preserves evidence before any base write.'}]}
(root/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
with tarfile.open(root/'cpu_evidence.tar.gz','w:gz') as tar:
 for rel in list(pins)+['source_sha256.json','cpu_receipt.json','cpu_tests_initial.log','cpu_tests_final.log','cpu_tests_failure_safe.log','write_cpu_receipt.py']:
  tar.add(root/rel,arcname=rel)
print(json.dumps({'receipt_sha256':sha(root/'cpu_receipt.json'),'archive_sha256':sha(root/'cpu_evidence.tar.gz'),'archive_bytes':(root/'cpu_evidence.tar.gz').stat().st_size,'tests':receipt['tests']}))
