import hashlib,json,os,sys,time,unittest
from pathlib import Path
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'code'))
sys.path.insert(0,str(p/'code/tests'))
started=time.monotonic()
import torch
import test_openh3_train_trajectory
with (p/'cpu_tests.log').open('x') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_openh3_train_trajectory))
receipt={'schema':'openh3.checkpoint_options_snapshot_cpu.v1','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'passed':result.wasSuccessful() and not result.skipped,'seconds':time.monotonic()-started,'torch':torch.__version__,'torch_npu_imported':'torch_npu' in sys.modules,'NPU_calls':0,'source_sha256':{str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((p/'code').rglob('*.py'))},'cpu_tests_log_sha256':hashlib.sha256((p/'cpu_tests.log').read_bytes()).hexdigest(),'comfy_reference_root':os.environ.get('OPENH3_COMFY_REFERENCE_ROOT')}
(p/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
print((p/'cpu_tests.log').read_text(),flush=True)
raise SystemExit(0 if receipt['passed'] else 1)
