"""Run only the frozen nonzero probe's default CPU inspection, once.

No admission, device query, model import or execution flag is permitted here.
The separately frozen NPU launcher is never imported or called.
"""
import contextlib
import datetime
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path('/cache/zihaohe/omni_openh3_20261111')
PREP = ROOT / 'preparation/h3_nonzero_adapter_probe_20261112b_31112'
OUT = ROOT / 'preparation/h3_nonzero_adapter_probe_20261112b_31112_cpu_inspection'
QUEUE_SHA = 'c5e9e2a2b937994da827000d078df7ab572463a262f560f5c2f8be451d0f4598'
CONTRACT_SHA = '559e0fc5fa0ebd4d326ce0a364c258d90f08dd906f313f73847b3523ecc13f38'
PRODUCER_SHA = 'ab710d9a365ac8f502aea177f8993354e2a17d086de97cc71a0e4b265ae99f6b'
FORBIDDEN = ('torch', 'torch_npu', 'comfy', 'comfy_kitchen', 'h3_comfy')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


class NoModelImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise RuntimeError('CPU inspection forbids model/device import: ' + fullname)
        return None


def main():
    for path in (OUT/'receipt.json', OUT/'inspection', OUT/'inspection.log'):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
        raise ValueError('Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before Python starts')
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError('model/device module was already imported')
    guard = NoModelImports()
    receipt = {'schema':'openh3.nonzero_adapter_actual_CPU_inspection.v1',
        'complete':False,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':'huawei-modelarts-31112','wrapper_sha256':sha(__file__),
        'scope':'Only frozen default CPU CLI, metadata/pins and saved reference arrays; no model/device import, card query, NPU admission or execution.',
        'python':sys.executable,'python_version':sys.version,
        'TORCH_DEVICE_BACKEND_AUTOLOAD':os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],
        'NPU_launcher_called':False,'admission_created':False,'run_directory_created':False}
    started = time.monotonic()
    sys.meta_path.insert(0, guard)
    try:
        for path, expected in ((PREP/'queue.json',QUEUE_SHA),
                (PREP/'execution_contract.json',CONTRACT_SHA),
                (PREP/'code/check_h3_nonzero_adapter.py',PRODUCER_SHA)):
            if sha(path) != expected:
                raise ValueError('frozen source differs: ' + str(path))
        queue = json.loads((PREP/'queue.json').read_text())
        pins = queue['implementation_sha256']
        if len(pins) != 1289:
            raise ValueError('unexpected queue pin count')
        actual = {}
        for relative, expected in pins.items():
            path = ROOT/relative
            actual[relative] = sha(path)
            if actual[relative] != expected:
                raise ValueError('queue pin mismatch: ' + relative)
        receipt['queue_sha256'] = QUEUE_SHA
        receipt['execution_contract_sha256'] = CONTRACT_SHA
        receipt['queue_pins_checked'] = len(actual)
        receipt['actual_pin_map_sha256'] = hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        write(OUT/'verified_queue_pins.json',actual)
        receipt['verified_queue_pins_file_sha256'] = sha(OUT/'verified_queue_pins.json')
        argv = [str(PREP/'code/check_h3_nonzero_adapter.py'),
            '--pretrained-manifest',str(ROOT/'configs/openh3_20261111/refiner_turbo_30794.json'),
            '--execution-contract',str(PREP/'execution_contract.json'),
            '--out',str(OUT/'inspection')]
        receipt['argv'] = argv
        if '--execute' in argv:
            raise AssertionError('NPU execution flag forbidden')
        old_argv, old_path = sys.argv, list(sys.path)
        try:
            sys.argv = argv
            sys.path.insert(0,str(PREP/'code'))
            with (OUT/'inspection.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                runpy.run_path(argv[0],run_name='__main__')
        finally:
            sys.argv, sys.path = old_argv, old_path
        inspection = json.loads((OUT/'inspection/inspection.json').read_text())
        if (inspection.get('inspection_only') is not True or inspection.get('runtime_files_verified') != 1288
                or inspection.get('asset_bytes_sha_verified') is not False):
            raise ValueError('unexpected default inspection contract')
        receipt['inspection_file_sha256'] = sha(OUT/'inspection/inspection.json')
        receipt['inspection'] = inspection
        receipt['large_asset_verification_scope'] = 'All recipe files size-only; context and prior joint artifacts/arrays actual SHA verified by default producer. No full VAE/DiT/adapter weight SHA claim.'
        receipt['complete'] = True
    except BaseException as error:
        receipt['failure'] = {'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        raise
    finally:
        sys.meta_path.remove(guard)
        receipt['forbidden_import_attempts'] = guard.attempts
        receipt['actual_forbidden_modules_loaded'] = [name for name in sys.modules if name.split('.')[0] in FORBIDDEN]
        if guard.attempts or receipt['actual_forbidden_modules_loaded']:
            receipt['complete'] = False
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['NPU_run_directory_exists_after'] = (ROOT/'runs/h3_nonzero_adapter_probe_20261112b_31112').exists()
        receipt['NPU_admission_exists_after'] = (PREP/'admission.json').exists()
        if (OUT/'inspection.log').is_file():
            receipt['log_sha256'] = sha(OUT/'inspection.log')
        write(OUT/'receipt.json',receipt)
    print(json.dumps({'complete':receipt['complete'],'pins_checked':receipt['queue_pins_checked'],
        'receipt':str(OUT/'receipt.json'),'receipt_sha256':sha(OUT/'receipt.json')},sort_keys=True))


if __name__ == '__main__':
    main()
