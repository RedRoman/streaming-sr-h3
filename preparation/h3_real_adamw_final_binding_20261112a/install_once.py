"""Install root-authorized, unstarted candidate bindings; never launch a job."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil

ROOT=Path('/cache/zihaohe/omni_openh3_20261111')
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write_new(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(data)
def main():
    contract=json.loads((HERE/'install_manifest.json').read_text())
    if (HERE/'installation_receipt.json').exists():raise FileExistsError('installation already attempted')
    for run in contract['must_not_exist']:
        if (ROOT/run).exists():raise FileExistsError('run/queue already exists: '+run)
    # All old candidate content is verified before creating backups or replacing anything.
    for rel,h in contract['expected_existing'].items():
        if sha(ROOT/rel)!=h:raise ValueError('prior candidate SHA changed: '+rel)
    for rel,h in contract['payload_sha256'].items():
        source=HERE/'payload'/rel
        if sha(source)!=h:raise ValueError('payload SHA mismatch: '+rel)
        if not source.resolve().is_relative_to((HERE/'payload').resolve()):raise ValueError('payload escaped namespace')
    for prefix in contract['snapshot_prefixes']:
        destination=ROOT/prefix/'pending_snapshot'
        if destination.exists():raise FileExistsError('pending snapshot exists')
    for original,backup in contract['snapshot_files'].items():
        write_new(ROOT/backup,(ROOT/original).read_bytes())
        if sha(ROOT/backup)!=contract['expected_existing'][original]:raise ValueError('pending snapshot copy changed')
    for rel,h in contract['payload_sha256'].items():
        destination=ROOT/rel;data=(HERE/'payload'/rel).read_bytes()
        if destination.exists():
            if sha(destination)==h:continue
            if rel not in contract['replace_after_backup']:raise ValueError('unauthorized existing-file replacement: '+rel)
            if sha(destination)!=contract['expected_existing'][rel]:raise ValueError('old file changed after backup')
            temporary=destination.with_name(destination.name+'.final_binding_tmp')
            write_new(temporary,data);os.replace(temporary,destination)
        else:write_new(destination,data)
    for rel,h in contract['payload_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('installed SHA mismatch: '+rel)
    # Pure metadata/source validation: runner validate() has no hardware/model calls.
    runner=ROOT/contract['runner'];spec=importlib.util.spec_from_file_location('bound_adamw_runner',runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan,pins=module.validate(ROOT/contract['queue'])
    audit_root=ROOT/contract['auditor'];spec=importlib.util.spec_from_file_location('bound_adamw_finalizer',audit_root/'finalize_once.py')
    import sys
    sys.path.insert(0,str(audit_root))
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    config,paths=audit.admission(ROOT,audit_root,audit_root/'admission.json',contract['terminal_admission_sha256'])
    audit.verify(config,paths,audit_root)
    result={'schema':'openh3.real_AdamW_final_binding_installation.v1','complete':True,
      'prior_candidate_snapshots':contract['snapshot_files'],'installed_files_sha256':contract['payload_sha256'],
      'actual_queue_runtime_pins_verified':len(pins),'queue_sha256':sha(ROOT/contract['queue']),
      'execution_contract_sha256':sha(ROOT/contract['execution_contract']),
      'terminal_admission_sha256':contract['terminal_admission_sha256'],
      'card':plan['cards'],'metadata_only':True,'model_calls':0,'NPU_calls':0,'launches':0,
      'fresh_resource_checks_performed':False,'root_must_dispatch':True}
    write_new(HERE/'installation_receipt.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps({k:result[k] for k in ('complete','actual_queue_runtime_pins_verified','queue_sha256','execution_contract_sha256','terminal_admission_sha256','card','launches')}))
if __name__=='__main__':main()
