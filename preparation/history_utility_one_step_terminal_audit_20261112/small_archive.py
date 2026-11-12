"""Allowlisted JSON/log/source evidence only; no tensor arrays or weights."""
import hashlib
import json
from pathlib import Path
import tarfile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pack(out,report,source):
    out=Path(out);source=Path(source)
    files={'input/producer_report.json':Path(report),'audit/supervision.json':out/'supervision.json',
        'audit/invocation.json':out/'invocation.json','audit/worker.log':out/'worker.log'}
    worker=out/'arrays_audit/audit.json'
    if worker.is_file():files['audit/array_audit.json']=worker
    for name in ('audit_arrays.py','run_audit_once.py','small_archive.py','freeze.json','README.md'):
        files['source/'+name]=source/name
    total=0;records=[]
    for name,p in files.items():
        if not p.is_file() or p.is_symlink():raise ValueError('missing/unsafe evidence '+str(p))
        if p.suffix not in ('.json','.log','.py','.md'):raise ValueError('non-small evidence type')
        total+=p.stat().st_size
        if total>16<<20:raise ValueError('small archive payload exceeds16MiB')
        records.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'schema':'openh3.small_evidence_manifest.v1','array_payloads_included':False,
        'arrays_remain_remote':True,'quality_verdict':'requires_review','files':records}
    mp=out/'MANIFEST.json'
    with mp.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    archive=out/'evidence.tar.gz'
    with archive.open('xb') as target:
        with tarfile.open(fileobj=target,mode='w:gz') as tar:
            tar.add(mp,arcname='MANIFEST.json',recursive=False)
            for row in records:
                p=files[row['path']]
                if sha(p)!=row['sha256']:raise ValueError('evidence changed before archive')
                tar.add(p,arcname=row['path'],recursive=False)
    # Verify archived bytes against the manifest without extracting any paths.
    with tarfile.open(archive,'r:gz') as tar:
        if len(tar.getmembers())!=len(records)+1:raise ValueError('archive count')
        for row in records:
            member=tar.getmember(row['path'])
            if not member.isfile() or member.size!=row['bytes']:raise ValueError('archive member differs')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=row['sha256']:raise ValueError('archive SHA differs')
    result={'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'manifest_sha256':sha(mp),'payload_bytes':total,'files':len(records),'arrays_included':False}
    with (out/'archive_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result
