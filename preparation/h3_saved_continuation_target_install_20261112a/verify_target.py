"""Read-only full dependency SHA and package-metadata verification; no models.

Default is stat-only. --full-sha explicitly reads all pinned bytes, including
the four pre-existing large models. Neither mode admits or launches an NPU run.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

INDEX_SHA='d865e85ae3a458ad402f87e478bf27645065e2d3f2373f4013c843d0eff13dde'
RESOURCES={'wall_seconds':1800,'CPU_cores':2,'address_space_bytes':4 << 30}

def sha_file(path):
    h=hashlib.sha256()
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError('not a regular file')
        for chunk in iter(lambda:f.read(1 << 20),b''):h.update(chunk)
        if identity(os.fstat(f.fileno()))!=identity(before):raise ValueError('opened file changed during SHA read')
    return h.hexdigest()

def identity(st):return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns

def check_file(row,full_sha):
    result={'path':row['path'],'roles':row['roles'],'expected_sha256':row['expected_sha256'],'expected_bytes':row['expected_bytes']}
    path=Path(row['path'])
    try:
        before=path.lstat();result['actual_bytes']=before.st_size
        if not stat.S_ISREG(before.st_mode):raise ValueError('not a regular non-symlink file')
        if row['expected_bytes'] is not None and before.st_size!=row['expected_bytes']:raise ValueError('size mismatch')
        if full_sha:
            result['actual_sha256']=sha_file(path)
            if result['actual_sha256']!=row['expected_sha256']:raise ValueError('SHA mismatch')
            if identity(path.lstat())!=identity(before):raise ValueError('identity changed during SHA read')
        result['status']='SHA_verified' if full_sha else 'stat_only_SHA_pending'
    except (OSError,ValueError) as exc:
        result.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    return result

def runtime_metadata(expected):
    # Distribution metadata only: never import torch, torch_npu, Comfy or H3.
    actual={'python':sys.version}
    for key in ('torch','torch_npu'):
        try:actual[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:actual[key]=None
    return {'expected':expected,'actual':actual,'exact_match':actual==expected,
        'scope':'sys.version and installed distribution metadata only; not import/runtime/numerical validation',
        'forbidden_modules_imported':[k for k in ('torch','torch_npu','comfy','h3_comfy') if k in sys.modules]}

def limits():
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<2:raise RuntimeError('need two available CPU cores')
    os.sched_setaffinity(0,cpus[:2]);os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
    def stop(sig,frame):raise TimeoutError('fixed 1800-second verifier bound; no retry')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(1780)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--full-sha',action='store_true');a=p.parse_args()
    raw=Path(__file__).with_name('expected_dependencies.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INDEX_SHA:raise ValueError('dependency index changed')
    index=json.loads(raw);files=index['files']
    if index['schema']!='openh3.target_dependency_index.v1' or len(files)!=1406 or len({r['path'] for r in files})!=1406:raise ValueError('fixed dependency set changed')
    pins=[r for r in files if 'frozen_implementation_pin' in r['roles']]
    large=[r for r in files if r['expected_bytes'] is not None and r['expected_bytes']>=16 << 20]
    if len(pins)!=1364 or len([r for r in large if any(role.startswith('parent_model/') for role in r['roles'])])!=4:raise ValueError('fixed pin/model count changed')
    out=a.out.absolute()
    if out.exists() or out.is_symlink() or not out.parent.is_dir() or out.parent.resolve()!=out.parent:raise ValueError('fresh output with canonical existing parent required')
    if any(Path(r['path'])==out or out in Path(r['path']).parents for r in files):raise ValueError('output overlaps dependency')
    out.mkdir();started=time.monotonic()
    report={'schema':'openh3.target_full_dependency_verification.v1','complete':False,'full_sha':a.full_sha,
        'dependency_index_sha256':INDEX_SHA,'source_inventory_sha256':index['inventory_sha256'],
        'required_paths':1406,'frozen_implementation_pins':1364,'resources':RESOURCES,
        'model_calls':0,'NPU_calls':0,'NPU_admitted':False,'automatic_retry':False,'files':[]}
    try:
        limits();report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        report['runtime']=runtime_metadata(index['runtime'])
        for row in files:report['files'].append(check_file(row,a.full_sha))
        report['failed_files']=sum(r['status']=='failed' for r in report['files'])
        report['complete']=report['failed_files']==0 and report['runtime']['exact_match'] and not report['runtime']['forbidden_modules_imported']
        report['full_dependencies_ready']=bool(report['complete'] and a.full_sha)
        report['status']='full_SHA_and_metadata_verified_not_NPU_admission' if report['full_dependencies_ready'] else ('stat_only_all_SHA_pending' if report['complete'] else 'failed_not_ready')
    except BaseException as exc:
        report.update(status='failed_not_ready',complete=False,full_dependencies_ready=False,error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.alarm(0);report.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:v for k,v in report.items() if k!='files'},sort_keys=True))
    return 0 if report['complete'] else 2

if __name__=='__main__':raise SystemExit(main())
