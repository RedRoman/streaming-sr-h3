"""Verify a frozen content-addressed archive; explicit commit never overwrites.

Only stdlib. No transfer, model import or NPU. POSIX directory descriptors and
O_NOFOLLOW constrain installs below an existing project root. The transaction
has a rollback journal; it is not a globally atomic multi-file rename. A crash
can leave a partial install. Reconciliation requires a new explicit run, never
an automatic retry. Existing identical files are preserved and skipped.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import stat
import tarfile
import time

ARCHIVE_SHA = 'b26949c1b9c8b9fbd98d94249146c4135a19dda16ec824a465c3987124fd3f2a'
MANIFEST_SHA = 'b862d92543b1e994d8896d75936fb94c081ffd2cb2be4e8bd3de65e8e8e20d20'
ARCHIVE_BYTES = 718719172
RESOURCES = {'wall_seconds':900, 'CPU_cores':2, 'address_space_bytes':4 << 30}

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def stamp(st): return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

def fresh_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def paths_and_blobs(manifest):
    need(manifest['schema']=='openh3.content_addressed_missing_dependency_archive.v1', 'manifest schema')
    files = manifest['files']; paths = {}; blobs = {}
    for row in files:
        name = row['path']; p = PurePosixPath(name)
        need(isinstance(name,str) and name and not p.is_absolute() and '..' not in p.parts
             and '.' not in p.parts and str(p)==name and name not in paths, 'unsafe/duplicate destination')
        h = row['blob_sha256']; n = row['bytes']
        need(isinstance(h,str) and len(h)==64 and all(x in '0123456789abcdef' for x in h)
             and type(n) is int and n>=0, 'invalid blob identity')
        need(not any(x.startswith('parent_model/') for x in row['roles']), 'parent model must not be installed')
        need(h not in blobs or blobs[h]==n, 'same blob has inconsistent size')
        paths[name] = row; blobs[h] = n
    need(len(files)==manifest['path_count'] and len(blobs)==manifest['blob_count']
         and sum(blobs.values())==manifest['unique_source_bytes'] and sum(blobs.values())<=1300000000, 'manifest count/size cap')
    # A listed file cannot also be another listed file's directory.
    need(not any(str(p) in paths for name in paths for p in PurePosixPath(name).parents if str(p)!='.'), 'file/directory destination conflict')
    return paths, blobs

def verified_archive(archive, expected_sha, manifest_bytes, blobs, stage=None):
    """No extract/extractall: stream only exact declared regular members."""
    archive = Path(archive); before = archive.stat()
    need(not archive.is_symlink() and stat.S_ISREG(before.st_mode), 'archive must be a regular file')
    need(sha(archive)==expected_sha and stamp(archive.stat())==stamp(before), 'archive SHA or identity differs')
    allowed = {'blobs/'+h:n for h,n in blobs.items()}; allowed['manifest.json']=len(manifest_bytes)
    seen=set()
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            name=member.name
            need(name in allowed and name not in seen and member.isfile() and not member.issparse()
                 and not member.pax_headers and not tar.pax_headers and member.size==allowed[name], 'unexpected/unsafe/duplicate archive member')
            seen.add(name); src=tar.extractfile(member); need(src is not None, 'missing regular member stream')
            h=hashlib.sha256(); n=0; parts=[]; target=None
            if stage is not None and name.startswith('blobs/'):
                target=(Path(stage)/name.split('/')[1]).open('xb')
            try:
                while True:
                    chunk=src.read(1 << 20)
                    if not chunk: break
                    n+=len(chunk); need(n<=member.size, 'member byte overflow'); h.update(chunk)
                    if name=='manifest.json': parts.append(chunk)
                    if target: target.write(chunk)
                if target: target.flush(); os.fsync(target.fileno())
            finally:
                src.close()
                if target: target.close()
            need(n==member.size, 'member truncated')
            if name=='manifest.json': need(b''.join(parts)==manifest_bytes, 'internal manifest differs')
            else:
                need(h.hexdigest()==name.split('/')[1], 'blob SHA differs')
                if stage is not None: os.chmod(Path(stage)/name.split('/')[1],0o444)
    need(seen==set(allowed) and stamp(archive.stat())==stamp(before), 'archive incomplete or changed')
    return {'members_verified':len(seen), 'unique_blob_SHA_verified':len(blobs), 'archive_SHA_verified':True}

def parent_fd(root_fd, relative, create=False, made=None):
    parts=PurePosixPath(relative).parts; fd=os.dup(root_fd); partial=[]
    try:
        for part in parts[:-1]:
            partial.append(part)
            try: child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            except FileNotFoundError:
                if not create: os.close(fd); return None
                os.mkdir(part,0o755,dir_fd=fd)
                child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                if made is not None: made.append(('/'.join(partial),os.fstat(child).st_dev,os.fstat(child).st_ino))
            os.close(fd); fd=child
        return fd
    except BaseException:
        os.close(fd); raise

def existing_identity(root_fd, row):
    fd=parent_fd(root_fd,row['path'])
    if fd is None:return None
    try:
        try:f=os.open(PurePosixPath(row['path']).name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        except FileNotFoundError:return None
        try:
            st=os.fstat(f);need(stat.S_ISREG(st.st_mode) and st.st_size==row['bytes'],'existing destination size/type mismatch: '+row['path'])
            h=hashlib.sha256()
            with os.fdopen(f,'rb',closefd=False) as handle:
                for chunk in iter(lambda:handle.read(1 << 20),b''):h.update(chunk)
            need(h.hexdigest()==row['blob_sha256'] and stamp(os.fstat(f))==stamp(st),'existing destination SHA/identity mismatch: '+row['path'])
            return st.st_dev,st.st_ino
        finally:os.close(f)
    finally:os.close(fd)

def journal(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def independent_copy(stage, row):
    """A separate inode for each logical destination; never hardlink the blob."""
    temporary=Path(stage)/('destination_'+hashlib.sha256(row['path'].encode()).hexdigest())
    h=hashlib.sha256();n=0
    with (Path(stage)/row['blob_sha256']).open('rb') as src,temporary.open('xb') as dst:
        for chunk in iter(lambda:src.read(1 << 20),b''):
            n+=len(chunk);need(n<=row['bytes'],'staged blob exceeds declared size')
            h.update(chunk);dst.write(chunk)
        dst.flush();os.fsync(dst.fileno())
    need(n==row['bytes'] and h.hexdigest()==row['blob_sha256'],'staged copy SHA/size mismatch')
    os.chmod(temporary,0o444)
    return temporary

def commit(root, paths, stage, journal_path, before_link=None):
    root=Path(root);need(root.is_absolute() and not root.is_symlink() and root.resolve()==root,'canonical existing root required')
    root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);created=[];made=[];skipped=[]
    try:
        # Check all conflicts before creating directories or committing any blob.
        pre={name:existing_identity(root_fd,row) for name,row in paths.items()}
        for name,row in paths.items():
            if pre[name] is not None:
                need(existing_identity(root_fd,row)==pre[name],'pre-existing destination changed');skipped.append(name);continue
            temporary=independent_copy(stage,row)
            fd=parent_fd(root_fd,name,create=True,made=made)
            try:
                if before_link is not None: before_link(name)
                # link is exclusive: EEXIST fails even if the racer has equal bytes.
                os.link(temporary,PurePosixPath(name).name,dst_dir_fd=fd,follow_symlinks=False)
                st=os.stat(PurePosixPath(name).name,dir_fd=fd,follow_symlinks=False)
                created.append((name,st.st_dev,st.st_ino));os.fsync(fd)
                journal(journal_path,{'action':'created','path':name,'device':st.st_dev,'inode':st.st_ino,'blob_sha256':row['blob_sha256']})
                temporary.unlink() # Only this newly created staging path, never an existing target.
            finally:os.close(fd)
        for name,row in paths.items(): need(existing_identity(root_fd,row) is not None,'final destination missing')
        need(len({(dev,ino) for _,dev,ino in created})==len(created),'new targets share inode')
        return {'installed_paths':len(created),'existing_identical_preserved':len(skipped),'final_paths_SHA_verified':len(paths),'new_destination_inodes_independent':True,'rollback':False}
    except BaseException as error:
        failures=[]
        for name,dev,ino in reversed(created):
            try:
                fd=parent_fd(root_fd,name)
                if fd is None:raise RuntimeError('owned destination disappeared')
                try:
                    st=os.stat(PurePosixPath(name).name,dir_fd=fd,follow_symlinks=False)
                    need((st.st_dev,st.st_ino)==(dev,ino),'rollback refuses to remove replaced destination')
                    os.unlink(PurePosixPath(name).name,dir_fd=fd);os.fsync(fd)
                    journal(journal_path,{'action':'rollback_owned_file','path':name})
                finally:os.close(fd)
            except BaseException as exc:failures.append({'path':name,'error':str(exc)})
        for name,dev,ino in reversed(made):
            try:
                fd=parent_fd(root_fd,name)
                if fd is None:continue
                try:
                    st=os.stat(PurePosixPath(name).name,dir_fd=fd,follow_symlinks=False)
                    if (st.st_dev,st.st_ino)==(dev,ino):os.rmdir(PurePosixPath(name).name,dir_fd=fd)
                finally:os.close(fd)
            except OSError:pass # Non-empty or externally changed directory is preserved.
        journal(journal_path,{'action':'rollback_terminal','original_error':str(error),'rollback_failures':failures})
        if failures:raise RuntimeError('rollback incomplete; stop and reconcile owned journal: '+str(failures)) from error
        raise
    finally:os.close(root_fd)

def limits():
    cpus=sorted(os.sched_getaffinity(0));need(len(cpus)>=2,'need2CPU');os.sched_setaffinity(0,cpus[:2]);os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
    def stop(sig,frame):raise TimeoutError('fixed900s/signal, no automatic retry')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(890) # Reserve10seconds for rollback/report; outer guard is900seconds.

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    raw=Path(__file__).with_name('expected_manifest.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==MANIFEST_SHA,'frozen manifest changed')
    manifest=json.loads(raw);paths,blobs=paths_and_blobs(manifest)
    need(len(paths)==346 and len(blobs)==150 and a.archive.stat().st_size==ARCHIVE_BYTES,'fixed package count/size changed')
    root=Path(manifest['root']);out=a.out.absolute();need(not out.exists() and not out.is_symlink(),'fresh receipt directory required')
    need(root.is_absolute() and root.resolve()==root and not root.is_symlink(),'canonical existing root required')
    need(out.parent.is_dir() and out.parent.resolve()==out.parent and out.parent.is_relative_to(root),'new output must have existing canonical parent below root')
    need(not any(out==root/n or out in (root/n).parents for n in paths),'transaction output overlaps destination')
    out.mkdir();start=time.monotonic();result={'schema':'openh3.missing_dependency_target_install.v1','complete':False,'execute':a.execute,'resources':RESOURCES,'archive_sha256':ARCHIVE_SHA,'manifest_sha256':MANIFEST_SHA,'model_calls':0,'NPU_calls':0,'existing_files_overwritten':0,'automatic_retry':False,'global_atomic_visibility':False}
    try:
        limits();result['cpu_affinity']=sorted(os.sched_getaffinity(0));stage=None
        if a.execute:
            disk=os.statvfs(out);result['free_disk_bytes_before']=disk.f_bavail*disk.f_frsize
            need(result['free_disk_bytes_before']>=3*(1 << 30),'need at least3GiB free for staging and independent target copies')
        if a.execute:stage=out/'staged_blobs';stage.mkdir()
        result['archive_verification']=verified_archive(a.archive,ARCHIVE_SHA,raw,blobs,stage)
        if a.execute:result['transaction']=commit(root,paths,stage,out/'transaction.jsonl')
        else:
            fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:identical=sum(existing_identity(fd,row) is not None for row in paths.values())
            finally:os.close(fd)
            result['destination_inspection']={'identical_existing_paths':identical,'missing_paths':len(paths)-identical,'installed_paths':0}
        result.update(complete=True,status='installed_missing_only_no_NPU_admission' if a.execute else 'archive_and_destinations_verified_no_install')
    except BaseException as exc:result.update(status='failed_preserved_no_retry',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.alarm(0);result.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        fresh_json(out/'receipt.json',result)
    print(json.dumps(result,sort_keys=True));return 0 if result['complete'] else 2

if __name__=='__main__':raise SystemExit(main())
