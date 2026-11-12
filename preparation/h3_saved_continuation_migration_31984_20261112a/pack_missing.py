"""One bounded read-only source pack. No model import, transfer or extraction."""
import argparse,datetime,gzip,hashlib,io,json,os,resource,signal,stat,tarfile,time
from pathlib import Path

def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fresh_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def signature(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
class Bounded(Exception):pass
class HashReader:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.bytes=0
    def read(self,n=-1):
        require(0<=n<=16<<20,'unbounded tar read forbidden');b=self.f.read(n);self.h.update(b);self.bytes+=len(b);return b

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--config-sha256',required=True);args=a.parse_args()
    require(digest(args.config)==args.config_sha256,'config changed');c=json.loads(Path(args.config).read_text())
    require(c['schema']=='openh3.missing_dependency_pack.v1' and c['resources']=={'wall_seconds':600,'internal_deadline_seconds':590,'CPU_cores':2,'address_space_bytes':4<<30,'unique_source_byte_limit':1300000000},'fixed bounds differ')
    require(c['source_sha256']==digest(__file__),'source changed');root=Path(c['root']);out=Path(c['output'])
    require(out==Path(__file__).resolve().parent/'output' and not out.exists(),'fresh isolated output only')
    require(len(c['missing'])==346 and len({x['path'] for x in c['missing']})==346,'missing exact coverage differs')
    out.mkdir(exist_ok=False);started=time.monotonic();events=out/'events.jsonl';phase='starting'
    result={'schema':'openh3.missing_dependency_pack_receipt.v1','complete':False,'source_host':c['source_host'],'target_host':c['target_host'],'started_utc':now(),'PID':os.getpid(),'PGID':os.getpgrp(),'config_sha256':args.config_sha256,'source_sha256':c['source_sha256'],'inventory_sha256':c['inventory_sha256'],'resources':c['resources'],'original_files_written':0,'NPU_calls':0,'model_imports':0,'cross_host_asset_transfer':False,'extracted':False,'retries':0}
    fresh_json(out/'startup.json',result)
    def event(name,**extra):
        nonlocal phase;phase=name
        with events.open('a') as f:f.write(json.dumps({'utc':now(),'elapsed_seconds':time.monotonic()-started,'phase':name,**extra},sort_keys=True)+'\n');f.flush()
    def stop(sig,frame):raise Bounded('fixed pack limit/signal '+str(sig))
    try:
        cpus=sorted(os.sched_getaffinity(0));require(len(cpus)>=2,'two CPU cores unavailable');os.sched_setaffinity(0,set(cpus[:2]));os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.alarm(590)
        result['CPU_affinity']=sorted(os.sched_getaffinity(0));result['RLIMIT_AS']=list(resource.getrlimit(resource.RLIMIT_AS))
        rows=[];bysha={};snapshots={};event('source_stat')
        for item in c['missing']:
            path=Path(item['path']);require(path.is_absolute() and path.is_relative_to(root) and not path.is_symlink(),'source path outside root/symlink')
            require(path.resolve().is_relative_to(root.resolve()),'source parent escapes root')
            require(not any(x.startswith('parent_model/') for x in item['roles']),'parent model asset must not be copied')
            st=path.stat();require(stat.S_ISREG(st.st_mode),'source not regular');require(item['expected_bytes'] is None or st.st_size==item['expected_bytes'],'source size differs '+str(path))
            h=item['expected_sha256'];require(len(h)==64 and all(x in '0123456789abcdef' for x in h),'invalid expectedSHA')
            if h in bysha:require(bysha[h]['bytes']==st.st_size,'sameSHA inconsistent size')
            else:bysha[h]={'source_path':str(path),'bytes':st.st_size}
            snapshots[str(path)]=signature(st);rows.append({'path':str(path.relative_to(root)),'blob_sha256':h,'bytes':st.st_size,'roles':item['roles']})
        unique_bytes=sum(x['bytes'] for x in bysha.values());require(unique_bytes<=1300000000,'unique source cap exceeded')
        result.update(missing_paths=len(rows),unique_blobs=len(bysha),source_path_bytes=sum(x['bytes'] for x in rows),unique_source_bytes=unique_bytes)
        event('verifying_all_missing_source_paths',**{k:result[k] for k in ['missing_paths','unique_blobs','source_path_bytes','unique_source_bytes']})
        for index,row in enumerate(rows,1):
            path=root/row['path'];require(signature(path.stat())==snapshots[str(path)],'source changed before SHA')
            require(digest(path)==row['blob_sha256'],'source SHA differs '+str(path));require(signature(path.stat())==snapshots[str(path)],'source changed during SHA')
            if index%25==0 or index==len(rows):event('verifying_all_missing_source_paths',verified_paths=index,total_paths=len(rows))
        manifest={'schema':'openh3.content_addressed_missing_dependency_archive.v1','root':str(root),'source_host':c['source_host'],'target_host':c['target_host'],'inventory_sha256':c['inventory_sha256'],'all_source_paths_SHA_verified':True,'files':rows,'unique_source_bytes':unique_bytes,'blob_count':len(bysha),'path_count':len(rows),'no_extraction_performed':True,'four_existing_parent_models_excluded':True}
        manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();fresh_json(out/'manifest.json',manifest)
        archive=out/'missing_dependencies.tar.gz';event('archiving_unique_blobs',unique_blobs=len(bysha),unique_source_bytes=unique_bytes)
        with archive.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w|') as tar:
                    for index,(h,item) in enumerate(sorted(bysha.items()),1):
                        path=Path(item['source_path']);require(signature(path.stat())==snapshots[str(path)],'source changed before archive')
                        info=tarfile.TarInfo('blobs/'+h);info.size=item['bytes'];info.mode=0o444;info.mtime=0
                        with path.open('rb') as f:
                            hr=HashReader(f);tar.addfile(info,hr);require(hr.bytes==item['bytes'] and hr.h.hexdigest()==h,'archive source stream differs')
                        require(signature(path.stat())==snapshots[str(path)],'source changed during archive')
                        if index%10==0 or index==len(bysha):event('archiving_unique_blobs',packed_blobs=index,total_blobs=len(bysha),archive_bytes_so_far=raw.tell())
                    info=tarfile.TarInfo('manifest.json');info.size=len(manifest_bytes);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(manifest_bytes))
            raw.flush();os.fsync(raw.fileno())
        event('hashing_closed_archive',archive_bytes=archive.stat().st_size)
        ah=digest(archive);require(digest(args.config)==args.config_sha256 and digest(__file__)==c['source_sha256'],'pack source/config changed')
        for row in rows:require(signature((root/row['path']).stat())==snapshots[str(root/row['path'])],'source metadata changed after archive')
        result.update(complete=True,status='verified_source_pack_ready_no_transfer',archive={'path':str(archive),'bytes':archive.stat().st_size,'sha256':ah,'compression':'gzip level1'},manifest={'path':str(out/'manifest.json'),'sha256':digest(out/'manifest.json')},all_missing_paths_SHA_verified=True,all_archive_source_streams_SHA_verified=True)
        event('completed',archive_sha256=ah,archive_bytes=archive.stat().st_size)
    except BaseException as exc:
        result.update(complete=False,status='failed_preserved_no_retry',failure={'phase':phase,'type':type(exc).__name__,'message':str(exc)})
        event('failed',error=result['failure'])
    finally:
        signal.alarm(0);result['elapsed_seconds']=time.monotonic()-started;result['finished_utc']=now();result['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        fresh_json(out/'receipt.json',result)
    print(json.dumps({k:result.get(k) for k in ['complete','status','archive','failure','elapsed_seconds']},sort_keys=True),flush=True)
    return 0 if result['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
