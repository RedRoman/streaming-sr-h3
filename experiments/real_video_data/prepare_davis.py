#!/usr/bin/env python3
"""Fetch selected official DAVIS ZIP members by bounded HTTP byte ranges.

Original JPEG files are retained bit-for-bit and CRC checked. HR is only modcrop;
4x LR uses an explicitly versioned PIL bicubic operator. No model is involved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,hashlib,json,os,platform,struct,subprocess,sys,time,zlib
from datetime import datetime,timezone
from pathlib import Path
from threading import Lock,Semaphore
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'data'/'real_video'
URL='https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
ARCHIVE_BYTES=832766765
SELECT={'train':['bear','bus','car-turn','flamingo'],'val':['cows','dance-twirl'],'test':['bmx-trees','soapbox']}
LOCK=Lock();HTTP_SLOTS=Semaphore(16);NETWORK=[];PART_BYTES=512*1024


def sha(b):return hashlib.sha256(b).hexdigest()


def live(**kwargs):
    with LOCK:
        p=DEST/'LIVE_STATE.json'
        try:state=json.loads(p.read_text())
        except Exception:state={}
        state.update(kwargs);state.update(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
        p.write_text(json.dumps(state,indent=2))


def range_piece(low,high,path,proxy,max_time=180):
    expected=high-low+1
    if path.exists() and path.stat().st_size==expected:return path.read_bytes()
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=path.stat().st_size if path.exists() else 0
    assert existing<expected
    resume_low=low+existing
    output=path.with_suffix(path.suffix+'.resume') if existing else path
    need=high-resume_low+1
    headers=path.with_suffix(path.suffix+'.headers')
    started=time.time()
    with HTTP_SLOTS:
        proc=subprocess.run(['curl','--proxy',proxy,'--silent','--show-error','--fail','--location','--range',f'{resume_low}-{high}','--max-filesize',str(need),'--retry','1','--max-time',str(max_time),'--dump-header',str(headers),'--output',str(output),URL],capture_output=True,text=True)
    actual=output.stat().st_size if output.exists() else 0
    record=dict(low=resume_low,high=high,expected_bytes=need,received_file_bytes=actual,resumed_prefix_bytes=existing,returncode=proc.returncode,elapsed_seconds=time.time()-started,headers=str(headers.relative_to(DEST)))
    with LOCK:
        NETWORK.append(record)
        (DEST/'network_ranges.json').write_text(json.dumps(NETWORK,indent=2))
    if proc.returncode:raise RuntimeError(proc.stderr[-1000:])
    text=headers.read_text()
    assert f'Content-Range: bytes {resume_low}-{high}/{ARCHIVE_BYTES}'.lower() in text.lower(),text
    assert actual==need,(actual,need)
    if existing:
        with path.open('ab') as f:f.write(output.read_bytes())
        output.unlink()
    assert path.stat().st_size==expected
    return path.read_bytes()


def network_range(low,high,path,proxy,max_time=180):
    expected=high-low+1
    if path.exists() and path.stat().st_size==expected:return path.read_bytes()
    if expected<=PART_BYTES:return range_piece(low,high,path,proxy,max_time)
    partdir=path.with_suffix(path.suffix+'.parts');partdir.mkdir(parents=True,exist_ok=True)
    # Reuse every byte of the confirmed partial response from the first run.
    # It is only accepted as a ZIP member after the final CRC check.
    if path.exists():
        partial=path.read_bytes()
        assert len(partial)<expected
        for start in range(0,len(partial),PART_BYTES):
            pp=partdir/f'{start:010d}.bin'
            if not pp.exists():pp.write_bytes(partial[start:start+PART_BYTES])
        with LOCK:
            NETWORK.append(dict(low=low,high=low+len(partial)-1,expected_bytes=len(partial),received_file_bytes=len(partial),returncode=0,resumed_from_initial_monolithic_attempt=True))
            (DEST/'network_ranges.json').write_text(json.dumps(NETWORK,indent=2))
        path.unlink()
    ranges=[(start,min(start+PART_BYTES,expected)) for start in range(0,expected,PART_BYTES)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(range_piece,low+start,low+end-1,partdir/f'{start:010d}.bin',proxy,max_time) for start,end in ranges]
        for f in as_completed(futures):f.result()
    with path.open('wb') as target:
        for start,end in ranges:
            pp=partdir/f'{start:010d}.bin';target.write(pp.read_bytes());pp.unlink()
    # Headers are retained for network provenance; only payload files removed.
    return path.read_bytes()


def central_index():
    tail=(DEST/'archive_tail.bin').read_bytes()
    eocd=tail.rfind(b'PK\x05\x06')
    entry=struct.unpack_from('<4s4H2IH',tail,eocd)
    count,cdsize,cdoffset=entry[4],entry[5],entry[6]
    cd=(DEST/'archive_central_prefix.bin').read_bytes()+tail
    assert len(cd)>=cdsize
    cd=cd[:cdsize];pos=0;entries=[]
    while pos<len(cd):
        h=struct.unpack_from('<4s6H3I5H2I',cd,pos)
        assert h[0]==b'PK\x01\x02'
        name=cd[pos+46:pos+46+h[10]].decode('utf-8')
        entries.append(dict(name=name,method=h[4],crc32=h[7],compressed_bytes=h[8],uncompressed_bytes=h[9],local_offset=h[16]))
        pos+=46+h[10]+h[11]+h[12]
    assert len(entries)==count
    (DEST/'archive_index.json').write_text(json.dumps(dict(url=URL,archive_bytes=ARCHIVE_BYTES,central_offset=cdoffset,central_sha256=sha(cd),entries=entries),indent=2))
    return entries,cdoffset


def extract_member(blob,start,item):
    offset=item['local_offset']-start
    h=struct.unpack_from('<4s5H3I2H',blob,offset)
    assert h[0]==b'PK\x03\x04'
    name=blob[offset+30:offset+30+h[9]].decode('utf-8')
    assert name==item['name']
    begin=offset+30+h[9]+h[10]
    compressed=blob[begin:begin+item['compressed_bytes']]
    raw=zlib.decompress(compressed,-15) if item['method']==8 else compressed
    assert item['method'] in (0,8)
    assert len(raw)==item['uncompressed_bytes']
    assert zlib.crc32(raw)&0xffffffff==item['crc32']
    return raw


def members_blob(entries,selected,central_offset,path,proxy):
    selected=sorted(selected,key=lambda e:e['local_offset'])
    start=selected[0]['local_offset'];last=selected[-1]['local_offset']
    nextoffset=min([e['local_offset'] for e in entries if e['local_offset']>last]+[central_offset])
    return network_range(start,nextoffset-1,path,proxy),start,nextoffset-start


def prepare_sequence(split,seq,entries,central_offset,proxy,include_masks):
    selected=[e for e in entries if e['name'].startswith(f'DAVIS/JPEGImages/480p/{seq}/') and e['name'].endswith('.jpg')]
    assert selected,seq
    folder=DEST/'davis480p'/split/seq
    for sub in ('source_jpeg','hr','lr_x4'):(folder/sub).mkdir(parents=True,exist_ok=True)
    chunk=DEST/'_ranges'/f'{seq}_jpeg.bin'
    blob,start,cbytes=members_blob(entries,selected,central_offset,chunk,proxy)
    frames=[];dimensions=set()
    for item in sorted(selected,key=lambda e:e['name']):
        raw=extract_member(blob,start,item);basename=Path(item['name']).name
        assert basename[:-4].isdigit()
        source=folder/'source_jpeg'/basename;source.write_bytes(raw)
        with Image.open(source) as image:
            image=image.convert('RGB');w,h=image.size;dimensions.add((w,h))
            cw,ch=w-w%4,h-h%4
            hr=image.crop((0,0,cw,ch));lr=hr.resize((cw//4,ch//4),Image.Resampling.BICUBIC)
            hrpath=folder/'hr'/(Path(basename).stem+'.png');lrpath=folder/'lr_x4'/hrpath.name
            hr.save(hrpath);lr.save(lrpath)
        frames.append(dict(frame_index=int(Path(basename).stem),timestamp_seconds=int(Path(basename).stem)/24,source_zip_member=item['name'],zip_crc32=f"{item['crc32']:08x}",source_jpeg=str(source.relative_to(DEST)),source_sha256=sha(raw),hr=str(hrpath.relative_to(DEST)),hr_sha256=sha(hrpath.read_bytes()),lr_x4=str(lrpath.relative_to(DEST)),lr_sha256=sha(lrpath.read_bytes()),source_wh=[w,h],hr_wh=[cw,ch],lr_wh=[cw//4,ch//4]))
    del blob
    chunk.unlink()
    masks=[]
    if include_masks:
        selected_masks=[e for e in entries if e['name'].startswith(f'DAVIS/Annotations/480p/{seq}/') and e['name'].endswith('.png')]
        if selected_masks:
            chunk=DEST/'_ranges'/f'{seq}_mask.bin'
            blob,start,maskbytes=members_blob(entries,selected_masks,central_offset,chunk,proxy)
            (folder/'segmentation_gt_only').mkdir(exist_ok=True)
            for item in selected_masks:
                raw=extract_member(blob,start,item);mp=folder/'segmentation_gt_only'/Path(item['name']).name;mp.write_bytes(raw)
                masks.append(dict(source_zip_member=item['name'],path=str(mp.relative_to(DEST)),sha256=sha(raw)))
            del blob;chunk.unlink();cbytes+=maskbytes
    indices=[f['frame_index'] for f in frames]
    assert indices==list(range(indices[0],indices[0]+len(indices)))
    result=dict(source_video_id='DAVIS/'+seq,sequence=seq,project_split=split,official_davis2016_split='train' if split=='train' else 'val',official_davis2017_split='train' if split=='train' else 'val',frames=len(frames),fps=24,fps_evidence='All selected clips belong to DAVIS2016; CVPR2016 paper section3 states all captured at24fps',duration_seconds=len(frames)/24,source_dimensions=list(dimensions),source_kind='real natural-motion video frames; official author-provided 480p JPEG release',source_url=URL,downloaded_range_bytes=cbytes,frames_metadata=frames,segmentation_masks=masks,segmentation_usage='evaluation/optional explicitly labeled oracle only; not ordinary model input',sequence_source_hash=sha(''.join(f['source_sha256'] for f in frames).encode()))
    (folder/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(sequence=seq,split=split,frames=len(frames),range_bytes=cbytes,status='verified_and_prepared')),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proxy',default='http://127.0.0.1:7897');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--index-only',action='store_true');ap.add_argument('--skip-masks',action='store_true');args=ap.parse_args()
    begin=time.time();DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'network_ranges.json').exists():NETWORK.extend(json.loads((DEST/'network_ranges.json').read_text()))
    live(status='indexing_official_archive',scope='<=3GB network and <=2GB local including temporary data')
    entries,cdoffset=central_index()
    plan=[]
    for split,seqs in SELECT.items():
        for seq in seqs:
            sel=[e for e in entries if e['name'].startswith(f'DAVIS/JPEGImages/480p/{seq}/') and e['name'].endswith('.jpg')]
            plan.append(dict(sequence=seq,split=split,frames=len(sel),jpeg_bytes=sum(e['uncompressed_bytes'] for e in sel)))
    (DEST/'download_plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(dict(plan=plan,total_jpeg_bytes=sum(r['jpeg_bytes'] for r in plan))),flush=True)
    if args.index_only:
        print('metadata_members',[e['name'] for e in entries if 'ImageSets' in e['name'] or 'LICENSE' in e['name'] or 'README' in e['name']]);return
    # Small archive metadata files verify source split membership directly.
    metadata=[e for e in entries if e['name'] in ['DAVIS/ImageSets/2016/train.txt','DAVIS/ImageSets/2016/val.txt','DAVIS/ImageSets/2017/train.txt','DAVIS/ImageSets/2017/val.txt','DAVIS/SOURCES.md'] or ('/' not in e['name'].strip('/')) or e['name'].endswith('/README.md')]
    splits={}
    (DEST/'source_metadata').mkdir(exist_ok=True)
    for item in metadata:
        if item['uncompressed_bytes']==0 or item['name'].endswith('/'):continue
        existing=DEST/'source_metadata'/item['name'].replace('/','__')
        if existing.exists():
            raw=existing.read_bytes()
            assert len(raw)==item['uncompressed_bytes'] and zlib.crc32(raw)&0xffffffff==item['crc32']
            if item['name'].endswith('.txt'):
                parts=item['name'].split('/');splits[(parts[-2],parts[-1][:-4])]=set(raw.decode().split())
            continue
        chunk=DEST/'_ranges'/('metadata_'+str(item['local_offset'])+'.bin')
        blob,start,_=members_blob(entries,[item],cdoffset,chunk,args.proxy);raw=extract_member(blob,start,item)
        name=item['name'].replace('/','__');(DEST/'source_metadata'/name).write_bytes(raw)
        if item['name'].endswith('.txt'):
            parts=item['name'].split('/');splits[(parts[-2],parts[-1][:-4])]=set(raw.decode().split())
        chunk.unlink()
    for split,seqs in SELECT.items():
        source_split='train' if split=='train' else 'val'
        for seq in seqs:
            for year in ('2016','2017'):
                if (year,source_split) in splits:assert seq in splits[(year,source_split)],(seq,year,source_split)
    live(status='downloading_selected_sequences',plan=plan,completed_sequences=[])
    completed=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(prepare_sequence,split,seq,entries,cdoffset,args.proxy,not args.skip_masks):(split,seq) for split,seqs in SELECT.items() for seq in seqs}
        for job in as_completed(jobs):
            completed.append(job.result());live(status='downloading_selected_sequences',completed_sequences=[r['sequence'] for r in completed])
            diskbytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
            assert diskbytes<2_000_000_000,diskbytes
    videos=sorted(completed,key=lambda r:(r['project_split'],r['sequence']))
    manifest=dict(dataset='DAVIS real-motion small subset',created_utc=datetime.now(timezone.utc).isoformat(),source_url=URL,official_download_page='https://davischallenge.org/davis2017/code.html',usage_url=URL+'#DAVIS/README.md',license='CC BY-NC4.0 per dataset archive Terms of Use',license_url='https://creativecommons.org/licenses/by-nc/4.0/',license_evidence='source_metadata/DAVIS__README.md; dataset archive notice takes precedence over BSD helper-code license',license_note='README Credits says Attribution4.0 for author-owned sequences; explicit Terms of Use says CC BY-NC4.0. Use the stricter explicit dataset Terms. Selected8 sequences are absent from external exceptions in DAVIS/SOURCES.md.',fps_source='https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf',fps=24,selection=SELECT,source_split_policy='Complete DAVIS source video remains in one project split. Official train clips go to train; official val partitioned into disjoint project val/test. Project test is not official DAVIS hidden test.',degradation=dict(scale=4,operator='PIL.Image.Resampling.BICUBIC antialiased resize',pillow=__import__('PIL').__version__,hr='RGB decode original JPEG, top-left modcrop to multiple of4, PNG lossless',lr='resize HR to exact quarter dimensions, PNG; no added noise or models',matlab_bicubic_equivalence_claimed=False),limits=dict(network_download_max_bytes=3_000_000_000,local_max_bytes=2_000_000_000),source_clips=len(videos),total_frames=sum(v['frames'] for v in videos),network_range_received_bytes=sum(x['received_file_bytes'] for x in NETWORK)+1_401_051,numeric_runtime_seconds=time.time()-begin,python=sys.version,platform=platform.platform(),script_sha256=sha(Path(__file__).read_bytes()),limitations=['Natural motion is real, LR degradation is synthetic','Author release is480p JPEG, not RAW camera or uncompressed1080p','Short complete clips, about2-4s: not a long-revisit benchmark','No loops/ping-pong repetition or synthetic occlusion was added','Only8 independent source clips; insufficient as a large training benchmark','Segmentation annotations, if used as masks, are oracle supervision and must be declared'],videos=videos)
    ids=[v['source_video_id'] for v in videos];assert len(ids)==len(set(ids))
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2))
    allframes=[dict(sequence=v['sequence'],split=v['project_split'],**f) for v in videos for f in v['frames_metadata']]
    with (DEST/'frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allframes[0]));w.writeheader();w.writerows(allframes)
    footprint=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    live(status='complete_local_verified',active_tool_session_ids=[],local_footprint_bytes=footprint,total_frames=manifest['total_frames'],completed_sequences=ids,manifest='data/real_video/manifest.json')
    print(json.dumps(dict(done=True,clips=len(videos),frames=manifest['total_frames'],local_bytes=footprint,elapsed_seconds=time.time()-begin)),flush=True)


if __name__=='__main__':main()
