"""Bounded, pinned-mirror LaSOT metadata/preview inspection; no bulk dataset fetch.

Official papers and usage statement are recorded separately. Mirror bytes are
pinned and CRC-checked, not claimed independently equal to an official archive.
Annotations are benchmark selection/evaluation metadata, never model inputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import threading
import urllib.request
import zlib
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'data/long_video_candidates'
REV='a97464600e8c1ab91eb91a47332a20732830585e'
PROXY='http://127.0.0.1:7897'
LOCK=threading.Lock()
NETWORK=[]


def sha(b):return hashlib.sha256(b).hexdigest()


def fetch(category,size,lo,hi,label):
    assert 0<=lo<=hi<size and hi-lo+1<20_000_000
    folder=DEST/category/'ranges';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{lo}_{hi}.bin'
    if path.exists():return path.read_bytes()
    url=f'https://hf-mirror.com/datasets/l-lt/LaSOT/resolve/{REV}/{category}.zip'
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':PROXY,'http':PROXY}))
    req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
    with op.open(req,timeout=45) as response:
        expected=f'bytes {lo}-{hi}/{size}'
        assert response.status==206 and response.headers.get('Content-Range')==expected,(response.status,dict(response.headers))
        raw=response.read(hi-lo+2)
        assert len(raw)==hi-lo+1
    path.write_bytes(raw)
    with LOCK:
        NETWORK.append({'category':category,'low':lo,'high':hi,'bytes':len(raw),'sha256':sha(raw),'label':label})
        (DEST/'network_ranges.json').write_text(json.dumps(NETWORK,indent=2))
        assert sum(x['bytes'] for x in NETWORK)<250_000_000
    return raw


def index(category,size):
    tail=fetch(category,size,size-65557,size-1,'central directory locator')
    pos=tail.rfind(b'PK\x05\x06');assert pos>=0
    record=struct.unpack_from('<4s4H2IH',tail,pos)
    count,cdsize,offset=record[4:7]
    if count==65535 or cdsize==0xffffffff or offset==0xffffffff:
        pos64=tail.rfind(b'PK\x06\x06');assert pos64>=0
        large=struct.unpack_from('<4sQ2H2I4Q',tail,pos64)
        count,cdsize,offset=large[7:10]
    central=fetch(category,size,offset,offset+cdsize-1,'central directory')
    entries=[];pos=0
    while pos<len(central):
        h=struct.unpack_from('<4s6H3I5H2I',central,pos);assert h[0]==b'PK\x01\x02'
        name=central[pos+46:pos+46+h[10]].decode('utf-8')
        entries.append(dict(name=name,method=h[4],crc32=h[7],compressed_bytes=h[8],uncompressed_bytes=h[9],local_offset=h[16]))
        pos+=46+h[10]+h[11]+h[12]
    assert len(entries)==count
    (DEST/category/'index.json').write_text(json.dumps({'archive_bytes':size,'revision':REV,'central_sha256':sha(central),'entries':entries},indent=2))
    return entries


def member(category,size,item):
    lo=item['local_offset'];hi=min(size-1,lo+30+len(item['name'].encode())+512+item['compressed_bytes'])
    blob=fetch(category,size,lo,hi,item['name'])
    header=struct.unpack_from('<4s5H3I2H',blob,0);assert header[0]==b'PK\x03\x04'
    name=blob[30:30+header[9]].decode();assert name==item['name']
    begin=30+header[9]+header[10]
    assert begin+item['compressed_bytes']<=len(blob)
    zipped=blob[begin:begin+item['compressed_bytes']]
    raw=zlib.decompress(zipped,-15) if item['method']==8 else zipped
    assert item['method'] in (0,8) and len(raw)==item['uncompressed_bytes'] and zlib.crc32(raw)&0xffffffff==item['crc32']
    return raw


def inspect(category,size):
    entries=index(category,size)
    annotations=[e for e in entries if Path(e['name']).name in ['full_occlusion.txt','out_of_view.txt','groundtruth.txt','nlp.txt']]
    def download(e):
        raw=member(category,size,e);path=DEST/category/'members'/e['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,annotations))
    sources=sorted({str(Path(e['name']).parent) for e in annotations if e['name'].endswith('full_occlusion.txt')})
    rows=[];previews=[]
    for source in sources:
        folder=DEST/category/'members'/source
        def flags(name):return [int(x) for x in (folder/name).read_text().replace('\n',',').split(',') if x.strip()]
        full,out=flags('full_occlusion.txt'),flags('out_of_view.txt');assert len(full)==len(out)
        absent=[bool(a or b) for a,b in zip(full,out)];events=[];begin=None
        for i,value in enumerate(absent+[False]):
            if value and begin is None:begin=i
            if not value and begin is not None:
                if begin>0 and i<len(absent):events.append({'start_frame0':begin,'return_frame0':i,'absent_frames':i-begin})
                begin=None
        images=sorted([e for e in entries if e['name'].startswith(source+'/img/') and e['name'].lower().endswith('.jpg')],key=lambda e:e['name'])
        assert len(images)==len(full),(source,len(images),len(full))
        selected=max(events,key=lambda e:e['absent_frames']) if events else None
        rows.append({'source':source,'frames':len(images),'first_image_member':images[0]['name'],'reappearances':events,'longest_absence':selected})
        raw=member(category,size,images[0]);path=folder/'preview_first.jpg';path.write_bytes(raw)
        image=Image.open(path).convert('RGB');image.thumbnail((256,144));previews.append((source,image))
    canvas=Image.new('RGB',(256*4,172*((len(previews)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,image) in enumerate(previews):
        x,y=i%4*256,i//4*172;draw.text((x+4,y+3),name,fill='black');canvas.paste(image,(x,y+24))
    canvas.save(DEST/category/'previews.png')
    result={'category':category,'revision':REV,'archive_bytes':size,'scope':'metadata and first-frame preview only','sequences':rows,
            'annotation_use':'selection/evaluation oracle only; not model input','fps':'nominal30 for paper duration convention; per-source fps not verified',
            'limitations':['Third-party mirror, CRC/pinned hash recorded; full archive official checksum not verified','No method evaluation or test-based sequence selection']}
    (DEST/category/'inspection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'category':category,'sequences':len(rows),'events_ge60frames':sum(any(e['absent_frames']>=60 for e in r['reappearances']) for r in rows)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--categories',nargs='+',default=['book','hat']);args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'network_ranges.json').exists():NETWORK.extend(json.loads((DEST/'network_ranges.json').read_text()))
    tree=json.loads((ROOT/'research/long_video/lasot_mirror_tree.json').read_text())
    for category in args.categories:
        size=next(x['size'] for x in tree if x.get('path')==category+'.zip');inspect(category,size)
