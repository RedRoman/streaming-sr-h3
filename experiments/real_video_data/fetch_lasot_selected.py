"""Download the predeclared contiguous long clips; retain original JPEG bytes.

No temporal looping, synthetic occlusion, SR, model fitting or HR upsampling.
Full-resolution HR/LR pairs can be derived lazily by a declared modcrop/resize.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import zlib
from PIL import Image
import inspect_lasot as remote

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'data/long_video'


def run(plan):
    cat,seq=plan['category'],plan['sequence']
    index=json.loads((remote.DEST/cat/'index.json').read_text());size=index['archive_bytes']
    entries={e['name']:e for e in index['entries']}
    low,high=plan['local_offset_start'],plan['local_offset_end']
    ranges=[(lo,min(lo+8*1024**2-1,high)) for lo in range(low,high+1,8*1024**2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        blobs=list(pool.map(lambda r:remote.fetch(cat,size,r[0],r[1],'selected clip '+seq),ranges))
    blob=b''.join(blobs);assert len(blob)==high-low+1
    folder=DEST/plan['split']/seq;original=folder/'source_jpeg';original.mkdir(parents=True,exist_ok=True)
    rows=[];sizes=set()
    for relative,name in enumerate(plan['source_members']):
        e=entries[name];off=e['local_offset']-low;header=struct.unpack_from('<4s5H3I2H',blob,off)
        assert header[0]==b'PK\x03\x04';localname=blob[off+30:off+30+header[9]].decode();assert localname==name
        begin=off+30+header[9]+header[10];compressed=blob[begin:begin+e['compressed_bytes']]
        raw=zlib.decompress(compressed,-15) if e['method']==8 else compressed
        assert e['method'] in (0,8) and len(raw)==e['uncompressed_bytes'] and zlib.crc32(raw)&0xffffffff==e['crc32']
        path=original/Path(name).name;path.write_bytes(raw)
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:sizes.add(image.size)
        rows.append({'local_frame0':relative,'source_frame0':plan['start_frame0']+relative,'archive_member':name,'zip_crc32':f"{e['crc32']:08x}",'sha256':remote.sha(raw),'path':str(path.relative_to(DEST))})
    evaluation=folder/'annotations_evaluation_only';evaluation.mkdir(exist_ok=True)
    for file in (remote.DEST/cat/'members'/seq).glob('*.txt'):(evaluation/file.name).write_bytes(file.read_bytes())
    record={**{k:v for k,v in plan.items() if k!='source_members'},'source_dimensions_wh':sorted(sizes),'source_frames':rows,
            'nominal_duration_seconds_at30fps':len(rows)/30,'source_actual_fps_verified':False,
            'source_archive_revision':remote.REV,'source_archive_size':size,
            'scope':'Real contiguous source video; original JPEG, no fabricated repeats; no model results',
            'annotation_policy':'Full-source metadata for evaluation/selection only; indices use source_frame0; not restoration input',
            'degradation':'Not yet generated; keep raw originals for predeclared native mod32 HR and Pillow x4 LR'}
    (folder/'manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps({'sequence':seq,'split':plan['split'],'frames':len(rows),'source_wh':sorted(sizes),'status':'CRC_SHA_verified'}),flush=True)
    return record


if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    if (remote.DEST/'network_ranges.json').exists():remote.NETWORK.extend(json.loads((remote.DEST/'network_ranges.json').read_text()))
    plan=json.loads((remote.DEST/'selected_plan.json').read_text())
    records=[run(p) for p in plan]
    (DEST/'manifest.json').write_text(json.dumps({'dataset':'LaSOT selected long clips, pinned third-party mirror','videos':records,'total_frames':sum(len(r['source_frames']) for r in records),'source_evidence':'research/long_video/SOURCES.md','network_range_bytes_total_including_candidate_inspection':sum(n['bytes'] for n in remote.NETWORK),'limit':'Only4 sources; official archive checksum not independently verified; fps nominal30, not source-measured'},indent=2))
