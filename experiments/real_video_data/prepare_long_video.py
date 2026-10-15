"""Create declared native-sized long-clip HR/LR pairs, never upsample HR."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from PIL import Image
import PIL

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/long_video'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(video):
    folder=DATA/video['split']/video['sequence']
    for sub in ['hr','lr_x4']:(folder/sub).mkdir(exist_ok=True)
    records=[];sizes=set()
    for frame in video['source_frames']:
        original=DATA/frame['path'];assert sha(original)==frame['sha256']
        im=Image.open(original).convert('RGB');w,h=im.size;scale=min(1.,480/h)
        resized=(round(w*scale),round(h*scale))
        if resized!=im.size:im=im.resize(resized,Image.Resampling.LANCZOS)
        cw,ch=im.width//32*32,im.height//32*32;x,y=(im.width-cw)//2,(im.height-ch)//2
        high=im.crop((x,y,x+cw,y+ch));low=high.resize((cw//4,ch//4),Image.Resampling.BICUBIC)
        hp=folder/'hr'/original.with_suffix('.png').name;lp=folder/'lr_x4'/hp.name
        high.save(hp,compress_level=3);low.save(lp,compress_level=3)
        sizes.add((cw,ch))
        records.append({**frame,'hr':str(hp.relative_to(DATA)),'lr_x4':str(lp.relative_to(DATA)),
                        'hr_sha256':sha(hp),'lr_sha256':sha(lp),'source_resize_wh':list(resized),
                        'resize_scale':scale,'resized_crop_xywh':[x,y,cw,ch],'hr_wh':[cw,ch],'lr_wh':[cw//4,ch//4]})
    assert len(sizes)==1
    result={k:v for k,v in video.items() if k!='source_frames'}
    result.update(frames_metadata=records,frame_count=len(records),hr_dimensions_wh=list(sizes)[0],
                  degradation='Source downsample ONLY if height>480 using PillowLanczos, then center mod32 crop; HR→LR x4 PillowBICUBIC; no added noise/compression',
                  pillow=PIL.__version__,pixels_from_annotation=False)
    (folder/'derived_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'sequence':video['sequence'],'frames':len(records),'hr_wh':list(sizes)[0],'status':'prepared'}),flush=True)
    return result


if __name__=='__main__':
    begin=time.perf_counter();source=json.loads((DATA/'manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:videos=list(pool.map(prepare,source['videos']))
    result={'dataset':'LaSOT four contiguous long clips; synthetic4x LR','source_manifest_sha256':sha(DATA/'manifest.json'),
            'videos':videos,'total_frames':sum(v['frame_count'] for v in videos),'seconds':time.perf_counter()-begin,
            'script_sha256':sha(Path(__file__)),'split_policy':'1train /1val /2test source-disjoint, explicit official split in each video',
            'fps_policy':'Nominal30 only for frame timestamps/display; actual source fps unavailable',
            'annotation_policy':'Selection/evaluation only; never part of observed LR inputs',
            'limits':['4source pilot, insufficient large benchmark','HR sourced from compressed author JPEG; not raw sensor ground truth','Third-party pinned mirror; original archive official checksum not independently verified']}
    (DATA/'derived_h3_manifest.json').write_text(json.dumps(result,indent=2))
