#!/usr/bin/env python3
"""Make model-ready39-frame windows only from completely verified source clips."""
import hashlib,json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'real_video'


def main():
    windows=[]
    for path in sorted((DATA/'davis480p').glob('*/*/manifest.json')):
        clip=json.loads(path.read_text());frames=clip['frames_metadata'][:39]
        assert len(frames)==39
        w,h=frames[0]['hr_wh'];left=((w-256)//2)//4*4;top=((h-256)//2)//4*4
        folder=DATA/'h3_256'/clip['project_split']/clip['sequence']
        (folder/'hr').mkdir(parents=True,exist_ok=True);(folder/'lr_x4').mkdir(exist_ok=True)
        records=[]
        for f in frames:
            with Image.open(DATA/f['hr']) as im:
                hr=im.crop((left,top,left+256,top+256));lr=hr.resize((64,64),Image.Resampling.BICUBIC)
                hp=folder/'hr'/Path(f['hr']).name;lp=folder/'lr_x4'/hp.name
                hr.save(hp);lr.save(lp)
            records.append(dict(frame_index=f['frame_index'],hr=str(hp.relative_to(DATA)),lr_x4=str(lp.relative_to(DATA)),hr_sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),lr_sha256=hashlib.sha256(lp.read_bytes()).hexdigest()))
        win=dict(source_video_id=clip['source_video_id'],split=clip['project_split'],sequence=clip['sequence'],start=0,frames=39,fps=24,hr_crop_xywh=[left,top,256,256],hr_paths=[f['hr'] for f in records],lr_paths=[f['lr_x4'] for f in records],frames_metadata=records,lr_operator='Pillow bicubic applied after fixedHR crop',block_starts=[0,17],block_frames=22,overlap_frames=5,wait_budget='bounded chunk lookahead<=21 frames; not zero-delay',oracle_geometry=False,oracle_visibility=False)
        (folder/'manifest.json').write_text(json.dumps(win,indent=2));windows.append(win)
    (DATA/'h3_windows_completed.json').write_text(json.dumps(dict(status='Only completed source clips included; acquisition may still be running',windows=windows),indent=2))
    print(json.dumps(dict(completed_source_clips=[w['sequence'] for w in windows],h3_frames_per_clip=39)))


if __name__=='__main__':main()
