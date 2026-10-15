#!/usr/bin/env python3
"""Independent data integrity/4x degradation audit and source-motion preview."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'real_video'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.time();manifest=json.loads((DATA/'manifest.json').read_text())
    source_sets={};total=0;motion=[];windows=[];checks={}
    board=Image.new('RGB',(720,160*len(manifest['videos'])),(250,250,250));draw=ImageDraw.Draw(board)
    for row,clip in enumerate(manifest['videos']):
        source_sets.setdefault(clip['project_split'],set()).add(clip['source_video_id'])
        frames=clip['frames_metadata'];last=None;diffs=[]
        for mask in clip['segmentation_masks']:
            assert digest(DATA/mask['path'])==mask['sha256']
        for f in frames:
            source=DATA/f['source_jpeg'];hrp=DATA/f['hr'];lrp=DATA/f['lr_x4']
            assert digest(source)==f['source_sha256']
            assert digest(hrp)==f['hr_sha256']
            assert digest(lrp)==f['lr_sha256']
            with Image.open(source) as im:
                im=im.convert('RGB');w,h=im.size;expected=im.crop((0,0,w-w%4,h-h%4))
                with Image.open(hrp) as gt:assert np.array_equal(np.asarray(gt),np.asarray(expected))
                down=expected.resize((expected.width//4,expected.height//4),Image.Resampling.BICUBIC)
                with Image.open(lrp) as lr:assert np.array_equal(np.asarray(lr),np.asarray(down))
                small=np.asarray(expected.resize((64,36)),np.float32)/255
            if last is not None:diffs.append(float(np.mean((small-last)**2)))
            last=small;total+=1
        assert any(v>0 for v in diffs),'Entire source sequence has no visible change'
        motion.append(dict(sequence=clip['sequence'],consecutive_thumbnail_mse_mean=float(np.mean(diffs)),exact_repeated_thumbnail_pairs=sum(v==0 for v in diffs),note='Appearance-change evidence only, not optical-flow magnitude'))
        draw.text((5,row*160+3),f"{clip['project_split']} / {clip['sequence']} / {len(frames)} frames / 24fps",fill=(0,0,0))
        for col,index in enumerate([0,len(frames)//2,len(frames)-1]):
            with Image.open(DATA/frames[index]['hr']) as im:
                im=im.resize((240,135),Image.Resampling.LANCZOS)
                board.paste(im,(col*240,row*160+23))
        # 39-frame no-loop clips on H3 native17k+5 grid. Whole sources stay in
        # one split, even if multiple training windows are later sampled.
        if len(frames)>=39:
            win=frames[:39]
            w,h=win[0]['hr_wh'];size=256
            # Pixel-aligned ROI crop; do not resize high-quality GT to create
            # fictitious motion. Its exact quarter crop uses the same coordinates.
            left=((w-size)//2)//4*4;top=((h-size)//2)//4*4
            cropdir=DATA/'h3_256'/clip['project_split']/clip['sequence']
            (cropdir/'hr').mkdir(parents=True,exist_ok=True);(cropdir/'lr_x4').mkdir(exist_ok=True)
            cropframes=[]
            for f in win:
                with Image.open(DATA/f['hr']) as im:
                    cropped=im.crop((left,top,left+size,top+size))
                    lr=cropped.resize((size//4,size//4),Image.Resampling.BICUBIC)
                    hpath=cropdir/'hr'/Path(f['hr']).name;lpath=cropdir/'lr_x4'/hpath.name
                    cropped.save(hpath);lr.save(lpath)
                cropframes.append(dict(frame_index=f['frame_index'],hr=str(hpath.relative_to(DATA)),lr_x4=str(lpath.relative_to(DATA)),hr_sha256=digest(hpath),lr_sha256=digest(lpath)))
            windows.append(dict(source_video_id=clip['source_video_id'],split=clip['project_split'],sequence=clip['sequence'],start=0,frames=39,fps=24,hr_crop_xywh=[left,top,size,size],hr_paths=[x['hr'] for x in cropframes],lr_paths=[x['lr_x4'] for x in cropframes],frames_metadata=cropframes,lr_operator='Bicubic applied after fixedHR crop, so no boundary-halo mismatch from cropping wholeframe LR',block_starts=[0,17],block_frames=22,overlap_frames=5,wait_budget='bounded chunk lookahead<=21 frames; not zero-delay',oracle_geometry=False,oracle_visibility=False))
    for a in source_sets:
        for b in source_sets:
            if a!=b:assert not source_sets[a]&source_sets[b]
    assert total==manifest['total_frames']
    footprint=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
    assert footprint<2_000_000_000
    checks.update(passed=True,all_source_sha256_verified=True,all_hr_lr_sha256_verified=True,all_source_zip_crc32_verified_by_acquisition=True,all_hr_equal_to_modcropped_original=True,all_lr_equal_to_exact_documented_bicubic=True,source_video_split_disjoint=True,total_frames=total,total_clips=len(manifest['videos']),local_footprint_bytes=footprint,elapsed_seconds=time.time()-started,motion_diagnostics=motion,limits='Thumbnail differences only verify nonidentical real frames; no inferred claim of long revisit')
    (DATA/'verification.json').write_text(json.dumps(checks,indent=2));board.save(DATA/'contact_sheet.png')
    (DATA/'h3_windows_39.json').write_text(json.dumps(dict(protocol='Real natural motion; synthetic4x LR; fixed spatial crop; no source crosssplit',windows=windows),indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
