"""Small target-only contact sheets for human continuity review; no approval."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import numpy as np
from PIL import Image, ImageDraw

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def require(ok,msg):
    if not ok:raise ValueError(msg)

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--complete',type=Path,required=True)
    p.add_argument('--complete-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(sha(a.complete)==a.complete_sha256,'aggregate SHA changed')
    aggregate=json.loads(a.complete.read_text());require(aggregate['complete'] is True and aggregate['status']=='completed_engineering_only','not complete')
    a.out.mkdir(parents=True,exist_ok=False)
    report={'schema':'openh3.history_utility_target_contact_sheets.v1','complete':False,'aggregate_sha256':a.complete_sha256,
        'entry_sha256':sha(__file__),'scientific_continuity_approved':False,'natural_cut_review':'pending human visual review',
        'image_semantics':'original decoded/cropped target only; no degraded frames; area-downsampled thumbnails',
        'timestamps':'actual target FFmpeg showinfo window-relative grid; not original source-PTS correspondence',
        'frames_per_sheet':96,'model_calls':0,'NPU_calls':0,'sheets':[]}
    try:
        for split in ('train','dev'):
            for entry in aggregate['receipt_indices'][split]:
                require(sha(entry['path'])==entry['sha256'],'source receipt changed');r=json.loads(Path(entry['path']).read_text())
                target=r['target'];require(sha(target['path'])==target['file_sha256'],'target NPZ changed')
                with np.load(target['path'],allow_pickle=False) as z:frames=z['frames']
                require(frames.dtype==np.uint8 and frames.shape==(96,480,832,3) and frames.flags.c_contiguous,'target pixels differ')
                require(hashlib.sha256(memoryview(frames).cast('B')).hexdigest()==target['pixel_sha256'],'target pixel SHA changed')
                times=r['decode']['timestamps']['target'];require([v['index'] for v in times]==list(range(96)),'target timestamp indices differ')
                canvas=Image.new('RGB',(832,1000),'white');draw=ImageDraw.Draw(canvas)
                sid=r['source']['source_id'];draw.text((4,4),sid+'  '+split+'  TARGET ONLY / continuity pending',fill='black')
                for i,frame in enumerate(frames):
                    x=(i%8)*104;y=24+(i//8)*80
                    canvas.paste(Image.fromarray(frame).resize((104,60),Image.Resampling.BOX),(x,y))
                    draw.text((x+1,y+61),'%02d +%.3fs'%(i,times[i]['pts_time_seconds']),fill='black')
                dest=a.out/(sid+'_target96.png');canvas.save(dest,format='PNG')
                report['sheets'].append({'source_id':sid,'split':split,'source_receipt_sha256':entry['sha256'],
                    'target_file_sha256':target['file_sha256'],'target_pixel_sha256':target['pixel_sha256'],
                    'png':dest.name,'png_sha256':sha(dest),'bytes':dest.stat().st_size,'timestamps':times})
                del frames,canvas
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                require(peak<=768*(1<<20),'contact-sheet CPU RSS exceeded768MiB')
                report['peak_RSS_bytes']=peak;write(a.out/'partial.json',report)
        require(len(report['sheets'])==4 and 'torch' not in sys.modules and 'torch_npu' not in sys.modules,'scope differs')
        report['complete']=True;write(a.out/'complete.json',report);(a.out/'partial.json').unlink()
    except BaseException as e:
        report.update(complete=False,error=repr(e));write(a.out/'failure.json',report);raise

if __name__=='__main__':main()
