"""Fixed-source diagnostic only: characterize all completed native guide cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def read(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/255

def psnr(x,y):return float(-10*np.log10(max(np.mean((x-y)**2),1e-12)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for case in sorted(a.root.iterdir()):
        if not case.is_dir() or not (case/'decode_result.json').exists():continue
        manifest=json.loads((case/'guide_manifest.json').read_text())
        outputs=sorted(case.glob('output_*.png'))
        if len(outputs)!=manifest['target_frames']:continue
        hr=read(case/'source_hr_for_evaluation_only.png')
        values=[psnr(read(file),hr) for file in outputs]
        guides=sorted(case.glob('guide_roundtrip_*.png'))
        row={'case':case.name,'oracle_HR':manifest['oracle_HR_guide'],'frames':manifest['target_frames'],'guide_frames':manifest['guide_input_frames'],
             'output_first_psnr':values[0],'output_last_psnr':values[-1],'output_mean_frame_psnr':float(np.mean(values)),
             'output_frame_psnr_db':values,
             'vae_guide_first_psnr':psnr(read(guides[0]),hr),'vae_guide_last_psnr':psnr(read(guides[-1]),hr),
             'bicubic_psnr':psnr(read(case/'lr_upsampled_guide.png'),hr),'context_sha256':manifest['context_sha256'],
             'context_is_LR_vision':'lrvision' in case.name,'output_png_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in outputs}}
        rows.append(row)
    result={'scope':'Single BSDS train100075 static repeated-image diagnostic, same neutral prompt/seed101/Euler49evaluations/exactnormal50sigma. HR is explicit interface oracle, never causal method. LRvision context is identical across LR/HRoracle guide and reads only first observed LR upsample. No prompt/seed/sigma selection.', 'rows':rows}
    (a.root/'characterization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='output_png_sha256'} for row in rows]))

if __name__=='__main__':main()
