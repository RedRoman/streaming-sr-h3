"""One-clip RGB characterization; read GT only after restoration is finished."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hr',type=Path,required=True)
    p.add_argument('--lr',type=Path,required=True)
    p.add_argument('--frames',type=int,default=22)
    args=p.parse_args()
    load=lambda path:np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/255
    records=[]
    for i in range(args.frames):
        hr=load(args.hr/f'{i:05d}.png');lr=load(args.lr/f'{i:05d}.png')
        bic=F.interpolate(torch.from_numpy(lr).permute(2,0,1)[None],size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)[0].permute(1,2,0).numpy()
        # Compare all outputs on identical 8-bit RGB storage.
        bic=np.round(bic*255)/255
        methods={'bicubic':bic,'vae_guide_roundtrip':load(args.output/f'guide_roundtrip_{i:02d}.png'),'h3_bf16_normal49':load(args.output/f'output_{i:02d}.png')}
        row={'frame':i,'methods':{}}
        for name,rgb in methods.items():
            mse=float(np.mean((rgb-hr)**2))
            red=np.asarray(Image.fromarray(np.uint8(np.round(rgb*255))).resize((64,64),Image.Resampling.BICUBIC),dtype=np.float64)/255
            row['methods'][name]={'rgb_mse':mse,'rgb_psnr_db':float(-10*np.log10(mse)),'lr_operator_mse':float(np.mean((red-lr)**2))}
        records.append(row)
    summary={}
    for name in records[0]['methods']:
        mean_mse=float(np.mean([r['methods'][name]['rgb_mse'] for r in records]))
        summary[name]={'mean_frame_psnr_db':float(np.mean([r['methods'][name]['rgb_psnr_db'] for r in records])),
                       'pooled_rgb_psnr_db':float(-10*np.log10(mean_mse)),
                       'lr_operator_rmse':float(np.sqrt(np.mean([r['methods'][name]['lr_operator_mse'] for r in records])))}
    result={'scope':'One held-out DAVIS/bmx-trees native22 block, 64LR/256HR; interface/conditioning characterization only, no method or dataset claim',
            'protocol':'Observed LR0..21 only; fixed neutral real text; BF16 DiT and FP16 VAE; output all22 after observation21, bounded block lookahead<=21; no later frames, GT pixels, oracle geometry or visibility in restoration',
            'limitation':'Custom LR-VAE guide protocol, no Qwen vision input; shorter/smaller than official4-15sec768p generation. Singleblock has no persistent reference update comparison.',
            'metrics':'RGB0..1 PNG8bit, fullcrop/no border shaving; LR consistency uses same Pillow bicubic4x reduction; bicubic control uses torch bicubic antialias as runner then PNG8bit rounding',
            'summary':summary,'frames':records}
    (args.output/'rgb_characterization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
