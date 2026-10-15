"""Separate FP16 finite-difference noise from real decoder curvature.

Uses the same FP16 checkpoint values for both FP16 and FP32 compute. No model
weights or restoration losses are tuned. The native 22-frame static context
contains copies of one already observed LR anchor, not 22 independent packets.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from memory_adapter import CanonicalChart
from h3_interface import load_frozen_vae,decoder_teacher


def main():
    p=argparse.ArgumentParser();p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--source',required=True)
    p.add_argument('--out',default='results/h3_core/teacher_audit.json');p.add_argument('--size',type=int,default=256);p.add_argument('--context',type=int,default=22);p.add_argument('--output-frame',type=int,default=21)
    args=p.parse_args();sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    image=Image.open(args.source).convert('RGB');left=(image.width-args.size)//2;top=(image.height-args.size)//2;image=image.crop((left,top,left+args.size,top+args.size))
    hr=torch.from_numpy(np.asarray(image).copy()).cuda().permute(2,0,1)[None].float()/255
    lr=F.interpolate(hr,size=(args.size//4,)*2,mode='bicubic',align_corners=False,antialias=True)
    up=F.interpolate(lr,size=(args.size,)*2,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    model,info=load_frozen_vae(args.vae);reports={}
    with torch.no_grad():
        anchor=model.encode((2*up-1)[:,:,None].repeat(1,1,args.context,1,1).half().clone())
        chart=CanonicalChart(temporal=anchor.shape[2]).cuda();roi=((anchor.shape[-1]-4)//2,)*2
        p0=chart.p0[:,:4]
        kwargs=dict(canonical_rectangle=(0,0,args.size,args.size),view_rectangle=(0,0,args.size,args.size),view_size=(args.size,)*2,lr_size=(args.size//4,)*2)
        reference=None
        for dtype in [torch.float16,torch.float32]:
            model=model.to(dtype=dtype);z=anchor.to(dtype);cache={}
            for epsilon in [.1,.3,1.]:
                base,directions=decoder_teacher(model,z,p0,roi,epsilon,args.output_frame)
                observed=camera_footprint(directions,**kwargs)
                delta=torch.zeros_like(z);delta[:,:,:,roi[0]:roi[0]+4,roi[1]:roi[1]+4]=p0[:,0].reshape(1,24,z.shape[2],4,4).to(z)
                actual=model.decode(z+epsilon*delta)[:,:,args.output_frame].to(device=z.device,dtype=torch.float32)
                predicted=base+epsilon*directions[:1]
                cache[str(epsilon)]={'rgb_derivative':directions.cpu(),'lr_derivative':observed.cpu(),
                                     'rgb_derivative_rms':float(directions.square().mean().sqrt()),
                                     'lr_derivative_rms':float(observed.square().mean().sqrt()),
                                     'rgb_forward_linearization_rmse':float((actual-predicted).square().mean().sqrt()),
                                     'lr_forward_linearization_rmse':float(camera_footprint(actual-predicted,**kwargs).square().mean().sqrt())}
            for epsilon in [.1,.3,1.]:
                for other in [.1,.3,1.]:
                    a,b=cache[str(epsilon)],cache[str(other)]
                    a.setdefault('lr_relative_to_epsilon',{})[str(other)]=float((a['lr_derivative']-b['lr_derivative']).norm()/b['lr_derivative'].norm().clamp_min(1e-12))
            if dtype==torch.float16:reference=cache
            else:
                for epsilon in [.1,.3,1.]:
                    a,b=reference[str(epsilon)],cache[str(epsilon)]
                    b['fp16_lr_derivative_relative_error_same_epsilon']=float((a['lr_derivative']-b['lr_derivative']).norm()/b['lr_derivative'].norm().clamp_min(1e-12))
            reports[str(dtype)]={epsilon:{k:v for k,v in r.items() if not torch.is_tensor(v)} for epsilon,r in cache.items()}
    result={'config':vars(args),'vae':info,'anchor_latent_shape':list(anchor.shape),'reports':reports,'note':'FP32 compute uses the exact same FP16-stored weights cast to float32; final rollout must still audit actual target precision.'}
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
