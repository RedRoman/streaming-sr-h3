"""Audit a distributed, train-fitted decoder direction at actual LR precision.

This uses a cached LR anchor and fixed P0 only, with no held-out restoration
loss or hyperparameter selection. The same FP16-stored weights compute FP32.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_interface import load_frozen_vae,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    item=torch.load(args.cache,map_location='cpu',weights_only=False)
    model,info=load_frozen_vae(args.vae);model.decoder.float();model.post_quant_conv.float()
    z=item['inputs']['anchor'].cuda().float();y,x=item['inputs']['roi'];p0=item['teacher']['p0'][:,:2].cuda()
    frame=item['source']['output_frame'];size=item['supervision']['canonical_hr'].shape[0]
    kwargs=dict(canonical_rectangle=(0,0,size,size),view_rectangle=(0,0,size,size),view_size=(size,size),lr_size=(size//4,size//4))
    reports={};all_lr={};all_rgb={}
    # Explicit strict TF32 flags are disclosed independently from eager RoPE.
    old_mm=torch.backends.cuda.matmul.allow_tf32;old_conv=torch.backends.cudnn.allow_tf32
    try:
        for mode in ['native','strict_fp32','strict_fp32_eager']:
            torch.backends.cuda.matmul.allow_tf32=old_mm if mode=='native' else False
            torch.backends.cudnn.allow_tf32=old_conv if mode=='native' else False
            context=differentiable_decoder_math() if mode.endswith('eager') else nullcontext()
            with context,torch.no_grad():
                def decode(value):return model.decode(value)[:,:,frame].to(z.device,dtype=torch.float32)
                base=decode(z);rgb_derivatives={};lr_derivatives={};local={}
                repeat=decode(z)
                local['identical_input_repeat_rgb_rmse']=float((base-repeat).square().mean().sqrt())
                for epsilon in [.05,.1,.3]:
                    deriv=[];forward=[]
                    for atom in p0.T:
                        delta=torch.zeros_like(z);delta[:,:,:,y:y+4,x:x+4]=atom.reshape(1,24,z.shape[2],4,4)
                        plus=decode(z+epsilon*delta);minus=decode(z-epsilon*delta)
                        d=(plus-minus)/(2*epsilon);deriv.append(d)
                        forward.append(camera_footprint(plus-base-epsilon*d,**kwargs).square().mean())
                    rgb_derivatives[epsilon]=torch.cat(deriv);lr_derivatives[epsilon]=camera_footprint(torch.cat(deriv),**kwargs)
                    local[str(epsilon)]={'rgb_derivative_rms':float(rgb_derivatives[epsilon].square().mean().sqrt()),'lr_derivative_rms':float(lr_derivatives[epsilon].square().mean().sqrt()),'lr_forward_linearization_rmse':float(torch.stack(forward).mean().sqrt())}
                for epsilon in [.05,.1,.3]:
                    local[str(epsilon)]['lr_relative_to_epsilon']={str(other):float((lr_derivatives[epsilon]-lr_derivatives[other]).norm()/lr_derivatives[other].norm().clamp_min(1e-12)) for other in [.05,.1,.3]}
                    local[str(epsilon)]['rgb_relative_to_epsilon']={str(other):float((rgb_derivatives[epsilon]-rgb_derivatives[other]).norm()/rgb_derivatives[other].norm().clamp_min(1e-12)) for other in [.05,.1,.3]}
                reports[mode]=local;all_lr[mode]=lr_derivatives;all_rgb[mode]=rgb_derivatives
        for mode in reports:
            for epsilon in [.05,.1,.3]:
                reports[mode][str(epsilon)]['lr_relative_to_strict_eager']=float((all_lr[mode][epsilon]-all_lr['strict_fp32_eager'][epsilon]).norm()/all_lr['strict_fp32_eager'][epsilon].norm().clamp_min(1e-12))
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_mm;torch.backends.cudnn.allow_tf32=old_conv
    result={'config':vars(args),'vae':info,'p0':'fixed train-only PCA; first two atoms; no restoration loss selection','original_tf32':{'matmul':old_mm,'cudnn':old_conv},'reports':reports}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
