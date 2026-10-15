import json,time,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from pixel_dc import Projection
from native_vae import decode_last

FP=[(32,32),(32,96),(96,32),(96,96)]


def batch(items,device,rgb=True):
    inputs={k:torch.stack([x['inputs'][k] for x in items]).to(device) for k in items[0]['inputs'] if rgb or k!='query_base_rgb'}
    inputs={k:(v.float() if v.is_floating_point() else v) for k,v in inputs.items()}
    targets={k:torch.stack([x['supervision'][k] for x in items]).to(device).float() for k in items[0]['supervision'] if rgb or k!='rgb'}
    masks=torch.zeros(len(items),8,1,16,16,device=device)
    for b in range(len(items)):
        for j,fp in enumerate(inputs['packet_fp'][b].tolist()):top,left=FP[fp];masks[b,j,:,top//16:top//16+8,left//16:left//16+8]=1
    inputs['packet_masks']=masks
    # Raw LR only used as training/evaluation observations, not exposed to the
    # learned persistent-state transition after lift/controller construction.
    targets['observed_near_LR']=torch.stack([x['classical_inputs']['near_LR'] for x in items]).to(device).float()
    return inputs,targets


def live(out,**extra):
    p=Path(out).parent/'LIVE_STATE.json';value=dict(command=' '.join(sys.argv),pid=os.getpid(),unix_time=time.time(),foreground=True,**extra);p.write_text(json.dumps(value,indent=2))


class Readout:
    def __init__(self,vae):
        self.vae=vae;self.far=Projection(256,4);self.near=Projection(128,2);self.pf=self.far.right@self.far.matrix;self.pn=self.near.right@self.near.matrix
    def null(self,x):return x-self.pf@x@self.pf.T
    def historical_band(self,x,inputs,query_index):
        step=int(inputs['query_steps'][0,query_index]);r=self.null(x);total=torch.zeros_like(x);count=torch.zeros_like(x[:,:1])
        for j,fpi in enumerate(inputs['packet_fp'][0].tolist()):
            weight=inputs['weights'][:,step,j]
            if not bool(weight.any()):continue
            top,left=FP[fpi];mask=torch.zeros_like(count);mask[...,top:top+128,left:left+128]=1;weight=F.interpolate(weight,size=(256,256),mode='bilinear',align_corners=False)*mask;v=torch.zeros_like(x);v[...,top:top+128,left:left+128]=self.pn@r[...,top:top+128,left:left+128]@self.pn.T;total+=weight*v;count+=weight
        return self.null(total/count.clamp_min(.001))
    def rgb(self,inputs,delta,query_index):
        base=inputs['query_base_rgb'][:,query_index];z=inputs['query_latents'][:,query_index];decoded=decode_last(self.vae,z+delta);return base+self.historical_band(decoded-base,inputs,query_index)
