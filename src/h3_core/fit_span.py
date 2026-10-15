"""Fit a fixed decoder-teacher span from TRAIN-only actual H3 SR residuals.

No validation/test source fits the span. It is fixed before decoder response
caching. Later conditional C_phi is learned inside this span, so K0 C remains
consistent as C_phi changes. DCT remains an explicit untrained-span control.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_interface import load_frozen_vae


def main():
    p=argparse.ArgumentParser();p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True)
    p.add_argument('--train',default='data/BSR/BSDS500/data/images/train');p.add_argument('--out',default='results/h3_core/span_train200.pt')
    p.add_argument('--count',type=int,default=200);p.add_argument('--size',type=int,default=256);p.add_argument('--context',type=int,default=22);p.add_argument('--atoms',type=int,default=32)
    args=p.parse_args();sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    paths=sorted(Path(args.train).glob('*.jpg'))
    if Path(args.train).name!='train':raise ValueError('Span fitting must point to the official train folder')
    if len(paths)<args.count:raise ValueError(f'Need {args.count} train images, found {len(paths)}')
    if args.count<args.atoms:raise ValueError('Need at least as many source images as span atoms')
    model,info=load_frozen_vae(args.vae);rows=[];anchors=[];sources=[]
    with torch.no_grad():
        for i,path in enumerate(paths[:args.count]):
            image=Image.open(path).convert('RGB');left=(image.width-args.size)//2;top=(image.height-args.size)//2
            image=image.crop((left,top,left+args.size,top+args.size))
            hr=torch.from_numpy(np.asarray(image).copy()).cuda().permute(2,0,1)[None].float()/255
            rectangle=(0.,0.,float(args.size),float(args.size))
            lr=camera_footprint(hr,rectangle,rectangle,view_size=(args.size,)*2,lr_size=(args.size//4,)*2,blur=.6)
            up=F.interpolate(lr,size=(args.size,)*2,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            z0=model.encode((2*up-1)[:,:,None].repeat(1,1,args.context,1,1).half().clone()).float()
            z1=model.encode((2*hr-1)[:,:,None].repeat(1,1,args.context,1,1).half().clone()).float()
            y,x=((z0.shape[-1]-4)//2,)*2
            rows.append((z1-z0)[:,:,:,y:y+4,x:x+4].flatten().cpu())
            anchors.append(z0[:,:,:,y:y+4,x:x+4].flatten().cpu())
            sources.append({'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            if (i+1)%25==0:print(json.dumps({'event':'span_encode','sources':i+1}),flush=True)
        residuals=torch.stack(rows).cuda();_,singular,vh=torch.linalg.svd(residuals,full_matrices=False)
        basis=vh[:args.atoms].T.contiguous();coverage=float(singular[:args.atoms].square().sum()/singular.square().sum())
    result={'p0':basis.cpu(),'singular_values':singular.cpu(),'train_energy_fraction':coverage,'config':vars(args),'vae':info,
            'train_anchor_rois':torch.stack(anchors),'train_detail_p0':(residuals@basis).cpu(),
            'train_coverage_curve':{str(k):float(singular[:k].square().sum()/singular.square().sum()) for k in [8,16,32,64] if k<=len(singular)},
            'sources':sources,'definition':'uncentered train-only low-rank SR detail span; not a latent inverse operator'}
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)
    (output.with_suffix('.json')).write_text(json.dumps({k:v for k,v in result.items() if not torch.is_tensor(v)},indent=2))
    print(json.dumps({'event':'complete','train_sources':len(sources),'atoms':args.atoms,'energy_fraction':coverage,'shape':list(basis.shape)}),flush=True)


if __name__=='__main__':main()
