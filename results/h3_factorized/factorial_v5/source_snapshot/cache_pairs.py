"""Actual H3 caches for content/appearance counterfactual SR rollouts.

Uses official disjoint BSDS source splits. Ground truth is exclusively simulator
or training/evaluation supervision; model inputs are blurred/downsampled LR only.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from native_vae import load_vae,encode_repeated,decode_last,CANONICAL_SHA


def live(out,**fields):
    p=out.parent/'LIVE_STATE.json'
    try:s=json.loads(p.read_text())
    except Exception:s={}
    s.update(fields);s['pid']=os.getpid();s['unix_time']=time.time();p.write_text(json.dumps(s,indent=2))


def photo(x,q):
    h,w=x.shape[-2:];yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
    modes=torch.stack((xx,yy,xx*yy,xx.square()-1/3,yy.square()-1/3,torch.cos(torch.pi*xx/2)*torch.cos(torch.pi*yy/2)))
    shade=1+(q[6:,None,None]*modes).sum(0)
    return (x*(1+q[:3,None,None])*shade+q[3:6,None,None]).clamp(0,1)


def degrade(x,factor):
    coords=torch.arange(-2,3,device=x.device,dtype=x.dtype);k=torch.exp(-coords.square()/(2*.6**2));k=k/k.sum();kernel=(k[:,None]*k[None,:])[None,None].repeat(3,1,1,1)
    x=F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),kernel,groups=3)
    return F.interpolate(x,size=(x.shape[-2]//factor,x.shape[-1]//factor),mode='bicubic',align_corners=False,antialias=True)


def image(path,size,device):
    im=Image.open(path).convert('RGB');w,h=im.size
    if min(w,h)<size:raise ValueError('Source too small')
    left=(w-size)//2;top=(h-size)//2
    arr=np.asarray(im.crop((left,top,left+size,top+size))).copy()
    return torch.from_numpy(arr).to(device).permute(2,0,1).float()/255


def make_pair(path,donor,size,seed,device):
    rng=np.random.default_rng(seed);a=image(path,size,device);don=image(donor,size,device)
    lo,hi=size//4,size*3//4;b=a.clone();b[:,lo:hi,lo:hi]=don[:,lo:hi,lo:hi]
    q0=torch.zeros(12,device=device)
    qs=[q0]
    for _ in range(2):
        q=torch.tensor(np.r_[rng.uniform(-.18,.18,3),rng.uniform(-.025,.025,3),rng.uniform(-.09,.09,6)],device=device,dtype=torch.float32)
        qs.append(q)
    targets=torch.stack([photo(x,q) for x in (a,b) for q in qs])
    generator=torch.Generator(device=device).manual_seed(seed)
    def observed(x,near=False):
        lr=degrade(x[None],4)[0];lr=(lr+torch.randn(lr.shape,device=device,generator=generator)*.003).clamp(0,1)
        far=F.interpolate(lr[None],size=(size,size),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
        if near:
            crop=x[:,lo:hi,lo:hi];yl=degrade(crop[None],2)[0]
            yl=(yl+torch.randn(yl.shape,device=device,generator=generator)*.003).clamp(0,1)
            far[:,lo:hi,lo:hi]=F.interpolate(yl[None],size=(hi-lo,hi-lo),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
        return far
    unique=[observed(targets[0]),observed(targets[1]),observed(targets[1],True),observed(targets[0]),observed(targets[3],True),observed(targets[5])]
    # The B near view's outside pixels equal A under neutral illumination; no
    # HR target, donor patch, or neutral ground truth enters inference tensors.
    unique[4][:,:lo]=unique[3][:,:lo];unique[4][:,hi:]=unique[3][:,hi:]
    unique[4][:,lo:hi,:lo]=unique[3][:,lo:hi,:lo];unique[4][:,lo:hi,hi:]=unique[3][:,lo:hi,hi:]
    unique=torch.stack(unique)
    order=[0,1,2,2,3,4,5,5]
    visible=torch.ones(8,1,size,size,device=device);visible[3]=0
    reveal=torch.zeros_like(visible);reveal[2,:,lo:hi,lo:hi]=1;reveal[5,:,lo:hi,lo:hi]=1
    visible[2]=reveal[2];visible[5]=reveal[5]
    return unique,targets,torch.stack([q for _ in range(2) for q in qs]),visible,reveal,order


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    p.add_argument('--counts',type=int,nargs=3,default=[32,8,16]);p.add_argument('--size',type=int,default=256);p.add_argument('--context',type=int,default=22);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    live(out,status='checking_complete_H3_VAE',command=' '.join(sys.argv),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    start=time.time();vae=load_vae(args.vae,args.comfy)
    files={s:sorted((Path(args.data)/s).glob('*.jpg'))[:n] for s,n in zip(('train','val','test'),args.counts)}
    for s,n in zip(files,args.counts):assert len(files[s])==n,(s,len(files[s]),n,args.data)
    assert not set(x.name for x in files['train'])&set(x.name for x in files['test'])
    assert not set(x.name for x in files['train'])&set(x.name for x in files['val'])
    manifest=dict(config=vars(args),native_H3_VAE=True,vae_sha256=CANONICAL_SHA,device=torch.cuda.get_device_name(0),encoder_dtype='float16',decoder_audit_dtype='float16',latent_shape=[1,24,7,args.size//16,args.size//16],scope='Controlled static canonical content and simulated photometric/local-content changes in actual H3 representation; no DiT or real-motion effectiveness claim',causality='Each stage uses current/past actual LR only; HR targets live in a separate supervision dict',copied_context='22 temporal copies of one image only supply native VAE context; never counted as22 measurements',degradation='Gaussian sigma.6 kernel5 reflection padding + torch bicubic4x antialias align_cornersFalse; independent LR noise.003; near ROI128 gives genuine2x canonical samples',source_split_policy='BSDS official source-level train/val/test, donor restricted to SAME split and selected source pool',target_order=['A0','A1','A2','B0','B1','B2'],stage_order=['far_A0','far_A1','near_A1','query_no_visible_evidence','return_A0','near_B0','far_B2','duplicate_far_B2'],items=[])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with torch.no_grad():
        for split,paths in files.items():
            for i,path in enumerate(paths):
                destination=out/f'{split}_{path.stem}.pt'
                if destination.exists():
                    cached=torch.load(destination,map_location='cpu',weights_only=False);manifest['items'].append(cached['provenance']);continue
                donor=paths[(i+1)%len(paths)];seed=int(hashlib.sha256(path.name.encode()).hexdigest()[:8],16)
                observed,targets,appearance,visible,reveal,order=make_pair(path,donor,args.size,seed,'cuda')
                zl=[encode_repeated(vae,x[None],args.context)[0].cpu().half() for x in observed]
                zt=[encode_repeated(vae,x[None],args.context)[0].cpu().half() for x in targets]
                # Frozen decoder is actually executed during caching, so even
                # before training its representation floor is recorded in RGB.
                decoded=decode_last(vae,zt[0][None].cuda()).cpu()[0]
                mse=float((decoded-targets[0].cpu()).square().mean())
                prov=dict(id=f'{split}_{path.stem}',split=split,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),donor_source=str(donor),donor_sha256=hashlib.sha256(donor.read_bytes()).hexdigest(),path=str(destination),seed=seed,target_roundtrip_psnr=float(-10*np.log10(max(mse,1e-12))))
                inputs=dict(observation_latents=torch.stack([zl[j] for j in order]),observation_rgb=observed[order].cpu().half(),visible=visible.cpu().half(),reveal=reveal.cpu().half(),footprint=torch.tensor([.25,.25,.5,.5,.25,.5,.25,.25]))
                supervision=dict(latents=torch.stack(zt),rgb=targets.cpu().half(),appearance=appearance.cpu())
                torch.save(dict(inputs=inputs,supervision=supervision,provenance=prov),destination)
                manifest['items'].append(prov);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
                live(out,status='caching_actual_H3_latents',cached=len(manifest['items']),last_source=prov['id'],elapsed_seconds=time.time()-start)
                print(json.dumps(dict(event='cached',**prov,elapsed_seconds=round(time.time()-start,2))),flush=True)
    manifest['runtime_seconds']=time.time()-start;manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    live(out,status='cache_complete',cached=len(manifest['items']),elapsed_seconds=time.time()-start)
    print(json.dumps(dict(event='complete',items=len(manifest['items']),seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
