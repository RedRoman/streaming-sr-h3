"""Recover exact simulator measurements, verifying original cached LR inputs.

This is data construction only: HR creates the measurement exactly as during
the original cache build. Saved runtime measurement dictionaries contain no HR.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cache_pairs import image,photo,degrade


def collect(prov,size=256):
    seed=prov['seed'];device='cuda';rng=np.random.default_rng(seed)
    source=Path(prov['source']);donor=Path(prov['donor_source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==prov['source_sha256']
    assert hashlib.sha256(donor.read_bytes()).hexdigest()==prov['donor_sha256']
    a=image(source,size,device);b=a.clone();lo,hi=size//4,size*3//4;d=image(donor,size,device);b[:,lo:hi,lo:hi]=d[:,lo:hi,lo:hi]
    qs=[torch.zeros(12,device=device)]
    for _ in range(2):qs.append(torch.tensor(np.r_[rng.uniform(-.18,.18,3),rng.uniform(-.025,.025,3),rng.uniform(-.09,.09,6)],device=device,dtype=torch.float32))
    targets=torch.stack([photo(x,q) for x in (a,b) for q in qs]);generator=torch.Generator(device=device).manual_seed(seed)
    obs=[];far=[];near=[]
    for target,close in zip([0,1,1,0,3,5],[False,False,True,False,True,False]):
        x=targets[target];lr=degrade(x[None],4)[0];lr=(lr+torch.randn(lr.shape,device=device,generator=generator)*.003).clamp(0,1);far.append(lr.cpu())
        rgb=F.interpolate(lr[None],size=(size,size),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1);yl=torch.zeros_like(lr)
        if close:
            yl=degrade(x[:,lo:hi,lo:hi][None],2)[0];yl=(yl+torch.randn(yl.shape,device=device,generator=generator)*.003).clamp(0,1)
            rgb[:,lo:hi,lo:hi]=F.interpolate(yl[None],size=(hi-lo,hi-lo),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
        near.append(yl.cpu());obs.append(rgb)
    obs[4][:,:lo]=obs[3][:,:lo];obs[4][:,hi:]=obs[3][:,hi:];obs[4][:,lo:hi,:lo]=obs[3][:,lo:hi,:lo];obs[4][:,lo:hi,hi:]=obs[3][:,lo:hi,hi:]
    order=[0,1,2,2,3,4,5,5]
    inputs=dict(far_y=torch.stack(far)[order],near_y=torch.stack(near)[order],near_stage=torch.tensor([False,False,True,True,False,True,False,False]),visible_stage=torch.tensor([True,True,True,False,True,True,True,True]),roi=(lo,hi))
    return torch.stack(obs)[order].cpu().half(),inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4)
    cache=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((cache/'manifest.json').read_text());records=[];t=time.time()
    with torch.no_grad():
        for prov in manifest['items']:
            reconstructed,measurements=collect(prov,manifest['config']['size']);old=torch.load(cache/(prov['id']+'.pt'),weights_only=False)['inputs']['observation_rgb']
            assert torch.equal(reconstructed,old),('Changed simulator input!',prov['id'],float((reconstructed-old).abs().max()))
            path=out/(prov['id']+'.pt');torch.save(dict(inputs=measurements,source_id=prov['id'],no_HR_stored=True),path)
            records.append(dict(source_id=prov['id'],split=prov['split'],cached_LR_RGB_bitwise_equal=True,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(passed=True,items=records,seconds=time.time()-t,scope='Exact physical LR measurements from original observation simulator, no reconstruction result or HR in saved inputs',operator='Gaussian sigma.6 kernel5 reflect + torch bicubic antialias align_cornersFalse; observed noise.003/clipped',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,items=len(records),seconds=time.time()-t)),flush=True)


if __name__=='__main__':main()
