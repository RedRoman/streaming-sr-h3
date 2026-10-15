"""Projected nonlinear H3 memory, with no raw HR-latent regression target.

Train a shared prior using actual projected decoder RGB, then equal-capacity
causal heads with the same prior/span/ledger/coarse/DC and actual RGB+witness
loss. Initial validation is a structural pilot, not a new held-out test claim.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import torch
from torch import nn
from PIL import Image
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint,nuisance_project
from memory_adapter import SpectralDetailTransport
from nonlinear_memory import NonlinearMemoryHead
from observation_quotient import ObservationQuotient
from h3_interface import load_frozen_vae,differentiable_decoder_math
from train_nonlinear import masks,rgb_metrics


class ProjectedPrior(nn.Module):
    def __init__(self,p0):
        super().__init__();self.register_buffer('p0',p0)
        self.context=nn.Sequential(nn.Linear(2688,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.output=nn.Linear(64,32);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,anchor,roi):
        y,x=roi;context=self.context(anchor[:,:,:,y:y+4,x:x+4].flatten(1));return self.output(context),context
    def embed(self,anchor,coefficient,roi):
        y,x=roi;out=anchor.clone();out[:,:,:,y:y+4,x:x+4]+=(coefficient@self.p0.T).reshape(-1,24,7,4,4);return out


def witness(rgb,item,t,nuisance=True):
    raw=[];clean=[]
    for index in sorted(set([0,min(t,17),t])):
        packet=item['inputs']['packets'][index];kwargs=dict(canonical_rectangle=item['source']['canonical_rectangle'],view_rectangle=packet['world_rectangle'],view_size=(256,256),lr_size=(64,64))
        predicted=camera_footprint(rgb,**kwargs);coverage=camera_footprint(torch.ones_like(rgb[:,:1]),**kwargs)[0,0]
        valid=(coverage>.999)&(coverage<1.001)&(packet['visible_mask'].cuda()>0);mask=valid[None].expand(3,-1,-1)
        observed=packet['rgb'].cuda().permute(2,0,1).float()/255;residual=(predicted[0]-observed)[mask]
        raw.append(residual.square().mean())
        if nuisance:
            bias=torch.eye(3,device='cuda')[:,:,None,None].expand(-1,-1,64,64);n=torch.cat([bias,bias*observed[None]],0)[:,mask].T
            # No target HR or unknown appearance parameter defines this Schur basis.
            _,residual=nuisance_project(torch.zeros(len(residual),1,device='cuda'),residual,n,damping=1e-6)
        clean.append(residual.square().mean())
    return torch.stack(clean).mean(),torch.stack(raw).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--prior-steps',type=int,default=400);p.add_argument('--steps',type=int,default=400);p.add_argument('--eval-splits',nargs='+',default=['val']);p.add_argument('--limit',type=int);p.add_argument('--eval-only');args=p.parse_args()
    start=time.time();torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);sys.path.insert(0,str(Path(args.comfy).resolve()));vae,vae_info=load_frozen_vae(args.vae);q=ObservationQuotient()
    items=[torch.load(path,map_location='cpu',weights_only=False) for path in sorted(Path(args.cache).glob('*.pt'))]
    splits={s:[v for v in items if v['source']['split']==s][:args.limit] for s in ['train','val','test']};assert splits['train'] and all(splits[s] for s in args.eval_splits)
    assert not {v['source']['source_hash'] for v in splits['train']}&{v['source']['source_hash'] for s in args.eval_splits for v in splits[s]}
    prior=ProjectedPrior(items[0]['teacher']['p0']).cuda();heads={k:NonlinearMemoryHead(k,rank=32,hidden=27,initialize=True).cuda() for k in ['transport','stateless','gru','current_only']}
    restored=torch.load(args.eval_only,map_location='cpu',weights_only=False) if args.eval_only else None
    if restored:
        prior.load_state_dict(restored['prior'])
        for k,h in heads.items():h.load_state_dict(restored['heads'][k])
    def inputs(item):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in item['inputs'].items()}
    def readout(item,z,coefficient,t,backward=False):
        latent=prior.embed(z['anchor'],coefficient,z['roi'])
        if backward:
            with differentiable_decoder_math():decoded=vae.decode(latent.half())[:,:,21].cuda()
        else:decoded=vae.decode(latent.half())[:,:,21].cuda()
        return q.readout(decoded,z['coarse'][t:t+1])
    def loss_for(rgb,item,t):
        target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,'cuda');error=(rgb-target).square();physical,_=witness(rgb,item,t)
        return 100*error[:,:,roi].mean()+25*error[:,:,support].mean()+25*physical
    log=[]
    if not restored:
        optimizer=torch.optim.Adam(prior.parameters(),lr=1e-3)
        for step in range(args.prior_steps):
            item=random.choice(splits['train']);z=inputs(item);coefficient,_=prior(z['anchor'],z['roi']);rgb=readout(item,z,coefficient,0,True);loss=loss_for(rgb,item,0)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(prior.parameters(),1.);optimizer.step()
            if step%50==0 or step==args.prior_steps-1:
                event={'event':'prior_rgb','step':step+1,'loss':float(loss.detach()),'seconds':time.time()-start};log.append(event);print(json.dumps(event),flush=True)
    prior.eval().requires_grad_(False);optimizers={k:torch.optim.Adam(h.parameters(),lr=1e-3) for k,h in heads.items()};matrix=torch.eye(32,device='cuda')[None];scale=torch.ones(1,32,device='cuda')
    if not restored:
        for step in range(args.steps):
            item=random.choice(splits['train']);z=inputs(item);t=random.choice([0,8,17,24,30,38]);noise=torch.randn(1,32,device='cuda')
            with torch.no_grad():mu,context=prior(z['anchor'],z['roi'])
            common=(matrix,mu,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None]);losses={}
            for kind,head in heads.items():
                coefficients,_=head(*common,noise,scale);rgb=readout(item,z,coefficients[:,t],t,True);loss=loss_for(rgb,item,t)
                optimizers[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizers[kind].step();losses[kind]=float(loss.detach())
            if step%50==0 or step==args.steps-1:
                event={'event':'heads_rgb','step':step+1,'losses':losses,'seconds':time.time()-start};log.append(event);print(json.dumps(event),flush=True)
        torch.save({'prior':prior.state_dict(),'heads':{k:h.state_dict() for k,h in heads.items()},'config':vars(args)},out/'checkpoint.pt')
    result={'config':vars(args),'scope':'Projected frozenH3VAE mechanism pilot; not H3 DiT; controlled camera/oraclevisibility; currentonly means fine evidence only, shared classical coarse mean','loss':'100 nominalROI RGBMSE +25 teacher90support RGBMSE +25 LR-only-nuisance-projected past3witnessMSE; no rawHRlatent loss','prior_parameters':sum(p.numel() for p in prior.parameters()),'head_parameters':{k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},'source_hashes':{s:[v['source']['source_hash'] for v in values] for s,values in splits.items()},'vae':vae_info,'training_log':log,'evaluation':{}}
    analytic=SpectralDetailTransport(rank=32).cuda();generator=torch.Generator(device='cuda').manual_seed(args.seed+50000)
    with torch.no_grad():
        for split in args.eval_splits:
            for number,item in enumerate(splits[split]):
                z=inputs(item);mu,context=prior(z['anchor'],z['roi']);noise=torch.randn(1,32,device='cuda',generator=generator);common=(matrix,mu,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None])
                predictions={k:h(*common,noise,scale)[0] for k,h in heads.items()}
                for mode in ['map','shared_seed','bures_transport']:predictions[mode]=analytic(*common,mode=mode,identity_noise=noise)[0]
                predictions['anchor_only']=torch.zeros_like(predictions['map']);predictions['fixed_prior']=mu[:,None].expand_as(predictions['map'])
                roi,support=masks(item,'cuda');target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;metrics={};panels=[]
                for kind,coefficients in predictions.items():
                    values=[];rgbs=[]
                    for t in [0,17,38]:
                        rgb=readout(item,z,coefficients[:,t],t);bounded=q.bounded(rgb,z['coarse'][t:t+1]);clean,raw=witness(bounded,item,t)
                        values.append({'frame':t,'affine':rgb_metrics(rgb,target,roi,support),'bounded':rgb_metrics(bounded,target,roi,support),'initialfootprint_lr_rmse':float((q.observe(bounded)-z['coarse'][t:t+1]).square().mean().sqrt()),'past_witness_clean_rmse':float(clean.sqrt()),'past_witness_raw_rmse':float(raw.sqrt()),'coefficient_norm':float(coefficients[:,t].norm())});rgbs.append(bounded)
                    metrics[kind]={'audits':values,'rgb_revision_17_38':float((rgbs[2]-rgbs[1]).square().mean())}
                    if kind in ['transport','gru','map','anchor_only']:panels.append(rgbs[-1].cpu())
                key=f"{split}/{Path(item['source']['path']).stem}";result['evaluation'][key]=metrics;print(json.dumps({'event':'evaluation','source':key,'bounded_roi_psnr':{k:v['audits'][-1]['bounded']['psnr_nominal_roi_db'] for k,v in metrics.items()},'seconds':time.time()-start}),flush=True)
                if number<2:Image.fromarray(np.concatenate([item['supervision']['canonical_hr'].numpy()]+[(v[0].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8) for v in panels],1)).save(out/f'panel_{split}{number}.png')
                (out/'results.json').write_text(json.dumps(result,indent=2))
    result['seconds']=time.time()-start;(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'complete','seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
