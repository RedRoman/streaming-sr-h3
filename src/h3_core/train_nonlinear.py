"""Chart pretraining then nonlinear memory with genuine frozen-H3 RGB loss.

This is a frozen-H3-VAE mechanism experiment, not H3 DiT fine tuning. All
restoration losses use train sources only; validation/test are evaluated once
per fixed-budget run. Cached local derivatives are features, never RGB labels.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_data import NaturalCameraSequence
from memory_adapter import CanonicalChart,SpectralDetailTransport
from nonlinear_memory import NonlinearMemoryHead
from h3_interface import load_frozen_vae,differentiable_decoder_math


def move_inputs(item,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in item['inputs'].items()}


def masks(item,device):
    responses=item['teacher']['decoder_responses'].to(device)
    energy=responses.square().sum((0,1));values,indices=energy.flatten().sort(descending=True)
    keep=values.cumsum(0)<=values.sum()*.9;keep[0]=True
    support=torch.zeros_like(keep);support[indices[keep]]=True;support=support.reshape(energy.shape)
    roi=torch.zeros_like(support);y,x=item['inputs']['roi'];roi[y*16:(y+4)*16,x*16:(x+4)*16]=True
    return roi,support


def rgb_metrics(rgb,target,roi,support):
    error=(rgb-target).square()
    def psnr(value):return float(-10*torch.log10(value.clamp_min(1e-12)))
    gradient=F.l1_loss(rgb[:,:,:,1:]-rgb[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])+F.l1_loss(rgb[:,:,1:]-rgb[:,:,:-1],target[:,:,1:]-target[:,:,:-1])
    return {'psnr_full_db':psnr(error.mean()),'psnr_nominal_roi_db':psnr(error[:,:,roi].mean()),'psnr_teacher90_support_db':psnr(error[:,:,support].mean()),'gradient_l1_full':float(gradient),'nominal_roi_fraction':float(roi.float().mean()),'teacher90_support_fraction':float(support.float().mean())}


def packet_loss(rgb,item,t,sequence):
    errors=[]
    # Three past witnesses, with duplicates removed. A future packet is never used.
    for witness in sorted(set([0,min(t,17),t])):
        packet=sequence.render(witness)[0]
        kwargs=dict(canonical_rectangle=item['source']['canonical_rectangle'],view_rectangle=packet.world_rectangle,view_size=rgb.shape[-2:],lr_size=packet.rgb.shape[:2])
        observed=torch.as_tensor(packet.rgb.copy(),device=rgb.device).permute(2,0,1)[None].float()/255
        projected=camera_footprint(rgb,**kwargs)
        cover=camera_footprint(torch.ones_like(rgb[:,:1]),**kwargs)[0,0]
        valid=(cover>.999)&(cover<1.001)&(torch.as_tensor(packet.visible_mask.copy(),device=rgb.device)>.5)
        if valid.any():errors.append((projected-observed)[:,:,valid].square().mean())
    return torch.stack(errors).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--span',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--basis-steps',type=int,default=600);p.add_argument('--latent-steps',type=int,default=400);p.add_argument('--rgb-steps',type=int,default=100)
    p.add_argument('--basis-mode',choices=['learned','fixed_pca','random'],default='learned');p.add_argument('--eval-only')
    p.add_argument('--pipeline-only',action='store_true',help='Use only first train item to check gradients/metrics while larger cache is being generated')
    args=p.parse_args()
    start=time.time();torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed);device=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    items=[torch.load(path,map_location='cpu',weights_only=False) for path in sorted(Path(args.cache).glob('*.pt'))]
    splits={s:[v for v in items if v['source']['split']==s] for s in ['train','val','test']}
    if not args.pipeline_only and any(not x for x in splits.values()):raise ValueError('Need all three disjoint source splits')
    if args.pipeline_only:splits={'train':splits['train'][:1],'val':[],'test':[]}
    hashes={s:{v['source']['source_hash'] for v in values} for s,values in splits.items()}
    assert not hashes['train']&(hashes['val']|hashes['test'])
    span=torch.load(args.span,map_location='cpu',weights_only=False)
    chart=CanonicalChart(temporal=7,basis_mode=args.basis_mode).cuda();chart.p0.copy_(span['p0'].cuda())
    bank=span['train_anchor_rois'].cuda();targets=span['train_detail_p0'].cuda()
    scale32=targets.square().mean(0).sqrt().clamp_min(1.)
    optimizer=torch.optim.Adam(chart.parameters(),lr=1e-3)
    restored=torch.load(args.eval_only,map_location='cpu',weights_only=False) if args.eval_only else None
    if restored is None:
        for step in range(args.basis_steps):
            indices=torch.randint(len(bank),(16,),device=device);matrix,prior,_=chart(bank[indices])
            target=targets[indices];predicted=(matrix@prior[...,None])[...,0]
            projected=(matrix@(matrix.transpose(1,2)@target[...,None]))[...,0]
            loss=((predicted-target)/scale32).square().mean()+((projected-target)/scale32).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    else:chart.load_state_dict(restored['chart'])
    # Sharing one frozen chart gives every head identical representation and
    # likelihood. Learned vs fixed-basis runs use exactly the same outer budget.
    chart.eval().requires_grad_(False)
    with torch.no_grad():
        matrix,_,_=chart(bank);projected=torch.einsum('bkr,bk->br',matrix,targets)
        scale=projected.square().mean(0).sqrt().clamp_min(1.)[None]
    heads={kind:NonlinearMemoryHead(kind).cuda() for kind in ['transport','stateless','gru','current_only']}
    if restored:
        for k,h in heads.items():h.load_state_dict(restored['heads'][k])
        scale=restored['scale'].cuda()
    optimizers={k:torch.optim.Adam(h.parameters(),lr=1e-3) for k,h in heads.items()}
    vae=None;sequences={}
    def sequence_for(item):
        key=item['source']['source_hash']
        if key not in sequences:sequences[key]=NaturalCameraSequence(item['source']['path'],size=256,frames=39,protocol=item['source']['protocol'],seed=item['source']['sequence_seed'])
        return sequences[key]
    def features(item):
        z=move_inputs(item,device);y,x=z['roi']
        with torch.no_grad():matrix,prior,context=chart(z['anchor'][:,:,:,y:y+4,x:x+4])
        return z,matrix,prior,context
    training_log=[]
    if restored is None:
        for phase,steps in [('latent',args.latent_steps),('rgb',args.rgb_steps)]:
            if phase=='rgb':
                sys.path.insert(0,str(Path(args.comfy).resolve()));vae,vae_info=load_frozen_vae(args.vae)
            for step in range(steps):
                item=random.choice(splits['train']);z,matrix,prior,context=features(item);noise=torch.randn(1,8,device=device)
                head_args=(matrix,prior,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None],noise,scale)
                target=item['supervision']['target_p0'].cuda();losses={};t=random.choice([0,8,17,24,30,38])
                for kind,head in heads.items():
                    predicted,_=head(*head_args);represented=torch.einsum('bkr,btr->btk',matrix,predicted)
                    coefficient_loss=((represented-target[:,None])/scale32).square().mean()
                    loss=coefficient_loss
                    if phase=='rgb':
                        guide=chart.embed(z['anchor'],matrix,predicted[:,t],z['roi'])
                        with differentiable_decoder_math():rgb=vae.decode(guide.half())[:,:,item['source']['output_frame']].to(device)
                        gt=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255
                        roi,support=masks(item,device)
                        pixel=((rgb-gt).square()[:,:,support]).mean()
                        lr=packet_loss(rgb,item,t,sequence_for(item))
                        # Fixed, declared loss weights; no validation selection.
                        loss=.1*coefficient_loss+100*pixel+25*lr
                    optimizers[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizers[kind].step();losses[kind]=float(loss.detach())
                if step%50==0 or step==steps-1:
                    log={'event':'train','phase':phase,'step':step+1,'losses':losses,'seconds':time.time()-start};training_log.append(log);print(json.dumps(log),flush=True)
        torch.save({'chart':chart.state_dict(),'heads':{k:h.state_dict() for k,h in heads.items()},'scale':scale.cpu(),'config':vars(args)},out/'checkpoint.pt')
    if vae is None:
        sys.path.insert(0,str(Path(args.comfy).resolve()));vae,vae_info=load_frozen_vae(args.vae)
    result={'config':vars(args),'scope':'Frozen native H3 VAE nonlinear readout; no H3 DiT claim; controlled static-natural camera with oracle geometry/visibility','identity_noise':'One identical unit Gaussian rank8 draw for every method on each source; matches classical prior precision I. Train residual scale is only an output normalization.','source_hashes':{s:sorted(x) for s,x in hashes.items()},'source_protocols':{f"{v['source']['split']}/{Path(v['source']['path']).stem}":v['source']['protocol'] for v in items},'train_span_sources':len(bank),'parameters':{k:sum(p.numel() for p in h.parameters()) for k,h in heads.items()},'shared_chart_parameters':sum(p.numel() for p in chart.parameters()),'training_log':training_log,'evaluation':{}}
    analytic=SpectralDetailTransport().cuda()
    generator=torch.Generator(device=device).manual_seed(args.seed+50000)
    with torch.no_grad():
        for split in (['train'] if args.pipeline_only else ['val','test']):
            for number,item in enumerate(splits[split]):
                z,matrix,prior,context=features(item);noise=torch.randn(1,8,generator=generator,device=device)
                common=(matrix,prior,context,z['grams'][None],z['rhs'][None],z['packet_counts'][None])
                predictions={k:h(*common,noise,scale)[0] for k,h in heads.items()}
                for mode in ['map','shared_seed','bures_transport']:
                    predictions[mode]=analytic(*common,mode=mode,identity_noise=noise)[0]
                predictions['anchor_only']=torch.zeros_like(predictions['map'])
                predictions['fixed_reference']=(prior+noise)[:,None].expand_as(predictions['map'])
                # Supervised local-code transfer diagnostic, not an RGB bound;
                # it omits changes to the other latent tokens. Never an input
                # or a claimed restoration method. It is labeled in the output.
                target_p0=item['supervision']['target_p0'].cuda()
                oracle=(matrix.transpose(1,2)@target_p0[...,None])[...,0]
                predictions['oracle_local_hr_code_diagnostic']=oracle[:,None].expand_as(predictions['map'])
                target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,device);metrics={};panels=[]
                for kind,prediction in predictions.items():
                    values=[];rgbs=[]
                    for t in [0,17,38]:
                        guide=chart.embed(z['anchor'],matrix,prediction[:,t],z['roi'])
                        rgb=vae.decode(guide.half())[:,:,item['source']['output_frame']].to(device)
                        entry={'frame':t,**rgb_metrics(rgb,target,roi,support),'lr_witness_rmse':float(packet_loss(rgb,item,t,sequence_for(item)).sqrt()),'code_norm':float(prediction[:,t].norm())}
                        values.append(entry);rgbs.append(rgb)
                    metrics[kind]={'audits':values,'actual_canonical_rgb_drift_17_38':float((rgbs[2]-rgbs[1]).square().mean()),'code_drift_occluded24_30':float((prediction[:,30]-prediction[:,24]).square().mean())}
                    if kind in ['anchor_only','map','gru','transport']:panels.append((kind,rgbs[-1].cpu()))
                key=f"{split}/{Path(item['source']['path']).stem}";result['evaluation'][key]=metrics
                y,x=z['roi']
                full_delta=(item['supervision']['target_latent']-item['inputs']['anchor'])[:,:,:,y:y+4,x:x+4]
                result.setdefault('basis_diagnostics',{})[key]={'rank32_target_energy_fraction':float(target_p0.square().sum())/max(float(full_delta.square().sum()),1e-12),'conditional_rank8_fraction_of_rank32':float(oracle.square().sum()/target_p0.square().sum().clamp_min(1e-12)),'oracle_label':'Local HR-code transfer diagnostic; not an RGB upper bound because other latent tokens remain LR. Excluded from inference inputs and all fitting on val/test'}
                print(json.dumps({'event':'eval','item':key,'roi_psnr':{k:v['audits'][-1]['psnr_nominal_roi_db'] for k,v in metrics.items()},'seconds':time.time()-start}),flush=True)
                if number<2:
                    arrays=[item['supervision']['canonical_hr'].numpy()]+[(r[0].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8) for _,r in panels]
                    Image.fromarray(np.concatenate(arrays,1)).save(out/f'panel_{split}{number}.png')
                (out/'results.json').write_text(json.dumps(result,indent=2))
    result['seconds']=time.time()-start;(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'complete','seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
