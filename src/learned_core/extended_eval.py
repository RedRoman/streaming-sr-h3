"""Held-out stress tests and mechanism ablations of the fixed checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiment import Sources, CoreModel, pack, gaussian_map, simple_map, score


def simulate(target, condition, seed=62181):
    n,_,h,w=target.shape
    g=torch.Generator(device=target.device).manual_seed(seed)
    orders=torch.stack([torch.randperm(16,generator=g,device=target.device) for _ in range(n)])
    noise=.05 if condition=='high_noise_new' else .01
    fixed_noise=torch.randn(target.shape,generator=g,device=target.device)*noise
    seen=torch.zeros_like(target)
    ys,masks,fresh=[],[],[]
    steps=24 if condition=='duplicate_then_new' else (8 if condition in ['eight_new','high_noise_new','wrong25','checker_ood'] else 16)
    for t in range(steps):
        position=t
        if condition=='duplicate_then_new':
            position=0 if t<8 else t-7
        if condition in ('independent_noise_repeat','duplicate_repeat'):
            position=0
        position=min(position,15)
        mask=torch.zeros((n,16,1,1),device=target.device)
        mask[torch.arange(n,device=target.device),orders[:,position]]=1
        mask=mask.expand(-1,-1,h,w)
        if condition=='reveal_then_repeat' and t==4:
            mask=torch.ones_like(mask)
        if condition=='reveal_then_repeat' and t>4:
            mask=torch.zeros_like(mask)
            mask[torch.arange(n,device=target.device),orders[:,0]]=1
        eps=torch.randn(target.shape,generator=g,device=target.device)*noise if condition=='independent_noise_repeat' else fixed_noise
        values=target+eps
        if condition=='wrong25' and t%4==3:
            values=torch.roll(values,1,1)
        ys.append(values*mask)
        masks.append(mask)
        packet_fresh=mask if condition=='independent_noise_repeat' else mask*(1-seen)
        fresh.append(packet_fresh)
        seen=(seen+mask).clamp_max(1)
    return ys,masks,fresh,noise


def transport_ablation(model,ys,masks,fresh,bases,variant):
    sums=torch.zeros_like(ys[0]);counts=torch.zeros_like(ys[0]);detail=torch.zeros_like(ys[0]);out=[]
    for t,(y,mask,fresh_mask,base) in enumerate(zip(ys,masks,fresh,bases)):
        old_mean=sums/counts.clamp_min(1)
        novel=mask*(counts==0)
        innovation=mask*(y-old_mean)
        sums+=y*fresh_mask;counts+=fresh_mask
        if variant=='reset_detail':detail=detail*0
        proposal,gain=model.net(torch.cat([base,counts.clamp_max(4)/4,detail,innovation],1)).chunk(2,1)
        gain=gain.sigmoid()*novel.flatten(1).amax(1)[:,None,None,None]
        if variant=='replace_detail':gain=(gain>0).to(gain)
        if variant=='freeze_after_first' and t>0:gain=gain*0
        detail=detail+gain*(proposal-detail)
        out.append(base+detail if variant=='no_projection' else base+(counts==0)*detail)
    return out


def checker_targets(count,side,device):
    rng=np.random.default_rng(39971);yy,xx=np.mgrid[:side,:side];out=[]
    for _ in range(count):
        angle,freq,phase=rng.uniform(0,np.pi),rng.uniform(.07,.39),rng.uniform(0,6.28)
        carrier=np.sin(2*np.pi*freq*(xx*np.cos(angle)+yy*np.sin(angle))+phase)
        second=np.sin(2*np.pi*rng.uniform(.05,.3)*(xx*np.sin(angle)-yy*np.cos(angle)))
        # Discontinuous multiplicative checkers/chirps never occur in the
        # training generator (three additive continuous sinusoidal waves).
        x=.5+.3*np.sign(carrier*second)
        out.append(x.astype(np.float32))
    return pack(torch.from_numpy(np.stack(out)[:,None]).to(device))


def calibrate_residual(model,source,side,stats,device):
    """A fixed risk identity, fitted on VAL only, with no test feedback.

    R(a)-R(0) = 2*a E[e*d] + a^2 E[d^2]. The scalar a is
    clipped to [0,1]. This is a diagnostic control, not a new trained model.
    """
    target=pack(source.evaluation(100,side,198)).to(device)
    ys,masks,fresh,noise=simulate(target,'long_new',seed=82731)
    ys,masks,fresh=ys[:8],masks[:8],fresh[:8]
    bases=gaussian_map(ys,masks,stats,noise**2,fresh)
    predictions=model(ys,masks,bases,fresh)
    result={}
    for t,(base,pred) in enumerate(zip(bases,predictions)):
        error=base-target;delta=pred-base
        correlation=(error*delta).flatten(1).mean(1)
        energy=delta.square().flatten(1).mean(1)
        numerator=float(correlation.mean());denominator=float(energy.mean())
        alpha=float(np.clip(-numerator/max(denominator,1e-12),0,1))
        result[t+1]={'alpha_mse':alpha,'error_residual_correlation':numerator,'residual_energy':denominator,
                     'val_mse_before':float(error.square().mean()),
                     'val_mse_after':float((error+alpha*delta).square().mean()),
                     'val_psnr_before':float((-10*torch.log10(error.square().flatten(1).mean(1))).mean()),
                     'val_psnr_after':float((-10*torch.log10((error+alpha*delta).square().flatten(1).mean(1))).mean())}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/BSR/BSDS500/data/images');p.add_argument('--run',default='results/learned_core/main')
    p.add_argument('--out',default='results/learned_core/extended');p.add_argument('--count',type=int,default=200);p.add_argument('--device',default='cuda')
    p.add_argument('--conditions',nargs='+',default=['long_new','duplicate_repeat','independent_noise_repeat','duplicate_then_new','high_noise_new','reveal_then_repeat','wrong25','checker_ood'])
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device(args.device)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    stats=tuple(x.to(device) for x in torch.load(Path(args.run)/'gaussian_prior.pt',weights_only=True))
    source=Sources(args.data,'test');rows=[];all_results={}
    for checkpoint in sorted(Path(args.run).glob('checkpoint_seed*.pt')):
        loaded=torch.load(checkpoint,map_location=device,weights_only=False);seed=loaded['seed'];side=loaded['config']['side']
        models={kind:CoreModel(kind).to(device).eval() for kind in loaded['models']}
        for kind,model in models.items():model.load_state_dict(loaded['models'][kind])
        natural=pack(source.evaluation(args.count,side,269)).to(device)
        seed_result={}
        with torch.no_grad():
            calibration=calibrate_residual(models['map_transport'],Sources(args.data,'val'),side,stats,device)
            (out/f'calibration_seed{seed}.json').write_text(json.dumps(calibration,indent=2))
            for condition in args.conditions:
                target=checker_targets(args.count,side,device) if condition=='checker_ood' else natural
                ys,masks,fresh,noise=simulate(target,condition)
                bases=gaussian_map(ys,masks,stats,noise**2,fresh)
                outputs={kind:model(ys,masks,bases,fresh) for kind,model in models.items()}
                outputs['gaussian_all_history_map']=bases
                outputs['flat_all_history_map']=simple_map(ys,masks,fresh_masks=fresh)
                outputs['current_prior_all_history_map']=simple_map(ys,masks,outputs['current'],fresh_masks=fresh)
                outputs['gru_prior_all_history_map']=simple_map(ys,masks,outputs['gru'],fresh_masks=fresh)
                # Transfer the VAL risk coefficient once, including OOD
                # conditions; ranks absent from calibration use zero residual.
                seen=torch.zeros_like(target);calibrated=[]
                for base,pred,mask in zip(bases,outputs['map_transport'],masks):
                    seen=(seen+mask).clamp_max(1)
                    ranks=(seen[:,:,0,0]>0).sum(1).tolist()
                    alpha=torch.tensor([calibration.get(rank,{'alpha_mse':0})['alpha_mse'] for rank in ranks],device=device)[:,None,None,None]
                    calibrated.append(base+alpha*(pred-base))
                outputs['val_risk_calibrated_transport']=calibrated
                for variant in ['reset_detail','replace_detail','freeze_after_first','no_projection']:
                    outputs['map_transport/'+variant]=transport_ablation(models['map_transport'],ys,masks,fresh,bases,variant)
                metrics={kind:score(predictions,target,ys,masks) for kind,predictions in outputs.items()}
                seed_result[condition]=metrics
                print(json.dumps({'event':'extended_eval','seed':seed,'condition':condition,'psnr':{kind:round(v['psnr_final_db'],4) for kind,v in metrics.items()}}),flush=True)
                for kind,values in metrics.items():rows.append({'seed':seed,'condition':condition,'model':kind,**{k:v for k,v in values.items() if not isinstance(v,list)}})
        all_results[str(seed)]=seed_result
        (out/f'seed{seed}.json').write_text(json.dumps(seed_result,indent=2))
        (out/'all_results.json').write_text(json.dumps(all_results,indent=2))
        with (out/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'event':'complete','seeds':list(all_results)}),flush=True)

if __name__=='__main__':main()
