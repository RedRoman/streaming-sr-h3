"""Strong classical real-decoder LR inverse control. No HR inference objective.

Forty fixed Adam steps start from projected linear MAP, use all past real LR
packets and the exact same saved RGB-trained prior. Report decoder-call budget
and raw objective before/after. HR is loaded only after optimization for audit.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_interface import load_frozen_vae,differentiable_decoder_math
from observation_quotient import ObservationQuotient
from train_quotient import ProjectedPrior
from train_nonlinear import masks,rgb_metrics


class PastLRLikelihood:
    def __init__(self,inputs,source,q,stop=38):
        rectangle=source['canonical_rectangle'];wy=[];wx=[];observed=[];visible=[]
        def warp_axis(low,high,viewlow,viewhigh):
            u=(torch.arange(256,device='cuda',dtype=torch.float64)+.5)/256*(viewhigh-viewlow)+viewlow
            coordinate=(u-low)/(high-low)*256-.5;left=coordinate.floor().long();fraction=coordinate-left
            matrix=torch.zeros(256,256,device='cuda',dtype=torch.float64);rows=torch.arange(256,device='cuda')
            for index,value in [(left,1-fraction),(left+1,fraction)]:
                valid=(index>=0)&(index<256);matrix[rows[valid],index[valid]]=value[valid]
            return q.matrix@matrix
        for packet in inputs['packets'][:stop+1]:
            rect=packet['world_rectangle'];wx.append(warp_axis(rectangle[0],rectangle[2],rect[0],rect[2]));wy.append(warp_axis(rectangle[1],rectangle[3],rect[1],rect[3]));observed.append(packet['rgb'].cuda().permute(2,0,1).float()/255);visible.append(packet['visible_mask'].cuda()>0)
        self.wy=torch.stack(wy).float();self.wx=torch.stack(wx).float();self.y=torch.stack(observed)
        coverage=self.observe(torch.ones(1,1,256,256,device='cuda'))[:,0];self.mask=((coverage>.999)&(coverage<1.001)&torch.stack(visible))[:,None]
        # Fixed observed-LR nuisance basis, matching the classical cache. No HR.
        self.design=torch.stack([torch.ones_like(self.y),self.y],-1).double()*self.mask[...,None]
        self.gram=torch.einsum('bchwk,bchwl->bckl',self.design,self.design)+torch.eye(2,device='cuda',dtype=torch.float64)*1e-8
        self.inverse=torch.linalg.inv(self.gram)
        self.minimum_valid_pixels=int(self.mask.sum((-1,-2)).min())
    def observe(self,rgb):return self.wy[:,None]@rgb@self.wx[:,None].transpose(-1,-2)
    def residual(self,rgb):
        residual=(self.observe(rgb)-self.y)*self.mask
        natural=torch.einsum('bchwk,bchw->bck',self.design,residual.double());coefficient=(self.inverse@natural[...,None])[...,0]
        clean=residual-torch.einsum('bchwk,bck->bchw',self.design,coefficient).to(residual.dtype)
        return clean*self.mask
    def objective(self,rgb):return .5*self.residual(rgb).double().square().sum()/.005**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--wait-repair');p.add_argument('--max-sources',type=int);p.add_argument('--resume',action='store_true');args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if args.wait_repair:
        while not all((Path(args.wait_repair)/f'seed{s}'/'current_control_corrected.json').exists() for s in [17,29,43]):time.sleep(10)
    sys.path.insert(0,str(Path(args.comfy).resolve()));vae,info=load_frozen_vae(args.vae);q=ObservationQuotient();checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    prior=ProjectedPrior(checkpoint['prior']['p0']).cuda();prior.load_state_dict(checkpoint['prior']);prior.eval().requires_grad_(False)
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);result={'config':vars(args),'scope':'VAL-only classical nonlinearinverse diagnostic; ALL39 realLR targets+sharedprior, noHR optimization; extra40Dbackwardcalls vs neural1Dreadout','vae':info,'numerical_protocol':'Divide the entire unchanged sum objective by fixed valid RGB sample count for FP16 backward; retain best finite LR-objective iterate, never select using HR. Adam lr .05 /40 steps unchanged.','items':{}}
    if args.resume and (out/'results.json').exists():result['items']=json.loads((out/'results.json').read_text())['items']
    paths=sorted(Path(args.cache).glob('val_*.pt'))
    if args.max_sources is not None:paths=paths[:args.max_sources]
    for path in paths:
        if path.stem in result['items']:continue
        item=torch.load(path,map_location='cpu',weights_only=False);inputs=item['inputs'];z=inputs['anchor'].cuda()
        with torch.no_grad():mu,_=prior(z,inputs['roi']);gram=inputs['grams'][-1].cuda().double();rhs=inputs['rhs'][-1].cuda().double();initial=torch.linalg.solve(gram+torch.eye(32,device='cuda',dtype=torch.float64),rhs+mu[0].double()).float()
        likelihood=PastLRLikelihood(inputs,item['source'],q);coarse=inputs['coarse'][-1:].cuda()
        # Verify fast separable camera against the independently implemented
        # physical warp/blur/downsample before using its LR objective.
        probe=item['teacher']['decoder_responses'][:1].cuda();errors=[]
        for index in [0,17,38]:
            packet=inputs['packets'][index];truth=camera_footprint(probe,item['source']['canonical_rectangle'],packet['world_rectangle'])
            errors.append(float((likelihood.observe(probe)[index:index+1]-truth).abs().max()))
        assert max(errors)<2e-6,errors
        def decode(coefficient,grad=False):
            latent=prior.embed(z,coefficient[None],inputs['roi'])
            if grad:
                with differentiable_decoder_math():rgb=vae.decode(latent.half())[:,:,21].cuda()
            else:rgb=vae.decode(latent.half())[:,:,21].cuda()
            return q.readout(rgb,coarse)
        with torch.no_grad():before=decode(initial);objective_before=float(likelihood.objective(before)+.5*(initial-mu[0]).double().square().sum())
        coefficient=initial.detach().clone().requires_grad_(True);optimizer=torch.optim.Adam([coefficient],lr=.05)
        # Constant positive scaling preserves the inference objective while
        # preventing its million-scale sum from overflowing FP16 VAE backward.
        normalizer=float(likelihood.mask.sum()*3)
        best_objective=objective_before;best_coefficient=initial.detach().clone();best_rgb=before.detach().clone();best_step=0;trace=[]
        for step in range(args.steps):
            rgb=decode(coefficient,True);loss=likelihood.objective(rgb)+.5*(coefficient-mu[0]).double().square().sum();optimizer.zero_grad(set_to_none=True);(loss/normalizer).backward()
            if not torch.isfinite(loss) or coefficient.grad is None or not torch.isfinite(coefficient.grad).all():raise FloatingPointError(f'{path.stem} nonfinite objective/gradient at step{step}')
            if float(loss.detach())<best_objective:best_objective=float(loss.detach());best_coefficient=coefficient.detach().clone();best_rgb=rgb.detach().clone();best_step=step
            trace.append({'step':step,'objective':float(loss.detach()),'gradient_norm':float(coefficient.grad.norm())});optimizer.step()
        with torch.no_grad():
            final=decode(coefficient);final_objective=float(likelihood.objective(final)+.5*(coefficient-mu[0]).double().square().sum())
            if final_objective<best_objective:best_objective=final_objective;best_coefficient=coefficient.detach().clone();best_rgb=final.detach().clone();best_step=args.steps
            after=best_rgb;objective_after=best_objective;target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,'cuda')
            metrics={}
            for name,rgb in [('linear_map',before),('nonlinear_map40',after)]:
                bounded=q.bounded(rgb,coarse);metrics[name]={'affine':rgb_metrics(rgb,target,roi,support),'bounded':rgb_metrics(bounded,target,roi,support),'all39_lr_schur_rmse':float(likelihood.residual(bounded).square().sum().div(likelihood.mask.sum()*3).sqrt())}
        result['items'][path.stem]={'objective_before':objective_before,'objective_after':objective_after,'last_iterate_objective':final_objective,'best_lr_objective_step':best_step,'backward_constant_normalizer':normalizer,'operator_probe_maxabs':max(errors),'minimum_valid_pixels':likelihood.minimum_valid_pixels,'initial_code_norm':float(initial.norm()),'final_code_norm':float(best_coefficient.norm()),'gradient_decoder_calls':args.steps,'total_decoder_calls':args.steps+2,'optimization_trace':trace,'metrics':metrics}
        (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'nonlinear_lr_inverse','source':path.stem,'objective_before':objective_before,'objective_after':objective_after,'bounded_roi':{k:v['bounded']['psnr_nominal_roi_db'] for k,v in metrics.items()}}),flush=True)
    print(json.dumps({'event':'complete','val_sources':len(result['items'])}),flush=True)


if __name__=='__main__':main()
