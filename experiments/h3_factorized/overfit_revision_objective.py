"""Authorized fixed single-train-source objective diagnosis, not a method result.

Two arms, same existing checkpoint/input/architecture, 300 updates each.
Latent-only versus actual-native-D RGB-only, no selection or weight sweep.
"""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_model import build
from revision_utils import batch,Readout
from native_vae import load_vae,decode_last,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--lr',type=float,default=.0005);args=p.parse_args();assert args.steps==300 and args.lr==.0005;torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(17);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Path(args.cache);manifest=json.loads((cache/'manifest.json').read_text());train=[x for x in manifest['items'] if x['split']=='train'];prov=train[0];assert prov['id']=='train_100075';item=torch.load(cache/(prov['id']+'.pt'),map_location='cpu',weights_only=False);other=torch.load(cache/(train[1]['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');inputs['weights'][:,:,1:]=0;w=inputs['weights'][:,3,0].unsqueeze(2);q=targets['query_q'][:,0].clone();mask=inputs['packet_masks'][:,0];anchor=inputs['query_latents'][:,0];vae=load_vae(args.vae,args.comfy);read=Readout(vae);checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);start=time.time();rows=[];audits=[];history=[]
    with torch.no_grad():
        fullcode=item['classical_inputs']['packet_full_latents'][0][None].cuda().float();teacher_delta=fullcode-anchor;fullRGB=decode_last(vae,fullcode);teacherRGB=inputs['query_base_rgb'][:,0]+read.historical_band(fullRGB-inputs['query_base_rgb'][:,0],inputs,0);HR=targets['rgb'][:,0];far=item['classical_inputs']['query_LR'][0:1].cuda().float()
    def forward(model,delta=None):
        h=model.lift(inputs['packet_delta'][:,0] if delta is None else delta,inputs['packet_proxy'][:,0],mask);canonical=model.core.canonical(h,q);styled=model.core.style(canonical,q)*(w>0);return model.read(styled,anchor,q,(w/8).clamp(0,1)),h
    def evaluate(model,arm,step):
        with torch.no_grad():
            z,h=forward(model);rgb=read.rgb(inputs,z,0);zero,_=forward(model,inputs['packet_delta'][:,0]*0);shuffled,_=forward(model,other['inputs']['packet_delta'][0][None].cuda().float());zr=read.rgb(inputs,zero,0);sr=read.rgb(inputs,shuffled,0);region=torch.zeros_like(rgb[:,:1],dtype=torch.bool);region[...,32:160,32:160]=True
            for family,value in [('raw',rgb.clamp(0,1)),('DC',read.far.apply(rgb,far))]:
                for target_name,target in [('observed24_nativeD',teacherRGB),('HR_evaluation_only',HR)]:
                    if family=='DC' and target_name=='observed24_nativeD':target=read.far.apply(target,far)
                    err=(value-target).square();rows.append(dict(arm=arm,step=step,family=family,target=target_name,full_RGB_L1=float((value-target).abs().mean()),full_RGB_PSNR=float(-10*torch.log10(err.mean())),single_near_PSNR=float(-10*torch.log10(err[region.expand_as(err)].mean())),latent_teacher_L1=float((z-teacher_delta).abs().mean()),latent_teacher_outside_readout_bound_fraction=float((teacher_delta.abs()>2).float().mean()),feature_RMS=float(h.square().mean().sqrt()),detail_code_RMS=float(z.square().mean().sqrt()),no_measured_delta_RGB_RMSE=float((rgb-zr).square().mean().sqrt()),shuffle_delta_RGB_RMSE=float((rgb-sr).square().mean().sqrt()),current_far_RMSE=float((read.far.observe(value)-far).square().mean().sqrt())))
            for name,value in [('baseline_current_LRVAE',inputs['query_base_rgb'][:,0]),('teacher_observed24_nativeD',teacherRGB)]:
                value=read.far.apply(value,far);err=(value-HR).square();rows.append(dict(arm=arm,step=step,family='DC',target=name,full_RGB_L1=float((value-HR).abs().mean()),full_RGB_PSNR=float(-10*torch.log10(err.mean())),single_near_PSNR=float(-10*torch.log10(err[region.expand_as(err)].mean())),latent_teacher_L1=None,latent_teacher_outside_readout_bound_fraction=None,feature_RMS=None,detail_code_RMS=None,no_measured_delta_RGB_RMSE=None,shuffle_delta_RGB_RMSE=None,current_far_RMSE=float((read.far.observe(value)-far).square().mean().sqrt())))
        with (out/'fixed_readouts.csv').open('w') as f:wri=csv.DictWriter(f,fieldnames=list(rows[0]));wri.writeheader();wri.writerows(rows)
    for arm in ['latent_L1_only','actualD_RGB_only']:
        torch.manual_seed(17);model=build('closed_action').cuda();model.load_state_dict(checkpoint['model']);model.core.requires_grad_(False);model.appearance.requires_grad_(False);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr);directory=out/arm;directory.mkdir(exist_ok=True);evaluate(model,arm,0);torch.save(dict(model=model.state_dict(),step=0,arm=arm),directory/'step0.pt')
        for step in range(1,301):
            opt.zero_grad(set_to_none=True);z,h=forward(model)
            if arm=='latent_L1_only':loss=F.l1_loss(z,teacher_delta);loss.backward()
            else:
                with differentiable_decoder_math():rgb=read.rgb(inputs,z,0);loss=F.l1_loss(rgb,teacherRGB);loss.backward()
            if step==1:
                gradients={name:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(name) and p.grad is not None) for name in ['encoder','decoder','core','appearance']};assert gradients['encoder']>0 and gradients['decoder']>0 and gradients['core']==0 and gradients['appearance']==0;assert all(p.grad is None for p in vae.parameters());audits.append(dict(arm=arm,step=step,gradients=gradients,frozen_VAE_weight_gradients=0));(out/'gradient_audits.json').write_text(json.dumps(audits,indent=2))
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step%50==0:
                history.append(dict(arm=arm,step=step,train_loss=float(loss.detach()),seconds=time.time()-start));live=dict(status='fixed_single_train_source_objective_diagnosis',command=' '.join(sys.argv),pid=os.getpid(),arm=arm,step=step,steps=300,foreground=True,unix_time=time.time(),seconds=time.time()-start);(out/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(live),flush=True)
            if step in [50,150,300]:evaluate(model,arm,step);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,arm=arm),directory/f'step{step}.pt')
        del model,opt;torch.cuda.empty_cache()
    meta=dict(**vars(args),seconds=time.time()-start,source=prov['id'],source_sha256=prov['source_sha256'],packet=0,query_step=3,checkpoint_initial_step=checkpoint['step'],checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),steps_per_arm=300,arms=['latent_L1_only','actualD_RGB_only'],actual_D_backward_calls=300,readouts=[0,50,150,300],selection='none, fixed endpoint and fixed intermediate readouts',frozen='native VAE/core/appearance; only existing4ch encoder+decoder optimized; same true simulated nuisance q supplied to both diagnostic arms',teacher='actual measured-near prelift full24 nativecode and its nativeD RGB; supported current-anchor readout. No HR teaching in either arm. HR only metric.',scope='One fixed TRAIN source/packet and sameappearance; capacity/objective/optimization diagnostic, not evidence of generalization or state update performance. Reconstructing one input can memorize the anchor; delta ablations reported.',training_log=history,peak_GPU_bytes=torch.cuda.max_memory_allocated(),test_accessed=False);(out/'complete.json').write_text(json.dumps(meta,indent=2));(out/'LIVE_STATE.json').write_text(json.dumps(dict(status='fixed_two_arm300_complete',seconds=meta['seconds'],foreground=True,unix_time=time.time()),indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
