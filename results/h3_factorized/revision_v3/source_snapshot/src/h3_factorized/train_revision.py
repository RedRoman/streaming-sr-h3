"""Fixed B3R latent600 / actual frozen native-D RGB200 training, source-disjoint."""
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from revision_model import build,losses,KINDS
from revision_utils import batch,live,Readout,FP
from native_vae import load_vae,differentiable_decoder_math,CANONICAL_SHA
from cache_pairs import photo,degrade
from cache_revision import inverse_photo


def past_observation_loss(rgb,a,details,inputs,targets,qi):
    step=int(inputs['query_steps'][0,qi]);canonical=torch.stack([inverse_photo(rgb[i],a[i]) for i in range(len(rgb))]);weighted=rgb.new_zeros(());mass=rgb.new_zeros(())
    for j,fpi in enumerate(inputs['packet_fp'][0].tolist()):
        w=inputs['weights'][:,step,j]
        if not bool(w.any()):continue
        g=int(inputs['packet_groups'][0,j]);q=details['group_appearance'][:,step,g];styled=torch.stack([photo(canonical[i],q[i]) for i in range(len(rgb))]);top,left=FP[fpi];observed=degrade(styled[...,top:top+128,left:left+128],2);weight=F.interpolate(w,size=(256,256),mode='bilinear',align_corners=False)[...,top:top+128,left:left+128];weight=F.interpolate(weight,size=(64,64),mode='area');weighted=weighted+((observed-targets['observed_near_LR'][:,j]).abs()*weight).sum();mass=mass+weight.sum()*3
    return weighted/mass.clamp_min(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--stage',choices=['latent','rgb'],required=True);p.add_argument('--initial');p.add_argument('--initial-checkpoint',choices=['best','last'],default='best');p.add_argument('--vae');p.add_argument('--comfy');p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float);p.add_argument('--validate-every',type=int);p.add_argument('--kinds',nargs='+',default=KINDS);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.steps=args.steps or (600 if args.stage=='latent' else 200);args.lr=args.lr or (.002 if args.stage=='latent' else .0005);args.validate_every=args.validate_every or (100 if args.stage=='latent' else 50);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Path(args.cache);manifest=json.loads((cache/'manifest.json').read_text());assert len(manifest['items'])==40
    data={s:[torch.load(cache/(x['id']+'.pt'),map_location='cpu',weights_only=False) for x in manifest['items'] if x['split']==s] for s in ['train','val']};assert len(data['train'])==32 and len(data['val'])==8;start=time.time();reader=None;vae=None
    if args.stage=='rgb':
        assert args.initial and args.vae and args.comfy;vae=load_vae(args.vae,args.comfy);reader=Readout(vae)
    config=dict(**vars(args),capture_group_provenance=manifest['capture_group_provenance'],source_split='32officialtrain/8officialval,oldBSDS16testnotaccessed_by_B3R;BSDSsourceswereusedelsewhereearlier_notproject-freshheldout',train_sources=[x['provenance']['id'] for x in data['train']],val_sources=[x['provenance']['id'] for x in data['val']],source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('revision*.py')},loss='L1 supervised past-support native detail +10physical appearance MSE +.5 feature equivariance(no_equiv omits); RGB stage adds5*(trueD RGB L1+.1 confidence-weighted actual past-LR loss) every4steps',selection='shared latent reconstruction+10appearance for stage1; actualD seven-far-query RGB L1 forstage2',DiT_proposal='explicitzero',actual_native_D=bool(reader),VAE_sha256=CANONICAL_SHA if reader else None,TF32=False)
    (out/'config.json').write_text(json.dumps(config,indent=2));rows=list(csv.DictReader((out/'training.csv').open())) if (out/'training.csv').exists() else [];schedule=np.random.default_rng(args.seed+(10000 if reader else 0)).integers(0,32,size=(args.steps,args.batch))
    for kind in args.kinds:
        directory=out/f'{kind}_seed{args.seed}';directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);model=build(kind).cuda();opt=torch.optim.Adam(model.parameters(),lr=args.lr);best=float('inf');first=1;initial=None
        if reader:
            initial=Path(args.initial)/f'{kind}_seed{args.seed}'/(args.initial_checkpoint+'.pt');initial_data=torch.load(initial,map_location='cpu',weights_only=False)
            if args.initial_checkpoint=='last':assert initial_data['step']==600
            model.load_state_dict(initial_data['model'])
        if (directory/'last.pt').exists():
            ckpt=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer']);first=ckpt['step']+1;best=ckpt['best_score']
        t0=time.time();parameter_count=sum(p.numel() for p in model.parameters())
        for step in range(first,args.steps+1):
            chosen=[data['train'][i] for i in schedule[step-1]];inputs,targets=batch(chosen,'cuda',rgb=False);model.train();opt.zero_grad(set_to_none=True);loss,parts,pred,a,states,details=losses(model,inputs,targets);rgb_loss=loss.new_zeros(());past_loss=loss.new_zeros(())
            if reader and step%4==0:
                qi=(step//4-1)%7;one={k:v[:1] for k,v in inputs.items()};one['query_base_rgb']=chosen[0]['inputs']['query_base_rgb'][None].cuda().float();one_targets={k:v[:1] for k,v in targets.items()};one_targets['rgb']=chosen[0]['supervision']['rgb'][None].cuda().float();one_details={**details,'group_appearance':details['group_appearance'][:1]}
                with differentiable_decoder_math():
                    rgb=reader.rgb(one,pred[:1,qi],qi);rgb_loss=F.l1_loss(rgb,one_targets['rgb'][:,qi]);past_loss=past_observation_loss(rgb,a[:1,qi],one_details,one,one_targets,qi);total=loss+5*(rgb_loss+.1*past_loss);total.backward()
            else:total=loss;total.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite state gradient'
            if step in [1,2]:
                audit={branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']};(directory/f'gradient_step{step}.json').write_text(json.dumps(audit,indent=2));assert all(x>0 for x in audit.values())
            if vae:assert all(p.grad is None for p in vae.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step==1 or step%args.validate_every==0 or step==args.steps:
                model.eval();vals=[];features=[];energy=[]
                with torch.no_grad():
                    for item in data['val']:
                        vi,vt=batch([item],'cuda',rgb=bool(reader));_,vp,vz,va,vs,vd=losses(model,vi,vt);features.append(float(vd['packet_features'].square().mean().sqrt()));energy.append(float(vz.square().mean().sqrt()))
                        if reader:vals.extend(float(F.l1_loss(reader.rgb(vi,vz[:,qi],qi),vt['rgb'][:,qi])) for qi in range(7))
                        else:vals.append(float(vp['reconstruction']+10*vp['appearance']))
                score=float(np.mean(vals));best=min(best,score);payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=args.seed,step=step,best_score=best,validation_score=score,parameters=parameter_count,config=config,initial_checkpoint=str(initial) if initial else None)
                if score<=best:torch.save(payload,directory/'best.pt')
                torch.save(payload,directory/'last.pt');row=dict(kind=kind,seed=args.seed,stage=args.stage,step=step,seconds=time.time()-t0,train_total=float(total.detach()),train_reconstruction=float(parts['reconstruction'].detach()),train_appearance=float(parts['appearance'].detach()),train_equivariance=float(parts['equivariance'].detach()),train_RGB_L1=float(rgb_loss.detach()),train_past_LR_L1=float(past_loss.detach()),validation_score=score,val_factor_feature_RMS=float(np.mean(features)),val_detail_code_RMS=float(np.mean(energy)),parameters=parameter_count);rows.append(row)
                with (out/'training.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                live(out,status='B3R_fixed_'+args.stage+'_training',kind=kind,seed=args.seed,step=step,steps=args.steps,seconds=time.time()-start);print(json.dumps(row),flush=True)
        (directory/'complete.json').write_text(json.dumps(dict(kind=kind,seed=args.seed,steps=args.steps,parameters=parameter_count,best_score=best,seconds=time.time()-t0),indent=2));del model,opt;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(stage=args.stage,seed=args.seed,kinds=args.kinds,seconds=time.time()-start),indent=2));live(out,status='B3R_'+args.stage+'_training_complete',step=args.steps,steps=args.steps,seconds=time.time()-start)


if __name__=='__main__':main()
