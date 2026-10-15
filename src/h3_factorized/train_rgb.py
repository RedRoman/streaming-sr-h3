"""Stage2: equal-budget fine tuning through the actual frozen native H3 D."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactorizedMemory,KINDS
from train import batch,losses,write_live,TARGET
from native_vae import load_vae,decode_last,differentiable_decoder_math,CANONICAL_SHA

STAGES=[1,2,4,5,6]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[17]);p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.0005);p.add_argument('--rgb-every',type=int,default=4);p.add_argument('--rgb-weight',type=float,default=5);p.add_argument('--validate-every',type=int,default=50);p.add_argument('--kinds',nargs='+',default=KINDS);p.add_argument('--residual-v2',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Path(args.cache);start=time.time()
    model_class=FactorizedMemory;loss_function=losses;readout=None
    if args.residual_v2:
        from residual_model import ResidualMemory,residual_losses
        from residual_readout import ResidualReadout
        model_class=ResidualMemory;loss_function=residual_losses;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    vae=load_vae(args.vae,args.comfy);manifest=json.loads((cache/'manifest.json').read_text());data={s:[torch.load(cache/(i['id']+'.pt'),map_location='cpu',weights_only=False) for i in manifest['items'] if i['split']==s] for s in ['train','val']}
    if args.residual_v2:readout=ResidualReadout(vae)
    config=dict(**vars(args),torch=torch.__version__,vae_sha256=CANONICAL_SHA,actual_decoder_dtype='float16',test_accessed=False,selection='minimum mean actual-D validation RGB L1 across five unique new-evidence stages',source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2));rows=list(csv.DictReader((out/'training.csv').open())) if (out/'training.csv').exists() else []
    for seed in args.seeds:
        schedule=np.random.default_rng(seed+10000).integers(0,len(data['train']),size=(args.steps,args.batch))
        for kind in args.kinds:
            directory=out/f'{kind}_seed{seed}';directory.mkdir(exist_ok=True)
            if (directory/'complete.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);initial=Path(args.initial)/f'{kind}_seed{seed}'/'best.pt';ckpt=torch.load(initial,map_location='cpu',weights_only=False);model=model_class(kind).cuda();model.load_state_dict(ckpt['model']);opt=torch.optim.Adam(model.parameters(),lr=args.lr);best=float('inf');t0=time.time()
            for step in range(1,args.steps+1):
                model.train();inputs,targets=batch([data['train'][i] for i in schedule[step-1]],'cuda');opt.zero_grad(set_to_none=True);loss,parts,pred,appearance,_=loss_function(model,inputs,targets);rgb_loss=loss.new_zeros(())
                with differentiable_decoder_math():
                    if step%args.rgb_every==0:
                        stage=STAGES[(step//args.rgb_every-1)%len(STAGES)]
                        one={k:(v if k=='footprint' else v[:1]) for k,v in inputs.items()}
                        decoded=readout.rgb(model,one,pred[:1,stage],appearance[:1,stage],stage) if readout else decode_last(vae,pred[:1,stage]);rgb_loss=F.l1_loss(decoded,targets['rgb'][:1,TARGET[stage]])
                    total=loss+args.rgb_weight*rgb_loss;total.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite actual-D training gradient'
                assert all(p.grad is None for p in vae.parameters()),'Frozen pretrained VAE got gradient'
                torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if step==1 or step%args.validate_every==0 or step==args.steps:
                    model.eval();validation=[]
                    with torch.no_grad():
                        for item in data['val']:
                            vi,vt=batch([item],'cuda');_,_,vz,va,_=loss_function(model,vi,vt)
                            validation.extend(float(F.l1_loss(readout.rgb(model,vi,vz[:,stage],va[:,stage],stage) if readout else decode_last(vae,vz[:,stage]),vt['rgb'][:,TARGET[stage]])) for stage in STAGES)
                    score=float(np.mean(validation));payload=dict(model=model.state_dict(),kind=kind,seed=seed,step=step,parameters=sum(p.numel() for p in model.parameters()),validation_rgb_l1=score,initial_checkpoint=str(initial),initial_step=ckpt['step'],config=config)
                    if score<best:best=score;torch.save(payload,directory/'best.pt')
                    torch.save(payload,directory/'last.pt');row=dict(seed=seed,kind=kind,step=step,seconds=time.time()-t0,train_latent=float(loss.detach()),train_rgb_l1=float(rgb_loss.detach()),val_rgb_l1=score);rows.append(row)
                    with (out/'training.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                    print(json.dumps(row),flush=True);write_live(out,status='actual_H3_RGB_gradient_training',kind=kind,seed=seed,step=step,steps=args.steps,seconds=time.time()-start)
            (directory/'complete.json').write_text(json.dumps(dict(kind=kind,seed=seed,best_val_rgb_l1=best,seconds=time.time()-t0),indent=2));del model,opt;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,seeds=args.seeds,kinds=args.kinds),indent=2));write_live(out,status='RGB_training_complete',seconds=time.time()-start)


if __name__=='__main__':main()
