"""Matched-budget B candidate/control training on actual frozen-H3 codes."""
import argparse,csv,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactorizedMemory,KINDS,counterfactual_loss

TARGET=[0,1,1,1,0,3,5,5]


def batch(items,device):
    inputs={k:(items[0]['inputs'][k].to(device).float() if k=='footprint' else torch.stack([x['inputs'][k] for x in items]).to(device).float()) for k in items[0]['inputs']}
    targets={k:torch.stack([x['supervision'][k] for x in items]).to(device).float() for k in items[0]['supervision']}
    return inputs,targets


def losses(model,inputs,targets):
    out,a,states=model.rollout(inputs)
    # Stage0 is fixed LR initialization; query and duplicate incur no extra
    # training weight. Each genuinely new packet contributes equally.
    stages=[1,2,4,5,6]
    reconstruction=F.l1_loss(out[:,stages],targets['latents'][:,[TARGET[s] for s in stages]])
    appearance=F.mse_loss(a[:,stages],targets['appearance'][:,[TARGET[s] for s in stages]])
    swap=counterfactual_loss(model,targets)
    total=reconstruction+10*appearance+.5*swap
    return total,dict(total=total,reconstruction=reconstruction,appearance=appearance,swap=swap),out,a,states


def write_live(out,**fields):
    p=out.parent/'LIVE_STATE.json'
    try:s=json.loads(p.read_text())
    except Exception:s={}
    if s.get('command')!=' '.join(sys.argv):
        # Keep completed phase details in config/session history rather than
        # allowing cache counts or old training steps to masquerade as live.
        s={k:s[k] for k in ['host','gpu'] if k in s}
    s.update(fields);s.update(pid=os.getpid(),unix_time=time.time(),command=' '.join(sys.argv));p.write_text(json.dumps(s,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[17]);p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.002);p.add_argument('--validate-every',type=int,default=100);p.add_argument('--kinds',nargs='+',default=KINDS);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Path(args.cache)
    manifest=json.loads((cache/'manifest.json').read_text());assert len(manifest['items'])==sum(manifest['config']['counts']),'Cache is incomplete'
    data={s:[torch.load(cache/(i['id']+'.pt'),map_location='cpu',weights_only=False) for i in manifest['items'] if i['split']==s] for s in ['train','val']}
    config=dict(**vars(args),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),train_sources=[x['provenance']['id'] for x in data['train']],val_sources=[x['provenance']['id'] for x in data['val']],test_accessed=False,training_loss='L1 rollout latent +10*MSE physical appearance +.5 counterfactual L1; no_swap omits counterfactual; equal source batches and optimizer steps',candidate_scope='Frozen actual H3 representation; no DiT inference yet',source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    rows=list(csv.DictReader((out/'training.csv').open())) if (out/'training.csv').exists() else []
    start=time.time()
    for seed in args.seeds:
        rng=np.random.default_rng(seed);schedule=rng.integers(0,len(data['train']),size=(args.steps,args.batch))
        for kind in args.kinds:
            directory=out/f'{kind}_seed{seed}';directory.mkdir(exist_ok=True)
            if (directory/'complete.json').exists():continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=FactorizedMemory(kind).cuda();opt=torch.optim.Adam(model.parameters(),lr=args.lr);best=float('inf');t0=time.time()
            for step in range(1,args.steps+1):
                model.train();inputs,targets=batch([data['train'][i] for i in schedule[step-1]],'cuda');opt.zero_grad(set_to_none=True)
                loss,parts,_,_,_=losses(model,inputs,targets);loss.backward()
                if step in [1,2]:
                    audit={branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['appearance','chart','update_head']}
                    (directory/f'real_batch_gradient_step{step}.json').write_text(json.dumps(audit,indent=2))
                torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if step==1 or step%args.validate_every==0 or step==args.steps:
                    model.eval();vals=[]
                    with torch.no_grad():
                        for item in data['val']:
                            vi,vt=batch([item],'cuda');_,vl,_,_,_=losses(model,vi,vt);vals.append({k:float(v) for k,v in vl.items()})
                    val={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
                    # Select by shared recurrence error; no_swap's omitted
                    # auxiliary loss must not alter checkpoint selection.
                    score=val['reconstruction']+10*val['appearance']
                    payload=dict(model=model.state_dict(),kind=kind,seed=seed,step=step,parameters=sum(p.numel() for p in model.parameters()),validation=val,shared_validation_score=score,config=config)
                    if score<best:best=score;torch.save(payload,directory/'best.pt')
                    torch.save(payload,directory/'last.pt')
                    row=dict(seed=seed,kind=kind,step=step,seconds=time.time()-t0,**{'train_'+k:float(v.detach()) for k,v in parts.items()},**{'val_'+k:v for k,v in val.items()});rows.append(row)
                    with (out/'training.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                    write_live(out,status='training_actual_H3_latent_memory',kind=kind,seed=seed,step=step,steps=args.steps,seconds=time.time()-start,foreground=True)
                    print(json.dumps(row),flush=True)
            (directory/'complete.json').write_text(json.dumps(dict(seed=seed,kind=kind,steps=args.steps,best_shared_val=best,seconds=time.time()-t0),indent=2))
            del model,opt;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,seeds=args.seeds,kinds=args.kinds),indent=2));write_live(out,status='training_complete_awaiting_actual_H3_RGB_eval',seconds=time.time()-start)


if __name__=='__main__':main()
