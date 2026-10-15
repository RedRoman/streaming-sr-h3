"""Locked7x672 B3R training: balanced source/query, every-step nativeD, no latent loss."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from revision_dense import build_dense,DENSE_KINDS,auxiliary
from revision_utils import batch,Readout
from native_vae import load_vae,differentiable_decoder_math,CANONICAL_SHA
from train_revision import past_observation_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--query-labels',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=672);p.add_argument('--lr',type=float,default=.0005);a=p.parse_args();assert a.steps==672 and a.lr==.0005;torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']=='train'];assert len(sources)==32;data=[]
    for prov in sources:
        item=torch.load(cache/(prov['id']+'.pt'),map_location='cpu',weights_only=False);item['supervision']['query_neutral_latents']=torch.load(Path(a.query_labels)/(prov['id']+'.pt'),map_location='cpu',weights_only=False)['query_neutral_latents'];data.append(item)
    pairs=[(s,q) for s in range(32) for q in range(7)];rng=np.random.default_rng(a.seed+20261017);schedule=[pairs[i] for _ in range(3) for i in rng.permutation(224)];assert len(schedule)==672;vae=load_vae(a.vae,a.comfy);read=Readout(vae);start=time.time();rows=[]
    config=dict(**vars(a),kinds=DENSE_KINDS,selection='fixedlast672; no val load/selection during training',schedule=schedule,loss='5*(eachstep nativeD RGB L1 +.1confidence-weighted past actualLR L1)+10appearance+.5equivariance(noeq=0); no latent L1 term',batch=1,actualD_backward_per_model=672,source_ids=[x['id'] for x in sources],runtime_no_qGT=True,anchor='samefull-stage1last600 initialization, currentquery highpass/lowpass encoder input; no historical images/codes; shared postdecoder LRcontroller explicit',source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*revision*.py')},VAE_sha256=CANONICAL_SHA,TF32=False);(out/'config.json').write_text(json.dumps(config,indent=2))
    for kind in DENSE_KINDS:
        directory=out/f'{kind}_seed{a.seed}';directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        torch.manual_seed(a.seed);model=build_dense(kind).cuda();initial=Path(a.initial)/f'{"closed_action" if kind=="anchor_only" else kind}_seed{a.seed}'/'last.pt';cp=torch.load(initial,map_location='cpu',weights_only=False);assert cp['step']==600;model.load_state_dict(cp['model']);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.lr);first=1
        if (directory/'last.pt').exists():
            saved=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);first=saved['step']+1
        t0=time.time();active=sum(p.numel() for p in model.parameters() if p.requires_grad)
        for step in range(first,673):
            s,qi=schedule[step-1];inputs,target=batch([data[s]],'cuda');model.train();opt.zero_grad(set_to_none=True);appearance,equiv,pred,q,states,details=auxiliary(model,inputs,target)
            with differentiable_decoder_math():
                rgb=read.rgb(inputs,pred[:,qi],qi);rgb_loss=F.l1_loss(rgb,target['rgb'][:,qi]);past=past_observation_loss(rgb,q[:,qi],details,inputs,target,qi);total=5*(rgb_loss+.1*past)+10*appearance+.5*equiv;total.backward()
            assert all(p.grad is None for p in vae.parameters());assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step==1 or step%56==0 or step==672:
                row=dict(kind=kind,seed=a.seed,step=step,steps=672,source=sources[s]['id'],query=int(inputs['query_steps'][0,qi]),total=float(total.detach()),RGB_L1=float(rgb_loss.detach()),past_LR_L1=float(past.detach()),appearance_MSE=float(appearance.detach()),equivariance_L1=float(equiv.detach()),factor_feature_RMS=float(details['packet_features'].detach().square().mean().sqrt()),detail_code_RMS=float(pred[:,qi].detach().square().mean().sqrt()),active_parameters=active,seconds=time.time()-t0);rows.append(row)
                with (out/'training.csv').open('a') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader() if f.tell()==0 else None;w.writerow(row)
                live=dict(status='B3R_dense_actualD7x672',command=' '.join(sys.argv),pid=os.getpid(),kind=kind,step=step,steps=672,actualD_backward_this_model=step,seconds=time.time()-start,foreground=True,unix_time=time.time());(out.parent/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(row),flush=True)
            if step%56==0 or step==672:
                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),kind=kind,seed=a.seed,step=step,parameters=sum(p.numel() for p in model.parameters()),active_parameters=active,initial_checkpoint=str(initial),config=config);torch.save(payload,directory/'last.pt')
                if step in [224,448,672]:torch.save(payload,directory/f'step{step}.pt')
        (directory/'complete.json').write_text(json.dumps(dict(kind=kind,steps=672,actualD_backward=672,active_parameters=active,seconds=time.time()-t0),indent=2));del model,opt;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(steps_per_model=672,kinds=DENSE_KINDS,actualD_backward=7*672,seconds=time.time()-start),indent=2));(out.parent/'LIVE_STATE.json').write_text(json.dumps(dict(status='B3R_dense_all7_training_complete',seconds=time.time()-start,foreground=True,unix_time=time.time()),indent=2))


if __name__=='__main__':main()
