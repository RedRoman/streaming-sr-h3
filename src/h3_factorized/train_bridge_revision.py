"""Locked4x672 actual-D factorial, no latent main loss or val selection."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from revision_bridge import build_bridge,BRIDGE_ARMS,attach_native,bridge_rgb
from revision_dense import auxiliary
from revision_utils import batch,Readout
from train_revision import past_observation_loss
from native_vae import load_vae,differentiable_decoder_math,CANONICAL_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--phi',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--seed',type=int,default=17);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']=='train'];assert len(sources)==32;data=[];phis=[]
    for source in sources:
        data.append(torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False));phis.append(torch.load(Path(a.phi)/(source['id']+'.pt'),map_location='cpu',weights_only=False))
    rng=np.random.default_rng(a.seed+20261017);pairs=[(s,q) for s in range(32) for q in range(7)];schedule=[pairs[i] for _ in range(3) for i in rng.permutation(224)];initial=torch.load(a.initial,map_location='cpu',weights_only=False);assert initial['step']==600;vae=load_vae(a.vae,a.comfy);read=Readout(vae);start=time.time();config=dict(**vars(a),arms=BRIDGE_ARMS,steps_per_arm=672,batch=1,lr=.0005,clip=5,selection='fixedlast672; no VAL data loaded',schedule=schedule,loss='5*(eachstepactualD_RGB_L1+.1past_actualLR_L1)+10appearanceMSE+.5equivarianceL1; no latent L1',VAE_sha256=CANONICAL_SHA,TF32=False,initial_sha256=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),source_ids=[s['id'] for s in sources],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('revision_bridge.py')]},physical_base='currentD or common fixed H3decodedatlas; external causal LR controller, detached; all arms retain/use full/proxy codes',RGB_graph='B_s+H_common[D(zcur+delta)-D(zcur)]; pixeladdition, no reencode; final DC evaluation only');(out/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(dict(status='B5_train_started',pid=os.getpid(),arms=BRIDGE_ARMS)),flush=True)
    for arm in BRIDGE_ARMS:
        directory=out/f'{arm}_seed{a.seed}';directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        model=build_bridge(arm,initial,a.seed).cuda();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.0005);first=1;t0=time.time();active=sum(p.numel() for p in model.parameters() if p.requires_grad);parameters=sum(p.numel() for p in model.parameters())
        def save(step,path):torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=arm,seed=a.seed,step=step,parameters=parameters,active_parameters=active,config=config),path)
        if (directory/'last.pt').exists():
            cp=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);first=cp['step']+1
        else:save(0,directory/'step0.pt')
        for step in range(first,673):
            si,qi=schedule[step-1];inputs,target=batch([data[si]],'cuda');attach_native(inputs,data[si]);base=phis[si]['bridge'][None].cuda();model.train();opt.zero_grad(set_to_none=True);appearance,equiv,pred,q,states,details=auxiliary(model,inputs,target)
            with differentiable_decoder_math():
                rgb=bridge_rgb(read,inputs,pred[:,qi],qi,base,model.bridge_skip);rgb_loss=F.l1_loss(rgb,target['rgb'][:,qi]);past=past_observation_loss(rgb,q[:,qi],details,inputs,target,qi);total=5*(rgb_loss+.1*past)+10*appearance+.5*equiv;total.backward()
            assert all(p.grad is None for p in vae.parameters());assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step==1 or step%56==0:
                row=dict(arm=arm,step=step,steps=672,source=sources[si]['id'],query_step=int(inputs['query_steps'][0,qi]),total=float(total.detach()),RGB_L1=float(rgb_loss.detach()),past_LR_L1=float(past.detach()),appearance_MSE=float(appearance.detach()),equivariance_L1=float(equiv.detach()),packet_h_RMS=float(details['packet_features'].detach().square().mean().sqrt()),delta_RMS=float(pred[:,qi].detach().square().mean().sqrt()),active_parameters=active,seconds=time.time()-t0)
                with (out/'training.csv').open('a') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader() if f.tell()==0 else None;w.writerow(row)
                live=dict(status='B5_fixed4x672_actualD',command=' '.join(sys.argv),pid=os.getpid(),arm=arm,step=step,steps=672,actualD_backward_this_arm=step,foreground=True,seconds=time.time()-start,unix_time=time.time());(out.parent/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(row),flush=True)
            if step%56==0:
                save(step,directory/'last.pt')
                if step in [224,448,672]:save(step,directory/f'step{step}.pt')
        (directory/'complete.json').write_text(json.dumps(dict(arm=arm,steps=672,actual_D_backwards=672,active_parameters=active,parameters=parameters,seconds=time.time()-t0),indent=2));del model,opt;torch.cuda.empty_cache()
    (out/'complete.json').write_text(json.dumps(dict(arms=BRIDGE_ARMS,steps_per_arm=672,actual_D_backwards=2688,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated()),indent=2));(out.parent/'LIVE_STATE.json').write_text(json.dumps(dict(status='B5_all4x672_complete',foreground=True,unix_time=time.time(),seconds=time.time()-start),indent=2))


if __name__=='__main__':main()
