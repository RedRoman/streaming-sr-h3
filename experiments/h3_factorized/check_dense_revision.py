"""Actual native-D seven-control checks before starting dense training."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_dense import build_dense,DENSE_KINDS,freeze_cancelling_biases,auxiliary
from revision_utils import batch,Readout
from native_vae import load_vae,differentiable_decoder_math
from train_revision import past_observation_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--query-labels',required=True);p.add_argument('--initial',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;manifest=json.loads((Path(a.cache)/'manifest.json').read_text());prov=next(x for x in manifest['items'] if x['split']=='train');item=torch.load(Path(a.cache)/(prov['id']+'.pt'),map_location='cpu',weights_only=False);item['supervision']['query_neutral_latents']=torch.load(Path(a.query_labels)/(prov['id']+'.pt'),weights_only=False)['query_neutral_latents'];inputs,target=batch([item],'cuda');vae=load_vae(a.vae,a.comfy);read=Readout(vae);rows=[];start=time.time()
    for kind in DENSE_KINDS:
        initial=Path(a.initial)/f'{"closed_action" if kind=="anchor_only" else kind}_seed17'/'last.pt';cp=torch.load(initial,map_location='cpu',weights_only=False);assert cp['step']==600;model=build_dense(kind).cuda();model.load_state_dict(cp['model']);model.zero_grad(set_to_none=True);app,equiv,pred,q,states,details=auxiliary(model,inputs,target)
        with differentiable_decoder_math():
            rgb=read.rgb(inputs,pred[:,0],0);rgb_loss=F.l1_loss(rgb,target['rgb'][:,0]);past=past_observation_loss(rgb,q[:,0],details,inputs,target,0);loss=5*(rgb_loss+.1*past)+10*app+.5*equiv;loss.backward()
        grads={branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']};assert all(x>0 for x in grads.values()),(kind,grads);assert all(p.grad is None for p in vae.parameters());assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        with torch.no_grad():
            poison={k:v.clone() for k,v in inputs.items()};poison['packet_delta'][:,2:]=123;poison['packet_proxy'][:,2:]=-91;pz,_,_=model.rollout(poison);future=float((pz[:,0]-pred[:,0]).abs().max());assert future==0
            empty={k:v.clone() for k,v in inputs.items()};empty['weights'].zero_();empty['packet_steps'].fill_(99);ez,_,_=model.rollout(empty);zero=float(ez.abs().max());assert zero==0
            zeroRGB=float((read.rgb(empty,ez[:,0],0)-inputs['query_base_rgb'][:,0]).abs().max());assert zeroRGB==0
            anchor_poison=None
            if kind=='anchor_only':
                poison['packet_delta'].fill_(123);poison['packet_proxy'].fill_(-91);poison['group_features'].fill_(77);poison['coarse_descriptors'].fill_(91);pz,_,_=model.rollout(poison);anchor_poison=float((pz-pred).abs().max());assert anchor_poison==0
        rows.append(dict(kind=kind,initial=str(initial),initial_step=cp['step'],architecture_parameters=sum(p.numel() for p in model.parameters()),active_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_cancelling_biases=freeze_cancelling_biases(model),actualD_RGB_L1=float(rgb_loss.detach()),gradients=grads,frozen_VAE_grads=0,future_packet_poison_maxabs=future,zero_evidence_delta_maxabs=zero,zero_evidence_actualD_RGB_maxabs=zeroRGB,anchor_all_history_poison_maxabs=anchor_poison));print(json.dumps(rows[-1]),flush=True);del model;torch.cuda.empty_cache()
    counts=[r['active_parameters'] for r in rows];assert max(counts)/min(counts)<1.015;Path(a.out).write_text(json.dumps(dict(passed=True,rows=rows,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),scope='ActualTRAINcache and nativeFP16D, all7controls. No update or validation selection.'),indent=2))


if __name__=='__main__':main()
