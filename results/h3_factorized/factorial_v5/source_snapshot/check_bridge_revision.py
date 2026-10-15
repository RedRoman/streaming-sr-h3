"""B5 independent exact-base, initialization, causality and actualD audits."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_bridge import build_bridge,BRIDGE_ARMS,attach_native,shared_bridge,bridge_rgb
from revision_utils import batch,Readout
from revision_classical import ClassicalRevision
from native_vae import load_vae,decode_last,differentiable_decoder_math


def grads(model):return {branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--phi',required=True);p.add_argument('--initial',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());source=next(s for s in manifest['items'] if s['split']=='train');item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');attach_native(inputs,item);phi=torch.load(Path(a.phi)/(source['id']+'.pt'),map_location='cpu',weights_only=False);base=phi['bridge'][None].cuda();decoded=[z[None].cuda() for z in phi['decoded']];near=item['classical_inputs']['near_LR'].cuda().float();far=item['classical_inputs']['query_LR'].cuda().float();vae=load_vae(a.vae,a.comfy);read=Readout(vae);classic=ClassicalRevision(read);initial=torch.load(a.initial,map_location='cpu',weights_only=False);assert initial['step']==600;start=time.time();rows=[];qi=4
    with torch.no_grad():
        direct=shared_bridge(read,inputs,near,decoded);independent=torch.stack([classic.decoded_atlas(inputs['query_base_rgb'][:,j],decoded,classic.packets(inputs,near,j)) for j in range(7)],1);phi_parity=float((direct-base).abs().max());assert torch.equal(direct,base) and torch.equal(direct,independent)
    for arm in BRIDGE_ARMS:
        model=build_bridge(arm,initial).cuda();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.0005);count=sum(p.numel() for p in model.parameters());active=sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            z=model.rollout(inputs)[0];assert torch.equal(z,torch.zeros_like(z));zero_raw=bridge_rgb(read,inputs,z[:,qi],qi,base,model.bridge_skip);specified=base[:,qi] if model.bridge_skip else inputs['query_base_rgb'][:,qi];rawparity=float((zero_raw-specified).abs().max());dcparity=float((read.far.apply(zero_raw,far[qi:qi+1])-read.far.apply(specified,far[qi:qi+1])).abs().max());assert rawparity==dcparity==0
        audits=[]
        for step in [1,2]:
            model.zero_grad(set_to_none=True);z=model.rollout(inputs)[0]
            with differentiable_decoder_math():rgb=bridge_rgb(read,inputs,z[:,qi],qi,base,model.bridge_skip);loss=F.l1_loss(rgb,targets['rgb'][:,qi]);loss.backward()
            g=grads(model);assert g['decoder']>0 and all(p.grad is None for p in vae.parameters());assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step==1:assert g['encoder']==g['core']==0
            else:assert g['encoder']>0 and g['core']>0
            audits.append(dict(step=step,loss=float(loss),gradients=g,VAE_weight_gradients=0));torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        with torch.no_grad():
            z=model.rollout(inputs)[0];vi={k:v.clone() for k,v in inputs.items()};vi['query_latents']=vi['query_latents'].flip(-1)*-1;otherz=model.rollout(vi)[0];anchorchange=float((z-otherz).abs().max())
            if not model.anchor_condition:assert anchorchange==0
            future={k:v.clone() for k,v in inputs.items()};future['packet_native'][:,2:]=101;future['packet_delta'][:,2:]=101;future['packet_proxy'][:,2:]=-81;pz=model.rollout(future)[0];futurechange=float((z[:,0]-pz[:,0]).abs().max());assert futurechange==0
            baddecoded=[v.clone() for v in decoded]
            for j in range(2,8):baddecoded[j].fill_(101)
            futurebase=shared_bridge(read,inputs,near,baddecoded);futurebasechange=float((futurebase[:,0]-base[:,0]).abs().max());assert futurebasechange==0
            empty={k:v.clone() for k,v in inputs.items()};empty['weights'].zero_();ez=model.rollout(empty)[0];eb=shared_bridge(read,empty,near,decoded);er=bridge_rgb(read,empty,ez[:,qi],qi,eb,model.bridge_skip);emptychange=float((er-inputs['query_base_rgb'][:,qi]).abs().max());assert emptychange==0
        rows.append(dict(arm=arm,architecture_parameters=count,active_parameters=active,decoder_width=model.decoder.net[0].out_channels,zero_correction_raw_pixel_maxabs=rawparity,zero_correction_common_single_DC_maxabs=dcparity,actualD_RGB_only_steps=audits,neural_anchor_poison_maxabs=anchorchange,future_neural_firstquery_maxabs=futurechange,future_bridge_firstquery_maxabs=futurebasechange,no_evidence_vs_current_pixel_maxabs=emptychange));del model,opt;torch.cuda.empty_cache()
    counts=[r['active_parameters'] for r in rows];assert (max(counts)-min(counts))/min(counts)<.015
    (out/'acceptance.json').write_text(json.dumps(dict(arms=rows,common_cached_phi_route_pixel_maxabs=phi_parity,source=source['id'],actual_D_backward_calls=8,diagnostic_optimizer_updates=8,main_training_updates=0,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),TF32=False,scope='Independent pretraining checks, models discarded; raw base and finalDC zero parity both exact; q11 mixed groups, first zerohead coregrad expected0 then nonzero after update'),indent=2));print(json.dumps(dict(status='B5_all_acceptance_passed',seconds=time.time()-start,arms=BRIDGE_ARMS)),flush=True)


if __name__=='__main__':main()
