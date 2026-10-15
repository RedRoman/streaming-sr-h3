"""Actual train-cache B2 zero-anchor/support/gradient correctness checks."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from model import KINDS
from residual_model import ResidualMemory,residual_losses
from residual_readout import ResidualReadout
from native_vae import load_vae,differentiable_decoder_math
from train import batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(813);t=time.time()
    item=torch.load(Path(args.cache)/'train_100075.pt',weights_only=False);inputs,targets=batch([item],'cuda');vae=load_vae(args.vae,args.comfy);reader=ResidualReadout(vae);results=[]
    for kind in KINDS:
        model=ResidualMemory(kind).cuda();opt=torch.optim.Adam(model.parameters(),lr=.002);before={n:p.detach().clone() for n,p in model.named_parameters()};grads=[]
        for step in range(2):
            opt.zero_grad();loss,_,_,_,states=residual_losses(model,inputs,targets);loss.backward();grads.append({k:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(k) and p.grad is not None) for k in ['appearance','chart','update_head']});opt.step()
        changes={k:sum(float((p.detach()-before[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(k)) for k in ['appearance','chart','update_head']};assert all(x>0 for x in changes.values()) and all(x>0 for x in grads[-1].values()),(kind,grads,changes)
        with torch.no_grad():
            r,a,states=model.rollout(inputs);assert not bool(states[0].any());assert torch.equal(states[2],states[3]) and torch.equal(states[2],states[4]);assert torch.equal(states[5],states[6]) and torch.equal(states[6],states[7])
            z=torch.zeros_like(inputs['observation_latents'][:,0]);assert not bool(model.read_detail(z,torch.randn(1,12,device='cuda')).any())
            zeroerr={str(stage):float((reader.rgb(model,inputs,z,a[:,stage],stage)-inputs['base_decoded_rgb'][:,stage]).abs().max()) for stage in [2,4]};assert max(zeroerr.values())<2e-6,(kind,zeroerr)
            near=reader.rgb(model,inputs,r[:,2],a[:,2],2);nearerr=float((near-inputs['base_decoded_rgb'][:,2]).abs().max());assert nearerr<2e-6,nearerr
            far=reader.rgb(model,inputs,r[:,4],a[:,4],4);anchorrms=float(reader.far.observe(far-inputs['base_decoded_rgb'][:,4]).square().mean().sqrt());assert anchorrms<2e-6,anchorrms
        results.append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),gradient_l1=grads,parameter_update_l1=changes,zero_residual_native_LR_VAE_maxabs=zeroerr,near_read_native_LR_VAE_maxabs=nearerr,far_anchor_measurement_difference_rms=anchorrms,zero_initial_detail=True,far_cannot_overwrite_near=True,query_read_only=True,duplicate_idempotent=True))
    # Actual native D derivative under the new readout: same far operator is
    # blind to its detail correction, whereas old near evidence can see it.
    model=ResidualMemory('factorized_swap').cuda();detail=(torch.randn_like(inputs['observation_latents'][:,4])*.1).requires_grad_(True);a=torch.zeros(1,12,device='cuda')
    with differentiable_decoder_math():
        rgb=reader.rgb(model,inputs,detail,a,4);far=reader.far.observe(rgb);near=reader.near.observe(rgb[:,:,64:192,64:192]);far_scalar=(far*torch.randn_like(far)).mean();near_scalar=(near*torch.randn_like(near)).mean();gf=torch.autograd.grad(far_scalar,detail,retain_graph=True)[0];gn=torch.autograd.grad(near_scalar,detail)[0]
    fargrad=float(gf.square().mean().sqrt());neargrad=float(gn.square().mean().sqrt());assert neargrad>0 and fargrad<neargrad*.001,(fargrad,neargrad)
    result=dict(passed=True,source='actual train_100075 cache; no test/val for checks',initial_proposal='zero',results=results,true_D_same_far_detail_gradient_rms=fargrad,true_D_historical_near_detail_gradient_rms=neargrad,seconds=time.time()-t)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
