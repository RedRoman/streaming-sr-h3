import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_model import build,losses,KINDS
from revision_utils import batch,Readout
from native_vae import load_vae,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;start=time.time();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.cache)/'manifest.json').read_text());assert len(manifest['items'])==40;item=next(x for x in manifest['items'] if x['split']=='train');data=torch.load(Path(args.cache)/(item['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([data],'cuda');results=[];full=None
    for kind in KINDS:
        torch.manual_seed(17);model=build(kind).cuda();parameters=sum(p.numel() for p in model.parameters());opt=torch.optim.Adam(model.parameters(),lr=.002)
        with torch.no_grad():
            pred,a,states,details=model.rollout(inputs,True);assert torch.equal(states[8],states[9]);assert torch.equal(states[14],states[15]);zero_inputs={k:v.clone() for k,v in inputs.items()};zero_inputs['weights'].zero_();zero=model.rollout(zero_inputs)[0];assert float(zero.abs().max())==0
            poison={k:v.clone() for k,v in inputs.items()};poison['packet_delta'][:,2:]=torch.randn_like(poison['packet_delta'][:,2:])*100;poison['packet_proxy'][:,2:]=torch.randn_like(poison['packet_proxy'][:,2:])*100;future=model.rollout(poison)[0];future_error=float((future[:,0]-pred[:,0]).abs().max());assert future_error<1e-6
            algebra=None;withdraw=None
            if kind.startswith('closed'):
                def explicit(inp,detail,step):
                    h=detail['packet_features'];q=detail['group_appearance'][:,step];w=inp['weights'][:,step].unsqueeze(3);g=inp['packet_groups'][0];c=model.core.canonical(h.flatten(0,1),q[:,g].flatten(0,1)).reshape_as(h);den=w.sum(1);return (w*c).sum(1)/den.clamp_min(.001)*(den>0)
                algebra=float((explicit(inputs,details,13)-states[13]).abs().max());assert algebra<2e-5
                removed={k:v.clone() for k,v in inputs.items()};removed['weights'][:,13:,0]=0;_,_,rs,rd=model.rollout(removed,True);withdraw=float((explicit(removed,rd,17)-rs[17]).abs().max());assert withdraw<2e-5
        audits=[]
        for step in range(2):
            opt.zero_grad(set_to_none=True);before={name:p.detach().clone() for name,p in model.named_parameters()};loss,parts,*_=losses(model,inputs,targets);loss.backward();grad={branch:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']};assert all(x>0 for x in grad.values()),(kind,grad);opt.step();changed={branch:sum(float((p-before[name]).abs().sum()) for name,p in model.named_parameters() if name.startswith(branch)) for branch in grad};assert all(x>0 for x in changed.values());audits.append(dict(step=step+1,gradient_L1=grad,parameter_delta_L1=changed,loss=float(loss)))
        result=dict(kind=kind,parameters=parameters,duplicate_idempotent=True,query_readonly=True,zero_state_output_maxabs=0,future_packet_poison_query0_maxabs=future_error,group_rebase_vs_replay_maxabs=algebra,withdrawal_vs_replay_maxabs=withdraw,gradient_audits=audits);results.append(result)
        if kind=='closed_action':full=model
        else:del model,opt
        print(json.dumps(result),flush=True)
    counts=[r['parameters'] for r in results];assert max(abs(x-counts[0])/counts[0] for x in counts)<.015
    vae=load_vae(args.vae,args.comfy);reader=Readout(vae)
    with torch.no_grad():
        z=torch.zeros_like(inputs['query_latents'][:,1]);rgb=reader.rgb(inputs,z,1);native_zero=float((rgb-inputs['query_base_rgb'][:,1]).abs().max());assert native_zero==0
    full.zero_grad(set_to_none=True)
    with differentiable_decoder_math():
        loss,parts,pred,a,states,details=losses(full,inputs,targets);rgb=reader.rgb(inputs,pred[:,3],3);rgb_loss=F.l1_loss(rgb,targets['rgb'][:,3]);total=loss+5*rgb_loss;total.backward()
    true_grad={branch:sum(float(p.grad.abs().sum()) for name,p in full.named_parameters() if name.startswith(branch) and p.grad is not None) for branch in ['encoder','decoder','core','appearance']};assert all(x>0 for x in true_grad.values());assert all(p.grad is None for p in vae.parameters());report=dict(passed=True,source=item['id'],kinds=results,native_zero_RGB_maxabs=native_zero,actual_native_D_backward_RGB_L1=float(rgb_loss),actual_D_total_gradient_L1=true_grad,frozen_VAE_weight_gradients=0,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),scope='Actual train cache and actual frozen H3 D; checks do not train a selected checkpoint');out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
