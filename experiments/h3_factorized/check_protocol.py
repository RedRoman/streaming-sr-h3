"""Small correctness audit, not a performance benchmark."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from model import FactorizedMemory,KINDS,counterfactual_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(713)
    batch=2;h=4;t=2
    inputs=dict(observation_latents=torch.randn(batch,8,24,t,h,h),observation_rgb=torch.rand(batch,8,3,64,64),visible=torch.ones(batch,8,1,64,64),reveal=torch.zeros(batch,8,1,64,64),footprint=torch.tensor([.25,.25,.5,.5,.25,.5,.25,.25]))
    inputs['visible'][:,3]=0;inputs['observation_latents'][:,7]=inputs['observation_latents'][:,6];inputs['observation_rgb'][:,7]=inputs['observation_rgb'][:,6]
    targets=dict(latents=torch.randn(batch,6,24,t,h,h),appearance=torch.randn(batch,6,12)*.1)
    results=[]
    for kind in KINDS:
        model=FactorizedMemory(kind);opt=torch.optim.Adam(model.parameters(),lr=.002)
        before={n:p.detach().clone() for n,p in model.named_parameters()};steps=[]
        for step in range(2):
            opt.zero_grad();out,a,states=model.rollout(inputs)
            loss=F.mse_loss(out,inputs['observation_latents']+.05)+F.mse_loss(a,torch.ones_like(a)*.1)+counterfactual_loss(model,targets)
            loss.backward()
            grads={branch:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(branch) and p.grad is not None) for branch in ['appearance','chart','update_head']}
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();steps.append(dict(step=step,loss=float(loss.detach()),grad_l1=grads))
        deltas={branch:sum(float((p.detach()-before[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(branch)) for branch in ['appearance','chart','update_head']}
        assert all(v>0 for v in deltas.values()),(kind,deltas)
        assert all(steps[-1]['grad_l1'][b]>0 for b in deltas),(kind,steps)
        out,a,states=model.rollout(inputs)
        assert torch.equal(states[2],states[3]),'No-visible query mutated memory'
        assert torch.equal(states[6],states[7]),'Duplicate packet mutated memory'
        assert torch.equal(out[:,6],out[:,7]),'Duplicate packet changed readout'
        inverse=None
        if kind.startswith('factorized'):
            z=torch.randn(batch,24,t,h,h);q=torch.randn(batch,12)*.2
            inverse=float((model.chart.read(model.chart.canonicalize(z,q),q)-z).abs().max().detach())
            assert inverse<2e-5,inverse
        results.append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),steps=steps,parameter_update_l1=deltas,inverse_max_abs=inverse,query_read_only=True,duplicate_idempotent=True))
    result=dict(passed=True,seed=713,torch=torch.__version__,scope='Synthetic gradient/protocol correctness only; no image-quality result',results=results)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
