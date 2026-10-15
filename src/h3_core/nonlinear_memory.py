"""Evidence-driven nonlinear coordinates, trained through the frozen H3 decoder.

The cached decoder derivative supplies LR evidence features only. It does not
define the nonlinear RGB readout or restrict learned residuals to an artificially
small local Gaussian covariance. No exact latent nullspace is assumed.
"""
import torch
from torch import nn


def symmetric_power(matrix,power):
    values,vectors=torch.linalg.eigh(.5*(matrix+matrix.transpose(-1,-2)))
    return (vectors*values.clamp_min(1e-6).pow(power)[...,None,:])@vectors.transpose(-1,-2)


class NonlinearMemoryHead(nn.Module):
    def __init__(self,kind='transport',rank=8,context=64,hidden=28,initialize=False):
        super().__init__();self.kind=kind;self.rank=rank;self.hidden=hidden;self.initialize=initialize
        # Every head receives identical all-history likelihood and anchor context.
        features=context+rank*4+2
        if kind=='gru':
            self.cell=nn.GRUCell(features,hidden);self.output=nn.Linear(hidden,rank)
        else:
            self.net=nn.Sequential(nn.Linear(features,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,rank))
        final=self.output if kind=='gru' else self.net[-1]
        nn.init.zeros_(final.weight);nn.init.zeros_(final.bias)

    def forward(self,matrix,prior,context,grams,rhs,counts,noise,scale,already_incremental=False):
        batch=matrix.shape[0];eye=torch.eye(self.rank,device=matrix.device)[None].expand(batch,-1,-1)
        old_gram=torch.zeros_like(eye);old_mean=prior;state=prior+noise
        old_count=torch.zeros_like(counts[:,0]);hidden=torch.zeros(batch,self.hidden,device=matrix.device)
        output=[];means=[];gains=[]
        # Batched likelihood algebra is causal: each row sees only its own
        # prefix statistics. This avoids 39 tiny solver launches per head/step.
        if self.kind=='current_only' and not already_incremental:
            grams=grams-torch.cat([torch.zeros_like(grams[:,:1]),grams[:,:-1]],1)
            rhs=rhs-torch.cat([torch.zeros_like(rhs[:,:1]),rhs[:,:-1]],1)
        gram_all=matrix[:,None].transpose(-1,-2)@grams@matrix[:,None]
        natural_all=(matrix[:,None].transpose(-1,-2)@rhs[...,None])[...,0]
        precision_all=gram_all+eye[:,None]
        mean_all=torch.linalg.solve(precision_all,(natural_all+prior[:,None])[...,None])[...,0]
        previous=torch.cat([torch.zeros_like(gram_all[:,:1]),gram_all[:,:-1]],1)
        if self.kind=='current_only':previous=torch.zeros_like(previous)
        with torch.no_grad():
            delta_all=gram_all.detach()-previous.detach()
            delta_all=.5*(delta_all+delta_all.transpose(-1,-2))
            invsqrt_all=symmetric_power(precision_all.detach(),-.5)
            gain_all,vectors_all=torch.linalg.eigh(invsqrt_all@delta_all@invsqrt_all)
            gain_all=gain_all.clamp(0,1)
            eigenlog_all=torch.linalg.eigvalsh(gram_all.detach()).clamp_min(0).log1p()/10
        for t in range(grams.shape[1]):
            gram=gram_all[:,t];mean=mean_all[:,t]
            # A relative precision increment is bounded in [0,1]. It accounts
            # for fresh noisy repeated views, not only a binary coverage mask.
            gain=gain_all[:,t];vectors=vectors_all[:,t]
            identity=(state-old_mean)/scale if self.kind=='transport' else noise/scale
            innovation=(mean-old_mean)/scale if self.kind in ('transport','gru') else torch.zeros_like(mean)
            features=torch.cat([context,mean/scale,identity,
                                innovation,eigenlog_all[:,t],
                                torch.log1p(counts[:,t].float())[:,None]/4,gain.mean(-1,keepdim=True)],1)
            if self.kind=='gru':
                proposed_hidden=self.cell(features,hidden)
                hidden=torch.where((counts[:,t]>old_count)[:,None],proposed_hidden,hidden)
                correction=self.output(hidden)
            else:correction=self.net(features)
            proposed=mean+scale*correction
            if self.kind=='transport':
                # Preserve the old nonlinear residual while new evidence revises
                # directions in proportion to its relative precision increment.
                residual=state-old_mean
                difference=proposed-(mean+residual)
                revision=(vectors@(gain[...,None]*(vectors.transpose(1,2)@difference[...,None])))[...,0]
                new_state=mean+residual+revision
            else:new_state=proposed
            fresh=counts[:,t]>old_count
            if self.initialize:new_state=torch.where((old_count==0)[:,None],proposed+noise,new_state)
            state=torch.where(fresh[:,None],new_state,state)
            output.append(state);means.append(mean);gains.append(gain)
            old_gram=gram;old_mean=mean;old_count=counts[:,t]
        return torch.stack(output,1),{'mean':torch.stack(means,1),'relative_information_gain':torch.stack(gains,1)}
