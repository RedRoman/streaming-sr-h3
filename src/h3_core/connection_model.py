"""Matched causal/invariant endpoint heads in shared immutable coordinates.

U is computed from the initial actual H3 proposal, never claimed to identify
semantic independence. Private latent inheritance is shared by every head.
Only the proposed head's TRAINING loss differs from basic sequential transport.
"""
import torch
from torch import nn


class ConnectionHead(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.phi=nn.Conv3d(5,8,1)
        self.local=nn.Conv3d(37,32,(1,3,3),padding=(0,1,1))
        self.global_context=nn.Linear(37,32)
        self.output=nn.Conv3d(32,24,(1,3,3),padding=(0,1,1))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def step(self,previous,initial,basis,responses,cue_features,current,arrived):
        subset='AB' if len(arrived)==2 else arrived[0]
        response=responses[subset]
        first=torch.einsum('ck,bcthw->bkthw',basis,response[0:1]);second=torch.einsum('ck,bcthw->bkthw',basis,response[1:2])
        ordered=sorted(arrived);aggregate=sum(cue_features[k] for k in ordered)
        if self.kind=='set':carrier=sum(self.phi(cue_features[k]) for k in ordered)
        else:carrier=previous-initial+self.phi(cue_features[current])
        features=torch.cat([initial,first,second,carrier,aggregate],1)
        hidden=torch.nn.functional.silu(self.local(features)+self.global_context(features.mean((2,3,4)))[:,:,None,None,None])
        first_gate,second_gate,proposal=self.output(hidden).chunk(3,1)
        if self.kind=='gru':
            delta=previous-initial;candidate=torch.tanh(proposal+torch.sigmoid(first_gate)*delta);return previous+torch.sigmoid(second_gate)*(candidate-delta)
        update=torch.sigmoid(first_gate)*torch.sigmoid(second_gate)*proposal
        return initial+update if self.kind=='set' else previous+update


def rollout(head,item,path):
    initial=torch.einsum('ck,bcthw->bkthw',item['basis'],item['z0']);state=initial;arrived=[];outputs={}
    for cue in path:
        if cue in arrived:continue  # copied packet IDs are a common input rule
        arrived.append(cue);state=head.step(state,initial,item['basis'],item['responses'],item['cue_features'],cue,arrived)
        key=''.join(arrived);outputs[key]=state
    return initial,outputs


def embed(item,state,initial):
    return item['z0']+torch.einsum('ck,bkthw->bcthw',item['basis'],state-initial)
