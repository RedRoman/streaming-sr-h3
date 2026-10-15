"""Native H3 reference/cue adapter with explicit revisable content state.

Physical support protection is a separate common classical operation. This
module makes no exact latent-nullspace claim. Every output has the same native
condition-token count. Zero content state exactly reproduces the reference
guide; a copied packet or a far-only noise observation cannot write fine state.
"""
from dataclasses import dataclass,field

import torch
from torch import nn


class NativeProposalCondition(nn.Module):
    def __init__(self,code=32,hidden=32):
        super().__init__();self.code=code
        self.update=nn.Sequential(nn.Linear(code+24+3*8*8+4,64),nn.SiLU(),nn.Linear(64,code))
        nn.init.zeros_(self.update[-1].weight);nn.init.zeros_(self.update[-1].bias)
        self.read=nn.Sequential(nn.Conv3d(48+code,hidden,(1,3,3),padding=(0,1,1)),nn.SiLU(),nn.Conv3d(hidden,24,(1,3,3),padding=(0,1,1)))
    def transition(self,previous,reference,innovation,geometry,new_fine_evidence=True):
        if not new_fine_evidence:return previous
        features=torch.cat([previous,reference.float().mean((2,3,4)),innovation.float().flatten(1),geometry.float()],1)
        return previous+self.update(features)
    def condition(self,reference,current,state):
        assert reference.shape==current.shape and reference.shape[1:]==(24,7,16,16)
        shape=(state.shape[0],self.code,*reference.shape[2:]);expanded=state[:,:,None,None,None].expand(shape)
        common=torch.cat([reference.float(),current.float()],1)
        # Shared readout subtraction gives exact zero-state identity while its
        # state Jacobian is nonzero, allowing the zero-initialized update to learn.
        correction=self.read(torch.cat([common,expanded],1))-self.read(torch.cat([common,torch.zeros_like(expanded)],1))
        return reference.float()+correction


@dataclass
class CausalProposalState:
    reference:torch.Tensor
    code:torch.Tensor
    packet_ids:set=field(default_factory=set)
    last_index:int=-1
    def observe(self,module,packet_id,index,innovation,geometry,new_fine_evidence):
        if packet_id in self.packet_ids:return False
        if index<self.last_index:raise ValueError('A new packet cannot arrive with an unavailable earlier chronology')
        self.packet_ids.add(packet_id);self.last_index=index
        self.code=module.transition(self.code,self.reference,innovation,geometry,new_fine_evidence)
        return bool(new_fine_evidence)
