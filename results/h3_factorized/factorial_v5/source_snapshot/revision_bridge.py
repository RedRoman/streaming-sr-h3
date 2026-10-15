"""B5 causal2x2 diagnosis, fixed observed pixel bridge + learned residual.

The pixel bridge and nullspace correction are established constructions. This
module tests information pathways, not a novelty claim.
"""
import torch
from torch import nn
from revision_model import RevisionMemory,FieldNet
from revision_dense import freeze_cancelling_biases,auxiliary
from model import flatten_time,restore_time
from revision_classical import ClassicalRevision
from native_vae import decode_last

BRIDGE_ARMS=['anchor_skip0','noanchor_skip0','anchor_skip1','noanchor_skip1']


class BridgeMemory(RevisionMemory):
    def __init__(self,anchor_condition=True):
        super().__init__('closed_action');self.anchor_condition=anchor_condition
        if not anchor_condition:
            target=sum(p.numel() for p in self.decoder.parameters());widths=[]
            for w in range(32,97):
                # Conv3x3, Conv3x3, Conv1x1 with their actual biases.
                count=17*w*9+w+w*w*9+w+24*w+24;widths.append((abs(count-target),w))
            self.decoder=FieldNet(17,24,min(widths)[1])
        nn.init.zeros_(self.decoder.net[-1].weight);nn.init.zeros_(self.decoder.net[-1].bias);freeze_cancelling_biases(self)
    def read(self,h,anchor,a,confidence):
        x,shape=flatten_time(h);b,c,t,hh,ww=shape;q=a[:,None,:,None,None].expand(-1,t,-1,hh,ww).reshape(b*t,12,hh,ww);w=confidence.expand(-1,1,t,-1,-1).permute(0,2,1,3,4).reshape(b*t,1,hh,ww)
        if self.anchor_condition:
            z,_=flatten_time(anchor);other=[z,w,q]
        else:other=[w,q]
        out=self.decoder(torch.cat([x,*other],1))-self.decoder(torch.cat([torch.zeros_like(x),*other],1));return restore_time(2*out.tanh(),shape)


def build_bridge(arm,initial=None,seed=17):
    assert arm in BRIDGE_ARMS;torch.manual_seed(seed);model=BridgeMemory(not arm.startswith('noanchor'));model.arm=arm;model.bridge_skip=int(arm.endswith('1'))
    if initial is not None:
        state=initial['model'];own=model.state_dict()
        for name,value in state.items():
            if not name.startswith('decoder.'):own[name]=value
        model.load_state_dict(own)
    return model


def attach_native(inputs,item):
    """All arms use the saved native full/proxy, not rounded old half-delta."""
    native=item['classical_inputs']['packet_full_latents'][None].to(inputs['packet_proxy'].device).float();inputs['packet_native']=native;inputs['packet_delta']=native-inputs['packet_proxy'];return inputs


def shared_bridge(read,inputs,near_LR,decoded):
    """Frozen external causal controller; no model/GT is accepted here."""
    classic=ClassicalRevision(read);values=[]
    for qi in range(inputs['query_latents'].shape[1]):values.append(classic.decoded_atlas(inputs['query_base_rgb'][:,qi],decoded,classic.packets(inputs,near_LR,qi)))
    return torch.stack(values,1).detach()


def bridge_rgb(read,inputs,delta,qi,base,skip):
    current=inputs['query_base_rgb'][:,qi];z=inputs['query_latents'][:,qi];value=decode_last(read.vae,z+delta);return (base[:,qi] if skip else current)+read.historical_band(value-current,inputs,qi)
