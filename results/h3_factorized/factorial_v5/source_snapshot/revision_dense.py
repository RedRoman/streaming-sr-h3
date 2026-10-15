"""Decoder-aware B3R protocol and a parameter-matched current-image control."""
import torch
from torch.nn import functional as F
from revision_model import RevisionMemory,build,KINDS,losses
from model import flatten_time,restore_time

DENSE_KINDS=[*KINDS,'anchor_only']


def geometry_mass(inputs,step):
    masks=inputs['packet_masks'];arrived=(inputs['packet_steps']<=step).to(masks.dtype)
    return (masks*arrived[:,:,None,None,None]).sum(1).unsqueeze(2)


class AnchorOnly(RevisionMemory):
    """Current native code and query metadata only; no packet image/code read.

Shared post-decoding reliability/support is outside this neural model and is
reported as shared historical controller, not as a completely memoryless system.
"""
    def __init__(self):super().__init__('closed_action');self.kind='anchor_only'
    def current_lift(self,z):
        flat,shape=flatten_time(z);smooth=restore_time(F.avg_pool2d(flat,3,stride=1,padding=1,count_include_pad=False),shape);mask=torch.ones(len(z),1,16,16,device=z.device);return self.lift(z-smooth,smooth,mask)
    def rollout(self,inputs,return_details=False):
        outputs=[];appearances=[];features=[];info=[]
        for qi,step in enumerate(inputs['query_steps'][0].tolist()):
            z=inputs['query_latents'][:,qi];h=self.current_lift(z);mass=geometry_mass(inputs,step);a=torch.zeros(len(z),12,device=z.device) if int(inputs['query_groups'][0,qi])<0 else self.appearance_value(inputs['query_features'][:,qi]);styled=self.core.style(h,a)*(mass>0);outputs.append(self.read(styled,z,a,(mass/8).clamp(0,1)));appearances.append(a);features.append(h);info.append(dict(step=step,feature_RMS=h.square().mean().sqrt(),evidence_mass=mass.mean(),active_factor_count=0))
        states=[torch.zeros_like(features[0]) for _ in range(18)];result=(torch.stack(outputs,1),torch.stack(appearances,1),states)
        if return_details:return (*result,dict(packet_features=torch.stack(features,1),query_features=torch.stack(features,1),information=info))
        return result


def freeze_cancelling_biases(model):
    """Algebraically cancelled final biases are not counted as active parameters."""
    frozen=[]
    for name,p in model.named_parameters():
        cancel=(name=='decoder.net.4.bias' or (name.startswith('core.') and name.endswith('net.4.bias')))
        if cancel:p.requires_grad_(False);frozen.append(name)
    return frozen


def build_dense(kind):
    model=AnchorOnly() if kind=='anchor_only' else build(kind);freeze_cancelling_biases(model);return model


def auxiliary(model,inputs,targets):
    if model.kind!='anchor_only':
        _,parts,pred,a,states,details=losses(model,inputs,targets)
        return parts['appearance'],parts['equivariance'],pred,a,states,details
    pred,a,states,details=model.rollout(inputs,True)
    # Past controller metadata is used only by the common training supervision
    # and physical loss, never by the current-image neural readout.
    valid=inputs['group_features'].abs().sum(-1)>0;groupa=model.appearance_value(inputs['group_features'])*valid[:,:,:,None];details['group_appearance']=groupa
    appearance=F.mse_loss(a,targets['query_q']);truth=targets['q_groups'][:,None].expand(-1,18,-1,-1);appearance+=((groupa-truth).square()*valid[:,:,:,None]).sum()/(valid.sum()*12).clamp_min(1)
    neutral=targets['query_neutral_latents'];b,n=neutral.shape[:2];hn=model.current_lift(neutral.flatten(0,1));styled=model.core.style(hn,targets['query_q'].flatten(0,1));equivariance=F.l1_loss(styled,details['packet_features'].flatten(0,1))
    return appearance,equivariance,pred,a,states,details
