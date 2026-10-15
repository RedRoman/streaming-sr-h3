"""Conditional reversible H3 identity code and LR-evidence-only state writes.

All models have24-channel persistent state and the same LR appearance encoder
and update head. Only the canonical/read chart differs. No inference HR inputs.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

APPEARANCE=12


def flatten_time(z):
    b,c,t,h,w=z.shape
    return z.permute(0,2,1,3,4).reshape(b*t,c,h,w),(b,c,t,h,w)


def restore_time(x,shape):
    b,c,t,h,w=shape
    return x.reshape(b,t,x.shape[1],h,w).permute(0,2,1,3,4)


class ConditionalNet(nn.Module):
    def __init__(self,inc,outc,width,appearance=APPEARANCE):
        super().__init__()
        self.first=nn.Conv2d(inc,width,3,padding=1)
        self.condition=nn.Linear(appearance,2*width,bias=False)
        self.last=nn.Conv2d(width,outc,3,padding=1)
        # Keep the neutral chart exactly identity by subtraction in forward,
        # but do not zero this end as well as AppearanceEncoder.head: that
        # would give both branches zero gradient at a=0.
        nn.init.normal_(self.last.weight,std=.005);nn.init.zeros_(self.last.bias)

    def forward(self,x,a,zero_at_neutral=True):
        h=F.silu(self.first(x))
        gain,bias=self.condition(a).chunk(2,1)
        out=self.last(F.silu(h*(1+.2*gain.tanh()[:,:,None,None])+bias[:,:,None,None]))
        if zero_at_neutral:out=out-self.last(F.silu(h))
        return out


class ConditionalCoupling(nn.Module):
    def __init__(self,width=40,flip=False):
        super().__init__();self.net=ConditionalNet(12,24,width);self.flip=flip

    def forward(self,z,a,inverse=False):
        if self.flip:z=z.flip(1)
        left,right=z.chunk(2,1)
        scale,shift=self.net(left,a).chunk(2,1)
        scale=.20*scale.tanh();shift=.40*shift
        right=(right-shift)*torch.exp(-scale) if inverse else right*torch.exp(scale)+shift
        z=torch.cat((left,right),1)
        return z.flip(1) if self.flip else z


class ReversibleChart(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([ConditionalCoupling(flip=bool(i%2)) for i in range(4)])

    def transform(self,z,a,inverse=False):
        flat,shape=flatten_time(z);aa=a[:,None].expand(-1,shape[2],-1).reshape(-1,APPEARANCE)
        for layer in (reversed(self.layers) if inverse else self.layers):flat=layer(flat,aa,inverse)
        return restore_time(flat,shape)

    def canonicalize(self,z,a):return self.transform(z,a,True)
    def read(self,c,a):return self.transform(c,a,False)


class NoninvertibleChart(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=ConditionalNet(24,24,61);self.decoder=ConditionalNet(24,24,61)

    def apply_net(self,z,a,net):
        flat,shape=flatten_time(z);aa=a[:,None].expand(-1,shape[2],-1).reshape(-1,APPEARANCE)
        return restore_time(flat+.4*net(flat,aa),shape)

    def canonicalize(self,z,a):return self.apply_net(z,a,self.encoder)
    def read(self,c,a):return self.apply_net(c,a,self.decoder)


class MixedChart(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([ConditionalNet(24,24,30) for _ in range(4)])

    def canonicalize(self,z,a):return z

    def read(self,c,a):
        flat,shape=flatten_time(c);aa=a[:,None].expand(-1,shape[2],-1).reshape(-1,APPEARANCE)
        for layer in self.layers:flat=flat+.1*layer(flat,aa,False)
        return restore_time(flat,shape)


class AppearanceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(6,16,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Linear(48,APPEARANCE)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,reference,current):
        x=F.interpolate(torch.cat((reference,current),1),size=(64,64),mode='area')
        return .5*self.head(self.net(x).flatten(1)).tanh()


class EvidenceUpdate(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(24*2+3+APPEARANCE,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,48,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,old,observed,a,visible,reveal,footprint):
        o,shape=flatten_time(old);z,_=flatten_time(observed)
        b,c,t,h,w=shape
        visible=F.interpolate(visible,size=(h,w),mode='area')
        reveal=F.interpolate(reveal,size=(h,w),mode='area')
        channels=torch.cat((visible,reveal,torch.full_like(visible,footprint)),1)
        channels=channels[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,3,h,w)
        aa=a[:,None,:,None,None].expand(-1,t,-1,h,w).reshape(b*t,APPEARANCE,h,w)
        delta,gate=self.net(torch.cat((o,z,channels,aa),1)).chunk(2,1)
        visibility=channels[:,:1]
        # Actual train-only H3 codes have SR residual RMS about1; a .2 bound
        # would prevent learning their correction. Shared2.0 bound for every
        # candidate/control, with no target-dependent inference scaling.
        return restore_time(o+visibility*gate.sigmoid()*(z-o+2.0*delta.tanh()),shape)


@dataclass
class IdentityState:
    content:torch.Tensor
    reference:torch.Tensor
    seen:frozenset


class FactorizedMemory(nn.Module):
    def __init__(self,kind='factorized_swap'):
        super().__init__();self.kind=kind
        self.appearance=AppearanceEncoder();self.update_head=EvidenceUpdate()
        self.chart=MixedChart() if kind=='mixed_memory' else NoninvertibleChart() if kind=='noninvertible_swap' else ReversibleChart()

    def begin(self,z_observed,reference,packet_id):
        a=torch.zeros((reference.shape[0],APPEARANCE),device=reference.device)
        return IdentityState(self.chart.canonicalize(z_observed,a),reference,frozenset([packet_id]))

    def observe(self,state,z_observed,rgb_observed,packet_id,visible,reveal,footprint):
        a=self.appearance(state.reference,rgb_observed)
        if packet_id in state.seen or not bool(visible.any()):return state,a
        canonical=self.chart.canonicalize(z_observed,a)
        content=self.update_head(state.content,canonical,a,visible,reveal,footprint)
        return IdentityState(content,state.reference,state.seen|{packet_id}),a

    def query(self,state,appearance):
        # Pure read: no packet, precision, reference or content mutation.
        return self.chart.read(state.content,appearance)

    def rollout(self,inputs):
        obs=inputs['observation_latents'];rgb=inputs['observation_rgb']
        state=self.begin(obs[:,0],rgb[:,0],'far_A0')
        zero=torch.zeros((obs.shape[0],APPEARANCE),device=obs.device)
        outputs=[self.query(state,zero)];appearances=[zero];states=[state.content]
        # Stage3 is a query without visibility; stage7 repeats stage6's packet.
        ids=['far_A0','far_A1','near_A1','query','return_A0','near_B0','far_B2','far_B2']
        for step in range(1,8):
            state,a=self.observe(state,obs[:,step],rgb[:,step],ids[step],inputs['visible'][:,step],inputs['reveal'][:,step],float(inputs['footprint'][step]))
            outputs.append(self.query(state,a));appearances.append(a);states.append(state.content)
        return torch.stack(outputs,1),torch.stack(appearances,1),states


KINDS=['factorized_swap','factorized_no_swap','noninvertible_swap','mixed_memory']


def counterfactual_loss(model,targets):
    if model.kind=='factorized_no_swap':return targets['latents'].new_zeros(())
    # GT latents/appearance parameters are training supervision only. They are
    # absent from observe/query, which only consume actual LR cache tensors.
    z=targets['latents'];q=targets['appearance']
    ca=model.chart.canonicalize(z[:,1],q[:,1])
    cb=model.chart.canonicalize(z[:,5],q[:,5])
    swap_a=model.chart.read(ca,q[:,2]);swap_b=model.chart.read(cb,q[:,4])
    return F.l1_loss(swap_a,z[:,2])+F.l1_loss(swap_b,z[:,4])+.5*(F.l1_loss(ca,z[:,0])+F.l1_loss(cb,z[:,3]))
